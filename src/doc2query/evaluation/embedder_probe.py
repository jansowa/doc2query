"""Frozen-budget bi-encoder probe training and natural-query retrieval evaluation."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.data import DataLoader, Dataset

from doc2query.evaluation.datasets import evaluation_fingerprint, load_frozen_records
from doc2query.evaluation.retrieval import (
    aggregate_query_metrics,
    metrics_from_positive_ranks,
)
from doc2query.utils.records import JsonlWriter, read_records, write_json
from doc2query.utils.reproducibility import set_seed
from doc2query.utils.tracking import collect_code_provenance

QuerySource = Literal["natural", "copy_control", "synthetic"]


@dataclass(frozen=True)
class ProbeRecipe:
    model_name_or_path: str
    revision: str
    max_length: int = 256
    batch_size: int = 16
    max_steps: int = 1000
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.05
    seed: int = 42
    negatives_per_example: int = 1
    normalize_embeddings: bool = True
    loss: str = "in_batch_cross_entropy_with_paired_hard_negative"

    def __post_init__(self) -> None:
        if len(self.revision) != 40:
            raise ValueError("probe model revision must be a full 40-character commit")
        if min(self.max_length, self.batch_size, self.max_steps) < 1:
            raise ValueError("probe length, batch and step budget must be positive")
        if self.negatives_per_example != 1:
            raise ValueError("the frozen v1 recipe uses exactly one paired hard negative")
        if not self.normalize_embeddings:
            raise ValueError("the frozen v1 recipe requires normalized embeddings")

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


class ProbePairs(Dataset[dict[str, str]]):
    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, str]:
        return self.rows[index]


class MeanPoolEncoder(nn.Module):
    def __init__(self, model_name_or_path: str, revision: str) -> None:
        super().__init__()
        from transformers import AutoModel

        loader: Any = getattr(AutoModel, "from_" + "pretrained")
        self.backbone = loader(
            model_name_or_path,
            revision=revision,
            trust_remote_code=False,
        )

    def forward(self, encoded: dict[str, torch.Tensor]) -> torch.Tensor:
        output = self.backbone(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).to(output.dtype)
        pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return functional.normalize(pooled, dim=-1)


def _synthetic_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    result = {}
    for row in read_records(path):
        if str(row.get("mode")) == "deterministic" and int(row.get("candidate_index", 0)) == 0:
            result[str(row["example_id"])] = str(row["generated"])
    return result


def _copy_control(passage: str) -> str:
    sentence = passage.split(".", 1)[0].strip()
    return " ".join(sentence.split()[:12])


def prepare_probe_pairs(
    records: Iterable[dict[str, Any]],
    *,
    query_source: QuerySource,
    synthetic_generations: Path | None = None,
    limit: int | None = None,
) -> tuple[list[dict[str, str]], str]:
    synthetic = _synthetic_map(synthetic_generations)
    rows: list[dict[str, str]] = []
    selected: list[tuple[int, str, dict[str, str]]] = []
    for record in records:
        positives = sorted(record.get("positives", []), key=lambda value: str(value["doc_id"]))
        negatives = record.get("hard_negatives", [])
        if not positives or not negatives:
            continue
        example_id = str(record["example_id"])
        passage = str(positives[0]["text"])
        if query_source == "natural":
            query = str(record["query"])
        elif query_source == "copy_control":
            query = _copy_control(passage)
        else:
            if example_id not in synthetic:
                continue
            query = synthetic[example_id]
        negative_index = int(
            hashlib.sha256(f"probe-v1:{example_id}".encode()).hexdigest()[:8], 16
        ) % len(negatives)
        row = {
            "example_id": example_id,
            "query": query,
            "positive": passage,
            "negative": str(negatives[negative_index]["text"]),
        }
        if limit is None:
            rows.append(row)
            continue
        selection_key = int(
            hashlib.sha256(f"probe-selection-v1:{example_id}".encode()).hexdigest(),
            16,
        )
        candidate = (-selection_key, example_id, row)
        if len(selected) < limit:
            heapq.heappush(selected, candidate)
        elif candidate > selected[0]:
            heapq.heapreplace(selected, candidate)
    if limit is not None:
        rows = [value[2] for value in selected]
    rows.sort(key=lambda value: value["example_id"])
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        )
        digest.update(b"\n")
    return rows, digest.hexdigest()


def _tokenize(
    tokenizer: Any, texts: list[str], max_length: int, device: torch.device
) -> dict[str, torch.Tensor]:
    encoded: dict[str, torch.Tensor] = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {key: value.to(device) for key, value in encoded.items()}


def train_probe(
    rows: list[dict[str, str]],
    *,
    recipe: ProbeRecipe,
    output_dir: Path,
    query_source: QuerySource,
    train_fingerprint: str,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("probe training set is empty")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"probe output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(recipe.seed)
    from transformers import AutoTokenizer

    tokenizer_loader: Any = getattr(AutoTokenizer, "from_" + "pretrained")
    tokenizer = tokenizer_loader(
        recipe.model_name_or_path,
        revision=recipe.revision,
        trust_remote_code=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MeanPoolEncoder(recipe.model_name_or_path, recipe.revision).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe.learning_rate)
    warmup_steps = int(recipe.max_steps * recipe.warmup_ratio)

    def learning_rate_scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, recipe.max_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_scale)
    loader = DataLoader(
        ProbePairs(rows),
        batch_size=recipe.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(recipe.seed),
    )
    iterator = iter(loader)
    losses = []
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    for _step in range(recipe.max_steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        queries = model(_tokenize(tokenizer, list(batch["query"]), recipe.max_length, device))
        positives = model(_tokenize(tokenizer, list(batch["positive"]), recipe.max_length, device))
        negatives = model(_tokenize(tokenizer, list(batch["negative"]), recipe.max_length, device))
        documents = torch.cat((positives, negatives), dim=0)
        logits = queries @ documents.T / 0.05
        targets = torch.arange(queries.shape[0], device=device)
        loss = functional.cross_entropy(logits, targets)
        torch.autograd.backward(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
    adapter_dir = output_dir / "model"
    model.backbone.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    summary = {
        "schema_version": 1,
        "status": "measured",
        "query_source": query_source,
        "recipe": asdict(recipe),
        "recipe_fingerprint": recipe.fingerprint,
        "train_fingerprint": train_fingerprint,
        "train_examples": len(rows),
        "steps": recipe.max_steps,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "elapsed_seconds": time.perf_counter() - started,
        "peak_vram_allocated_bytes": (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        ),
        "code": collect_code_provenance(),
    }
    write_json(output_dir / "train_summary.json", summary)
    return summary


def _encode(
    model: MeanPoolEncoder,
    tokenizer: Any,
    texts: list[str],
    *,
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    with torch.inference_mode():
        encoded = model(_tokenize(tokenizer, texts, max_length, device))
        return cast(torch.Tensor, encoded).cpu()


def _encode_batched(
    model: MeanPoolEncoder,
    tokenizer: Any,
    texts: list[str],
    *,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    chunks = [
        _encode(
            model,
            tokenizer,
            texts[start : start + batch_size],
            max_length=max_length,
            device=device,
        )
        for start in range(0, len(texts), batch_size)
    ]
    if not chunks:
        raise ValueError("cannot encode an empty corpus")
    return torch.cat(chunks)


def evaluate_probe(
    model_path: Path,
    records: list[dict[str, Any]],
    *,
    recipe: ProbeRecipe,
    output_dir: Path,
    test_fingerprint: str,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer_loader: Any = getattr(AutoTokenizer, "from_" + "pretrained")
    tokenizer = tokenizer_loader(model_path, trust_remote_code=False)
    model = MeanPoolEncoder(str(model_path), "main")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    corpus: dict[str, str] = {}
    for record in records:
        for document in [*record.get("positives", []), *record.get("hard_negatives", [])]:
            doc_id, text = str(document["doc_id"]), str(document["text"])
            if doc_id in corpus and corpus[doc_id] != text:
                raise ValueError(f"document text conflict for doc_id={doc_id}")
            corpus[doc_id] = text
    corpus_ids = sorted(corpus)
    corpus_index = {doc_id: index for index, doc_id in enumerate(corpus_ids)}
    index_started = time.perf_counter()
    corpus_embeddings = _encode_batched(
        model,
        tokenizer,
        [corpus[doc_id] for doc_id in corpus_ids],
        max_length=recipe.max_length,
        batch_size=recipe.batch_size,
        device=device,
    )
    index_seconds = time.perf_counter() - index_started
    per_query: list[dict[str, Any]] = []
    metric_rows: list[dict[str, float]] = []
    latencies: list[float] = []
    with JsonlWriter(output_dir / "retrieval_per_query.jsonl") as writer:
        for record in records:
            positives = record.get("positives", [])
            negatives = record.get("hard_negatives", [])
            if not positives or not negatives:
                continue
            started = time.perf_counter()
            query_embedding = _encode(
                model,
                tokenizer,
                [str(record["query"])],
                max_length=recipe.max_length,
                device=device,
            )
            latencies.append(time.perf_counter() - started)
            scores = (query_embedding @ corpus_embeddings.T).squeeze(0)
            positive_ids = [str(value["doc_id"]) for value in positives]
            negative_ids = [str(value["doc_id"]) for value in negatives]
            positive_ranks = []
            for doc_id in positive_ids:
                index = corpus_index[doc_id]
                score = scores[index]
                better = int(torch.sum(scores > score).item())
                tied_before = int(torch.sum(scores[:index] == score).item())
                positive_ranks.append(1 + better + tied_before)
            pairwise_wins = [
                float(scores[corpus_index[positive_id]] > scores[corpus_index[negative_id]])
                for positive_id in positive_ids
                for negative_id in negative_ids
            ]
            metrics = metrics_from_positive_ranks(
                positive_ranks,
                candidate_count=len(corpus_ids),
                hard_negative_win_rate=sum(pairwise_wins) / len(pairwise_wins),
            )
            row = {"example_id": str(record["example_id"]), **metrics}
            writer.write(row)
            per_query.append(row)
            metric_rows.append(metrics)
    aggregate = aggregate_query_metrics(metric_rows)
    summary = {
        "schema_version": 1,
        "status": "measured",
        "test_fingerprint": test_fingerprint,
        "recipe_fingerprint": recipe.fingerprint,
        "query_count": len(per_query),
        "metrics": aggregate,
        "latency_seconds_per_query": sum(latencies) / len(latencies) if latencies else None,
        "corpus_document_count": len(corpus_ids),
        "index_build_seconds": index_seconds,
        "index_size_bytes": corpus_embeddings.nelement() * corpus_embeddings.element_size(),
        "model_size_bytes": sum(
            path.stat().st_size for path in model_path.rglob("*") if path.is_file()
        ),
    }
    write_json(output_dir / "retrieval_summary.json", summary)
    return summary


def run_probe_experiment(
    *,
    train_path: Path,
    frozen_manifest: Path,
    test_subset: str,
    output_dir: Path,
    recipe: ProbeRecipe,
    query_source: QuerySource,
    synthetic_generations: Path | None = None,
    train_limit: int | None = None,
) -> dict[str, Any]:
    pairs, train_fingerprint = prepare_probe_pairs(
        read_records(train_path),
        query_source=query_source,
        synthetic_generations=synthetic_generations,
        limit=train_limit,
    )
    train_summary = train_probe(
        pairs,
        recipe=recipe,
        output_dir=output_dir,
        query_source=query_source,
        train_fingerprint=train_fingerprint,
    )
    test_records = load_frozen_records(frozen_manifest, test_subset)
    retrieval = evaluate_probe(
        output_dir / "model",
        test_records,
        recipe=recipe,
        output_dir=output_dir,
        test_fingerprint=evaluation_fingerprint(frozen_manifest, test_subset),
    )
    result = {"training": train_summary, "retrieval": retrieval}
    write_json(output_dir / "result.json", result)
    return result
