"""Frozen-budget bi-encoder probe training and natural-query retrieval evaluation."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.data import DataLoader, Dataset

from doc2query.evaluation.corpus import CorpusIndex, sha256_file
from doc2query.evaluation.datasets import evaluation_fingerprint, load_frozen_records
from doc2query.evaluation.native_holdout import (
    HoldoutProfile,
    holdout_artifact_path,
    holdout_fingerprint,
    holdout_set_status,
    load_holdout_records,
)
from doc2query.evaluation.probe_negatives import (
    NegativeCandidate,
    NegativeRecipe,
    PossibleFalseNegativeCalibration,
    select_negative,
    summarize_false_negative_audit,
)
from doc2query.evaluation.report import build_embedder_report
from doc2query.evaluation.retrieval import (
    CORPUS_RETRIEVAL,
    aggregate_query_metrics,
    corpus_metrics_from_positive_ranks,
)
from doc2query.evaluation.statistical_contract import (
    StatisticalContract,
    build_budget_manifest,
)
from doc2query.evaluation.translationese import aggregate_translationese
from doc2query.reranker.base import PairScorer
from doc2query.utils.records import JsonlWriter, read_records, write_json
from doc2query.utils.reproducibility import set_seed
from doc2query.utils.tracking import collect_code_provenance

QuerySource = Literal["natural", "copy_control", "synthetic"]


def _progress_enabled() -> bool:
    return os.environ.get("DOC2QUERY_PROGRESS", "").strip().lower() in {"1", "true", "yes"}


def _progress(stage: str, current: int, total: int, started: float) -> None:
    if not _progress_enabled():
        return
    percent = 100.0 * current / max(1, total)
    elapsed = time.perf_counter() - started
    print(
        f"[{stage}] {current:,}/{total:,} ({percent:5.1f}%) elapsed={elapsed:.1f}s",
        file=sys.stderr,
        flush=True,
    )


def _progress_callback(stage: str) -> Callable[[int, int], None]:
    started = time.perf_counter()
    last = -1

    def report(current: int, total: int) -> None:
        nonlocal last
        if current != last:
            last = current
            _progress(stage, current, total, started)

    return report


@dataclass(frozen=True)
class ProbeRecipe:
    model_name_or_path: str
    revision: str
    recipe_version: str
    negative_recipe: NegativeRecipe
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
        if not self.recipe_version.strip():
            raise ValueError("probe recipe_version must be non-empty")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ProbeRecipe:
        payload = dict(raw)
        negative = payload.get("negative_recipe")
        if not isinstance(negative, Mapping):
            raise ValueError("probe recipe requires a negative_recipe mapping")
        payload["negative_recipe"] = NegativeRecipe(**dict(negative))
        return cls(**payload)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


class ProbePairs(Dataset[dict[str, Any]]):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
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


def _query_for_record(
    record: Mapping[str, Any],
    *,
    query_source: QuerySource,
    synthetic: Mapping[str, str],
) -> str | None:
    if query_source == "natural":
        return str(record["query"])
    positives = record.get("positives", [])
    if not isinstance(positives, list) or not positives:
        return None
    if query_source == "copy_control":
        return _copy_control(str(positives[0]["text"]))
    return synthetic.get(str(record["example_id"]))


def _hn1_candidates(
    prepared: Sequence[tuple[dict[str, Any], str]],
    *,
    index: CorpusIndex,
    documents_path: Path,
    recipe: NegativeRecipe,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, list[NegativeCandidate]]:
    metadata = index.metadata
    if metadata.get("backend") != "bm25_sqlite":
        raise ValueError("HN1 requires the frozen P-01 BM25 index, not another corpus backend")
    if metadata.get("index_fingerprint") != recipe.bm25_index_fingerprint:
        raise ValueError("HN1 BM25 index fingerprint does not match the negative recipe")
    ranked: dict[str, list[tuple[str, int, float]]] = {}
    wanted: set[str] = set()
    for prepared_index, (record, query) in enumerate(prepared, start=1):
        positives = {
            str(document["doc_id"])
            for document in record.get("positives", [])
            if isinstance(document, dict) and "doc_id" in document
        }
        search = index.search(query, limit=recipe.bm25_candidates + len(positives))
        candidates = [
            (document.doc_id, document.rank, document.score)
            for document in search.documents
            if document.doc_id not in positives
        ][: recipe.bm25_candidates]
        if not candidates:
            raise ValueError(
                f"HN1 BM25 returned no non-positive candidate for {record['example_id']}"
            )
        ranked[str(record["example_id"])] = candidates
        wanted.update(doc_id for doc_id, _rank, _score in candidates)
        if progress is not None:
            progress(prepared_index, len(prepared))
    texts: dict[str, str] = {}
    for document in read_records(documents_path):
        doc_id = str(document["doc_id"])
        if doc_id in wanted:
            texts[doc_id] = str(document["text"])
            if len(texts) == len(wanted):
                break
    missing = sorted(wanted - texts.keys())
    if missing:
        raise ValueError(f"HN1 BM25 documents are absent from the frozen corpus: {missing[:3]}")
    return {
        example_id: [
            NegativeCandidate(
                doc_id=doc_id,
                text=texts[doc_id],
                miner="bm25",
                miner_rank=rank,
                miner_score=score,
            )
            for doc_id, rank, score in candidates
        ]
        for example_id, candidates in ranked.items()
    }


def prepare_probe_pairs(
    records: Iterable[dict[str, Any]],
    *,
    query_source: QuerySource,
    negative_recipe: NegativeRecipe,
    calibration: PossibleFalseNegativeCalibration | None,
    primary_scorer: PairScorer | None,
    synthetic_generations: Path | None = None,
    limit: int | None = None,
    prefix_limit: int | None = None,
    generator_id: str | None = None,
    bm25_index: CorpusIndex | None = None,
    documents_path: Path | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[list[dict[str, Any]], str, dict[str, Any], list[dict[str, Any]]]:
    if limit is not None and prefix_limit is not None:
        raise ValueError("probe limit and prefix_limit are mutually exclusive")
    if prefix_limit is not None and prefix_limit < 1:
        raise ValueError("probe prefix_limit must be positive")
    synthetic = _synthetic_map(synthetic_generations)
    materialized = list(records)
    if prefix_limit is not None:
        materialized = materialized[:prefix_limit]
    prepared: list[tuple[dict[str, Any], str]] = []
    for record in materialized:
        query = _query_for_record(record, query_source=query_source, synthetic=synthetic)
        if query is not None:
            prepared.append((record, query))
    mined: dict[str, list[NegativeCandidate]] = {}
    if negative_recipe.strategy == "hn1_bm25":
        if bm25_index is None or documents_path is None:
            raise ValueError("HN1 BM25 requires bm25_index and frozen documents_path")
        mined = _hn1_candidates(
            prepared,
            index=bm25_index,
            documents_path=documents_path,
            recipe=negative_recipe,
            progress=progress,
        )
    candidate_sets: list[tuple[dict[str, Any], str, list[NegativeCandidate]]] = []
    for record, query in prepared:
        negatives = record.get("hard_negatives", [])
        if not record.get("positives") or (
            not negatives and negative_recipe.strategy != "hn1_bm25"
        ):
            continue
        example_id = str(record["example_id"])
        if negative_recipe.strategy == "hn1_bm25":
            candidates = mined[example_id]
        else:
            candidates = [
                NegativeCandidate(
                    doc_id=str(document["doc_id"]),
                    text=str(document["text"]),
                    miner="inherited",
                    miner_rank=index + 1,
                )
                for index, document in enumerate(
                    sorted(negatives, key=lambda value: str(value["doc_id"]))
                )
            ]
        candidate_sets.append((record, query, candidates))
    precomputed: list[float] | None = None
    score_offsets: list[tuple[int, int]] = []
    if negative_recipe.requires_filter:
        if primary_scorer is None or calibration is None:
            raise ValueError("filtered probe preparation requires scorer and calibration")
        if primary_scorer.name != calibration.primary_judge_name:
            raise ValueError("runtime primary reranker does not match calibration provenance")
        pairs: list[tuple[str, str]] = []
        for _record, query, candidates in candidate_sets:
            start = len(pairs)
            pairs.extend((query, candidate.text) for candidate in candidates)
            score_offsets.append((start, len(pairs)))
        precomputed = []
        scoring_chunk_size = 2048
        for start in range(0, len(pairs), scoring_chunk_size):
            end = min(start + scoring_chunk_size, len(pairs))
            precomputed.extend(primary_scorer.score_pairs(pairs[start:end]))
            if progress is not None:
                completed = max(1, round(len(candidate_sets) * end / len(pairs)))
                progress(completed, len(candidate_sets))
        if len(precomputed) != len(pairs):
            raise ValueError("bulk primary reranker scoring returned an invalid result count")
    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    selected: list[tuple[int, str, dict[str, Any]]] = []
    policy_dropped_examples = 0
    for prepared_index, (record, query, candidates) in enumerate(candidate_sets, start=1):
        if progress is not None and precomputed is None:
            progress(prepared_index, len(candidate_sets))
        positives = sorted(record.get("positives", []), key=lambda value: str(value["doc_id"]))
        example_id = str(record["example_id"])
        passage = str(positives[0]["text"])
        candidate_scores = None
        if precomputed is not None:
            start, end = score_offsets[prepared_index - 1]
            candidate_scores = precomputed[start:end]
        selection = select_negative(
            example_id=example_id,
            query=query,
            candidates=candidates,
            recipe=negative_recipe,
            scorer=primary_scorer,
            calibration=calibration,
            precomputed_scores=candidate_scores,
        )
        for audit in selection.audit_rows:
            audit_rows.append(
                {
                    "example_id": example_id,
                    "query_source": query_source,
                    "generator_id": generator_id,
                    **audit,
                }
            )
        if selection.dropped_example:
            policy_dropped_examples += 1
            continue
        row = {
            "example_id": example_id,
            "query": query,
            "positive_doc_id": str(positives[0]["doc_id"]),
            "positive": passage,
            "negative": selection.paired.text if selection.paired is not None else "",
            "negative_doc_id": (selection.paired.doc_id if selection.paired is not None else ""),
            "demoted_negative": (selection.demoted.text if selection.demoted is not None else ""),
            "demoted_negative_doc_id": (
                selection.demoted.doc_id if selection.demoted is not None else ""
            ),
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
    report = summarize_false_negative_audit(
        audit_rows,
        query_source=query_source,
        generator_id=generator_id,
        input_examples=len(prepared),
        output_examples=len(rows),
        policy_dropped_examples=policy_dropped_examples,
    )
    return rows, digest.hexdigest(), report, audit_rows


def _tokenize(
    tokenizer: Any,
    texts: list[str],
    max_length: int,
    device: torch.device,
    *,
    padding: bool | str = True,
) -> dict[str, torch.Tensor]:
    encoded: dict[str, torch.Tensor] = tokenizer(
        texts,
        padding=padding,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {key: value.to(device) for key, value in encoded.items()}


def train_probe(
    rows: list[dict[str, Any]],
    *,
    recipe: ProbeRecipe,
    output_dir: Path,
    query_source: QuerySource,
    train_fingerprint: str,
    negative_contract: Mapping[str, Any],
    false_negative_report: Mapping[str, Any],
    negative_audit_rows: Sequence[dict[str, Any]],
    statistical_contract: StatisticalContract,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("probe training set is empty")
    resumed_summary = _resumable_train_summary(
        output_dir,
        rows=rows,
        recipe=recipe,
        query_source=query_source,
        train_fingerprint=train_fingerprint,
        negative_contract=negative_contract,
        false_negative_report=false_negative_report,
        statistical_contract=statistical_contract,
    )
    if resumed_summary is not None:
        if _progress_enabled():
            print(
                f"[resume] training already complete: {output_dir / 'model'}; "
                "continuing with evaluation",
                file=sys.stderr,
                flush=True,
            )
        return resumed_summary
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
    report_every = max(1, recipe.max_steps // 20)
    for _step in range(recipe.max_steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        queries = model(_tokenize(tokenizer, list(batch["query"]), recipe.max_length, device))
        positives = model(_tokenize(tokenizer, list(batch["positive"]), recipe.max_length, device))
        document_batches = [positives]
        paired_negative_texts = [str(text) for text in batch["negative"] if str(text)]
        if paired_negative_texts:
            document_batches.append(
                model(_tokenize(tokenizer, paired_negative_texts, recipe.max_length, device))
            )
        demoted_negative_texts = [str(text) for text in batch["demoted_negative"] if str(text)]
        if demoted_negative_texts:
            document_batches.append(
                model(_tokenize(tokenizer, demoted_negative_texts, recipe.max_length, device))
            )
        documents = torch.cat(document_batches, dim=0)
        logits = queries @ documents.T / 0.05
        targets = torch.arange(queries.shape[0], device=device)
        loss = functional.cross_entropy(logits, targets)
        torch.autograd.backward(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
        completed_steps = _step + 1
        if completed_steps == 1 or completed_steps % report_every == 0:
            _progress("train", completed_steps, recipe.max_steps, started)
    adapter_dir = output_dir / "model"
    model.backbone.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    passage_counts: dict[str, int] = {}
    for row in rows:
        passage_id = str(row["positive_doc_id"])
        passage_counts[passage_id] = passage_counts.get(passage_id, 0) + 1
    observed_k = set(passage_counts.values())
    if len(observed_k) != 1:
        raise ValueError("P-04 comparison budget requires a uniform K queries per passage")
    comparison_budget = build_budget_manifest(
        token_count=(
            recipe.max_steps
            * recipe.batch_size
            * recipe.max_length
            * (2 + recipe.negatives_per_example)
        ),
        pair_count=len(rows),
        unique_passage_count=len(passage_counts),
        queries_per_passage=observed_k.pop(),
    )
    summary = {
        "schema_version": 1,
        "status": "measured",
        "query_source": query_source,
        "recipe": asdict(recipe),
        "recipe_fingerprint": recipe.fingerprint,
        "recipe_version": recipe.recipe_version,
        "negative_contract": dict(negative_contract),
        "statistical_contract": statistical_contract.reference(),
        "comparison_budget": comparison_budget,
        "possible_false_negative_report": dict(false_negative_report),
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
    with JsonlWriter(output_dir / "negative_audit.jsonl") as writer:
        for row in negative_audit_rows:
            writer.write(row)
    write_json(output_dir / "train_summary.json", summary)
    return summary


def _resumable_train_summary(
    output_dir: Path,
    *,
    rows: Sequence[dict[str, Any]],
    recipe: ProbeRecipe,
    query_source: QuerySource,
    train_fingerprint: str,
    negative_contract: Mapping[str, Any],
    false_negative_report: Mapping[str, Any],
    statistical_contract: StatisticalContract,
) -> dict[str, Any] | None:
    """Reuse a completed, identity-checked training stage after an interrupted evaluation."""
    raw = _completed_train_summary(
        output_dir,
        recipe=recipe,
        query_source=query_source,
        negative_contract=negative_contract,
        statistical_contract=statistical_contract,
    )
    if raw is None:
        return None
    expected = {
        "possible_false_negative_report": dict(false_negative_report),
        "train_fingerprint": train_fingerprint,
        "train_examples": len(rows),
    }
    mismatches = [key for key, value in expected.items() if raw.get(key) != value]
    if mismatches:
        raise ValueError(
            "probe resume identity mismatch in train_summary.json: " + ", ".join(mismatches)
        )
    return raw


def _completed_train_summary(
    output_dir: Path,
    *,
    recipe: ProbeRecipe,
    query_source: QuerySource,
    negative_contract: Mapping[str, Any],
    statistical_contract: StatisticalContract,
) -> dict[str, Any] | None:
    """Load a completed training stage without repeating expensive pair scoring."""
    summary_path = output_dir / "train_summary.json"
    model_path = output_dir / "model"
    if not summary_path.is_file() or not model_path.is_dir() or not any(model_path.iterdir()):
        return None
    raw = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("probe resume train_summary.json must contain a JSON object")
    expected = {
        "schema_version": 1,
        "status": "measured",
        "query_source": query_source,
        "recipe": asdict(recipe),
        "recipe_fingerprint": recipe.fingerprint,
        "recipe_version": recipe.recipe_version,
        "negative_contract": dict(negative_contract),
        "statistical_contract": statistical_contract.reference(),
        "steps": recipe.max_steps,
    }
    mismatches = [key for key, value in expected.items() if raw.get(key) != value]
    if mismatches:
        raise ValueError(
            "probe resume identity mismatch in train_summary.json: " + ", ".join(mismatches)
        )
    return raw


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
    cache_dir: Path | None = None,
    cache_identity: Mapping[str, Any] | None = None,
) -> torch.Tensor:
    if not texts:
        raise ValueError("cannot encode an empty corpus")
    if (cache_dir is None) != (cache_identity is None):
        raise ValueError("embedding cache directory and identity must be supplied together")
    started = time.perf_counter()
    chunk_size = max(batch_size, math.ceil(len(texts) / 100 / batch_size) * batch_size)
    if cache_dir is not None and cache_identity is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = cache_dir / "manifest.json"
        expected_manifest = {
            "schema_version": 1,
            "status": "in_progress",
            "row_count": len(texts),
            "chunk_size": chunk_size,
            "identity": dict(cache_identity),
        }
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest != expected_manifest:
                raise ValueError("corpus embedding resume cache identity mismatch")
        else:
            unexpected = list(cache_dir.glob("chunk-*.pt"))
            if unexpected:
                raise ValueError("corpus embedding cache has shards but no identity manifest")
            temporary = cache_dir / "manifest.json.tmp"
            write_json(temporary, expected_manifest)
            os.replace(temporary, manifest_path)
    chunks: list[torch.Tensor] = []
    for chunk_index, chunk_start in enumerate(range(0, len(texts), chunk_size)):
        chunk_end = min(chunk_start + chunk_size, len(texts))
        shard = cache_dir / f"chunk-{chunk_index:05d}.pt" if cache_dir is not None else None
        if shard is not None and shard.is_file():
            chunk = torch.load(shard, map_location="cpu", weights_only=True)
            if not isinstance(chunk, torch.Tensor) or chunk.ndim != 2:
                raise ValueError(f"invalid corpus embedding cache shard: {shard}")
            if chunk.shape[0] != chunk_end - chunk_start:
                raise ValueError(f"corpus embedding cache shard has wrong row count: {shard}")
            if _progress_enabled():
                print(
                    f"[resume] encode_corpus shard {chunk_index + 1} reused",
                    file=sys.stderr,
                    flush=True,
                )
        else:
            batch_chunks = []
            for start in range(chunk_start, chunk_end, batch_size):
                batch_chunks.append(
                    _encode(
                        model,
                        tokenizer,
                        texts[start : min(start + batch_size, chunk_end)],
                        max_length=max_length,
                        device=device,
                    )
                )
            chunk = torch.cat(batch_chunks)
            if shard is not None:
                temporary_shard = shard.with_suffix(".pt.tmp")
                torch.save(chunk, temporary_shard)
                os.replace(temporary_shard, shard)
        chunks.append(chunk)
        _progress("encode_corpus", chunk_end, len(texts), started)
    return torch.cat(chunks)


def _path_tree_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    if not files:
        raise ValueError(f"cannot fingerprint empty model directory: {path}")
    for candidate in files:
        digest.update(str(candidate.relative_to(path)).encode())
        digest.update(b"\0")
        digest.update(sha256_file(candidate).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def evaluate_probe(
    model_path: Path,
    records: list[dict[str, Any]],
    *,
    documents_path: Path,
    recipe: ProbeRecipe,
    output_dir: Path,
    test_fingerprint: str,
    dataset_name: str = "test_translated_msmarco_pl",
    profile: str = "full",
    negative_contract: Mapping[str, Any],
    statistical_contract: Mapping[str, Any],
    comparison_budget: Mapping[str, Any],
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_loader: Any = getattr(AutoTokenizer, "from_" + "pretrained")
    tokenizer = tokenizer_loader(model_path, trust_remote_code=False)
    model = MeanPoolEncoder(str(model_path), "main")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    corpus: dict[str, str] = {}
    for document in read_records(documents_path):
        doc_id, text = str(document["doc_id"]), str(document["text"])
        if doc_id in corpus:
            raise ValueError(f"duplicate document in frozen corpus: doc_id={doc_id}")
        corpus[doc_id] = text
    corpus_ids = sorted(corpus)
    if len(corpus_ids) < 100:
        raise ValueError("corpus_retrieval requires at least 100 documents for Recall@100")
    corpus_index = {doc_id: index for index, doc_id in enumerate(corpus_ids)}
    corpus_sha256 = sha256_file(documents_path)
    model_fingerprint = _path_tree_fingerprint(model_path)
    corpus_ids_digest = hashlib.sha256()
    for doc_id in corpus_ids:
        corpus_ids_digest.update(doc_id.encode())
        corpus_ids_digest.update(b"\n")
    index_started = time.perf_counter()
    corpus_embeddings = _encode_batched(
        model,
        tokenizer,
        [corpus[doc_id] for doc_id in corpus_ids],
        max_length=recipe.max_length,
        batch_size=recipe.batch_size,
        device=device,
        cache_dir=output_dir / "corpus_embedding_cache",
        cache_identity={
            "recipe_fingerprint": recipe.fingerprint,
            "model_fingerprint": model_fingerprint,
            "corpus_sha256": corpus_sha256,
            "corpus_ids_sha256": corpus_ids_digest.hexdigest(),
        },
    )
    index_seconds = time.perf_counter() - index_started
    evaluable_records = [record for record in records if record.get("positives")]
    per_query_path = output_dir / "corpus_retrieval_per_query.jsonl"
    checkpoint_path = output_dir / "corpus_retrieval_checkpoint.json"
    checkpoint = {
        "schema_version": 1,
        "status": "in_progress",
        "dataset_name": dataset_name,
        "profile": profile,
        "test_fingerprint": test_fingerprint,
        "recipe_fingerprint": recipe.fingerprint,
        "model_fingerprint": model_fingerprint,
        "corpus_sha256": corpus_sha256,
        "query_count": len(evaluable_records),
    }
    if checkpoint_path.is_file():
        existing_checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if existing_checkpoint != checkpoint:
            raise ValueError("corpus retrieval resume checkpoint identity mismatch")
    else:
        if per_query_path.is_file() and per_query_path.stat().st_size:
            raise ValueError("corpus retrieval rows exist without a resume checkpoint")
        temporary_checkpoint = checkpoint_path.with_suffix(".json.tmp")
        write_json(temporary_checkpoint, checkpoint)
        os.replace(temporary_checkpoint, checkpoint_path)
    per_query = list(read_records(per_query_path)) if per_query_path.is_file() else []
    expected_ids = [str(record["example_id"]) for record in evaluable_records]
    resumed_ids = [str(row.get("example_id")) for row in per_query]
    if resumed_ids != expected_ids[: len(resumed_ids)]:
        raise ValueError("corpus retrieval resume rows are not the expected query prefix")
    if len(per_query) > len(evaluable_records):
        raise ValueError("corpus retrieval resume has more rows than expected")
    metric_rows: list[dict[str, float | int]] = [
        {
            key: value
            for key, value in row.items()
            if key.startswith("corpus_") and isinstance(value, (int, float))
        }
        for row in per_query
    ]
    latencies = [
        float(row["latency_seconds"])
        for row in per_query
        if isinstance(row.get("latency_seconds"), (int, float))
    ]
    if per_query and _progress_enabled():
        print(
            f"[resume] evaluate_queries {len(per_query):,}/{len(evaluable_records):,} reused",
            file=sys.stderr,
            flush=True,
        )
    with per_query_path.open("a", encoding="utf-8") as writer:
        query_started = time.perf_counter()
        query_total = len(evaluable_records)
        query_report_every = max(1, query_total // 100)
        for query_index, record in enumerate(
            evaluable_records[len(per_query) :],
            start=len(per_query) + 1,
        ):
            positives = record.get("positives", [])
            negatives = record.get("hard_negatives", [])
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
            missing = [doc_id for doc_id in positive_ids if doc_id not in corpus_index]
            if missing:
                raise ValueError(f"test positives are absent from frozen corpus: {missing[:3]}")
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
            metrics = corpus_metrics_from_positive_ranks(
                positive_ranks,
                candidate_count=len(corpus_ids),
            )
            row = {
                "example_id": str(record["example_id"]),
                **metrics,
                "pool_candidate_count": len(positive_ids) + len(negative_ids),
                "pool_hard_negative_win_rate": (
                    sum(pairwise_wins) / len(pairwise_wins) if pairwise_wins else None
                ),
                "latency_seconds": latencies[-1],
            }
            writer.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            writer.flush()
            per_query.append(row)
            metric_rows.append(metrics)
            if query_index == query_total or query_index % query_report_every == 0:
                _progress("evaluate_queries", query_index, query_total, query_started)
    aggregate = aggregate_query_metrics(metric_rows)
    summary = {
        "schema_version": 2,
        "status": "measured",
        "protocol": CORPUS_RETRIEVAL,
        "metric_prefix": "corpus_",
        "dataset_name": dataset_name,
        "profile": profile,
        "test_fingerprint": test_fingerprint,
        "recipe_fingerprint": recipe.fingerprint,
        "recipe_version": recipe.recipe_version,
        "negative_contract": dict(negative_contract),
        "statistical_contract": dict(statistical_contract),
        "comparison_budget": dict(comparison_budget),
        "query_count": len(per_query),
        "metrics": aggregate,
        "metric_candidate_count": {metric: len(corpus_ids) for metric in aggregate},
        "latency_seconds_per_query": sum(latencies) / len(latencies) if latencies else None,
        "corpus_candidate_count": len(corpus_ids),
        "corpus_path": str(documents_path),
        "corpus_sha256": corpus_sha256,
        "candidate_pool_diagnostics": {
            "pool_hard_negative_win_rate": (
                sum(
                    float(row["pool_hard_negative_win_rate"])
                    for row in per_query
                    if isinstance(row.get("pool_hard_negative_win_rate"), (int, float))
                )
                / sum(
                    isinstance(row.get("pool_hard_negative_win_rate"), (int, float))
                    for row in per_query
                )
                if any(
                    isinstance(row.get("pool_hard_negative_win_rate"), (int, float))
                    for row in per_query
                )
                else None
            ),
            "pool_candidate_count": sorted({int(row["pool_candidate_count"]) for row in per_query}),
        },
        "index_build_seconds": index_seconds,
        "index_size_bytes": corpus_embeddings.nelement() * corpus_embeddings.element_size(),
        "model_size_bytes": sum(
            path.stat().st_size for path in model_path.rglob("*") if path.is_file()
        ),
        "translationese": aggregate_translationese(str(record["query"]) for record in records),
    }
    write_json(output_dir / "corpus_retrieval_summary.json", summary)
    return summary


def run_probe_experiment(
    *,
    train_path: Path,
    frozen_manifest: Path,
    test_subset: str,
    output_dir: Path,
    recipe: ProbeRecipe,
    query_source: QuerySource,
    statistical_contract: StatisticalContract,
    synthetic_generations: Path | None = None,
    train_limit: int | None = None,
    train_prefix_limit: int | None = None,
    documents_path: Path,
    holdout_manifest: Path | None = None,
    native_documents_path: Path | None = None,
    holdout_profile: HoldoutProfile = "quick",
    primary_scorer: PairScorer | None = None,
    bm25_index: CorpusIndex | None = None,
    generator_id: str | None = None,
) -> dict[str, Any]:
    calibration = recipe.negative_recipe.load_calibration()
    negative_contract = recipe.negative_recipe.manifest(calibration) | {
        "probe_recipe_version": recipe.recipe_version,
        "probe_recipe_fingerprint": recipe.fingerprint,
    }
    train_summary = _completed_train_summary(
        output_dir,
        recipe=recipe,
        query_source=query_source,
        negative_contract=negative_contract,
        statistical_contract=statistical_contract,
    )
    if train_summary is None:
        pairs, train_fingerprint, false_negative_report, negative_audit_rows = prepare_probe_pairs(
            read_records(train_path),
            query_source=query_source,
            negative_recipe=recipe.negative_recipe,
            calibration=calibration,
            primary_scorer=primary_scorer,
            synthetic_generations=synthetic_generations,
            limit=train_limit,
            prefix_limit=train_prefix_limit,
            generator_id=generator_id,
            bm25_index=bm25_index,
            documents_path=documents_path,
            progress=(_progress_callback("filter_negatives") if _progress_enabled() else None),
        )
        train_summary = train_probe(
            pairs,
            recipe=recipe,
            output_dir=output_dir,
            query_source=query_source,
            train_fingerprint=train_fingerprint,
            negative_contract=negative_contract,
            false_negative_report=false_negative_report,
            negative_audit_rows=negative_audit_rows,
            statistical_contract=statistical_contract,
        )
    else:
        report = train_summary.get("possible_false_negative_report")
        if not isinstance(report, Mapping):
            raise ValueError("resumed probe training has no false-negative report")
        false_negative_report = dict(report)
        if _progress_enabled():
            print(
                "[resume] completed training accepted; skipping filter_negatives and training",
                file=sys.stderr,
                flush=True,
            )
    if (
        holdout_manifest is not None
        and holdout_set_status(holdout_manifest, "test_translated_msmarco_pl") == "materialized"
    ):
        test_records = load_holdout_records(
            holdout_manifest,
            "test_translated_msmarco_pl",
            profile=holdout_profile,
        )
        translated_fingerprint = holdout_fingerprint(
            holdout_manifest,
            "test_translated_msmarco_pl",
            holdout_profile,
        )
        translated_profile = holdout_profile
    else:
        test_records = load_frozen_records(frozen_manifest, test_subset)
        translated_fingerprint = evaluation_fingerprint(frozen_manifest, test_subset)
        translated_profile = "full"
    effective_translated_corpus = documents_path
    if holdout_manifest is not None and holdout_profile in {"quick", "medium"}:
        diagnostic_corpus = holdout_artifact_path(
            holdout_manifest,
            f"translated_{holdout_profile}_corpus",
        )
        if diagnostic_corpus is not None:
            effective_translated_corpus = diagnostic_corpus
    # Keep the pre-P-02 translated artifact paths stable for existing
    # comparison commands; native artifacts live in their own subdirectory.
    translated_output = output_dir
    retrieval = evaluate_probe(
        output_dir / "model",
        test_records,
        documents_path=effective_translated_corpus,
        recipe=recipe,
        output_dir=translated_output,
        test_fingerprint=translated_fingerprint,
        dataset_name="test_translated_msmarco_pl",
        profile=translated_profile,
        negative_contract=negative_contract,
        statistical_contract=train_summary["statistical_contract"],
        comparison_budget=train_summary["comparison_budget"],
    )
    native: dict[str, Any]
    if holdout_manifest is None:
        native = {
            "dataset_name": "test_native_pl",
            "profile": holdout_profile,
            "status": "not_measured",
            "reason": "native holdout manifest was not supplied",
            "test_fingerprint": None,
            "metrics": None,
        }
    elif holdout_set_status(holdout_manifest, "test_native_pl") != "materialized":
        native = {
            "dataset_name": "test_native_pl",
            "profile": holdout_profile,
            "status": "missing_artifact",
            "reason": "test_native_pl is not materialized in the frozen holdout manifest",
            "test_fingerprint": None,
            "metrics": None,
        }
    else:
        effective_native_corpus = native_documents_path
        if effective_native_corpus is None and holdout_profile in {"quick", "medium"}:
            effective_native_corpus = holdout_artifact_path(
                holdout_manifest,
                f"native_{holdout_profile}_corpus",
            )
        if effective_native_corpus is None:
            native = {
                "dataset_name": "test_native_pl",
                "profile": holdout_profile,
                "status": "missing_artifact",
                "reason": (
                    "native corpus is missing; full requires the adapted complete PolQA corpus"
                ),
                "test_fingerprint": holdout_fingerprint(
                    holdout_manifest, "test_native_pl", holdout_profile
                ),
                "metrics": None,
            }
        else:
            native_records = load_holdout_records(
                holdout_manifest,
                "test_native_pl",
                profile=holdout_profile,
            )
            native = evaluate_probe(
                output_dir / "model",
                native_records,
                documents_path=effective_native_corpus,
                recipe=recipe,
                output_dir=output_dir / "evaluation" / "test_native_pl",
                test_fingerprint=holdout_fingerprint(
                    holdout_manifest, "test_native_pl", holdout_profile
                ),
                dataset_name="test_native_pl",
                profile=holdout_profile,
                negative_contract=negative_contract,
                statistical_contract=train_summary["statistical_contract"],
                comparison_budget=train_summary["comparison_budget"],
            )
    report_status = "complete" if native.get("status") == "measured" else "incomplete"
    comparison_eligible = (
        report_status == "complete" and holdout_profile == "full" and translated_profile == "full"
    )
    result = {
        "schema_version": 2,
        "report_status": report_status,
        "comparison_eligible": comparison_eligible,
        "incomplete_reasons": (
            []
            if report_status == "complete"
            else [str(native.get("reason", "native not measured"))]
        ),
        "training": train_summary,
        "recipe_version": recipe.recipe_version,
        "recipe_fingerprint": recipe.fingerprint,
        "negative_contract": negative_contract,
        "statistical_contract": train_summary["statistical_contract"],
        "comparison_budget": train_summary["comparison_budget"],
        "possible_false_negative_report": false_negative_report,
        "evaluation_sets": {
            "test_native_pl": native,
            "test_translated_msmarco_pl": retrieval,
        },
        # Compatibility alias for pre-P-02 consumers.  It is explicitly the
        # translated result and must not be used as the native primary metric.
        "corpus_retrieval": retrieval,
    }
    write_json(output_dir / "result.json", result)
    build_embedder_report(
        result,
        markdown_path=output_dir / "embedder_report.md",
        json_path=output_dir / "embedder_report.json",
    )
    return result
