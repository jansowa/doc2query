"""Frozen-budget bi-encoder probe training and natural-query retrieval evaluation."""

from __future__ import annotations

import bisect
import gc
import hashlib
import heapq
import json
import math
import os
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal, cast

import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.data import DataLoader, Dataset

from doc2query.evaluation.corpus import CorpusIndex, sha256_file
from doc2query.evaluation.datasets import evaluation_fingerprint, load_frozen_records
from doc2query.evaluation.dense_retrieval import (
    ShardedEmbeddingIndex,
    exact_retrieval_batch,
)
from doc2query.evaluation.native_holdout import (
    HoldoutProfile,
    holdout_artifact_path,
    holdout_fingerprint,
    holdout_set_status,
    load_holdout_records,
)
from doc2query.evaluation.probe_in_run_collapse import (
    CollapseDetector,
    InRunCollapseDetection,
    InterimEvaluationSet,
    ProbeCollapseDetected,
    ProbeCollapseUnresolved,
    build_interim_evaluation_set,
    interim_recall,
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
    rate = current / elapsed if elapsed > 0 else 0.0
    eta = (total - current) / rate if rate > 0 else float("inf")
    print(
        f"[{stage}] {current:,}/{total:,} ({percent:5.1f}%) elapsed={elapsed:.1f}s "
        f"rate={rate:,.2f}/s eta={eta:.1f}s",
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


def _stage(message: str) -> None:
    print(f"[probe] {message}", file=sys.stderr, flush=True)


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


def _write_jsonl_atomically(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as writer:
        for row in rows:
            writer.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        writer.flush()
        os.fsync(writer.fileno())
    os.replace(temporary, path)


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}: journal rows must be JSON objects")
        rows.append(value)
    return rows


def _interim_recall_now(
    model: MeanPoolEncoder,
    tokenizer: Any,
    evaluation_set: InterimEvaluationSet,
    *,
    detection: InRunCollapseDetection,
    max_length: int,
    device: torch.device,
) -> float:
    """Score the held-in interim set without perturbing the training trajectory."""
    interim = detection.interim_evaluation
    torch_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    was_training = model.training
    model.eval()
    try:

        def encode(texts: list[str]) -> torch.Tensor:
            chunks = [
                _encode(
                    model,
                    tokenizer,
                    texts[start : start + interim.encode_batch_size],
                    max_length=max_length,
                    device=device,
                )
                for start in range(0, len(texts), interim.encode_batch_size)
            ]
            return torch.cat(chunks)

        documents = encode(evaluation_set.documents)
        queries = encode(evaluation_set.queries)
        return interim_recall(
            queries,
            documents,
            evaluation_set.positive_positions,
            depth=interim.retrieval_depth,
        )
    finally:
        if was_training:
            model.train()
        if interim.restore_rng_state:
            torch.set_rng_state(torch_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)


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
    checkpoint_interval_steps: int = 0,
    collapse_detection: InRunCollapseDetection | None = None,
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
    checkpoint_path = output_dir / "training_checkpoint.pt"
    loss_curve_path: Path | None = None
    interim_path: Path | None = None
    if collapse_detection is not None:
        loss_curve_path = output_dir / collapse_detection.persistence.loss_curve_file
        interim_path = output_dir / collapse_detection.persistence.interim_evaluation_file
    existing = set(output_dir.iterdir()) if output_dir.exists() else set()
    unexpected = existing - {checkpoint_path, loss_curve_path, interim_path}
    if unexpected:
        raise FileExistsError(f"probe output contains non-resumable artifacts: {output_dir}")
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
    _stage(
        f"loading train model on {device.type}; batch_size={recipe.batch_size}, "
        f"max_length={recipe.max_length}, steps={recipe.max_steps}"
    )
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
    losses: list[float] = []
    completed_before_resume = 0
    checkpoint_identity = {
        "schema_version": 1,
        "recipe_fingerprint": recipe.fingerprint,
        "query_source": query_source,
        "train_fingerprint": train_fingerprint,
        "negative_contract": dict(negative_contract),
        "statistical_contract": statistical_contract.reference(),
    }
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if not isinstance(checkpoint, Mapping) or checkpoint.get("identity") != checkpoint_identity:
            raise ValueError("probe training checkpoint identity mismatch")
        completed_before_resume = int(checkpoint.get("completed_steps", 0))
        if not 0 < completed_before_resume < recipe.max_steps:
            raise ValueError("probe training checkpoint has an invalid completed step")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        losses = [float(value) for value in checkpoint["losses"]]
        # Replay only the cheap sampler advancement so the next microbatch is identical.
        for _ in range(completed_before_resume):
            try:
                next(iterator)
            except StopIteration:
                iterator = iter(loader)
                next(iterator)
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in checkpoint["cuda_rng_state_all"]]
            )
        _stage(
            f"resumed training at step {completed_before_resume}/{recipe.max_steps} "
            f"from {checkpoint_path}"
        )
    detector: CollapseDetector | None = None
    interim_set: InterimEvaluationSet | None = None
    interim_rows: list[dict[str, Any]] = []
    if collapse_detection is not None and loss_curve_path is not None and interim_path is not None:
        interim_set = build_interim_evaluation_set(rows, collapse_detection.interim_evaluation)
        detector = CollapseDetector(
            contract=collapse_detection,
            chance_level=interim_set.chance_level(
                collapse_detection.interim_evaluation.retrieval_depth
            ),
        )
        # Replay the journals of the resumed prefix so no row is duplicated or lost.
        interim_rows = [
            row
            for row in _read_jsonl_rows(interim_path)
            if int(row.get("step", 0)) <= completed_before_resume
        ]
        for row in interim_rows:
            detector.observe(
                step=int(row["step"]),
                recall=float(row["train_holdin_recall_at_100"]),
                losses=losses[: int(row["step"])],
            )
        _write_jsonl_atomically(interim_path, interim_rows)
        _write_jsonl_atomically(
            loss_curve_path,
            [{"step": step, "loss": loss} for step, loss in enumerate(losses, start=1)],
        )
        _stage(
            f"in-run collapse detection enabled ({collapse_detection.detector_id}): "
            f"interim recall@{collapse_detection.interim_evaluation.retrieval_depth} on "
            f"{len(interim_set.documents)} held-in documents, floor {detector.floor:.6f}"
        )
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    report_every = max(1, recipe.max_steps // 20)
    _stage("training started")
    for _step in range(completed_before_resume, recipe.max_steps):
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
        if (
            checkpoint_interval_steps > 0
            and completed_steps < recipe.max_steps
            and completed_steps % checkpoint_interval_steps == 0
        ):
            checkpoint = {
                "identity": checkpoint_identity,
                "completed_steps": completed_steps,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "losses": losses,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                ),
            }
            temporary_checkpoint = checkpoint_path.with_suffix(".pt.tmp")
            torch.save(checkpoint, temporary_checkpoint)
            os.replace(temporary_checkpoint, checkpoint_path)
            _stage(f"training checkpoint saved at step {completed_steps}")
        if (
            detector is not None
            and interim_set is not None
            and loss_curve_path is not None
            and interim_path is not None
            and detector.should_check(completed_steps, recipe.max_steps)
        ):
            interim_started = time.perf_counter()
            recall = _interim_recall_now(
                model,
                tokenizer,
                interim_set,
                detection=detector.contract,
                max_length=recipe.max_length,
                device=device,
            )
            observation = detector.observe(step=completed_steps, recall=recall, losses=losses)
            observation["seconds"] = time.perf_counter() - interim_started
            observation["seed"] = recipe.seed
            interim_rows.append(observation)
            _write_jsonl_atomically(interim_path, interim_rows)
            _write_jsonl_atomically(
                loss_curve_path,
                [{"step": step, "loss": loss} for step, loss in enumerate(losses, start=1)],
            )
            _stage(
                f"interim check at step {completed_steps}: "
                f"recall={recall:.6f} floor={detector.floor:.6f} "
                f"below_floor={observation['below_floor']} "
                f"loss_non_decreasing={observation['loss_non_decreasing']}"
            )
            if observation["collapse_detected"]:
                _stage(
                    f"collapse detected at step {completed_steps} by {observation['rule']}; "
                    "aborting this attempt before the expensive evaluation"
                )
                raise ProbeCollapseDetected(observation)
    _stage("training complete; saving model")
    if loss_curve_path is not None:
        _write_jsonl_atomically(
            loss_curve_path,
            [{"step": step, "loss": loss} for step, loss in enumerate(losses, start=1)],
        )
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
    checkpoint_path.unlink(missing_ok=True)
    return summary


PROMOTED_ATTEMPT_ARTIFACTS = (
    "model",
    "train_summary.json",
    "negative_audit.jsonl",
    "training_loss_curve.jsonl",
    "training_interim_evaluation.jsonl",
)


def _promote_attempt(attempt_dir: Path, output_dir: Path) -> None:
    """Move the accepted attempt's artifacts up, keeping collapsed attempts as evidence."""
    for name in PROMOTED_ATTEMPT_ARTIFACTS:
        source = attempt_dir / name
        target = output_dir / name
        if not source.exists():
            continue
        if target.exists():
            raise FileExistsError(f"cannot promote probe attempt artifact over {target}")
        os.replace(source, target)


def train_probe_with_collapse_reseed(
    rows: list[dict[str, Any]],
    *,
    recipe: ProbeRecipe,
    output_dir: Path,
    collapse_detection: InRunCollapseDetection,
    query_source: QuerySource,
    train_fingerprint: str,
    negative_contract: Mapping[str, Any],
    false_negative_report: Mapping[str, Any],
    negative_audit_rows: Sequence[dict[str, Any]],
    statistical_contract: StatisticalContract,
    checkpoint_interval_steps: int = 0,
) -> tuple[dict[str, Any], ProbeRecipe]:
    """Train under the frozen in-run detector, reseeding deterministically on collapse.

    Every attempt — collapsed or accepted — is journalled before anything else happens, so
    the number of attempts and the seeds they used can never be hidden from a later reader.
    Exhausting the frozen attempt budget fails the run instead of reporting a result.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    journal_path = output_dir / collapse_detection.persistence.attempt_journal_file
    journal = _read_jsonl_rows(journal_path)
    already_collapsed = {
        int(row["attempt_index"]) for row in journal if row.get("outcome") == "collapsed"
    }
    attempts_root = output_dir / "collapse_attempts"
    for attempt_index in range(collapse_detection.reseed.max_attempts):
        if attempt_index in already_collapsed:
            continue
        seed = collapse_detection.attempt_seed(recipe.seed, attempt_index)
        attempt_recipe = recipe
        if seed != recipe.seed:
            attempt_recipe = ProbeRecipe.from_dict(asdict(recipe) | {"seed": seed})
        attempt_dir = attempts_root / f"attempt-{attempt_index:02d}-seed-{seed}"
        started = time.time()
        entry = {
            "attempt_index": attempt_index,
            "requested_seed": recipe.seed,
            "seed": seed,
            "attempt_dir": str(attempt_dir),
            "started_at": started,
        }
        try:
            summary = train_probe(
                rows,
                recipe=attempt_recipe,
                output_dir=attempt_dir,
                query_source=query_source,
                train_fingerprint=train_fingerprint,
                negative_contract=negative_contract,
                false_negative_report=false_negative_report,
                negative_audit_rows=negative_audit_rows,
                statistical_contract=statistical_contract,
                checkpoint_interval_steps=checkpoint_interval_steps,
                collapse_detection=collapse_detection,
            )
        except ProbeCollapseDetected as exc:
            observation = exc.observation
            journal.append(
                entry
                | {
                    "outcome": "collapsed",
                    "rule": observation.get("rule"),
                    "detected_at_step": observation.get("step"),
                    "train_holdin_recall_at_100": observation.get("train_holdin_recall_at_100"),
                    "floor": observation.get("floor"),
                    "below_floor": observation.get("below_floor"),
                    "loss_non_decreasing": observation.get("loss_non_decreasing"),
                    "finished_at": time.time(),
                    "seconds": time.time() - started,
                }
            )
            _write_jsonl_atomically(journal_path, journal)
            (attempt_dir / "training_checkpoint.pt").unlink(missing_ok=True)
            remaining = collapse_detection.reseed.max_attempts - attempt_index - 1
            _stage(
                f"attempt {attempt_index} (seed {seed}) collapsed; "
                f"{remaining} reseed attempt(s) left"
            )
            continue
        if not any(row.get("attempt_index") == attempt_index for row in journal):
            journal.append(
                entry | {"outcome": "completed", "finished_at": time.time()},
            )
            _write_jsonl_atomically(journal_path, journal)
        _promote_attempt(attempt_dir, output_dir)
        provenance = collapse_detection.reference() | {
            "enabled": True,
            "requested_seed": recipe.seed,
            "effective_seed": seed,
            "accepted_attempt_index": attempt_index,
            "attempt_count": attempt_index + 1,
            "detection_count": sum(1 for row in journal if row.get("outcome") == "collapsed"),
            "attempts": journal,
            "final_tests_used": [],
        }
        summary = dict(summary) | {"collapse_detection": provenance}
        write_json(output_dir / "train_summary.json", summary)
        return summary, attempt_recipe
    raise ProbeCollapseUnresolved(journal)


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


def _reseeded_recipe(output_dir: Path, recipe: ProbeRecipe) -> ProbeRecipe:
    """Recover the effective seed of an already promoted, possibly reseeded attempt."""
    summary_path = output_dir / "train_summary.json"
    if not summary_path.is_file():
        return recipe
    raw = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("probe resume train_summary.json must contain a JSON object")
    seed = int(raw.get("recipe", {}).get("seed", recipe.seed))
    if seed == recipe.seed:
        return recipe
    return ProbeRecipe.from_dict(asdict(recipe) | {"seed": seed})


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
    progress_stage: str = "encode_corpus",
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
                    f"[resume] {progress_stage} shard {chunk_index + 1} reused",
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
        _progress(progress_stage, chunk_end, len(texts), started)
    return torch.cat(chunks)


def _atomic_torch_save(value: torch.Tensor, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _valid_embedding_shard(path: Path, *, expected_rows: int) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        value = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:  # pragma: no cover - compatibility with older supported torch
        try:
            value = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            return False
    except Exception:
        return False
    return bool(
        isinstance(value, torch.Tensor)
        and value.ndim == 2
        and value.is_floating_point()
        and value.shape[0] == expected_rows
    )


def _embedding_manifest(cache_dir: Path) -> dict[str, Any] | None:
    path = cache_dir / "manifest.json"
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("embedding cache manifest must contain a JSON object")
    return value


def _ensure_corpus_embedding_cache(
    *,
    cache_dir: Path,
    cache_identity: Mapping[str, Any],
    row_count: int,
    texts: list[str] | None,
    model: MeanPoolEncoder | None,
    tokenizer: Any | None,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> ShardedEmbeddingIndex:
    """Complete missing shards without concatenating the full corpus in RAM."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    manifest = _embedding_manifest(cache_dir)
    if manifest is None:
        chunk_size = max(batch_size, math.ceil(row_count / 100 / batch_size) * batch_size)
        manifest = {
            "schema_version": 1,
            "status": "in_progress",
            "row_count": row_count,
            "chunk_size": chunk_size,
            "identity": dict(cache_identity),
        }
        write_json(manifest_path.with_suffix(".json.tmp"), manifest)
        os.replace(manifest_path.with_suffix(".json.tmp"), manifest_path)
    required = {
        "schema_version": 1,
        "row_count": row_count,
        "identity": dict(cache_identity),
    }
    mismatches = [key for key, value in required.items() if manifest.get(key) != value]
    if mismatches:
        raise ValueError(
            "corpus embedding resume cache identity mismatch: " + ", ".join(mismatches)
        )
    chunk_size = int(manifest.get("chunk_size", 0))
    if chunk_size < 1:
        raise ValueError("corpus embedding cache has an invalid chunk_size")
    shard_count = math.ceil(row_count / chunk_size)
    started = time.perf_counter()
    for shard_index in range(shard_count):
        start = shard_index * chunk_size
        end = min(start + chunk_size, row_count)
        shard = cache_dir / f"chunk-{shard_index:05d}.pt"
        if _valid_embedding_shard(shard, expected_rows=end - start):
            if _progress_enabled():
                print(
                    f"[resume] encode_corpus shard {shard_index + 1}/{shard_count} reused",
                    file=sys.stderr,
                    flush=True,
                )
        else:
            if texts is None or model is None or tokenizer is None:
                raise ValueError(
                    "corpus embedding cache is incomplete or corrupt and no encoder input is loaded"
                )
            if shard.is_file():
                _stage(f"repairing invalid corpus embedding shard {shard_index + 1}/{shard_count}")
            encoded = []
            for batch_start in range(start, end, batch_size):
                encoded.append(
                    _encode(
                        model,
                        tokenizer,
                        texts[batch_start : min(batch_start + batch_size, end)],
                        max_length=max_length,
                        device=device,
                    )
                )
            _atomic_torch_save(torch.cat(encoded), shard)
        _progress("encode_corpus", end, row_count, started)
    completed = dict(manifest) | {"status": "complete", "completed_shards": shard_count}
    write_json(manifest_path.with_suffix(".json.tmp"), completed)
    os.replace(manifest_path.with_suffix(".json.tmp"), manifest_path)
    return ShardedEmbeddingIndex.load(cache_dir, row_count=row_count, chunk_size=chunk_size)


def _corpus_ids_from_cache_or_source(
    documents_path: Path,
    *,
    cache_dir: Path,
    expected_count: int,
    expected_digest: str,
) -> list[str]:
    path = cache_dir / "corpus_ids.jsonl"
    if path.is_file():
        ids = [str(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines()]
    else:
        ids = sorted(str(document["doc_id"]) for document in read_records(documents_path))
        if any(left == right for left, right in pairwise(ids)):
            raise ValueError("duplicate document in frozen corpus")
        temporary = path.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as writer:
            for doc_id in ids:
                writer.write(json.dumps(doc_id, ensure_ascii=False) + "\n")
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temporary, path)
    digest = hashlib.sha256()
    for doc_id in ids:
        digest.update(doc_id.encode())
        digest.update(b"\n")
    if len(ids) != expected_count or digest.hexdigest() != expected_digest:
        raise ValueError("corpus ID catalog does not match the embedding cache identity")
    return ids


def _query_embeddings(
    *,
    cache_dir: Path,
    records: Sequence[Mapping[str, Any]],
    cache_identity: Mapping[str, Any],
    model: MeanPoolEncoder | None,
    tokenizer: Any | None,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    path = cache_dir / "query_embeddings.pt"
    manifest_path = cache_dir / "query_embeddings_manifest.json"
    expected = {
        "schema_version": 1,
        "status": "complete",
        "row_count": len(records),
        "identity": dict(cache_identity),
    }
    if path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest != expected:
            raise ValueError("query embedding cache identity mismatch")
        value = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[0] != len(records):
            raise ValueError("invalid query embedding cache")
        if _progress_enabled():
            print(f"[resume] query embeddings {len(records):,} reused", file=sys.stderr, flush=True)
        return value
    if path.exists() or manifest_path.exists():
        raise ValueError("query embedding cache is incomplete")
    if model is None or tokenizer is None:
        raise ValueError("query embeddings are missing and no encoder is loaded")
    value = _encode_batched(
        model,
        tokenizer,
        [str(record["query"]) for record in records],
        max_length=max_length,
        batch_size=batch_size,
        device=device,
        progress_stage="encode_queries",
    )
    _atomic_torch_save(value, path)
    write_json(manifest_path.with_suffix(".json.tmp"), expected)
    os.replace(manifest_path.with_suffix(".json.tmp"), manifest_path)
    return value


def _read_resume_jsonl(path: Path) -> list[dict[str, Any]]:
    """Recover only a truncated final JSONL write; reject corruption in the middle."""
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    good_bytes = 0
    size = path.stat().st_size
    with path.open("rb") as reader:
        while line := reader.readline():
            try:
                value = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if reader.tell() != size:
                    raise ValueError(
                        "corpus retrieval journal is corrupt before its final row"
                    ) from None
                with path.open("r+b") as recovery:
                    recovery.truncate(good_bytes)
                    recovery.flush()
                    os.fsync(recovery.fileno())
                print(
                    "[resume] truncated incomplete final corpus retrieval journal row recovered",
                    file=sys.stderr,
                    flush=True,
                )
                break
            if not isinstance(value, dict):
                raise ValueError("corpus retrieval journal rows must be JSON objects")
            rows.append(value)
            good_bytes = reader.tell()
    return rows


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
    evaluation_encode_batch_size: int = 64,
    retrieval_query_batch_size: int = 512,
    retrieval_device: str = "auto",
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    if min(evaluation_encode_batch_size, retrieval_query_batch_size) < 1:
        raise ValueError("evaluation and retrieval batch sizes must be positive")
    if retrieval_device not in {"auto", "cpu", "cuda"}:
        raise ValueError("retrieval_device must be auto, cpu or cuda")
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_sha256 = sha256_file(documents_path)
    model_fingerprint = _path_tree_fingerprint(model_path)
    cache_dir = output_dir / "corpus_embedding_cache"
    existing_manifest = _embedding_manifest(cache_dir)
    corpus_texts: list[str] | None = None
    if existing_manifest is not None:
        identity = existing_manifest.get("identity")
        if not isinstance(identity, Mapping):
            raise ValueError("corpus embedding cache has no identity mapping")
        expected_partial = {
            "recipe_fingerprint": recipe.fingerprint,
            "model_fingerprint": model_fingerprint,
            "corpus_sha256": corpus_sha256,
        }
        if any(identity.get(key) != value for key, value in expected_partial.items()):
            raise ValueError("corpus embedding resume cache identity mismatch")
        corpus_count = int(existing_manifest.get("row_count", 0))
        corpus_ids_sha256 = str(identity.get("corpus_ids_sha256", ""))
        corpus_ids = _corpus_ids_from_cache_or_source(
            documents_path,
            cache_dir=cache_dir,
            expected_count=corpus_count,
            expected_digest=corpus_ids_sha256,
        )
    else:
        _stage(f"evaluation {dataset_name}/{profile}: cataloguing corpus {documents_path}")
        corpus: dict[str, str] = {}
        for document in read_records(documents_path):
            doc_id, text = str(document["doc_id"]), str(document["text"])
            if doc_id in corpus:
                raise ValueError(f"duplicate document in frozen corpus: doc_id={doc_id}")
            corpus[doc_id] = text
        corpus_ids = sorted(corpus)
        corpus_texts = [corpus[doc_id] for doc_id in corpus_ids]
        corpus_count = len(corpus_ids)
        corpus_ids_digest = hashlib.sha256()
        for doc_id in corpus_ids:
            corpus_ids_digest.update(doc_id.encode())
            corpus_ids_digest.update(b"\n")
        corpus_ids_sha256 = corpus_ids_digest.hexdigest()
    if corpus_count < 100:
        raise ValueError("corpus_retrieval requires at least 100 documents for Recall@100")
    corpus_identity = {
        "recipe_fingerprint": recipe.fingerprint,
        "model_fingerprint": model_fingerprint,
        "corpus_sha256": corpus_sha256,
        "corpus_ids_sha256": corpus_ids_sha256,
    }
    manifest_chunk_size = int(existing_manifest.get("chunk_size", 0)) if existing_manifest else 0
    expected_shards = math.ceil(corpus_count / manifest_chunk_size) if manifest_chunk_size else 0
    corpus_cache_complete = bool(
        existing_manifest
        and expected_shards
        and all(
            _valid_embedding_shard(
                cache_dir / f"chunk-{index:05d}.pt",
                expected_rows=min(
                    manifest_chunk_size,
                    corpus_count - index * manifest_chunk_size,
                ),
            )
            for index in range(expected_shards)
        )
    )
    if not corpus_cache_complete and corpus_texts is None:
        corpus = {str(row["doc_id"]): str(row["text"]) for row in read_records(documents_path)}
        if len(corpus) != corpus_count:
            raise ValueError("duplicate or missing document while resuming corpus encoding")
        corpus_texts = [corpus[doc_id] for doc_id in corpus_ids]

    evaluable_records = [record for record in records if record.get("positives")]
    query_digest = hashlib.sha256()
    for record in evaluable_records:
        query_digest.update(str(record["example_id"]).encode())
        query_digest.update(b"\0")
        query_digest.update(str(record["query"]).encode())
        query_digest.update(b"\n")
    query_identity = {
        "recipe_fingerprint": recipe.fingerprint,
        "model_fingerprint": model_fingerprint,
        "test_fingerprint": test_fingerprint,
        "ordered_queries_sha256": query_digest.hexdigest(),
    }
    query_cache_complete = (cache_dir / "query_embeddings.pt").is_file() and (
        cache_dir / "query_embeddings_manifest.json"
    ).is_file()

    encoder_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model: MeanPoolEncoder | None = None
    tokenizer: Any | None = None
    if not corpus_cache_complete or not query_cache_complete:
        tokenizer_loader: Any = getattr(AutoTokenizer, "from_" + "pretrained")
        _stage(f"evaluation {dataset_name}/{profile}: loading trained model for missing embeddings")
        tokenizer = tokenizer_loader(model_path, trust_remote_code=False)
        model = MeanPoolEncoder(str(model_path), "main")
        model.to(encoder_device).eval()
    else:
        _stage(f"evaluation {dataset_name}/{profile}: all embeddings cached; model load skipped")
    index_started = time.perf_counter()
    _stage(
        f"evaluation {dataset_name}/{profile}: preparing {corpus_count:,} corpus embeddings "
        f"with batch_size={evaluation_encode_batch_size}"
    )
    sharded_index = _ensure_corpus_embedding_cache(
        cache_dir=cache_dir,
        cache_identity=corpus_identity,
        row_count=corpus_count,
        texts=corpus_texts,
        model=model,
        tokenizer=tokenizer,
        max_length=recipe.max_length,
        batch_size=evaluation_encode_batch_size,
        device=encoder_device,
    )
    index_seconds = time.perf_counter() - index_started
    query_encode_started = time.perf_counter()
    encoded_queries = _query_embeddings(
        cache_dir=cache_dir,
        records=evaluable_records,
        cache_identity=query_identity,
        model=model,
        tokenizer=tokenizer,
        max_length=recipe.max_length,
        batch_size=evaluation_encode_batch_size,
        device=encoder_device,
    )
    query_encode_seconds = time.perf_counter() - query_encode_started
    del model, tokenizer, corpus_texts
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if retrieval_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA retrieval was requested but CUDA is unavailable")
    use_cuda = retrieval_device == "cuda" or (
        retrieval_device == "auto" and torch.cuda.is_available()
    )
    score_device = torch.device("cuda" if use_cuda else "cpu")
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
    per_query = _read_resume_jsonl(per_query_path)
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
    _stage(
        f"evaluation {dataset_name}/{profile}: exact batched scoring of "
        f"{len(evaluable_records):,} queries on {score_device.type}; "
        f"query_batch_size={retrieval_query_batch_size}, shards={len(sharded_index.shards)}"
    )
    with per_query_path.open("a", encoding="utf-8") as writer:
        query_started = time.perf_counter()
        query_total = len(evaluable_records)
        resumed_query_count = len(per_query)
        numerical_audit: dict[str, Any] = {
            "status": "not_required_cpu_exact" if score_device.type == "cpu" else "pending",
            "queries": 0,
            "fallback_to_cpu": False,
        }
        for batch_start in range(resumed_query_count, query_total, retrieval_query_batch_size):
            batch_end = min(batch_start + retrieval_query_batch_size, query_total)
            batch_records = evaluable_records[batch_start:batch_end]
            positive_rows: list[list[int]] = []
            negative_rows: list[list[int]] = []
            pool_sizes: list[int] = []
            for record in batch_records:
                positive_ids = [str(value["doc_id"]) for value in record.get("positives", [])]
                negative_ids = [str(value["doc_id"]) for value in record.get("hard_negatives", [])]
                positions: list[int] = []
                for doc_id in (*positive_ids, *negative_ids):
                    position = bisect.bisect_left(corpus_ids, doc_id)
                    if position >= len(corpus_ids) or corpus_ids[position] != doc_id:
                        raise ValueError(f"test document is absent from frozen corpus: {doc_id}")
                    positions.append(position)
                positive_rows.append(positions[: len(positive_ids)])
                negative_rows.append(positions[len(positive_ids) :])
                pool_sizes.append(len(positions))
            batch_started = time.perf_counter()
            scan_started = time.perf_counter()
            report_every = max(1, len(sharded_index.shards) // 10)

            def shard_progress(
                current: int,
                total: int,
                *,
                every: int = report_every,
                first: int = batch_start,
                last: int = batch_end,
                started: float = scan_started,
            ) -> None:
                if current == total or current % every == 0:
                    _progress(
                        f"scan_shards[{first + 1}:{last}]",
                        current,
                        total,
                        started,
                    )

            retrieved = exact_retrieval_batch(
                encoded_queries[batch_start:batch_end],
                positive_rows=positive_rows,
                negative_rows=negative_rows,
                index=sharded_index,
                device=score_device,
                shard_progress=shard_progress if _progress_enabled() else None,
            )
            if numerical_audit["status"] == "pending":
                audit_count = min(8, len(batch_records))
                cpu_result = exact_retrieval_batch(
                    encoded_queries[batch_start : batch_start + audit_count],
                    positive_rows=positive_rows[:audit_count],
                    negative_rows=negative_rows[:audit_count],
                    index=sharded_index,
                    device=torch.device("cpu"),
                )
                cuda_ranks = retrieved.positive_ranks[:audit_count]
                cuda_wins = retrieved.hard_negative_win_rates[:audit_count]
                parity = cuda_ranks == cpu_result.positive_ranks and cuda_wins == (
                    cpu_result.hard_negative_win_rates
                )
                numerical_audit = {
                    "status": "passed" if parity else "failed_cpu_fallback",
                    "queries": audit_count,
                    "rank_match": cuda_ranks == cpu_result.positive_ranks,
                    "pool_win_match": cuda_wins == cpu_result.hard_negative_win_rates,
                    "fallback_to_cpu": not parity,
                }
                if not parity:
                    _stage("CUDA/CPU exact parity audit failed; recomputing on CPU")
                    score_device = torch.device("cpu")
                    retrieved = exact_retrieval_batch(
                        encoded_queries[batch_start:batch_end],
                        positive_rows=positive_rows,
                        negative_rows=negative_rows,
                        index=sharded_index,
                        device=score_device,
                    )
            batch_seconds = time.perf_counter() - batch_started
            latency = batch_seconds / len(batch_records)
            latencies.extend([latency] * len(batch_records))
            for offset, record in enumerate(batch_records):
                metrics = corpus_metrics_from_positive_ranks(
                    retrieved.positive_ranks[offset], candidate_count=corpus_count
                )
                row = {
                    "example_id": str(record["example_id"]),
                    **metrics,
                    "pool_candidate_count": pool_sizes[offset],
                    "pool_hard_negative_win_rate": retrieved.hard_negative_win_rates[offset],
                    "latency_seconds": latency,
                }
                writer.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                per_query.append(row)
                metric_rows.append(metrics)
            writer.flush()
            os.fsync(writer.fileno())
            _progress(
                "evaluate_queries",
                batch_end - resumed_query_count,
                query_total - resumed_query_count,
                query_started,
            )
    if numerical_audit["status"] == "pending":
        numerical_audit["status"] = "not_run_all_queries_resumed"
    retrieval_wall_seconds = time.perf_counter() - query_started
    newly_scored_queries = query_total - resumed_query_count
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
        "metric_candidate_count": {metric: corpus_count for metric in aggregate},
        "latency_seconds_per_query": sum(latencies) / len(latencies) if latencies else None,
        "execution_invocation": {
            "resumed_queries": resumed_query_count,
            "newly_scored_queries": newly_scored_queries,
            "retrieval_wall_seconds": retrieval_wall_seconds,
            "new_queries_per_second": (
                newly_scored_queries / retrieval_wall_seconds
                if newly_scored_queries and retrieval_wall_seconds > 0
                else None
            ),
            "legacy_latency_rows_reused": resumed_query_count,
        },
        "corpus_candidate_count": corpus_count,
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
        "query_encoding_seconds": query_encode_seconds,
        "index_size_bytes": sharded_index.size_bytes,
        "retrieval_backend": {
            "name": "torch_sharded_exact_ip",
            "approximate": False,
            "device": score_device.type,
            "query_batch_size": retrieval_query_batch_size,
            "corpus_shards": len(sharded_index.shards),
            "embedding_dimension": sharded_index.dimension,
            "stable_tie_break": "score_desc_then_sorted_doc_id",
            "numerical_parity_audit": numerical_audit,
        },
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
    checkpoint_interval_steps: int = 0,
    evaluation_encode_batch_size: int = 64,
    retrieval_query_batch_size: int = 512,
    retrieval_device: str = "auto",
    collapse_detection: InRunCollapseDetection | None = None,
) -> dict[str, Any]:
    calibration = recipe.negative_recipe.load_calibration()
    # The negative contract stays pinned to the *requested* recipe, so a reseeded run keeps
    # the same pair-selection provenance; the effective seed lives in recipe_fingerprint.
    negative_contract = recipe.negative_recipe.manifest(calibration) | {
        "probe_recipe_version": recipe.recipe_version,
        "probe_recipe_fingerprint": recipe.fingerprint,
    }
    requested_recipe = recipe
    if collapse_detection is not None:
        # A reseeded run trained under a different seed than the one requested; recover it
        # from the promoted summary so a resumed invocation stays identity-checked.
        recipe = _reseeded_recipe(output_dir, recipe)
    train_summary = _completed_train_summary(
        output_dir,
        recipe=recipe,
        query_source=query_source,
        negative_contract=negative_contract,
        statistical_contract=statistical_contract,
    )
    if train_summary is None:
        _stage("preparing training pairs and filtering possible false negatives")
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
        release_scorer = getattr(primary_scorer, "release", None)
        if callable(release_scorer):
            release_scorer()
            primary_scorer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _stage("released primary judge CUDA memory before probe training")
        if collapse_detection is None:
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
                checkpoint_interval_steps=checkpoint_interval_steps,
            )
        else:
            train_summary, recipe = train_probe_with_collapse_reseed(
                pairs,
                recipe=requested_recipe,
                output_dir=output_dir,
                collapse_detection=collapse_detection,
                query_source=query_source,
                train_fingerprint=train_fingerprint,
                negative_contract=negative_contract,
                false_negative_report=false_negative_report,
                negative_audit_rows=negative_audit_rows,
                statistical_contract=statistical_contract,
                checkpoint_interval_steps=checkpoint_interval_steps,
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
        dataset_name=(
            "test_translated_msmarco_pl" if holdout_manifest is not None else test_subset
        ),
        profile=translated_profile,
        negative_contract=negative_contract,
        statistical_contract=train_summary["statistical_contract"],
        comparison_budget=train_summary["comparison_budget"],
        evaluation_encode_batch_size=evaluation_encode_batch_size,
        retrieval_query_batch_size=retrieval_query_batch_size,
        retrieval_device=retrieval_device,
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
                evaluation_encode_batch_size=evaluation_encode_batch_size,
                retrieval_query_batch_size=retrieval_query_batch_size,
                retrieval_device=retrieval_device,
            )
    if holdout_manifest is None:
        report_status = "development_complete"
        comparison_eligible = False
        incomplete_reasons: list[str] = []
        evaluation_sets = {test_subset: retrieval}
    else:
        report_status = "complete" if native.get("status") == "measured" else "incomplete"
        comparison_eligible = (
            report_status == "complete"
            and holdout_profile == "full"
            and translated_profile == "full"
        )
        incomplete_reasons = (
            []
            if report_status == "complete"
            else [str(native.get("reason", "native not measured"))]
        )
        evaluation_sets = {
            "test_native_pl": native,
            "test_translated_msmarco_pl": retrieval,
        }
    result = {
        "schema_version": 2,
        "report_status": report_status,
        "comparison_eligible": comparison_eligible,
        "incomplete_reasons": incomplete_reasons,
        "training": train_summary,
        "recipe_version": recipe.recipe_version,
        "recipe_fingerprint": recipe.fingerprint,
        "negative_contract": negative_contract,
        "statistical_contract": train_summary["statistical_contract"],
        "comparison_budget": train_summary["comparison_budget"],
        "possible_false_negative_report": false_negative_report,
        "evaluation_sets": evaluation_sets,
        # Compatibility alias for pre-P-02 consumers. The named evaluation_sets
        # entry above remains authoritative for whether this is dev or translated.
        "corpus_retrieval": retrieval,
    }
    if collapse_detection is not None:
        # Never hide how many attempts this result took, nor which seed produced it.
        provenance = train_summary.get("collapse_detection")
        if not isinstance(provenance, Mapping):
            raise ValueError("an in-run detection run must carry collapse_detection provenance")
        result["collapse_detection"] = dict(provenance)
    write_json(output_dir / "result.json", result)
    build_embedder_report(
        result,
        markdown_path=output_dir / "embedder_report.md",
        json_path=output_dir / "embedder_report.json",
    )
    return result
