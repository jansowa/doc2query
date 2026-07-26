"""Intrinsic generator scoring over self-contained generation records."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise
from pathlib import Path
from statistics import fmean
from typing import Any

from doc2query.data.invert import query_style
from doc2query.evaluation.corpus import CorpusIndex, evaluate_round_trip_queries
from doc2query.evaluation.diversity import diversity_metrics
from doc2query.evaluation.format import format_metrics
from doc2query.evaluation.retrieval import (
    CANDIDATE_POOL_RANKING,
    CORPUS_RETRIEVAL,
    CORPUS_ROUND_TRIP_CUTOFFS,
    candidate_pool_metrics_from_rank,
    distribution,
    pearson_correlation,
)
from doc2query.evaluation.slices import aggregate_slices, rank_buckets
from doc2query.evaluation.translationese import aggregate_translationese
from doc2query.reranker.base import PairScorer
from doc2query.reranker.focus import assign_focus, split_sentences
from doc2query.reranker.infer import score_group
from doc2query.rewards.lexical import lexical_metrics
from doc2query.text.normalization import SimplePolishNormalizer
from doc2query.utils.records import JsonlWriter, write_json

SLICE_FIELDS = [
    "natural_overlap_quantile",
    "passage_length",
    "sentence_count",
    "target_sentence_position",
    "domain",
    "query_style",
    "entity_or_number",
    "positive_count",
    "reranker_difficulty",
    "near_duplicate_cluster_size",
]

KEY_METRICS = [
    "pool_recall_at_1",
    "pool_recall_at_5",
    "pool_mrr",
    "pool_ndcg_at_10",
    "pool_margin",
    "content_jaccard",
    "normalized_lcs",
    "copy_density",
    "format_valid",
    "sentence_level_source_hit",
    "reference_focus_agreement",
]

POOL_METRICS = (
    "pool_recall_at_1",
    "pool_recall_at_5",
    "pool_mrr",
    "pool_ndcg_at_10",
    "pool_hard_negative_win_rate",
)
ROUND_TRIP_METRICS = tuple(f"corpus_round_trip_at_{cutoff}" for cutoff in CORPUS_ROUND_TRIP_CUTOFFS)


def _document_texts(record: dict[str, Any]) -> tuple[str, str, list[str], list[str]]:
    positive = record.get("positive")
    negatives = record.get("hard_negatives")
    if not isinstance(positive, dict) or not isinstance(positive.get("text"), str):
        raise ValueError("generation record requires positive{text,doc_id}")
    if not isinstance(negatives, list) or len(negatives) < 10:
        raise ValueError("intrinsic retrieval scoring requires at least 10 hard negatives")
    negative_texts = [str(value["text"]) for value in negatives]
    negative_ids = [str(value["doc_id"]) for value in negatives]
    return (
        str(positive["text"]),
        str(positive.get("doc_id", "")),
        negative_texts,
        negative_ids,
    )


def _bucket(index: int, count: int) -> str:
    relative = (index + 0.5) / count
    return "beginning" if relative <= 1 / 3 else "middle" if relative <= 2 / 3 else "end"


def _difficulty(rank: int) -> str:
    return "easy" if rank == 1 else "medium" if rank <= 5 else "hard"


def _slice_base(
    record: dict[str, Any],
    *,
    style: str,
    target_focus: int,
    sentence_count: int,
    reference_rank: int,
) -> dict[str, str]:
    metadata = record.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    passage = str(record["positive"]["text"])
    positive_count = int(record.get("positive_count", 1))
    near_duplicate = metadata.get("near_duplicate_cluster_size", "unknown")
    has_number = bool(SimplePolishNormalizer().analyze(passage).numbers)
    return {
        "passage_length": (
            "short"
            if len(passage.split()) < 64
            else "medium"
            if len(passage.split()) < 192
            else "long"
        ),
        "sentence_count": (
            "one" if sentence_count == 1 else "two_to_four" if sentence_count <= 4 else "five_plus"
        ),
        "target_sentence_position": _bucket(target_focus, sentence_count),
        "domain": str(metadata.get("domain", metadata.get("source", "unknown"))),
        "query_style": style,
        "entity_or_number": "number" if has_number else "none_detected_simple_backend",
        "positive_count": str(positive_count),
        "reranker_difficulty": _difficulty(reference_rank),
        "near_duplicate_cluster_size": str(near_duplicate),
    }


def _mean_rate(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if isinstance(row.get(field), (int, float))]
    return fmean(values) if values else None


def _entropy(values: list[str]) -> float | None:
    if not values:
        return None
    counts = Counter(values)
    return -sum((count / len(values)) * math.log2(count / len(values)) for count in counts.values())


def _gini(values: list[int]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    total = sum(ordered)
    if total == 0:
        return 0.0
    count = len(ordered)
    weighted = sum(index * value for index, value in enumerate(ordered, 1))
    return (2 * weighted) / (count * total) - (count + 1) / count


def _mode_summary(rows: list[dict[str, Any]], group_rows: list[dict[str, Any]]) -> dict[str, Any]:
    lexical_fields = (
        "content_jaccard",
        "query_precision",
        "passage_recall",
        "longest_copied_ngram",
        "normalized_lcs",
        "copy_density",
        "number_preservation",
        "unit_preservation",
    )
    focus_buckets = [str(row["predicted_focus_bucket"]) for row in rows]
    sentence_counts = Counter(int(row["predicted_sentence_index"]) for row in rows)
    diversity_fields = (
        "distinct_1",
        "distinct_2",
        "self_bleu",
        "mean_pairwise_lemma_jaccard",
        "max_pairwise_lemma_jaccard",
        "duplicate_rate",
        "style_entropy",
        "focus_entropy",
    )
    return {
        "generation_count": len(rows),
        "example_count": len({str(row["example_id"]) for row in rows}),
        "candidate_pool_ranking": {
            "protocol": CANDIDATE_POOL_RANKING,
            "candidate_count": distribution([float(row["pool_candidate_count"]) for row in rows]),
            "metrics": {field: _mean_rate(rows, field) for field in POOL_METRICS},
        },
        "corpus_retrieval": {
            "protocol": CORPUS_RETRIEVAL,
            "status": (
                "measured"
                if any(isinstance(row.get(ROUND_TRIP_METRICS[0]), (int, float)) for row in rows)
                else "not_measured"
            ),
            "candidate_count": distribution(
                [
                    float(row["corpus_candidate_count"])
                    for row in rows
                    if isinstance(row.get("corpus_candidate_count"), (int, float))
                ]
            ),
            "metrics": {field: _mean_rate(rows, field) for field in ROUND_TRIP_METRICS},
        },
        "reranker_margin": distribution([float(row["pool_margin"]) for row in rows]),
        "lexical": {
            **{
                field: distribution(
                    [float(row[field]) for row in rows if isinstance(row.get(field), (int, float))]
                )
                for field in lexical_fields
            },
            "entity_preservation": None,
        },
        "format": {
            "empty_rate": _mean_rate(rows, "empty"),
            "multiple_query_rate": _mean_rate(rows, "multiple_query"),
            "prefix_rate": _mean_rate(rows, "has_prefix"),
            "metacomment_rate": _mean_rate(rows, "has_metacomment"),
            "valid_rate": _mean_rate(rows, "format_valid"),
            "invalid_character_rate": fmean(
                int(float(row["invalid_character_count"]) > 0) for row in rows
            ),
            "length": distribution([float(row["word_length"]) for row in rows]),
        },
        "focus": {
            "control_accuracy": _mean_rate(rows, "focus_accuracy"),
            "reference_focus_agreement": _mean_rate(rows, "reference_focus_agreement"),
            "sentence_level_source_hit": _mean_rate(rows, "sentence_level_source_hit"),
            "first_sentence_concentration": fmean(
                int(row["predicted_sentence_index"] == 0) for row in rows
            ),
            "bucket_distribution": dict(Counter(focus_buckets)),
            "bucket_entropy": _entropy(focus_buckets),
            "sentence_index_gini": _gini(list(sentence_counts.values())),
        },
        "diversity": {
            field: distribution(
                [
                    float(row[field])
                    for row in group_rows
                    if isinstance(row.get(field), (int, float)) and math.isfinite(float(row[field]))
                ]
            )
            for field in diversity_fields
        },
    }


def mode_summaries(
    measured: list[dict[str, Any]], group_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    modes = sorted({str(row.get("mode", "unknown")) for row in measured})
    return {
        mode: _mode_summary(
            [row for row in measured if str(row.get("mode", "unknown")) == mode],
            [row for row in group_rows if str(row.get("mode", "unknown")) == mode],
        )
        for mode in modes
    }


class _FixedPairScorer:
    def __init__(self, name: str, scores: Sequence[float]) -> None:
        self.name = name
        self._scores = list(scores)

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        if len(pairs) != len(self._scores):
            raise RuntimeError("batched scorer replay shape mismatch")
        return self._scores


def _score_pair_groups(
    scorer: PairScorer, groups: Sequence[Sequence[tuple[str, str]]]
) -> list[list[float]]:
    offsets = [0]
    flattened: list[tuple[str, str]] = []
    for group in groups:
        flattened.extend(group)
        offsets.append(len(flattened))
    scores = scorer.score_pairs(flattened) if flattened else []
    if len(scores) != len(flattened):
        raise ValueError("scorer returned an invalid batched score count")
    return [scores[left:right] for left, right in pairwise(offsets)]


def _resume_identity(
    records: Sequence[dict[str, Any]],
    *,
    primary: PairScorer,
    shadow: PairScorer | None,
    test_fingerprint: str,
    experiment_id: str,
    corpus_index: CorpusIndex | None,
) -> dict[str, Any]:
    record_digest = hashlib.sha256()
    for record in records:
        record_digest.update(
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        )
        record_digest.update(b"\n")
    return {
        "schema_version": 1,
        "scoring_contract": "intrinsic-batched-resume-v1",
        "experiment_id": experiment_id,
        "test_fingerprint": test_fingerprint,
        "record_count": len(records),
        "records_sha256": record_digest.hexdigest(),
        "primary_judge": primary.name,
        "primary_config": repr(getattr(primary, "config", None)),
        "shadow_judge": shadow.name if shadow else None,
        "shadow_config": repr(getattr(shadow, "config", None)) if shadow else None,
        "corpus_index_fingerprint": (
            str(corpus_index.metadata.get("index_fingerprint")) if corpus_index else None
        ),
    }


def _append_checkpoint_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _archive_incompatible_scoring_state(
    output_dir: Path,
    *,
    previous_identity: dict[str, Any],
    next_identity: dict[str, Any],
) -> Path:
    """Move an incompatible scoring run aside without deleting its artifacts."""
    def fingerprint(value: dict[str, Any]) -> str:
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:12]
    root = output_dir / "interrupted-scoring"
    destination = root / (
        f"{fingerprint(previous_identity)}-to-{fingerprint(next_identity)}"
    )
    suffix = 1
    while destination.exists():
        destination = root / (
            f"{fingerprint(previous_identity)}-to-{fingerprint(next_identity)}-{suffix}"
        )
        suffix += 1
    destination.mkdir(parents=True)
    artifact_names = (
        "scoring.resume.json",
        "scoring.journal.jsonl",
        "summary.json",
        "per_generation.jsonl",
        "per_group_diversity.jsonl",
        "report.md",
        "report.html",
        "result.json",
    )
    for name in artifact_names:
        source = output_dir / name
        if source.exists():
            os.replace(source, destination / name)
    write_json(
        destination / "archive_manifest.json",
        {
            "schema_version": 1,
            "reason": "explicitly_archived_incompatible_scoring_identity",
            "previous_identity": previous_identity,
            "next_identity": next_identity,
        },
    )
    return destination


def _read_checkpoint_rows(path: Path) -> list[dict[str, Any]]:
    """Read a durable prefix and discard only a crash-truncated final line."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    valid_bytes = 0
    with path.open("rb") as handle:
        while line := handle.readline():
            complete = line.endswith(b"\n")
            if not complete:
                break
            try:
                value = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("intrinsic scoring journal contains a malformed row") from exc
            if not isinstance(value, dict):
                raise ValueError("intrinsic scoring journal row must be an object")
            rows.append(value)
            valid_bytes = handle.tell()
    if path.stat().st_size != valid_bytes:
        with path.open("r+b") as handle:
            handle.truncate(valid_bytes)
            handle.flush()
            os.fsync(handle.fileno())
    return rows


def evaluate_intrinsic_records(
    records: list[dict[str, Any]],
    *,
    primary: PairScorer,
    shadow: PairScorer | None,
    output_dir: Path,
    test_fingerprint: str,
    experiment_id: str,
    corpus_index: CorpusIndex | None = None,
    scoring_batch_size: int = 64,
    bm25_workers: int = 8,
    progress_every: int = 100,
    archive_incompatible_scoring: bool = False,
) -> dict[str, Any]:
    if not records:
        raise ValueError("intrinsic evaluation requires generations")
    if scoring_batch_size < 1 or bm25_workers < 1 or progress_every < 1:
        raise ValueError("scoring batch size, BM25 workers and progress interval must be positive")
    normalizer = SimplePolishNormalizer()
    output_dir.mkdir(parents=True, exist_ok=True)
    journal_path = output_dir / "scoring.journal.jsonl"
    identity_path = output_dir / "scoring.resume.json"
    identity = _resume_identity(
        records,
        primary=primary,
        shadow=shadow,
        test_fingerprint=test_fingerprint,
        experiment_id=experiment_id,
        corpus_index=corpus_index,
    )
    archived_scoring: Path | None = None
    if identity_path.exists():
        existing_identity = json.loads(identity_path.read_text(encoding="utf-8"))
        if existing_identity != identity:
            if not archive_incompatible_scoring:
                raise ValueError("intrinsic scoring resume identity mismatch")
            archived_scoring = _archive_incompatible_scoring_state(
                output_dir,
                previous_identity=existing_identity,
                next_identity=identity,
            )
            write_json(identity_path, identity)
    elif journal_path.exists() and journal_path.stat().st_size:
        raise ValueError("intrinsic scoring journal exists without resume identity")
    else:
        temporary_identity = identity_path.with_suffix(".json.tmp")
        write_json(temporary_identity, identity)
        os.replace(temporary_identity, identity_path)
    measured = _read_checkpoint_rows(journal_path)
    if len(measured) > len(records):
        raise ValueError("intrinsic scoring journal contains too many rows")
    for index, checkpoint_row in enumerate(measured):
        expected_id = str(records[index].get("evaluation_id", index))
        if str(checkpoint_row.get("evaluation_id", index)) != expected_id:
            raise ValueError("intrinsic scoring journal is not the expected generation prefix")
    reference_cache: dict[str, tuple[int, int]] = {}
    started = time.perf_counter()
    resumed_count = len(measured)
    print(
        f"[intrinsic resume] {resumed_count:,}/{len(records):,} rows durable; "
        f"batch={scoring_batch_size} bm25_workers={bm25_workers}",
        file=sys.stderr,
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=1) as corpus_executor:
        for batch_start in range(resumed_count, len(records), scoring_batch_size):
            batch = records[batch_start : batch_start + scoring_batch_size]
            documents = [_document_texts(record) for record in batch]
            corpus_future = (
                corpus_executor.submit(
                    evaluate_round_trip_queries,
                    corpus_index,
                    [
                        (str(record.get("generated", "")), (document[1],))
                        for record, document in zip(batch, documents, strict=True)
                    ],
                    workers=bm25_workers,
                )
                if corpus_index is not None
                else None
            )

            generated_groups = [
                [(str(record.get("generated", "")), passage)]
                + [(str(record.get("generated", "")), negative) for negative in negatives]
                for record, (passage, _positive_id, negatives, _negative_ids) in zip(
                    batch, documents, strict=True
                )
            ]
            generated_focus_groups = [
                [
                    (str(record.get("generated", "")), sentence)
                    for sentence in split_sentences(passage)
                ]
                for record, (passage, _positive_id, _negatives, _negative_ids) in zip(
                    batch, documents, strict=True
                )
            ]
            primary_group_scores = _score_pair_groups(primary, generated_groups)
            primary_focus_scores = _score_pair_groups(primary, generated_focus_groups)
            shadow_group_scores = (
                _score_pair_groups(shadow, generated_groups) if shadow is not None else None
            )

            new_references: dict[
                str, tuple[dict[str, Any], tuple[str, str, list[str], list[str]]]
            ] = {}
            for record, document in zip(batch, documents, strict=True):
                example_id = str(record["example_id"])
                if example_id not in reference_cache and example_id not in new_references:
                    new_references[example_id] = (record, document)
            reference_items = list(new_references.items())
            reference_group_pairs = [
                [(str(record.get("reference", "")), passage)]
                + [(str(record.get("reference", "")), negative) for negative in negatives]
                for _example_id, (
                    record,
                    (passage, _positive_id, negatives, _negative_ids),
                ) in reference_items
            ]
            reference_focus_pairs = [
                [
                    (str(record.get("reference", "")), sentence)
                    for sentence in split_sentences(passage)
                ]
                for _example_id, (
                    record,
                    (passage, _positive_id, _negatives, _negative_ids),
                ) in reference_items
            ]
            reference_group_scores = _score_pair_groups(primary, reference_group_pairs)
            reference_focus_scores = _score_pair_groups(primary, reference_focus_pairs)
            for position, (example_id, (record, document)) in enumerate(reference_items):
                passage, _positive_id, negatives, _negative_ids = document
                reference = score_group(
                    _FixedPairScorer(primary.name, reference_group_scores[position]),
                    example_id=f"reference::{example_id}",
                    query=str(record.get("reference", "")),
                    positive=passage,
                    negatives=negatives,
                )
                reference_focus = assign_focus(
                    _FixedPairScorer(primary.name, reference_focus_scores[position]),
                    str(record.get("reference", "")),
                    passage,
                ).focus_sentence_id
                reference_cache[example_id] = (reference.pool_rank, reference_focus)

            corpus_rows = (
                corpus_future.result() if corpus_future is not None else [{} for _ in batch]
            )
            batch_rows: list[dict[str, Any]] = []
            for offset, (record, document) in enumerate(zip(batch, documents, strict=True)):
                index = batch_start + offset
                passage, positive_doc_id, negatives, negative_ids = document
                generated = str(record.get("generated", ""))
                identifier = str(record.get("evaluation_id", index))
                example_id = str(record["example_id"])
                primary_score = score_group(
                    _FixedPairScorer(primary.name, primary_group_scores[offset]),
                    example_id=identifier,
                    query=generated,
                    positive=passage,
                    negatives=negatives,
                    query_id=example_id,
                    positive_doc_id=positive_doc_id,
                    negative_doc_ids=tuple(negative_ids),
                )
                reference_rank, reference_focus = reference_cache[example_id]
                focus = assign_focus(
                    _FixedPairScorer(primary.name, primary_focus_scores[offset]),
                    generated,
                    passage,
                )
                source_sentences = split_sentences(passage)
                sentence_source_hit = float(
                    focus.focus_score > primary_score.hardest_negative_score
                )
                lexical = lexical_metrics(
                    normalizer.analyze(generated), normalizer.analyze(passage)
                )
                natural_lexical = lexical_metrics(
                    normalizer.analyze(str(record.get("reference", ""))),
                    normalizer.analyze(passage),
                )
                format_result = format_metrics(
                    generated, multi_query_json=bool(record.get("multi_query_json", False))
                )
                pool_metrics = candidate_pool_metrics_from_rank(
                    primary_score.pool_rank,
                    candidate_count=len(primary_score.document_scores),
                )
                corpus_metrics = corpus_rows[offset]
                shadow_result = None
                if shadow is not None:
                    if shadow_group_scores is None:
                        raise RuntimeError("missing batched shadow scores")
                    shadow_result = score_group(
                        _FixedPairScorer(shadow.name, shadow_group_scores[offset]),
                        example_id=identifier,
                        query=generated,
                        positive=passage,
                        negatives=negatives,
                    )
                row: dict[str, Any] = {
                    **record,
                    **pool_metrics,
                    **corpus_metrics,
                    **lexical.to_dict(),
                    **format_result,
                    "primary_judge": primary.name,
                    "pool_positive_score": primary_score.positive_score,
                    "pool_margin": primary_score.pool_margin,
                    "shadow_judge": shadow.name if shadow else None,
                    "shadow_score": shadow_result.positive_score if shadow_result else None,
                    "shadow_pool_margin": shadow_result.pool_margin if shadow_result else None,
                    "judge_rank_disagreement": (
                        primary_score.pool_rank != shadow_result.pool_rank
                        if shadow_result
                        else None
                    ),
                    "predicted_sentence_index": focus.focus_sentence_id,
                    "predicted_focus_bucket": focus.focus_bucket,
                    "focus_accuracy": (
                        float(focus.focus_bucket == str(record["requested_focus_bucket"]))
                        if record.get("requested_focus_bucket") is not None
                        else None
                    ),
                    "reference_focus_agreement": float(focus.focus_sentence_id == reference_focus),
                    "natural_content_jaccard": natural_lexical.content_jaccard,
                    "sentence_level_source_hit": sentence_source_hit,
                    "sentence_count": len(source_sentences),
                    "predicted_style": query_style(generated),
                }
                row["slices"] = _slice_base(
                    record,
                    style=str(row["predicted_style"]),
                    target_focus=reference_focus,
                    sentence_count=len(source_sentences),
                    reference_rank=reference_rank,
                )
                batch_rows.append(row)
            _append_checkpoint_rows(journal_path, batch_rows)
            measured.extend(batch_rows)
            completed = len(measured)
            crossed_progress_boundary = completed // progress_every != batch_start // progress_every
            if completed == len(records) or crossed_progress_boundary:
                elapsed = time.perf_counter() - started
                rate = (completed - resumed_count) / elapsed if elapsed > 0 else 0.0
                remaining = (len(records) - completed) / rate if rate > 0 else float("inf")
                print(
                    f"[intrinsic progress] {completed:,}/{len(records):,} "
                    f"({completed / len(records):.1%}) rate={rate:.2f}/s "
                    f"eta={remaining / 60:.1f} min",
                    file=sys.stderr,
                    flush=True,
                )

    temporary_rows = output_dir / "per_generation.jsonl.tmp"
    with JsonlWriter(temporary_rows) as writer:
        overlaps = [float(row["natural_content_jaccard"]) for row in measured]
        overlap_buckets = rank_buckets(overlaps, ("low", "medium", "high"))
        for row, bucket in zip(measured, overlap_buckets, strict=True):
            row["slices"]["natural_overlap_quantile"] = bucket
            writer.write(row)
    os.replace(temporary_rows, output_dir / "per_generation.jsonl")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in measured:
        grouped[(str(row["example_id"]), str(row.get("mode", "unknown")))].append(row)
    group_rows = []
    with JsonlWriter(output_dir / "per_group_diversity.jsonl") as writer:
        for (example_id, mode), values in sorted(grouped.items()):
            diversity = diversity_metrics(
                [str(value["generated"]) for value in values],
                styles=[str(value["predicted_style"]) for value in values],
                focus_buckets=[str(value["predicted_focus_bucket"]) for value in values],
            )
            row = {"example_id": example_id, "mode": mode, **diversity}
            writer.write(row)
            group_rows.append(row)

    margin_values = [float(row["pool_margin"]) for row in measured]
    lexical_fields = (
        "content_jaccard",
        "query_precision",
        "passage_recall",
        "longest_copied_ngram",
        "normalized_lcs",
        "copy_density",
        "number_preservation",
        "unit_preservation",
        "entity_preservation",
    )
    summary = {
        "schema_version": 2,
        "status": "measured",
        "experiment_id": experiment_id,
        "test_fingerprint": test_fingerprint,
        "generation_count": len(measured),
        "example_count": len({str(row["example_id"]) for row in measured}),
        "elapsed_seconds": time.perf_counter() - started,
        "resume": {
            "journal": str(journal_path),
            "resumed_generation_count": resumed_count,
            "durable_generation_count": len(measured),
            "archived_incompatible_scoring": (
                str(archived_scoring) if archived_scoring is not None else None
            ),
        },
        "execution": {
            "scoring_batch_size": scoring_batch_size,
            "bm25_workers": bm25_workers,
            "progress_every": progress_every,
            "gpu_and_corpus_overlapped": corpus_index is not None,
            "primary_device": getattr(getattr(primary, "config", None), "device", None),
            "shadow_device": getattr(getattr(shadow, "config", None), "device", None),
        },
        "judges": {
            "primary": primary.name,
            "shadow": shadow.name if shadow else None,
            "shadow_status": "measured" if shadow else "not_measured",
        },
        "protocols": {
            CANDIDATE_POOL_RANKING: {
                "protocol": CANDIDATE_POOL_RANKING,
                "role": "generator_grounding_diagnostic",
                "metric_prefix": "pool_",
                "candidate_count": distribution(
                    [float(row["pool_candidate_count"]) for row in measured]
                ),
                "metrics": {field: _mean_rate(measured, field) for field in POOL_METRICS},
                "metric_candidate_count": {
                    field: distribution([float(row["pool_candidate_count"]) for row in measured])
                    for field in POOL_METRICS
                },
            },
            CORPUS_RETRIEVAL: {
                "protocol": CORPUS_RETRIEVAL,
                "role": "generator_comparison_basis_and_round_trip",
                "metric_prefix": "corpus_",
                "status": "measured" if corpus_index is not None else "not_measured",
                "index": dict(corpus_index.metadata) if corpus_index is not None else None,
                "candidate_count": (
                    corpus_index.candidate_count if corpus_index is not None else None
                ),
                "metrics": {field: _mean_rate(measured, field) for field in ROUND_TRIP_METRICS},
                "metric_candidate_count": {
                    field: corpus_index.candidate_count if corpus_index is not None else None
                    for field in ROUND_TRIP_METRICS
                },
                "effective_candidate_count": distribution(
                    [
                        float(row["corpus_effective_candidate_count"])
                        for row in measured
                        if isinstance(row.get("corpus_effective_candidate_count"), (int, float))
                    ]
                ),
                "margin_to_best_nonpositive": distribution(
                    [
                        float(row["corpus_margin_to_best_nonpositive"])
                        for row in measured
                        if isinstance(row.get("corpus_margin_to_best_nonpositive"), (int, float))
                    ]
                ),
                "possibly_ambiguous_query_rate": _mean_rate(
                    measured, "corpus_possibly_ambiguous_query"
                ),
                "round_trip_pool_margin_correlation": {
                    field: pearson_correlation(
                        [
                            float(row["pool_margin"])
                            for row in measured
                            if isinstance(row.get(field), (int, float))
                        ],
                        [
                            float(row[field])
                            for row in measured
                            if isinstance(row.get(field), (int, float))
                        ],
                    )
                    for field in ROUND_TRIP_METRICS
                },
            },
        },
        "reranker_margin": distribution(margin_values),
        "lexical": {
            **{
                field: distribution(
                    [
                        float(row[field])
                        for row in measured
                        if isinstance(row.get(field), (int, float))
                    ]
                )
                for field in lexical_fields
                if field != "entity_preservation"
            },
            "entity_preservation": None,
        },
        "format": {
            "empty_rate": _mean_rate(measured, "empty"),
            "multiple_query_rate": _mean_rate(measured, "multiple_query"),
            "prefix_rate": _mean_rate(measured, "has_prefix"),
            "metacomment_rate": _mean_rate(measured, "has_metacomment"),
            "valid_rate": _mean_rate(measured, "format_valid"),
            "length": distribution([float(row["word_length"]) for row in measured]),
            "language_confidence_pl": distribution(
                [
                    float(row["language_confidence_pl"])
                    for row in measured
                    if row["language_confidence_pl"] is not None
                ]
            ),
        },
        "translationese": {
            "generated": aggregate_translationese(str(row["generated"]) for row in measured),
            "natural_reference": aggregate_translationese(
                str(row["reference"]) for row in measured
            ),
            "warning": "surface diagnostic only; it is not proof of translation or naturalness",
        },
        "focus": {
            "sentence_level_source_hit": _mean_rate(measured, "sentence_level_source_hit"),
            "control_accuracy": _mean_rate(measured, "focus_accuracy"),
            "reference_focus_agreement": _mean_rate(measured, "reference_focus_agreement"),
            "first_sentence_concentration": fmean(
                int(row["predicted_sentence_index"] == 0) for row in measured
            ),
            "bucket_distribution": dict(
                Counter(str(row["predicted_focus_bucket"]) for row in measured)
            ),
            "bucket_entropy": _entropy([str(row["predicted_focus_bucket"]) for row in measured]),
            "sentence_index_gini": _gini(
                list(Counter(int(row["predicted_sentence_index"]) for row in measured).values())
            ),
        },
        "diversity": {
            field: distribution(
                [
                    float(row[field])
                    for row in group_rows
                    if isinstance(row.get(field), (int, float)) and math.isfinite(float(row[field]))
                ]
            )
            for field in (
                "distinct_1",
                "distinct_2",
                "self_bleu",
                "mean_pairwise_lemma_jaccard",
                "max_pairwise_lemma_jaccard",
                "duplicate_rate",
                "semantic_cluster_count",
                "style_entropy",
                "focus_entropy",
            )
        },
        "slices": aggregate_slices(measured, slice_fields=SLICE_FIELDS, metric_fields=KEY_METRICS),
        "modes": mode_summaries(measured, group_rows),
        "unmeasured": [
            "pairwise_embedding_cosine",
            "semantic_cluster_count_without_embedding_backend",
            "human_answerability",
            "probe_embedder",
            *(["corpus_retrieval"] if corpus_index is None else []),
        ],
    }
    write_json(output_dir / "summary.json", summary)
    return summary
