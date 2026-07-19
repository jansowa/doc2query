"""Intrinsic generator scoring over self-contained generation records."""

from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

from doc2query.data.invert import query_style
from doc2query.evaluation.corpus import CorpusIndex, evaluate_round_trip_query
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


def evaluate_intrinsic_records(
    records: list[dict[str, Any]],
    *,
    primary: PairScorer,
    shadow: PairScorer | None,
    output_dir: Path,
    test_fingerprint: str,
    experiment_id: str,
    corpus_index: CorpusIndex | None = None,
) -> dict[str, Any]:
    if not records:
        raise ValueError("intrinsic evaluation requires generations")
    normalizer = SimplePolishNormalizer()
    output_dir.mkdir(parents=True, exist_ok=True)
    measured: list[dict[str, Any]] = []
    reference_cache: dict[str, tuple[int, int]] = {}
    started = time.perf_counter()
    with JsonlWriter(output_dir / "per_generation.jsonl") as writer:
        for index, record in enumerate(records):
            passage, positive_doc_id, negatives, negative_ids = _document_texts(record)
            generated = str(record.get("generated", ""))
            identifier = str(record.get("evaluation_id", index))
            example_id = str(record["example_id"])
            primary_score = score_group(
                primary,
                example_id=identifier,
                query=generated,
                positive=passage,
                negatives=negatives,
                query_id=example_id,
                positive_doc_id=positive_doc_id,
                negative_doc_ids=tuple(negative_ids),
            )
            if example_id not in reference_cache:
                reference = score_group(
                    primary,
                    example_id=f"reference::{example_id}",
                    query=str(record.get("reference", "")),
                    positive=passage,
                    negatives=negatives,
                )
                reference_focus = assign_focus(
                    primary, str(record.get("reference", "")), passage
                ).focus_sentence_id
                reference_cache[example_id] = (reference.pool_rank, reference_focus)
            reference_rank, reference_focus = reference_cache[example_id]
            focus = assign_focus(primary, generated, passage)
            source_sentences = split_sentences(passage)
            sentence_source_hit = float(focus.focus_score > primary_score.hardest_negative_score)
            lexical = lexical_metrics(normalizer.analyze(generated), normalizer.analyze(passage))
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
            corpus_metrics = (
                evaluate_round_trip_query(
                    corpus_index,
                    query=generated,
                    positive_doc_ids=(positive_doc_id,),
                )
                if corpus_index is not None
                else {}
            )
            shadow_result = None
            if shadow is not None:
                shadow_result = score_group(
                    shadow,
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
                    primary_score.pool_rank != shadow_result.pool_rank if shadow_result else None
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
            measured.append(row)
        overlaps = [float(row["natural_content_jaccard"]) for row in measured]
        overlap_buckets = rank_buckets(overlaps, ("low", "medium", "high"))
        for row, bucket in zip(measured, overlap_buckets, strict=True):
            row["slices"]["natural_overlap_quantile"] = bucket
            writer.write(row)

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
