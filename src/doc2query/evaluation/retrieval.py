"""Strictly separated candidate-pool and corpus-retrieval metric contracts."""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import fmean
from typing import Any

CANDIDATE_POOL_RANKING = "candidate_pool_ranking"
CORPUS_RETRIEVAL = "corpus_retrieval"
POOL_RECALL_CUTOFFS = (1, 5)
CORPUS_RECALL_CUTOFFS = (1, 5, 10, 100)
CORPUS_ROUND_TRIP_CUTOFFS = (1, 5, 20, 100)


def validate_recall_cutoffs(candidate_count: int, cutoffs: Sequence[int]) -> tuple[int, ...]:
    """Reject misleading Recall@K instead of silently truncating a smaller pool."""
    normalized = tuple(int(cutoff) for cutoff in cutoffs)
    if candidate_count < 1:
        raise ValueError("retrieval candidate pool must contain at least one document")
    if not normalized or any(cutoff < 1 for cutoff in normalized):
        raise ValueError("recall cutoffs must be positive")
    invalid = [cutoff for cutoff in normalized if cutoff > candidate_count]
    if invalid:
        joined = ", ".join(str(value) for value in invalid)
        raise ValueError(
            f"cannot report recall@{joined}: candidate pool has only {candidate_count} documents"
        )
    return normalized


def reciprocal_rank(relevances: Sequence[int], cutoff: int | None = None) -> float:
    values = relevances if cutoff is None else relevances[:cutoff]
    return next((1.0 / rank for rank, value in enumerate(values, 1) if value > 0), 0.0)


def dcg(relevances: Sequence[int], cutoff: int) -> float:
    return float(
        sum(
            (2**value - 1) / math.log2(rank + 1)
            for rank, value in enumerate(relevances[:cutoff], 1)
            if value > 0
        )
    )


def ndcg(relevances: Sequence[int], cutoff: int = 10) -> float:
    actual = dcg(relevances, cutoff)
    ideal = dcg(sorted(relevances, reverse=True), cutoff)
    return actual / ideal if ideal else 0.0


def average_precision(relevances: Sequence[int]) -> float:
    relevant = sum(value > 0 for value in relevances)
    if relevant == 0:
        return 0.0
    hits = 0
    total = 0.0
    for rank, value in enumerate(relevances, 1):
        if value > 0:
            hits += 1
            total += hits / rank
    return total / relevant


def candidate_pool_metrics_from_rank(
    rank: int,
    *,
    candidate_count: int,
    recall_cutoffs: Sequence[int] = POOL_RECALL_CUTOFFS,
) -> dict[str, float | int]:
    """Metrics for one known positive ranked against a controlled candidate pool."""
    cutoffs = validate_recall_cutoffs(candidate_count, recall_cutoffs)
    if rank < 1 or rank > candidate_count:
        raise ValueError("pool rank must be within the candidate pool")
    result: dict[str, float | int] = {
        "pool_candidate_count": candidate_count,
        "pool_rank": rank,
        "pool_mrr": 1.0 / rank,
        "pool_ndcg_at_10": 1.0 / math.log2(rank + 1) if rank <= 10 else 0.0,
        "pool_hard_negative_win_rate": float(rank == 1),
    }
    result.update({f"pool_recall_at_{cutoff}": float(rank <= cutoff) for cutoff in cutoffs})
    return result


def corpus_metrics_from_positive_ranks(
    positive_ranks: Sequence[int],
    *,
    candidate_count: int,
    recall_cutoffs: Sequence[int] = CORPUS_RECALL_CUTOFFS,
) -> dict[str, float | int]:
    """Exact full-corpus metrics without materializing a corpus-sized relevance vector."""
    cutoffs = validate_recall_cutoffs(candidate_count, recall_cutoffs)
    if not positive_ranks or any(rank < 1 or rank > candidate_count for rank in positive_ranks):
        raise ValueError("positive ranks must be non-empty and within the corpus")
    ranks = sorted(positive_ranks)
    relevant = len(ranks)
    dcg_at_10 = sum(1 / math.log2(rank + 1) for rank in ranks if rank <= 10)
    ideal_at_10 = sum(1 / math.log2(rank + 1) for rank in range(1, min(relevant, 10) + 1))
    result: dict[str, float | int] = {
        "corpus_candidate_count": candidate_count,
        "corpus_mrr_at_10": 1 / ranks[0] if ranks[0] <= 10 else 0.0,
        "corpus_ndcg_at_10": dcg_at_10 / ideal_at_10 if ideal_at_10 else 0.0,
        "corpus_map": sum(index / rank for index, rank in enumerate(ranks, 1)) / relevant,
    }
    result.update(
        {
            f"corpus_recall_at_{cutoff}": sum(rank <= cutoff for rank in ranks) / relevant
            for cutoff in cutoffs
        }
    )
    return result


def corpus_round_trip_metrics(
    positive_ranks: Sequence[int],
    *,
    candidate_count: int,
    cutoffs: Sequence[int] = CORPUS_ROUND_TRIP_CUTOFFS,
) -> dict[str, float | int]:
    """Return source-document hit rates for a generated query over the full corpus."""
    normalized = validate_recall_cutoffs(candidate_count, cutoffs)
    ranks = sorted(rank for rank in positive_ranks if 1 <= rank <= candidate_count)
    return {
        "corpus_candidate_count": candidate_count,
        **{
            f"corpus_round_trip_at_{cutoff}": float(any(rank <= cutoff for rank in ranks))
            for cutoff in normalized
        },
    }


def aggregate_query_metrics(rows: Sequence[dict[str, float | int]]) -> dict[str, float | None]:
    if not rows:
        return {}
    metric_keys = sorted(
        key
        for key in set().union(*(row.keys() for row in rows))
        if not key.endswith("_candidate_count")
    )
    return {
        key: fmean(float(row[key]) for row in rows if key in row)
        if any(key in row for row in rows)
        else None
        for key in metric_keys
    }


def rank_from_scores(positive_score: float, negative_scores: Sequence[float]) -> int:
    """Use stable pessimistic ties: equal negative scores precede the positive."""
    return 1 + sum(score >= positive_score for score in negative_scores)


def pearson_correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right):
        raise ValueError("correlation inputs must have the same length")
    if len(left) < 2:
        return None
    left_mean, right_mean = fmean(left), fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True))
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    return numerator / (left_scale * right_scale) if left_scale and right_scale else None


def percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between 0 and 1")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def distribution(values: Sequence[float]) -> dict[str, Any] | None:
    if not values:
        return None
    return {
        "count": len(values),
        "mean": fmean(values),
        "min": min(values),
        "p05": percentile(values, 0.05),
        "p25": percentile(values, 0.25),
        "p50": percentile(values, 0.50),
        "p75": percentile(values, 0.75),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }
