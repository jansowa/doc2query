"""Retrieval metrics shared by reranker and probe-embedder evaluation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import fmean
from typing import Any


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


def per_query_metrics(relevances: Sequence[int]) -> dict[str, float]:
    if not relevances:
        raise ValueError("retrieval ranking cannot be empty")
    relevant = sum(value > 0 for value in relevances)
    if relevant == 0:
        raise ValueError("retrieval ranking requires at least one positive")
    first_negative = next((index for index, value in enumerate(relevances) if value <= 0), None)
    positive_before_negative = (
        sum(value > 0 for value in relevances)
        if first_negative is None
        else sum(value > 0 for value in relevances[:first_negative])
    )
    return {
        "recall_at_1": sum(relevances[:1]) / relevant,
        "recall_at_5": sum(relevances[:5]) / relevant,
        "recall_at_10": sum(relevances[:10]) / relevant,
        "recall_at_100": sum(relevances[:100]) / relevant,
        "mrr_at_10": reciprocal_rank(relevances, 10),
        "mrr": reciprocal_rank(relevances),
        "ndcg_at_10": ndcg(relevances, 10),
        "map": average_precision(relevances),
        "hard_negative_win_rate": positive_before_negative / relevant,
    }


def aggregate_query_metrics(rows: Sequence[dict[str, float]]) -> dict[str, float | None]:
    if not rows:
        return {}
    keys = sorted(set().union(*(row.keys() for row in rows)))
    return {
        key: fmean(float(row[key]) for row in rows if key in row)
        if any(key in row for row in rows)
        else None
        for key in keys
    }


def rank_from_scores(positive_score: float, negative_scores: Sequence[float]) -> int:
    """Use stable pessimistic ties: equal negative scores precede the positive."""
    return 1 + sum(score >= positive_score for score in negative_scores)


def metrics_from_rank(rank: int, candidate_count: int) -> dict[str, float]:
    if rank < 1 or rank > candidate_count:
        raise ValueError("rank must be within the candidate pool")
    relevances = [0] * candidate_count
    relevances[rank - 1] = 1
    return per_query_metrics(relevances)


def metrics_from_positive_ranks(
    positive_ranks: Sequence[int],
    *,
    candidate_count: int,
    hard_negative_win_rate: float,
) -> dict[str, float]:
    """Compute exact metrics without materializing a corpus-sized relevance vector."""
    if not positive_ranks or any(rank < 1 or rank > candidate_count for rank in positive_ranks):
        raise ValueError("positive ranks must be non-empty and within the corpus")
    ranks = sorted(positive_ranks)
    relevant = len(ranks)
    dcg_at_10 = sum(1 / math.log2(rank + 1) for rank in ranks if rank <= 10)
    ideal_at_10 = sum(1 / math.log2(rank + 1) for rank in range(1, min(relevant, 10) + 1))
    return {
        "recall_at_1": sum(rank <= 1 for rank in ranks) / relevant,
        "recall_at_5": sum(rank <= 5 for rank in ranks) / relevant,
        "recall_at_10": sum(rank <= 10 for rank in ranks) / relevant,
        "recall_at_100": sum(rank <= 100 for rank in ranks) / relevant,
        "mrr_at_10": 1 / ranks[0] if ranks[0] <= 10 else 0.0,
        "mrr": 1 / ranks[0],
        "ndcg_at_10": dcg_at_10 / ideal_at_10 if ideal_at_10 else 0.0,
        "map": sum(index / rank for index, rank in enumerate(ranks, 1)) / relevant,
        "hard_negative_win_rate": hard_negative_win_rate,
    }


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
