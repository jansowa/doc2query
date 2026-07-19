"""Aggregation of retrieval metrics and explicit judge disagreement."""

from __future__ import annotations

import math
from collections.abc import Iterable
from statistics import fmean
from typing import Any, cast

from doc2query.evaluation.retrieval import CANDIDATE_POOL_RANKING, validate_recall_cutoffs
from doc2query.reranker.infer import GroupScore


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mx, my = fmean(xs), fmean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    denominator = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return numerator / denominator if denominator else None


def aggregate(scores: Iterable[GroupScore]) -> dict[str, Any]:
    rows = list(scores)
    if not rows:
        raise ValueError("cannot aggregate an empty benchmark")
    validate_recall_cutoffs(min(row.pool_candidate_count for row in rows), (1, 5))
    return {
        "count": len(rows),
        "protocol": CANDIDATE_POOL_RANKING,
        "pool_candidate_count": sorted({row.pool_candidate_count for row in rows}),
        "pool_recall_at_1": fmean(row.pool_recall_at_1 for row in rows),
        "pool_recall_at_5": fmean(cast(float, row.pool_recall_at_5) for row in rows),
        "pool_mrr": fmean(row.pool_mrr for row in rows),
        "pool_ndcg_at_10": fmean(row.pool_ndcg_at_10 for row in rows),
        "pool_mean_margin": fmean(row.pool_margin for row in rows),
        "pool_negative_margin_rate": fmean(row.pool_margin < 0 for row in rows),
        "pool_near_zero_margin_rate": fmean(row.pool_near_zero_margin for row in rows),
        "pool_all_scores_close_rate": fmean(row.pool_all_scores_close for row in rows),
    }


def aggregate_query_macro(scores: Iterable[GroupScore]) -> dict[str, Any]:
    """Give every query equal weight even when it has multiple positive passages."""
    by_query: dict[str, list[GroupScore]] = {}
    for row in scores:
        by_query.setdefault(row.query_id or row.example_id, []).append(row)
    if not by_query:
        raise ValueError("cannot aggregate an empty benchmark")
    per_query = [aggregate(rows) for rows in by_query.values()]
    metric_names = (
        "pool_recall_at_1",
        "pool_recall_at_5",
        "pool_mrr",
        "pool_ndcg_at_10",
        "pool_mean_margin",
        "pool_negative_margin_rate",
        "pool_near_zero_margin_rate",
        "pool_all_scores_close_rate",
    )
    return {
        "query_count": len(by_query),
        "pair_count": sum(len(rows) for rows in by_query.values()),
        **{name: fmean(float(row[name]) for row in per_query) for name in metric_names},
    }


def disagreement(primary: Iterable[GroupScore], shadow: Iterable[GroupScore]) -> dict[str, Any]:
    left = {row.example_id: row for row in primary}
    right = {row.example_id: row for row in shadow}
    shared = sorted(left.keys() & right.keys())
    if not shared:
        raise ValueError("judges have no shared examples")
    rank_disagree = [left[key].pool_rank != right[key].pool_rank for key in shared]
    winner_disagree = [
        (left[key].pool_margin >= 0) != (right[key].pool_margin >= 0) for key in shared
    ]
    return {
        "count": len(shared),
        "pool_rank_disagreement_rate": fmean(rank_disagree),
        "pool_winner_disagreement_rate": fmean(winner_disagree),
        "pool_margin_pearson": _pearson(
            [left[key].pool_margin for key in shared],
            [right[key].pool_margin for key in shared],
        ),
        "disagreed_example_ids": [
            key for key, flag in zip(shared, winner_disagree, strict=True) if flag
        ],
    }
