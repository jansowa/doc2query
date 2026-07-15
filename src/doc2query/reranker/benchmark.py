"""Aggregation of retrieval metrics and explicit judge disagreement."""

from __future__ import annotations

import math
from collections.abc import Iterable
from statistics import fmean
from typing import Any

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
    return {
        "count": len(rows),
        "recall_at_1": fmean(row.recall_at_1 for row in rows),
        "recall_at_5": fmean(row.recall_at_5 for row in rows),
        "mrr": fmean(row.reciprocal_rank for row in rows),
        "ndcg_at_10": fmean(row.ndcg_at_10 for row in rows),
        "mean_margin": fmean(row.margin for row in rows),
        "negative_margin_rate": fmean(row.margin < 0 for row in rows),
        "near_zero_margin_rate": fmean(row.near_zero_margin for row in rows),
        "all_scores_close_rate": fmean(row.all_scores_close for row in rows),
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
        "recall_at_1",
        "recall_at_5",
        "mrr",
        "ndcg_at_10",
        "mean_margin",
        "negative_margin_rate",
        "near_zero_margin_rate",
        "all_scores_close_rate",
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
    rank_disagree = [left[key].positive_rank != right[key].positive_rank for key in shared]
    winner_disagree = [(left[key].margin >= 0) != (right[key].margin >= 0) for key in shared]
    return {
        "count": len(shared),
        "positive_rank_disagreement_rate": fmean(rank_disagree),
        "positive_winner_disagreement_rate": fmean(winner_disagree),
        "margin_pearson": _pearson(
            [left[key].margin for key in shared], [right[key].margin for key in shared]
        ),
        "disagreed_example_ids": [
            key for key, flag in zip(shared, winner_disagree, strict=True) if flag
        ],
    }
