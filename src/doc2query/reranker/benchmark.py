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
