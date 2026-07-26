"""Dev-only hard-negative recipe gate (HN0 through HN3).

This module deliberately contains no optimizer or training entry point.  It
audits negative miners with frozen models on one predeclared development
cohort and records enough provenance to make accidental final-test use fail.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from statistics import fmean
from typing import Any

from doc2query.evaluation.bootstrap import paired_bootstrap
from doc2query.evaluation.retrieval import candidate_pool_metrics_from_rank

GATE_VERSION = "task04-hn-gate-v1"
ARMS = ("hn0", "hn0_filter", "hn1_bm25", "hn2_biencoder", "hn3_union_positive_filter")


def canonical_fingerprint(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def deduplicate_scoring_pairs(
    pairs: Sequence[tuple[str, str]],
) -> tuple[list[tuple[str, str]], list[int]]:
    """Deduplicate exact pairs and length-bucket them to reduce padded inference."""
    unique = set(pairs)
    ordered = sorted(
        unique,
        key=lambda pair: (
            (len(pair[0]) + len(pair[1])) // 256,
            len(pair[0]) + len(pair[1]),
            hashlib.sha256((pair[0] + "\0" + pair[1]).encode()).hexdigest(),
        ),
    )
    indexes = {pair: index for index, pair in enumerate(ordered)}
    return ordered, [indexes[pair] for pair in pairs]


def select_dev_records(
    records: Sequence[Mapping[str, Any]], *, limit: int, seed: int
) -> list[dict[str, Any]]:
    """Freeze a deterministic prefix without depending on source ordering."""
    if limit < 1 or limit > len(records):
        raise ValueError("dev gate limit must be within the frozen set")
    ranked = sorted(
        records,
        key=lambda row: (
            hashlib.sha256(f"{seed}:{row['example_id']}".encode()).hexdigest(),
            str(row["example_id"]),
        ),
    )
    selected = [dict(row) for row in ranked[:limit]]
    if any(len(row.get("hard_negatives", [])) < 10 for row in selected):
        raise ValueError("hard-negative gate requires at least ten inherited negatives")
    return selected


def assert_dev_only_contract(contract: Mapping[str, Any]) -> None:
    subset = str(contract.get("evaluation_subset", "")).lower()
    final_tests = contract.get("final_tests_used")
    if not subset.startswith("dev") or "test" in subset:
        raise ValueError("hard-negative gate accepts only a named frozen dev subset")
    if final_tests != []:
        raise ValueError("hard-negative gate must record final_tests_used=[]")
    if contract.get("training_runs") != []:
        raise ValueError("hard-negative gate is inference-only; training_runs must be empty")


def positive_aware_keep(
    *, negative_score: float, positive_score: float, absolute_threshold: float
) -> bool:
    """Conservative NV-style filter in a raw-logit space.

    Ratios are invalid for signed cross-encoder logits.  A candidate is kept
    only below both the dev-calibrated absolute threshold and its query's
    positive score.
    """
    if not all(math.isfinite(value) for value in (negative_score, positive_score)):
        raise ValueError("positive-aware filtering requires finite scores")
    return negative_score < absolute_threshold and negative_score < positive_score


def stable_union(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Interleave two ranked miners, preserving complete per-miner provenance."""
    result: list[dict[str, Any]] = []
    positions = {str(row["doc_id"]): index for index, row in enumerate(left)}
    positions.update(
        {
            str(row["doc_id"]): min(index, positions.get(str(row["doc_id"]), index))
            for index, row in enumerate(right)
        }
    )
    by_id: dict[str, dict[str, Any]] = {}
    for miner, rows in (("bm25", left), ("biencoder", right)):
        for row in rows:
            doc_id = str(row["doc_id"])
            entry = by_id.setdefault(doc_id, {"doc_id": doc_id, "miners": {}})
            entry["miners"][miner] = {
                "rank": int(row["rank"]),
                "score": float(row["score"]),
            }
    for doc_id in sorted(by_id, key=lambda value: (positions[value], value)):
        result.append(by_id[doc_id])
    return result


def pool_metrics(positive_score: float, negative_scores: Sequence[float]) -> dict[str, float]:
    if len(negative_scores) < 10:
        raise ValueError("gate metrics require at least ten negatives")
    rank = 1 + sum(score >= positive_score for score in negative_scores)
    metrics = candidate_pool_metrics_from_rank(rank, candidate_count=1 + len(negative_scores))
    return {key: float(value) for key, value in metrics.items()}


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty gate arm")
    metrics = ("pool_recall_at_1", "pool_recall_at_5", "pool_mrr", "pool_ndcg_at_10")
    summary = {
        "query_count": len(rows),
        **{metric: fmean(float(row[metric]) for row in rows) for metric in metrics},
        "pool_hard_negative_win_rate": fmean(
            float(row["positive_score"]) > max(float(value) for value in row["negative_scores"])
            for row in rows
        ),
        "mean_margin": fmean(
            float(row["positive_score"]) - max(float(value) for value in row["negative_scores"])
            for row in rows
        ),
        "possible_false_negative_rate": fmean(
            float(row["possible_false_negative_rate"]) for row in rows
        ),
    }
    if all(
        isinstance(row.get("shadow_pool_mrr"), (int, float))
        and isinstance(row.get("shadow_pool_ndcg_at_10"), (int, float))
        for row in rows
    ):
        summary.update(
            {
                "shadow_pool_mrr": fmean(float(row["shadow_pool_mrr"]) for row in rows),
                "shadow_pool_ndcg_at_10": fmean(
                    float(row["shadow_pool_ndcg_at_10"]) for row in rows
                ),
            }
        )
    return summary


def compare_to_reference(
    arms: Mapping[str, Sequence[Mapping[str, Any]]], *, samples: int, seed: int
) -> dict[str, Any]:
    if set(arms) != set(ARMS):
        raise ValueError("full gate requires exactly HN0, HN0+filter, HN1, HN2 and HN3")
    ids = [{str(row["example_id"]) for row in rows} for rows in arms.values()]
    if any(values != ids[0] for values in ids[1:]):
        raise ValueError("gate arms do not share an identical dev cohort")
    reference = {str(row["example_id"]): row for row in arms["hn0_filter"]}
    result: dict[str, Any] = {}
    for arm in ARMS:
        if arm == "hn0_filter":
            continue
        variant = {str(row["example_id"]): row for row in arms[arm]}
        result[f"{arm}_minus_hn0_filter"] = {}
        for metric in ("pool_mrr", "pool_ndcg_at_10"):
            result[f"{arm}_minus_hn0_filter"][metric] = paired_bootstrap(
                {key: float(value[metric]) for key, value in reference.items()},
                {key: float(value[metric]) for key, value in variant.items()},
                samples=samples,
                seed=seed,
            )
    return result
