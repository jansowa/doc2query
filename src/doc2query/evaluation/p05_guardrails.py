"""Assemble exact P-04 dev-screen guardrails from frozen per-query artifacts."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any

from doc2query.evaluation.bootstrap import paired_bootstrap
from doc2query.evaluation.p04_decision import REQUIRED_METRICS
from doc2query.evaluation.statistical_contract import StatisticalContract
from doc2query.utils.records import read_records


def query_id_fingerprint(query_ids: Sequence[str]) -> str:
    """Match the frozen-evaluation convention: sorted IDs, one per line."""
    digest = hashlib.sha256()
    for query_id in sorted(query_ids):
        digest.update(query_id.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _metric_map(path: Path, metric: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for row in read_records(path):
        query_id = str(row.get("example_id", ""))
        value = row.get(metric)
        if not query_id or isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{path} lacks numeric {metric} for a stable example_id")
        numeric = float(value)
        if not math.isfinite(numeric) or query_id in values:
            raise ValueError(f"{path} has invalid or duplicate {metric} row: {query_id}")
        values[query_id] = numeric
    if not values:
        raise ValueError(f"{path} contains no {metric} rows")
    return values


def _summary(values: Sequence[float]) -> dict[str, float]:
    mean = fmean(values)
    sample_sd = (
        math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
        if len(values) > 1
        else 0.0
    )
    return {
        "mean": mean,
        "sample_sd": sample_sd,
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
    }


def build_dev_screen_report(
    *,
    arm_id: str,
    control_id: str,
    arm_result: Mapping[str, Any],
    control_result: Mapping[str, Any],
    arm_per_query_path: Path,
    control_per_query_path: Path,
    arm_guardrails_path: Path,
    control_guardrails_path: Path,
    contract: StatisticalContract,
) -> dict[str, Any]:
    """Build the exact schema consumed by the fail-closed P-04 engine."""
    metric_paths = {
        "corpus_ndcg_at_10": (arm_per_query_path, control_per_query_path),
        "corpus_round_trip_at_20": (arm_guardrails_path, control_guardrails_path),
        "sentence_level_source_hit": (arm_guardrails_path, control_guardrails_path),
        "format_valid_rate": (arm_guardrails_path, control_guardrails_path),
    }
    arm_values: dict[str, dict[str, float]] = {}
    control_values: dict[str, dict[str, float]] = {}
    for metric, (arm_path, control_path) in metric_paths.items():
        arm_values[metric] = _metric_map(arm_path, metric)
        control_values[metric] = _metric_map(control_path, metric)
    id_sets = [set(values) for values in (*arm_values.values(), *control_values.values())]
    if any(ids != id_sets[0] for ids in id_sets[1:]):
        raise ValueError("P-04 report requires identical query IDs for every paired metric")
    ids = sorted(id_sets[0])
    budget = arm_result.get("comparison_budget")
    control_budget = control_result.get("comparison_budget")
    if not isinstance(budget, Mapping) or budget != control_budget:
        raise ValueError("P-04 report requires an identical comparison budget")
    if arm_result.get("statistical_contract") != contract.reference():
        raise ValueError("arm statistical contract does not match the pinned P-04 contract")
    if control_result.get("statistical_contract") != contract.reference():
        raise ValueError("control statistical contract does not match the pinned P-04 contract")

    means = {metric: fmean(values.values()) for metric, values in arm_values.items()}
    bootstraps = {
        metric: paired_bootstrap(
            control_values[metric],
            arm_values[metric],
            samples=10_000,
            seed=20_260_721,
        )
        for metric in REQUIRED_METRICS
    }
    return {
        "schema_version": 1,
        "arm_id": arm_id,
        "control_id": control_id,
        "stage": "dev_screen",
        "evaluation_sets": ["dev_intrinsic"],
        "data_sources": ["dev_intrinsic"],
        "actual_frozen_subset": "dev_intrinsic_rank10",
        "final_tests_used": [],
        "statistical_contract": contract.reference(),
        "comparison_budget": dict(budget),
        "per_seed": [{"seed": 42, "metrics": means}],
        "training_seed_summary": {metric: _summary([value]) for metric, value in means.items()},
        "paired_query_bootstrap": {
            "unit": "query",
            "paired": True,
            "query_id_fingerprint": query_id_fingerprint(ids),
            "query_count": len(ids),
            "includes_training_seed_variance": False,
            "samples": 10_000,
            "seed": 20_260_721,
            "metrics": {
                metric: {
                    "difference": result["difference"],
                    "ci_low": result["ci95_low"],
                    "ci_high": result["ci95_high"],
                    "variant_win_fraction": result["variant_win_fraction"],
                }
                for metric, result in bootstraps.items()
            },
        },
        "guardrail_semantics": {
            "corpus_round_trip_at_20": (
                "known-positive hit@20 of the arm's probe on the shared frozen natural dev query"
            ),
            "sentence_level_source_hit": (
                "frozen-primary best source sentence beats the hardest inherited negative; "
                "shared natural dev query, hence identical across probe-training arms"
            ),
            "format_valid_rate": (
                "single-query format validity of the shared frozen natural dev query; "
                "identical across probe-training arms"
            ),
        },
    }
