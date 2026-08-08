"""Assemble exact P-04 dev-screen guardrails from frozen per-query artifacts."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any

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


def _paired_bootstrap_pcg64(
    control: Mapping[str, float],
    arm: Mapping[str, float],
    *,
    samples: int,
    seed: int,
) -> dict[str, float | int]:
    """Reproduce the preregistered P-05 NumPy PCG64 query bootstrap."""
    import numpy as np

    ids = sorted(control.keys() & arm.keys())
    if not ids or set(control) != set(arm):
        raise ValueError("paired bootstrap requires the same non-empty query IDs")
    differences = np.asarray([arm[key] - control[key] for key in ids], dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(seed))
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        estimates[index] = float(np.mean(differences[rng.integers(0, len(ids), len(ids))]))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {
        "query_count": len(ids),
        "bootstrap_samples": samples,
        "seed": seed,
        "difference": float(np.mean(differences)),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "variant_win_fraction": float(np.mean(estimates > 0)),
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
        metric: _paired_bootstrap_pcg64(
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
            "rng": "numpy.random.PCG64",
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


def build_dev_confirm_report(
    *,
    arm_id: str,
    control_id: str,
    arm_results: Mapping[int, Mapping[str, Any]],
    control_results: Mapping[int, Mapping[str, Any]],
    arm_per_query_paths: Mapping[int, Path],
    control_per_query_paths: Mapping[int, Path],
    arm_guardrails_path: Path,
    control_guardrails_path: Path,
    contract: StatisticalContract,
) -> dict[str, Any]:
    """Build the fixed-seed P-04 dev-confirm report without pooling seed variance."""
    seed_sets = (
        set(arm_results),
        set(control_results),
        set(arm_per_query_paths),
        set(control_per_query_paths),
    )
    if not seed_sets[0] or any(seeds != seed_sets[0] for seeds in seed_sets[1:]):
        raise ValueError("dev-confirm requires the same non-empty seed set for both arms")
    seeds = sorted(seed_sets[0])

    guardrail_metrics = (
        "corpus_round_trip_at_20",
        "sentence_level_source_hit",
        "format_valid_rate",
    )
    arm_guardrails = {
        metric: _metric_map(arm_guardrails_path, metric) for metric in guardrail_metrics
    }
    control_guardrails = {
        metric: _metric_map(control_guardrails_path, metric) for metric in guardrail_metrics
    }

    arm_by_seed: dict[int, dict[str, dict[str, float]]] = {}
    control_by_seed: dict[int, dict[str, dict[str, float]]] = {}
    common_budget: dict[str, Any] | None = None
    common_ids: set[str] | None = None
    per_seed: list[dict[str, Any]] = []
    control_per_seed: list[dict[str, Any]] = []

    for seed in seeds:
        arm_result = arm_results[seed]
        control_result = control_results[seed]
        budget = arm_result.get("comparison_budget")
        control_budget = control_result.get("comparison_budget")
        if not isinstance(budget, Mapping) or budget != control_budget:
            raise ValueError(f"P-04 dev-confirm seed {seed} has a comparison-budget mismatch")
        if common_budget is None:
            common_budget = dict(budget)
        elif dict(budget) != common_budget:
            raise ValueError("P-04 dev-confirm comparison budget differs across seeds")
        if arm_result.get("statistical_contract") != contract.reference():
            raise ValueError(f"arm seed {seed} statistical contract does not match P-04")
        if control_result.get("statistical_contract") != contract.reference():
            raise ValueError(f"control seed {seed} statistical contract does not match P-04")

        arm_values = {
            "corpus_ndcg_at_10": _metric_map(
                arm_per_query_paths[seed], "corpus_ndcg_at_10"
            ),
            **arm_guardrails,
        }
        control_values = {
            "corpus_ndcg_at_10": _metric_map(
                control_per_query_paths[seed], "corpus_ndcg_at_10"
            ),
            **control_guardrails,
        }
        id_sets = [set(values) for values in (*arm_values.values(), *control_values.values())]
        if any(ids != id_sets[0] for ids in id_sets[1:]):
            raise ValueError(
                f"P-04 dev-confirm seed {seed} requires identical query IDs for every metric"
            )
        if common_ids is None:
            common_ids = id_sets[0]
        elif id_sets[0] != common_ids:
            raise ValueError("P-04 dev-confirm query IDs differ across training seeds")

        arm_by_seed[seed] = arm_values
        control_by_seed[seed] = control_values
        arm_means = {metric: fmean(values.values()) for metric, values in arm_values.items()}
        control_means = {
            metric: fmean(values.values()) for metric, values in control_values.items()
        }
        per_seed.append({"seed": seed, "metrics": arm_means})
        control_per_seed.append({"seed": seed, "metrics": control_means})

    assert common_budget is not None
    assert common_ids is not None
    ids = sorted(common_ids)

    arm_query_means: dict[str, dict[str, float]] = {}
    control_query_means: dict[str, dict[str, float]] = {}
    for metric in REQUIRED_METRICS:
        arm_query_means[metric] = {
            query_id: fmean(arm_by_seed[seed][metric][query_id] for seed in seeds)
            for query_id in ids
        }
        control_query_means[metric] = {
            query_id: fmean(control_by_seed[seed][metric][query_id] for seed in seeds)
            for query_id in ids
        }

    bootstraps = {
        metric: _paired_bootstrap_pcg64(
            control_query_means[metric],
            arm_query_means[metric],
            samples=10_000,
            seed=20_260_721,
        )
        for metric in REQUIRED_METRICS
    }
    training_seed_summary = {
        metric: _summary(
            [
                float(row["metrics"][metric])
                for row in per_seed
                if isinstance(row.get("metrics"), Mapping)
            ]
        )
        for metric in REQUIRED_METRICS
    }
    control_training_seed_summary = {
        metric: _summary(
            [
                float(row["metrics"][metric])
                for row in control_per_seed
                if isinstance(row.get("metrics"), Mapping)
            ]
        )
        for metric in REQUIRED_METRICS
    }

    return {
        "schema_version": 1,
        "arm_id": arm_id,
        "control_id": control_id,
        "stage": "dev_confirm",
        "evaluation_sets": ["dev_intrinsic"],
        "data_sources": ["dev_intrinsic"],
        "actual_frozen_subset": "dev_intrinsic_rank10",
        "final_tests_used": [],
        "statistical_contract": contract.reference(),
        "comparison_budget": common_budget,
        "per_seed": per_seed,
        "training_seed_summary": training_seed_summary,
        "control_per_seed": control_per_seed,
        "control_training_seed_summary": control_training_seed_summary,
        "paired_query_bootstrap": {
            "unit": "query",
            "paired": True,
            "query_id_fingerprint": query_id_fingerprint(ids),
            "query_count": len(ids),
            "includes_training_seed_variance": False,
            "fixed_training_seed_aggregation": "per_query_mean_before_query_resampling",
            "samples": 10_000,
            "seed": 20_260_721,
            "rng": "numpy.random.PCG64",
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
                "known-positive hit@20 on the shared frozen natural dev query; "
                "frozen independently of probe-training seed"
            ),
            "sentence_level_source_hit": (
                "frozen-primary best source sentence beats the hardest inherited negative; "
                "shared natural dev query"
            ),
            "format_valid_rate": (
                "single-query format validity of the shared frozen natural dev query"
            ),
        },
    }
