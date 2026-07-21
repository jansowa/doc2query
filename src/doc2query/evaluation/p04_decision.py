"""CPU-only, dev-only fail-closed P-04 comparison decision engine."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from doc2query.evaluation.statistical_contract import (
    BUDGET_FIELDS,
    CONTRACT_REFERENCE_FIELDS,
    StatisticalContract,
    assert_same_comparison_contract,
)

DecisionStatus = Literal["eligible", "non_inferior_only", "rejected", "incomplete"]
ALLOWED_STAGES = ("dev_screen", "dev_confirm")
REQUIRED_METRICS = (
    "corpus_ndcg_at_10",
    "corpus_round_trip_at_20",
    "sentence_level_source_hit",
    "format_valid_rate",
)


@dataclass(frozen=True)
class MetricRule:
    name: str
    threshold: float
    primary: bool


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _sample_sd(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _stage_seeds(contract: StatisticalContract, stage: str) -> list[int]:
    halving = contract.payload["successive_halving"]
    assert isinstance(halving, Mapping)
    stages = halving["stages"]
    assert isinstance(stages, list)
    for value in stages:
        if isinstance(value, Mapping) and value.get("name") == stage:
            seeds = value.get("seeds")
            if isinstance(seeds, list) and all(isinstance(seed, int) for seed in seeds):
                return list(seeds)
    raise ValueError(f"unsupported P-04 stage: {stage}")


def _rules(contract: StatisticalContract) -> list[MetricRule]:
    practical = contract.payload["minimum_practical_effect"]
    guardrails = contract.payload["non_inferiority"]
    assert isinstance(practical, Mapping)
    assert isinstance(guardrails, Mapping)
    values = [
        MetricRule("corpus_ndcg_at_10", float(practical["absolute"]), True),
    ]
    for dimension in ("grounding", "answerability", "format"):
        rule = guardrails[dimension]
        assert isinstance(rule, Mapping)
        values.append(MetricRule(str(rule["metric"]), -float(rule["margin"]), False))
    return values


def _validate_contract_reference(
    report: Mapping[str, Any], contract: StatisticalContract, errors: list[str]
) -> None:
    reference = report.get("statistical_contract")
    expected = contract.reference()
    if not isinstance(reference, Mapping):
        errors.append("missing statistical_contract reference")
        return
    for field in CONTRACT_REFERENCE_FIELDS:
        if reference.get(field) != expected.get(field):
            errors.append(f"statistical contract mismatch: {field}")


def _validate_budget(
    report: Mapping[str, Any], control: Mapping[str, Any], errors: list[str]
) -> None:
    try:
        assert_same_comparison_contract(control, report)
    except ValueError as exc:
        errors.append(str(exc))
    budget = report.get("comparison_budget")
    control_budget = control.get("comparison_budget")
    if isinstance(budget, Mapping):
        for field in BUDGET_FIELDS:
            if not isinstance(budget.get(field), int):
                errors.append(f"comparison budget lacks integer {field}")
            elif not isinstance(control_budget, Mapping) or budget.get(field) != control_budget.get(
                field
            ):
                errors.append(f"comparison budget mismatch: {field}")


def _validate_seed_reporting(
    report: Mapping[str, Any], required_seeds: Sequence[int], errors: list[str]
) -> None:
    per_seed = report.get("per_seed")
    summary = report.get("training_seed_summary")
    if not isinstance(per_seed, list):
        errors.append("missing per_seed values")
        return
    observed_seeds = [
        seed
        for item in per_seed
        if isinstance(item, Mapping) and isinstance((seed := item.get("seed")), int)
    ]
    if sorted(observed_seeds) != sorted(required_seeds):
        errors.append(
            f"seed set mismatch: expected {list(required_seeds)!r}, observed {observed_seeds!r}"
        )
        return
    if not isinstance(summary, Mapping):
        errors.append("missing training_seed_summary")
        return
    for metric in REQUIRED_METRICS:
        values: list[float] = []
        for row in per_seed:
            assert isinstance(row, Mapping)
            metrics = row.get("metrics")
            value = metrics.get(metric) if isinstance(metrics, Mapping) else None
            numeric = _as_float(value)
            if numeric is None:
                errors.append(f"missing per-seed metric {metric} for seed {row.get('seed')}")
                break
            values.append(numeric)
        if len(values) != len(required_seeds):
            continue
        aggregate = summary.get(metric)
        if not isinstance(aggregate, Mapping):
            errors.append(f"missing training seed aggregate for {metric}")
            continue
        expected = {
            "mean": sum(values) / len(values),
            "sample_sd": _sample_sd(values),
            "min": min(values),
            "max": max(values),
            "range": max(values) - min(values),
        }
        for name, value in expected.items():
            observed = aggregate.get(name)
            numeric = _as_float(observed)
            if numeric is None or not math.isclose(numeric, value, rel_tol=1e-9, abs_tol=1e-12):
                errors.append(f"invalid {name} across training seeds for {metric}")


def _validate_bootstrap(report: Mapping[str, Any], errors: list[str]) -> dict[str, float]:
    bootstrap = report.get("paired_query_bootstrap")
    if not isinstance(bootstrap, Mapping):
        errors.append("missing paired_query_bootstrap")
        return {}
    if bootstrap.get("unit") != "query" or bootstrap.get("paired") is not True:
        errors.append("bootstrap must be paired over query IDs")
    query_fingerprint = bootstrap.get("query_id_fingerprint")
    if not isinstance(query_fingerprint, str) or len(query_fingerprint) != 64:
        errors.append("paired bootstrap requires a 64-character query_id_fingerprint")
    if not isinstance(bootstrap.get("query_count"), int) or bootstrap.get("query_count", 0) < 1:
        errors.append("paired bootstrap requires a positive query_count")
    if bootstrap.get("includes_training_seed_variance") is not False:
        errors.append("query bootstrap must not aggregate training-seed variance")
    if bootstrap.get("samples") != 10_000 or bootstrap.get("seed") != 20_260_721:
        errors.append("bootstrap samples/seed do not match P-04")
    metrics = bootstrap.get("metrics")
    if not isinstance(metrics, Mapping):
        errors.append("bootstrap metrics are missing")
        return {}
    lower_bounds: dict[str, float] = {}
    for name in REQUIRED_METRICS:
        value = metrics.get(name)
        if not isinstance(value, Mapping):
            errors.append(f"missing paired-query CI for {name}")
            continue
        difference = value.get("difference")
        low = value.get("ci_low")
        high = value.get("ci_high")
        numeric_difference = _as_float(difference)
        numeric_low = _as_float(low)
        numeric_high = _as_float(high)
        if numeric_difference is None or numeric_low is None or numeric_high is None:
            errors.append(f"invalid paired-query CI for {name}")
            continue
        if numeric_low > numeric_high or not numeric_low <= numeric_difference <= numeric_high:
            errors.append(f"inconsistent paired-query CI for {name}")
            continue
        lower_bounds[name] = numeric_low
    return lower_bounds


def evaluate_p04_comparison(
    report: Mapping[str, Any], *, control_manifest: Mapping[str, Any], contract: StatisticalContract
) -> dict[str, Any]:
    """Return a preregistered dev decision without accessing any dataset files."""
    errors: list[str] = []
    stage = report.get("stage")
    if stage not in ALLOWED_STAGES:
        errors.append("stage must be dev_screen or dev_confirm")
        required_seeds: list[int] = []
    else:
        required_seeds = _stage_seeds(contract, str(stage))
    evaluation_sets = report.get("evaluation_sets")
    if evaluation_sets != ["dev_intrinsic"]:
        errors.append("P-04 preflight permits only dev_intrinsic")
    data_sources = report.get("data_sources")
    if data_sources != ["dev_intrinsic"]:
        errors.append(
            "P-04 preflight data_sources must contain only dev_intrinsic; final test data forbidden"
        )
    if report.get("final_tests_used") != []:
        errors.append("final_tests_used must be empty")
    _validate_contract_reference(report, contract, errors)
    _validate_budget(report, control_manifest, errors)
    if required_seeds:
        _validate_seed_reporting(report, required_seeds, errors)
    lower_bounds = _validate_bootstrap(report, errors)
    if errors:
        return {
            "schema_version": 1,
            "status": "incomplete",
            "stage": stage,
            "selection_claim": None,
            "errors": sorted(set(errors)),
            "criteria": {},
        }

    criteria = {
        rule.name: {
            "ci_low": lower_bounds[rule.name],
            "required_minimum": rule.threshold,
            "passed": lower_bounds[rule.name] >= rule.threshold,
            "kind": "minimum_practical_effect" if rule.primary else "non_inferiority",
        }
        for rule in _rules(contract)
    }
    guardrails_pass = all(
        bool(value["passed"]) for value in criteria.values() if value["kind"] == "non_inferiority"
    )
    primary_pass = bool(criteria["corpus_ndcg_at_10"]["passed"])
    status: DecisionStatus
    if not guardrails_pass:
        status = "rejected"
    elif primary_pass:
        status = "eligible"
    else:
        status = "non_inferior_only"
    return {
        "schema_version": 1,
        "status": status,
        "stage": stage,
        "selection_claim": None,
        "errors": [],
        "criteria": criteria,
    }
