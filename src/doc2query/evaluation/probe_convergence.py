"""M-03 probe convergence guardrail and paired per-seed decision statistic.

The probe budget sweep measured two things that force this module to exist:

* seed-to-seed spread of ``corpus_ndcg_at_10`` at a fixed arm and budget (0.0011 to 0.0826)
  can exceed the effect the closed confirm reports (+0.0479);
* training loss is **not** a convergence signal (``r = -0.199``, n=12), so a collapsed
  run cannot be detected from ``last_loss``.

The guardrail therefore works on a *retrieval* signal, and the comparison statistic works
on paired per-seed differences over at least five converged seed pairs.  Every threshold
comes from an externally frozen contract; this module never invents one.

Two design constraints keep the filter from biasing a comparison:

* a seed is dropped as a whole **pair**, never for one arm only;
* the floor is arm-independent — the random-retrieval chance level and a fraction of the
  median pooled across both arms — and the unfiltered result is always reported too.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from itertools import product
from pathlib import Path
from statistics import fmean, median, stdev
from typing import Any, Literal

import yaml
from pydantic import Field, model_validator

from doc2query.evaluation.bootstrap import paired_bootstrap
from doc2query.schemas import StrictModel

GUARDRAIL_CONTRACT = "task04-m03-probe-convergence-guardrail-v1"
GUARDRAIL_STATUS = "frozen_for_future_comparisons"
RETRIEVAL_SUMMARY = "corpus_retrieval_summary.json"
TRAIN_SUMMARY = "train_summary.json"
MAX_EXACT_PERMUTATION_SEEDS = 20


class ConvergenceSignal(StrictModel):
    """The retrieval signal that decides whether a probe run converged at all."""

    metric: Literal["corpus_recall_at_100"]
    source_file: Literal["corpus_retrieval_summary.json"]
    retrieval_depth: int = Field(ge=1)
    min_chance_multiple: float = Field(gt=1.0)
    min_fraction_of_pooled_median: float = Field(gt=0.0, lt=1.0)
    loss_based_guardrail_permitted: Literal[False]


class SeedAggregation(StrictModel):
    decision_metric: Literal["corpus_ndcg_at_10"]
    min_converged_seed_pairs: int = Field(ge=5)
    drop_policy: Literal["drop_whole_seed_pair"]
    superiority_threshold: float
    bootstrap_samples: int = Field(ge=1000)
    bootstrap_seed: int = Field(ge=0)
    permutation: Literal["exact_sign_flip_one_sided"]
    max_permutation_p: float = Field(gt=0.0, lt=1.0)
    report_unfiltered_result: Literal[True]


class ProbeConvergenceGuardrail(StrictModel):
    schema_version: Literal[1]
    contract: Literal["task04-m03-probe-convergence-guardrail-v1"]
    guardrail_id: str = Field(min_length=1)
    status: Literal["frozen_for_future_comparisons"]
    adr: str = Field(min_length=1)
    signal: ConvergenceSignal
    aggregation: SeedAggregation
    calibration_run_groups: list[str] = Field(min_length=1)
    final_tests_used: list[str] = Field(max_length=0)


class ProbeRunMetrics(StrictModel):
    run_id: str = Field(min_length=1)
    arm: str = Field(min_length=1)
    seed: int = Field(ge=0)
    metrics: dict[str, float]
    candidate_count: int = Field(ge=1)
    query_count: int = Field(ge=1)
    first_loss: float
    last_loss: float


class ConvergenceVerdict(StrictModel):
    run_id: str = Field(min_length=1)
    arm: str = Field(min_length=1)
    seed: int = Field(ge=0)
    signal_value: float
    chance_level: float
    chance_floor: float
    median_floor: float
    applied_floor: float
    converged: bool
    failure_reason: str | None = None

    @model_validator(mode="after")
    def verdict_is_consistent(self) -> ConvergenceVerdict:
        if self.converged != (self.failure_reason is None):
            raise ValueError("converged must equal the absence of a failure reason")
        return self


def load_guardrail(path: Path) -> ProbeConvergenceGuardrail:
    """Load an externally frozen M-03 guardrail; thresholds are never derived here."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: guardrail contract must be a mapping")
    return ProbeConvergenceGuardrail.model_validate(raw)


def read_probe_run(run_dir: Path, *, arm: str) -> ProbeRunMetrics:
    """Read the finished probe artifacts of one run without recomputing anything."""
    retrieval_path = run_dir / RETRIEVAL_SUMMARY
    train_path = run_dir / TRAIN_SUMMARY
    for path in (retrieval_path, train_path):
        if not path.is_file():
            raise ValueError(f"missing probe artifact: {path}")
    retrieval = json.loads(retrieval_path.read_text(encoding="utf-8"))
    train = json.loads(train_path.read_text(encoding="utf-8"))
    if retrieval.get("status") != "measured" or train.get("status") != "measured":
        raise ValueError(f"{run_dir}: probe run is not measured")
    metrics = retrieval["metrics"]
    candidate_counts = {int(value) for value in retrieval["metric_candidate_count"].values()}
    if len(candidate_counts) != 1:
        raise ValueError(f"{run_dir}: retrieval metrics disagree on the candidate count")
    return ProbeRunMetrics(
        run_id=run_dir.name,
        arm=arm,
        seed=int(train["recipe"]["seed"]),
        metrics={str(key): float(value) for key, value in metrics.items()},
        candidate_count=next(iter(candidate_counts)),
        query_count=int(retrieval["query_count"]),
        first_loss=float(train["first_loss"]),
        last_loss=float(train["last_loss"]),
    )


def apply_convergence_guardrail(
    runs: Sequence[ProbeRunMetrics], guardrail: ProbeConvergenceGuardrail
) -> list[ConvergenceVerdict]:
    """Flag collapsed runs using an arm-independent retrieval floor."""
    if not runs:
        raise ValueError("the guardrail needs at least one run")
    signal = guardrail.signal
    values = [run.metrics[signal.metric] for run in runs]
    candidate_counts = {run.candidate_count for run in runs}
    if len(candidate_counts) != 1:
        raise ValueError("convergence calibration requires one shared corpus size")
    corpus_size = next(iter(candidate_counts))
    chance = signal.retrieval_depth / corpus_size
    chance_floor = chance * signal.min_chance_multiple
    median_floor = median(values) * signal.min_fraction_of_pooled_median
    applied = max(chance_floor, median_floor)
    verdicts: list[ConvergenceVerdict] = []
    for run in sorted(runs, key=lambda item: (item.arm, item.seed)):
        value = run.metrics[signal.metric]
        converged = value >= applied
        verdicts.append(
            ConvergenceVerdict(
                run_id=run.run_id,
                arm=run.arm,
                seed=run.seed,
                signal_value=value,
                chance_level=chance,
                chance_floor=chance_floor,
                median_floor=median_floor,
                applied_floor=applied,
                converged=converged,
                failure_reason=None if converged else f"{signal.metric}_below_floor",
            )
        )
    return verdicts


def _sign_flip_p_value(differences: Sequence[float], threshold: float) -> dict[str, Any]:
    """Exact one-sided sign-flip randomization test for mean(d) > threshold."""
    centered = [value - threshold for value in differences]
    observed = fmean(centered)
    count = len(centered)
    if count > MAX_EXACT_PERMUTATION_SEEDS:
        raise ValueError("exact sign-flip enumeration is limited to twenty seed pairs")
    total = 0
    at_least = 0
    for signs in product((1.0, -1.0), repeat=count):
        total += 1
        candidate = fmean(sign * value for sign, value in zip(signs, centered, strict=True))
        if candidate >= observed - 1e-15:
            at_least += 1
    return {
        "test": "exact_sign_flip_one_sided",
        "seed_pair_count": count,
        "threshold": threshold,
        "observed_mean_minus_threshold": observed,
        "permutations": total,
        "p_value": at_least / total,
        "smallest_attainable_p_value": 1 / total,
    }


def paired_seed_comparison(
    variant: Sequence[ProbeRunMetrics],
    anchor: Sequence[ProbeRunMetrics],
    guardrail: ProbeConvergenceGuardrail,
) -> dict[str, Any]:
    """Decide a variant-minus-anchor comparison on paired per-seed differences."""
    rule = guardrail.aggregation
    verdicts = apply_convergence_guardrail([*variant, *anchor], guardrail)
    non_converged = {verdict.seed for verdict in verdicts if not verdict.converged}
    variant_by_seed = {run.seed: run.metrics[rule.decision_metric] for run in variant}
    anchor_by_seed = {run.seed: run.metrics[rule.decision_metric] for run in anchor}
    if len(variant_by_seed) != len(variant) or len(anchor_by_seed) != len(anchor):
        raise ValueError("each arm must contribute at most one run per seed")
    shared = sorted(set(variant_by_seed) & set(anchor_by_seed))
    if not shared:
        raise ValueError("the arms share no seed")

    def summarize(seeds: Sequence[int]) -> dict[str, Any]:
        differences = [variant_by_seed[seed] - anchor_by_seed[seed] for seed in seeds]
        payload: dict[str, Any] = {
            "seeds": list(seeds),
            "seed_pair_count": len(seeds),
            "per_seed_difference": dict(zip((str(s) for s in seeds), differences, strict=True)),
            "mean_difference": fmean(differences) if differences else None,
            "sd_difference": stdev(differences) if len(differences) > 1 else None,
        }
        if len(seeds) >= 2:
            payload["paired_bootstrap"] = paired_bootstrap(
                {str(seed): anchor_by_seed[seed] for seed in seeds},
                {str(seed): variant_by_seed[seed] for seed in seeds},
                samples=rule.bootstrap_samples,
                seed=rule.bootstrap_seed,
            )
            payload["sign_flip"] = _sign_flip_p_value(differences, rule.superiority_threshold)
        return payload

    converged_seeds = [seed for seed in shared if seed not in non_converged]
    filtered = summarize(converged_seeds)
    unfiltered = summarize(shared)

    if len(converged_seeds) < rule.min_converged_seed_pairs:
        status = "insufficient_converged_seeds"
    else:
        bootstrap = filtered["paired_bootstrap"]
        sign_flip = filtered["sign_flip"]
        superior = (
            float(bootstrap["ci95_low"]) >= rule.superiority_threshold
            and float(sign_flip["p_value"]) <= rule.max_permutation_p
        )
        status = "superior" if superior else "not_superior"
    return {
        "schema_version": 1,
        "contract": GUARDRAIL_CONTRACT,
        "guardrail_id": guardrail.guardrail_id,
        "decision_metric": rule.decision_metric,
        "superiority_threshold": rule.superiority_threshold,
        "min_converged_seed_pairs": rule.min_converged_seed_pairs,
        "status": status,
        "convergence_verdicts": [verdict.model_dump(mode="json") for verdict in verdicts],
        "dropped_seeds": sorted(seed for seed in shared if seed in non_converged),
        "converged": filtered,
        "unfiltered": unfiltered,
        "promotion_authorized": False,
        "final_tests_used": [],
    }


def convergence_report(
    verdicts: Sequence[ConvergenceVerdict], *, group_id: str
) -> dict[str, Any]:
    """Summarize a calibration group without deciding anything about it."""
    values = [verdict.signal_value for verdict in verdicts]
    flagged = [verdict for verdict in verdicts if not verdict.converged]
    return {
        "group_id": group_id,
        "run_count": len(verdicts),
        "applied_floor": verdicts[0].applied_floor if verdicts else None,
        "chance_level": verdicts[0].chance_level if verdicts else None,
        "median_signal": median(values) if values else None,
        "min_signal": min(values) if values else None,
        "max_signal": max(values) if values else None,
        "non_converged_run_count": len(flagged),
        "non_converged_runs": [verdict.run_id for verdict in flagged],
        "final_tests_used": [],
    }


def load_run_group(
    root: Path, arms: Mapping[str, str], *, include: str = ""
) -> list[ProbeRunMetrics]:
    """Read the runs of one comparison group, mapping a name fragment to its arm.

    ``include`` restricts the group to run directories whose name contains it, which is how
    a sweep holding several budgets is split into one comparison per budget.
    """
    runs: list[ProbeRunMetrics] = []
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if include and include not in run_dir.name:
            continue
        arm = next((label for prefix, label in arms.items() if prefix in run_dir.name), None)
        if arm is None:
            continue
        runs.append(read_probe_run(run_dir, arm=arm))
    if not runs:
        raise ValueError(f"no probe run of the requested arms under {root}")
    return runs
