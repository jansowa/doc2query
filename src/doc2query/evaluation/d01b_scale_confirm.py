"""Fail-closed ID-only audit and sensitivity planning for D01b 4.5B confirm."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import NormalDist
from typing import Any, cast

import yaml

from doc2query.evaluation.datasets import load_frozen_records
from doc2query.utils.records import read_records

AUDIT_CONTRACT = "task05-d01b-scale-interaction-4.5b-dev-confirm-feasibility-v1"
PILOT_CONTRACT = "task05-d01b-scale-interaction-4.5b-pilot-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _ids_sha256(ids: Sequence[str], *, sort_ids: bool = False) -> str:
    values = sorted(ids) if sort_ids else list(ids)
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def _root(path: Path) -> Path:
    root = next(
        (parent for parent in path.resolve().parents if (parent / "AGENTS.md").is_file()), None
    )
    if root is None:
        raise ValueError("cannot resolve repository root")
    return root


def _load(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping: {path}")
    return value


def _ordered_eligible_ids(
    records: Sequence[Mapping[str, Any]],
    *,
    excluded: set[str],
    minimum_hard_negatives: int,
    seed: int,
) -> list[str]:
    """Return quality-blind IDs; passage/query text is neither read nor returned."""
    eligible: list[tuple[str, str]] = []
    for row in records:
        example_id = str(row["example_id"])
        negatives = row.get("hard_negatives")
        if example_id in excluded or not isinstance(negatives, list):
            continue
        if len(negatives) < minimum_hard_negatives:
            continue
        digest = hashlib.sha256(f"{seed}:{example_id}".encode()).hexdigest()
        eligible.append((digest, example_id))
    eligible.sort()
    return [example_id for _digest, example_id in eligible]


def _reconstruct_prior_ids(
    records: Sequence[Mapping[str, Any]],
    *,
    rank10_ids: set[str],
    prior_specs: Sequence[Mapping[str, Any]],
    root: Path,
) -> set[str]:
    used: set[str] = set()
    for prior in prior_specs:
        manifest_path = root / str(prior["manifest"])
        if _sha256(manifest_path) != str(prior["manifest_sha256"]):
            raise ValueError("prior prospective cohort manifest drifted")
        ids = _ordered_eligible_ids(
            records,
            excluded=rank10_ids | used,
            minimum_hard_negatives=5,
            seed=int(prior["selection_seed"]),
        )[: int(prior["selected_count"])]
        if _ids_sha256(ids) != str(prior["selected_id_list_sha256"]):
            raise ValueError("prior prospective cohort reconstruction drifted")
        used.update(ids)
    return used


def audit_unseen_development(config_path: Path) -> dict[str, Any]:
    """Audit the complete legal reserve without emitting IDs or opening final-test files."""
    config = _load(config_path)
    if (
        config.get("contract") != PILOT_CONTRACT
        or config.get("final_tests_used") != []
        or cast(Mapping[str, Any], config.get("authorization", {})).get("final_tests") is not False
    ):
        raise ValueError("invalid or non-development pilot contract")
    root = _root(config_path)
    cohort = cast(Mapping[str, Any], config["cohort"])
    records = load_frozen_records(
        root / str(cohort["source_frozen_manifest"]), str(cohort["source_subset"])
    )
    rank10_path = root / "data/processed/v1/evaluation/task04-v1/dev_intrinsic_rank10.ids.jsonl"
    rank10_ids = {str(row["id"]) for row in read_records(rank10_path)}
    prior_specs = cast(Sequence[Mapping[str, Any]], cohort["prior_cohort_exclusions"])
    prior_ids = _reconstruct_prior_ids(
        records, rank10_ids=rank10_ids, prior_specs=prior_specs, root=root
    )
    eligible_ids = _ordered_eligible_ids(
        records,
        excluded=rank10_ids | prior_ids,
        minimum_hard_negatives=int(cohort["minimum_hard_negatives"]),
        seed=int(cohort["selection_seed"]),
    )
    if len(eligible_ids) != int(cohort["eligible_count"]):
        raise ValueError("pilot eligible development population drifted")
    generation_count = int(cohort["selected_count"])
    evaluation_offset = int(cohort["evaluation_offset"])
    evaluation_count = int(cohort["evaluation_count"])
    generation_ids = eligible_ids[:generation_count]
    evaluation_ids = eligible_ids[evaluation_offset : evaluation_offset + evaluation_count]
    reserve_ids = eligible_ids[evaluation_offset + evaluation_count :]
    if _ids_sha256(generation_ids) != str(cohort["selected_id_list_sha256"]):
        raise ValueError("pilot generation IDs drifted")
    if _ids_sha256(evaluation_ids) != str(cohort["evaluation_id_list_sha256"]):
        raise ValueError("pilot evaluation IDs drifted")
    expected_reserve = int(cohort["eligible_count"]) - generation_count - evaluation_count
    if len(reserve_ids) != expected_reserve or set(reserve_ids) & (
        rank10_ids | prior_ids | set(generation_ids) | set(evaluation_ids)
    ):
        raise ValueError("unseen development reserve is not legal and disjoint")
    return {
        "schema_version": 1,
        "contract": AUDIT_CONTRACT,
        "status": "id_only_audit_complete",
        "source_subset": "dev_intrinsic",
        "source_population_count": len(records),
        "exclusions": {
            "dev_intrinsic_rank10_count": len(rank10_ids),
            "prior_prospective_v1_v2_v3_union_count": len(prior_ids),
            "pilot_generation_count": len(generation_ids),
            "pilot_evaluation_count": len(evaluation_ids),
        },
        "minimum_hard_negatives": int(cohort["minimum_hard_negatives"]),
        "legal_unseen_eligible_count": len(reserve_ids),
        "legal_unseen_id_list_sha256_selection_order": _ids_sha256(reserve_ids),
        "legal_unseen_id_list_sha256_sorted": _ids_sha256(reserve_ids, sort_ids=True),
        "intersection_with_any_excluded_or_seen_id": 0,
        "quality_fields_used": ["hard_negative_count_gte_5"],
        "record_text_fields_used": [],
        "raw_ids_emitted": False,
        "final_test_manifests_opened": [],
        "final_tests_used": [],
    }


def sensitivity_from_pilot_ci(
    *,
    pilot_difference: float,
    pilot_ci95: tuple[float, float],
    pilot_queries: int,
    confirm_queries: int,
    practical_effect_threshold: float,
) -> dict[str, Any]:
    """Plan sensitivity from the aggregate pilot CI without selecting confirm records."""
    if pilot_queries <= 0 or confirm_queries <= 0:
        raise ValueError("query counts must be positive")
    low, high = pilot_ci95
    if not low < pilot_difference < high:
        raise ValueError("pilot difference must lie inside its CI")
    normal = NormalDist()
    z95 = normal.inv_cdf(0.975)
    z975 = normal.inv_cdf(0.9875)
    pilot_standard_error = (high - low) / (2.0 * z95)
    paired_query_sd = pilot_standard_error * math.sqrt(pilot_queries)
    confirm_half_width = z975 * paired_query_sd / math.sqrt(confirm_queries)
    optimistic_three_seed_half_width = confirm_half_width / math.sqrt(3.0)
    gap = pilot_difference - practical_effect_threshold

    def required(power: float) -> int | None:
        if gap <= 0.0:
            return None
        return math.ceil(((z975 + normal.inv_cdf(power)) * paired_query_sd / gap) ** 2)

    return {
        "method": "normal_approximation_from_aggregate_pilot_95pct_ci",
        "planning_only_no_threshold_tuning": True,
        "pilot_difference": pilot_difference,
        "pilot_ci95": [low, high],
        "pilot_queries": pilot_queries,
        "estimated_paired_query_sd": paired_query_sd,
        "confirm_queries": confirm_queries,
        "required_interval": "two_sided_97.5_percent",
        "practical_effect_threshold": practical_effect_threshold,
        "projected_half_width_one_seed_variance": confirm_half_width,
        "projected_ci_at_pilot_effect": [
            pilot_difference - confirm_half_width,
            pilot_difference + confirm_half_width,
        ],
        "optimistic_three_independent_seed_half_width": optimistic_three_seed_half_width,
        "optimistic_three_independent_seed_lower_bound": (
            pilot_difference - optimistic_three_seed_half_width
        ),
        "required_queries_for_80pct_power": required(0.8),
        "required_queries_for_90pct_power": required(0.9),
        "seed_variance_estimable_from_one_seed_pilot": False,
    }


def assess_confirm_feasibility(config_path: Path) -> dict[str, Any]:
    """Combine the legal-ID audit with the frozen, planning-only sensitivity analysis."""
    audit = audit_unseen_development(config_path)
    sensitivity = sensitivity_from_pilot_ci(
        pilot_difference=0.02073792007878962,
        pilot_ci95=(0.011055484860771694, 0.03017264616376007),
        pilot_queries=2000,
        confirm_queries=int(audit["legal_unseen_eligible_count"]),
        practical_effect_threshold=0.01,
    )
    feasible = (
        int(audit["legal_unseen_eligible_count"])
        >= int(sensitivity["required_queries_for_80pct_power"])
        and float(sensitivity["optimistic_three_independent_seed_lower_bound"]) >= 0.01
    )
    return {
        "schema_version": 1,
        "contract": AUDIT_CONTRACT,
        "status": "feasible" if feasible else "blocked_insufficient_unseen_development",
        "audit": audit,
        "sensitivity": sensitivity,
        "confirm_config_frozen": False,
        "confirm_runner_prepared": False,
        "expensive_run_authorized": False,
        "retained_for_finalist_freeze": False,
        "four_point_five_b_full_authorized": False,
        "selection_claim": None,
        "owner_decision_required": not feasible,
        "final_tests_used": [],
    }
