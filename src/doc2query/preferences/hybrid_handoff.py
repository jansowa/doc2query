"""Fail-closed model-free handoff of the confirmed D01b Hybrid to Task 06."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import yaml

from doc2query.evaluation.d01_pipeline import _artifact_fingerprint

CONTRACT = "task06-d01b-hybrid-handoff-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping: {path}")
    return value


def _root(path: Path) -> Path:
    root = next(
        (parent for parent in path.resolve().parents if (parent / "AGENTS.md").is_file()), None
    )
    if root is None:
        raise ValueError("cannot resolve repository root")
    return root


def _pin(root: Path, value: Mapping[str, Any], path_key: str, hash_key: str) -> Path:
    path = root / str(value[path_key])
    if not path.is_file() or _sha256(path) != str(value[hash_key]):
        raise ValueError(f"pinned handoff input drifted: {value[path_key]}")
    return path


def _assert_shape(config: Mapping[str, Any]) -> None:
    if (
        config.get("schema_version") != 1
        or config.get("contract") != CONTRACT
        or config.get("status") != "owner_approved_pending_task06_execution_design"
        or config.get("final_tests_used") != []
    ):
        raise ValueError("invalid Task 06 Hybrid handoff contract")
    confirm = cast(Mapping[str, Any], config.get("confirm", {}))
    if confirm != {
        **confirm,
        "required_status": "external_dev_confirm_complete",
        "required_decision": "eligible_for_finalist_freeze_review",
        "required_selection_claim": "external_dev_confirm_passed_pending_finalist_freeze_review",
        "retained_for_finalist_freeze": True,
        "task06_or_task09_promotion_authorized": False,
        "four_point_five_b_full_authorized": False,
        "final_tests_used": [],
    }:
        raise ValueError("confirm handoff boundary drifted")
    source = cast(Mapping[str, Any], config.get("source_procedure", {}))
    base = cast(Mapping[str, Any], source.get("base_model", {}))
    if (
        base
        != {
            "name_or_path": "speakleash/Bielik-4.5B-v3.0-Instruct",
            "revision": "4b1220a9d745bdd874c44347075ef25484ef322b",
            "trust_remote_code": False,
        }
        or source.get("candidate_count") != 8
        or source.get("selected_count") != 4
        or source.get("shadow_reserved_from_selection") is not True
    ):
        raise ValueError("two-generator Hybrid procedure drifted")
    w06 = cast(Mapping[str, Any], source.get("w06_anchor", {}))
    d01 = cast(Mapping[str, Any], source.get("d01_controlled", {}))
    if (
        w06.get("id") != "W06-4.5B-INSTRUCT-50K-8GB-BS8-L512"
        or w06.get("role") != "uncontrolled_safety_anchor_and_candidate_source"
        or d01.get("id") != "D01-4.5B-STYLE-50K-S42"
        or d01.get("role") != "controlled_candidate_source_and_task07_start"
    ):
        raise ValueError("Hybrid adapter roles drifted")
    selector = cast(Mapping[str, Any], source.get("selector", {}))
    if (
        selector.get("frozen_commit") != "2164822"
        or selector.get("anchor") != "all_four_uncontrolled_queries"
        or selector.get("enumerate_all_subsets") is not True
        or selector.get("deterministic_tie_break") != "lexicographic_candidate_identity"
    ):
        raise ValueError("safe-anchor selector drifted")
    task07 = cast(Mapping[str, Any], config.get("task07_start", {}))
    if (
        task07.get("adapter_role") != "d01_controlled"
        or task07.get("comparison_controls_required")
        != ["continued_sft", "score_weighted_continued_sft"]
        or task07.get("dpo_checkpoint_comparison_claim")
        != "not_established_by_task05_probe"
    ):
        raise ValueError("Task 07 start boundary drifted")
    authorization = cast(Mapping[str, Any], config.get("authorization", {}))
    if authorization != {
        "task06_handoff_approved": True,
        "task06_generation_authorized": False,
        "task06_scoring_authorized": False,
        "task06_preference_selection_authorized": False,
        "task07_training_authorized": False,
        "task09_promotion_authorized": False,
        "four_point_five_b_full_authorized": False,
        "final_tests_used": [],
    }:
        raise ValueError("Task 06/07 authorization boundary drifted")
    requirements = cast(Mapping[str, Any], config.get("next_stage_requirements", {}))
    if requirements != {
        "prospective_task06_execution_adr": True,
        "non_test_cohort_and_leakage_policy": True,
        "generation_request_matrix_k_4_to_8": True,
        "minimum_generation_seeds": 2,
        "frozen_primary_and_shadow": True,
        "frozen_component_calibration_and_human_evidence": True,
        "crash_safe_runner_and_single_operator_command": True,
        "triviaqa_training_forbidden": True,
    }:
        raise ValueError("next Task 06 stage requirements drifted")


def preflight_hybrid_handoff(config_path: Path) -> dict[str, Any]:
    """Verify the owner-approved handoff without loading either generator."""
    config = _load_yaml(config_path)
    _assert_shape(config)
    root = _root(config_path)
    adr = cast(Mapping[str, Any], config["adr"])
    _pin(root, adr, "path", "sha256")
    confirm = cast(Mapping[str, Any], config["confirm"])
    summary_path = _pin(root, confirm, "summary", "summary_sha256")
    _pin(root, confirm, "result_report", "result_report_sha256")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    guardrails = cast(Mapping[str, Any], summary.get("guardrails", {}))
    if (
        summary.get("status") != confirm["required_status"]
        or summary.get("decision") != confirm["required_decision"]
        or summary.get("selection_claim") != confirm["required_selection_claim"]
        or summary.get("retained_for_finalist_freeze") is not True
        or summary.get("task06_or_task09_promotion_authorized") is not False
        or summary.get("four_point_five_b_full_authorized") is not False
        or summary.get("final_tests_used") != []
        or cast(Mapping[str, Any], summary.get("primary_gate", {})).get("passed") is not True
        or not guardrails
        or any(
            cast(Mapping[str, Any], value).get("passed") is not True
            for value in guardrails.values()
        )
    ):
        raise ValueError("confirm result does not satisfy the frozen handoff gate")

    source = cast(Mapping[str, Any], config["source_procedure"])
    pilot_path = _pin(root, source, "pilot_config", "pilot_config_sha256")
    pilot = _load_yaml(pilot_path)
    pilot_arms = cast(Mapping[str, Any], pilot["arms"])
    for role, pilot_role in (("w06_anchor", "baseline"), ("d01_controlled", "controlled")):
        arm = cast(Mapping[str, Any], source[role])
        observed = cast(Mapping[str, Any], pilot_arms[pilot_role])
        if arm["id"] != observed["id"] or arm["adapter"] != observed["adapter"]:
            raise ValueError(f"{role} differs from the confirmed pilot identity")
        adapter = root / str(arm["adapter"])
        if _artifact_fingerprint(adapter) != arm["adapter_fingerprint"]:
            raise ValueError(f"{role} adapter fingerprint drifted")
        _pin(root, arm, "training_manifest", "training_manifest_sha256")
    selector = cast(Mapping[str, Any], source["selector"])
    _pin(root, selector, "implementation", "implementation_sha256")
    _pin(root, selector, "contract", "contract_sha256")
    if cast(Mapping[str, Any], pilot_arms["shared_model"]) != source["base_model"]:
        raise ValueError("shared base model differs from the confirmed pilot")

    return {
        "schema_version": 1,
        "contract": CONTRACT,
        "status": "verified_ready_for_task06_execution_design_not_generation",
        "config_sha256": _sha256(config_path),
        "confirm_summary_sha256": _sha256(summary_path),
        "candidate_pool": {"w06": 4, "d01_controlled": 4, "selected": 4},
        "task07_start_adapter_role": "d01_controlled",
        "model_loading_performed": False,
        "task06_generation_authorized": False,
        "task06_scoring_authorized": False,
        "task07_training_authorized": False,
        "task09_promotion_authorized": False,
        "four_point_five_b_full_authorized": False,
        "final_tests_used": [],
    }
