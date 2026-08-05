"""Fail-closed preregistered D01b equal-budget probe dev screen."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import yaml

from doc2query.evaluation.embedder_probe import ProbeRecipe
from doc2query.evaluation.p04_decision import evaluate_p04_comparison
from doc2query.evaluation.p05_guardrails import build_dev_screen_report
from doc2query.evaluation.statistical_contract import StatisticalContract
from doc2query.utils.records import read_records, write_json

D01B_PROBE_DEV_SCREEN_CONTRACT = "task05-d01b-probe-dev-screen-v1"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected mapping: {path}")
    return payload


def _root(path: Path) -> Path:
    root = next(
        (parent for parent in path.resolve().parents if (parent / "AGENTS.md").is_file()), None
    )
    if root is None:
        raise ValueError("cannot resolve repository root")
    return root


def _pinned(root: Path, section: Mapping[str, Any], path_key: str = "path") -> Path:
    path = root / str(section[path_key])
    if not path.is_file() or _file_sha256(path) != str(section["sha256"]):
        raise ValueError(f"D01b probe pin drifted: {path}")
    return path


def preflight_d01b_probe_dev_screen(config_path: Path) -> dict[str, Any]:
    """Validate every input, budget and dev-only boundary before GPU training."""
    config = _load_mapping(config_path)
    if (
        config.get("schema_version") != 1
        or config.get("contract") != D01B_PROBE_DEV_SCREEN_CONTRACT
        or config.get("status") != "amended_before_restart"
        or config.get("final_tests_used") != []
    ):
        raise ValueError("invalid D01b probe dev-screen contract")
    root = _root(config_path)
    _pinned(root, cast(Mapping[str, Any], config["adr"]))
    _pinned(root, cast(Mapping[str, Any], config["amendment"]))
    source_section = cast(Mapping[str, Any], config["source_materialization"])
    source_path = _pinned(root, source_section)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if (
        source.get("contract") != source_section.get("contract")
        or source.get("status") != source_section.get("status")
        or source.get("final_tests_used") != []
        or source.get("training_started") is not False
    ):
        raise ValueError("source materialization is not an authorized dev-only input")

    training = cast(Mapping[str, Any], config["training"])
    if (
        training.get("stage") != "dev_screen"
        or training.get("seed") != 42
        or training.get("max_steps") != 500
        or training.get("batch_size") != 4
        or training.get("train_prefix_pairs") != 1984
        or training.get("train_prefix_unique_passages") != 496
        or training.get("queries_per_passage") != 4
        or training.get("token_count") != 1_152_000
    ):
        raise ValueError("D01b probe dev-screen budget drifted")
    recipe_path = _pinned(
        root,
        {"path": training["probe_recipe"], "sha256": training["probe_recipe_sha256"]},
    )
    comparison_path = _pinned(
        root,
        {
            "path": training["comparison_contract"],
            "sha256": training["comparison_contract_sha256"],
        },
    )
    _pinned(
        root,
        {"path": training["primary_judge"], "sha256": training["primary_judge_sha256"]},
    )
    recipe = ProbeRecipe.from_dict(_load_mapping(recipe_path))
    runtime_recipe = ProbeRecipe.from_dict(
        asdict(recipe)
        | {
            "seed": int(training["seed"]),
            "max_steps": int(training["max_steps"]),
            "batch_size": int(training["batch_size"]),
        }
    )
    contract = StatisticalContract.load(comparison_path)
    if recipe.seed != 42 or recipe.batch_size != 8 or recipe.max_length != 192:
        raise ValueError("probe recipe execution budget drifted")
    if runtime_recipe.max_steps * runtime_recipe.batch_size * runtime_recipe.max_length * (
        2 + runtime_recipe.negatives_per_example
    ) != int(training["token_count"]):
        raise ValueError("D01b probe amended token budget drifted")
    if (
        recipe.negative_recipe.strategy != "hn0_filter"
        or recipe.negative_recipe.false_negative_policy != "drop"
    ):
        raise ValueError("D01b probe requires HN0+filter/drop")

    evaluation = cast(Mapping[str, Any], config["evaluation"])
    manifest_path = _pinned(
        root,
        {
            "path": evaluation["frozen_manifest"],
            "sha256": evaluation["frozen_manifest_sha256"],
        },
    )
    corpus_path = _pinned(
        root, {"path": evaluation["corpus"], "sha256": evaluation["corpus_sha256"]}
    )
    guardrails_path = _pinned(
        root,
        {
            "path": evaluation["natural_guardrails"],
            "sha256": evaluation["natural_guardrails_sha256"],
        },
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sets = manifest.get("sets")
    subset = sets.get(str(evaluation["subset"])) if isinstance(sets, Mapping) else None
    if (
        not isinstance(subset, Mapping)
        or subset.get("id_count") != evaluation["subset_id_count"]
        or subset.get("id_list_sha256") != evaluation["subset_id_list_sha256"]
        or evaluation.get("subset") != "dev_intrinsic_rank10"
        or evaluation.get("final_tests_forbidden") is not True
    ):
        raise ValueError("frozen dev evaluation panel drifted")
    evaluation_ids = {str(row["id"]) for row in read_records(root / str(subset["id_path"]))}
    if len(evaluation_ids) != int(evaluation["subset_id_count"]):
        raise ValueError("frozen dev evaluation ID count drifted")
    guardrail_ids = {str(row["example_id"]) for row in read_records(guardrails_path)}
    if guardrail_ids != evaluation_ids:
        raise ValueError("natural guardrails do not cover the exact dev panel")

    prefix_count = int(training["train_prefix_pairs"])
    arms = cast(Mapping[str, Any], config["arms"])
    if (
        cast(Mapping[str, Any], arms["control"]).get("id") != "D01B-PROBE-W05-DEV-SCREEN-S42-B4"
        or cast(Mapping[str, Any], arms["variant"]).get("id")
        != "D01B-PROBE-HYBRID-DEV-SCREEN-S42-B4"
    ):
        raise ValueError("D01b probe batch-4 arm identity drifted")
    outputs = cast(Mapping[str, Any], config["outputs"])
    if outputs != {
        "run_root": "runs/task05_d01b_probe_dev_screen_v2_batch4",
        "measurement_root": "reports/measurements/task05/d01b_probe_dev_screen_v2_batch4",
        "log_root": "logs/task05/d01b_probe_dev_screen_v2_batch4",
    }:
        raise ValueError("D01b probe batch-4 output namespace drifted")
    prefix_passages: dict[str, list[str]] = {}
    observed: dict[str, Any] = {}
    for role in ("control", "variant"):
        arm = cast(Mapping[str, Any], arms[role])
        input_path = _pinned(root, arm, path_key="input")
        rows = list(read_records(input_path))
        if len(rows) != int(source_section["full_pair_count"]):
            raise ValueError(f"{role} full probe input count drifted")
        if len({str(row.get("pair_id", "")) for row in rows}) != len(rows):
            raise ValueError(f"{role} probe pair identities are not unique")
        prefix = rows[:prefix_count]
        counts = Counter(str(row.get("source_passage_id", "")) for row in prefix)
        if len(counts) != int(training["train_prefix_unique_passages"]) or set(counts.values()) != {
            int(training["queries_per_passage"])
        }:
            raise ValueError(f"{role} prefix is not uniform exact K")
        if any(
            row.get("example_id") != row.get("pair_id")
            or row.get("generated") != row.get("query")
            or row.get("mode") != "deterministic"
            or row.get("candidate_index") != 0
            or not isinstance(row.get("positives"), list)
            or not row.get("hard_negatives")
            for row in prefix
        ):
            raise ValueError(f"{role} prefix is not compatible with the frozen probe loader")
        source_ids = {str(row["source_example_id"]) for row in prefix}
        if source_ids & evaluation_ids:
            raise ValueError(f"{role} training prefix overlaps the dev evaluation panel")
        prefix_passages[role] = list(counts)
        observed[role] = {
            "input": str(input_path.relative_to(root)),
            "sha256": _file_sha256(input_path),
            "full_pair_count": len(rows),
            "prefix_pair_count": len(prefix),
            "prefix_unique_passages": len(counts),
        }
    if prefix_passages["control"] != prefix_passages["variant"]:
        raise ValueError("probe arms do not use the same ordered passage prefix")

    authorization = cast(Mapping[str, Any], config["authorization"])
    if authorization != {
        "dev_screen_training": True,
        "dev_confirm": False,
        "four_point_five_b": False,
        "final_tests": False,
    }:
        raise ValueError("D01b probe authorization scope drifted")
    return {
        "schema_version": 1,
        "contract": config["contract"],
        "status": "verified",
        "config_sha256": _file_sha256(config_path),
        "arms": observed,
        "comparison_contract": contract.reference(),
        "base_probe_recipe_fingerprint": recipe.fingerprint,
        "probe_recipe_fingerprint": runtime_recipe.fingerprint,
        "evaluation_subset": evaluation["subset"],
        "evaluation_query_count": len(evaluation_ids),
        "corpus": str(corpus_path.relative_to(root)),
        "final_tests_used": [],
    }


def build_d01b_probe_dev_screen_decision(config_path: Path) -> dict[str, Any]:
    """Build the paired dev-only P-04 report after both probe runs complete."""
    preflight = preflight_d01b_probe_dev_screen(config_path)
    config = _load_mapping(config_path)
    root = _root(config_path)
    training = cast(Mapping[str, Any], config["training"])
    evaluation = cast(Mapping[str, Any], config["evaluation"])
    outputs = cast(Mapping[str, Any], config["outputs"])
    arms = cast(Mapping[str, Any], config["arms"])
    run_root = root / str(outputs["run_root"])
    measurement_root = root / str(outputs["measurement_root"])
    control = cast(Mapping[str, Any], arms["control"])
    variant = cast(Mapping[str, Any], arms["variant"])
    control_dir = run_root / str(control["id"])
    variant_dir = run_root / str(variant["id"])
    control_result = json.loads((control_dir / "result.json").read_text(encoding="utf-8"))
    variant_result = json.loads((variant_dir / "result.json").read_text(encoding="utf-8"))
    contract = StatisticalContract.load(root / str(training["comparison_contract"]))
    guardrails = root / str(evaluation["natural_guardrails"])
    report = build_dev_screen_report(
        arm_id=str(variant["id"]),
        control_id=str(control["id"]),
        arm_result=variant_result,
        control_result=control_result,
        arm_per_query_path=variant_dir / "corpus_retrieval_per_query.jsonl",
        control_per_query_path=control_dir / "corpus_retrieval_per_query.jsonl",
        arm_guardrails_path=guardrails,
        control_guardrails_path=guardrails,
        contract=contract,
    )
    decision = evaluate_p04_comparison(report, control_manifest=control_result, contract=contract)
    measurement_root.mkdir(parents=True, exist_ok=True)
    write_json(measurement_root / "comparison_report.json", report)
    write_json(measurement_root / "decision.json", decision)
    summary = {
        "schema_version": 1,
        "contract": config["contract"],
        "status": "dev_screen_complete",
        "preflight": preflight,
        "decision": decision,
        "dev_confirm_authorized": decision.get("status") == "eligible",
        "four_point_five_b_authorized": False,
        "final_tests_used": [],
    }
    write_json(measurement_root / "summary.json", summary)
    return summary
