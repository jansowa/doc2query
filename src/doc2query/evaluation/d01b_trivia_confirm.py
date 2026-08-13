"""Fail-closed preflight and comparison for the D01b 4.5B TriviaQA confirm."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, cast

import numpy as np
import yaml

from doc2query.evaluation.d01_pipeline import _artifact_fingerprint
from doc2query.evaluation.datasets import load_frozen_records
from doc2query.evaluation.embedder_probe import ProbeRecipe
from doc2query.utils.records import read_records, write_json

CONTRACT = "task05-d01b-scale-interaction-4.5b-trivia-dev-confirm-v1"
SUBSET = "dev_d01b_trivia_external_v1"
EXPECTED_METRICS = (
    "corpus_ndcg_at_10",
    "corpus_recall_at_1",
    "corpus_recall_at_5",
    "corpus_recall_at_10",
    "corpus_recall_at_100",
    "corpus_mrr_at_10",
    "corpus_map",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
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


def _assert_pin(root: Path, path: str, expected: str) -> Path:
    resolved = root / path
    if not resolved.is_file() or _sha256(resolved) != expected:
        raise ValueError(f"pinned file drifted: {path}")
    return resolved


def _assert_contract_shape(config: Mapping[str, Any]) -> None:
    if (
        config.get("schema_version") != 1
        or config.get("contract") != CONTRACT
        or config.get("status") != "preregistered_before_external_model_evaluation"
        or config.get("final_tests_used") != []
    ):
        raise ValueError("invalid TriviaQA confirm contract")
    source = cast(Mapping[str, Any], config.get("source_pilot", {}))
    if (
        source.get("required_status") != "pilot_complete"
        or source.get("required_decision_status") != "eligible"
        or source.get("required_dev_confirm_authorized") is not True
        or source.get("selection_claim") is not None
        or source.get("retained_for_finalist_freeze") is not False
        or source.get("four_point_five_b_full_authorized") is not False
        or source.get("final_tests_used") != []
    ):
        raise ValueError("source pilot boundary drifted")
    external = cast(Mapping[str, Any], config.get("external_development", {}))
    if (
        external.get("subset") != SUBSET
        or external.get("query_count") != 8000
        or external.get("positive_filter") != "pos_scores_stronger_reranker > 23.50"
        or external.get("bootstrap_unit") != "query_id"
        or external.get("corpus_document_count") != 139782
        or external.get("license_status")
        != "absent_from_downloaded_dataset_card_internal_evaluation_only"
    ):
        raise ValueError("external development contract drifted")
    training = cast(Mapping[str, Any], config.get("training", {}))
    if (
        training.get("seeds") != [42, 43, 44]
        or training.get("reused_without_retraining") != [42]
        or training.get("newly_trained") != [43, 44]
        or training.get("pair_count") != 3072
        or training.get("unique_passage_count") != 768
        or training.get("queries_per_passage") != 4
        or training.get("max_steps") != 1024
        or training.get("batch_size") != 2
        or training.get("max_length") != 192
        or training.get("negatives_per_example") != 1
        or training.get("token_count") != 1179648
    ):
        raise ValueError("matched training budget or seeds drifted")
    evaluation = cast(Mapping[str, Any], config.get("evaluation", {}))
    if (
        evaluation.get("primary_metric") != "corpus_ndcg_at_10"
        or evaluation.get("practical_effect_threshold") != 0.01
        or evaluation.get("interval") != "paired_query_percentile_two_sided_97_5"
        or evaluation.get("interval_quantiles") != [0.0125, 0.9875]
        or evaluation.get("bootstrap_samples") != 10000
        or evaluation.get("bootstrap_seed") != 20260721
        or evaluation.get("rng") != "numpy.random.PCG64"
        or evaluation.get("fixed_seed_aggregation") != "per_query_mean_before_query_resampling"
        or evaluation.get("resample_training_seeds") is not False
        or evaluation.get("required_metrics") != list(EXPECTED_METRICS)
        or evaluation.get("guardrails")
        != {
            "corpus_recall_at_10": {"lower_bound_minimum": -0.02},
            "corpus_mrr_at_10": {"lower_bound_minimum": -0.02},
            "corpus_map": {"lower_bound_minimum": -0.02},
        }
        or evaluation.get("all_gates_required") is not True
        or evaluation.get("final_tests_forbidden") is not True
    ):
        raise ValueError("97.5% statistical contract drifted")
    amendment = cast(Mapping[str, Any], config.get("execution_amendment", {}))
    if (
        amendment.get("reason") != "host_poweroff_risk_during_sustained_corpus_encoding"
        or amendment.get("prior_evaluation_encode_batch_size") != 32
    ):
        raise ValueError("batch-8 execution amendment drifted")
    execution = cast(Mapping[str, Any], config.get("execution", {}))
    if (
        execution.get("evaluation_encode_batch_size") != 8
        or execution.get("retrieval_query_batch_size") != 512
        or execution.get("retrieval_device") != "cuda"
        or execution.get("operator_command")
        != "bash scripts/run_task05_d01b_scale_interaction_4_5b_trivia_dev_confirm.sh run-all"
    ):
        raise ValueError("batch-8 execution contract drifted")
    authorization = cast(Mapping[str, Any], config.get("authorization", {}))
    if authorization != {
        "cohort_preparation": True,
        "cpu_preflight": True,
        "protocol_preregistered": True,
        "expensive_run_requires_explicit_operator_command": True,
        "expensive_run_started": False,
        "pilot_retraining": False,
        "task06_or_task09_promotion": False,
        "four_point_five_b_full_authorized": False,
        "final_tests": False,
    }:
        raise ValueError("confirm authorization boundary drifted")


def _arm_training_identity(path: Path) -> tuple[int, int, int, int]:
    rows = list(read_records(path))
    passage_counts: dict[str, int] = {}
    for row in rows:
        passage_id = str(row["source_passage_id"])
        passage_counts[passage_id] = passage_counts.get(passage_id, 0) + 1
        if str(row.get("mode")) != "deterministic":
            raise ValueError("pilot probe input mode drifted")
    counts = set(passage_counts.values())
    if counts != {4}:
        raise ValueError("pilot probe input no longer has exact K=4")
    return len(rows), len(passage_counts), min(counts), len({str(row["pair_id"]) for row in rows})


def _seed42_destination(config: Mapping[str, Any], root: Path, role: str) -> Path:
    execution = cast(Mapping[str, Any], config["execution"])
    arm = cast(Mapping[str, Any], cast(Mapping[str, Any], config["arms"])[role])
    runs = cast(Mapping[str, Any], arm["runs"])
    return root / str(execution["output_root"]) / str(runs["42"])


def preflight_trivia_confirm(
    config_path: Path, *, require_staged_seed42: bool = False
) -> dict[str, Any]:
    """Validate every cohort, input, model, budget, and authorization pin."""
    config = _load(config_path)
    _assert_contract_shape(config)
    root = _root(config_path)
    amendment = cast(Mapping[str, Any], config["execution_amendment"])
    _assert_pin(root, str(amendment["path"]), str(amendment["sha256"]))
    adr = cast(Mapping[str, Any], config["adr"])
    _assert_pin(root, str(adr["path"]), str(adr["sha256"]))
    source = cast(Mapping[str, Any], config["source_pilot"])
    _assert_pin(root, str(source["config"]), str(source["config_sha256"]))
    summary_path = _assert_pin(root, str(source["summary"]), str(source["summary_sha256"]))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    decision = cast(Mapping[str, Any], summary.get("decision", {}))
    if (
        summary.get("status") != "pilot_complete"
        or decision.get("status") != "eligible"
        or summary.get("dev_confirm_authorized") is not True
        or decision.get("selection_claim") is not None
        or summary.get("retained_for_finalist_freeze") is not False
        or summary.get("four_point_five_b_full_authorized") is not False
        or summary.get("final_tests_used") != []
    ):
        raise ValueError("completed scale pilot no longer authorizes only a confirm")

    external = cast(Mapping[str, Any], config["external_development"])
    _assert_pin(root, str(external["policy"]), str(external["policy_sha256"]))
    snapshot = _assert_pin(
        root, str(external["cohort_snapshot"]), str(external["cohort_snapshot_sha256"])
    )
    manifest_path = _assert_pin(root, str(external["manifest"]), str(external["manifest_sha256"]))
    if snapshot.read_bytes() != manifest_path.read_bytes():
        raise ValueError("tracked cohort snapshot differs from materialized manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "materialized_before_model_evaluation"
        or manifest.get("authorization")
        != {"final_tests": False, "model_evaluation": False, "probe_training": False}
        or manifest.get("final_tests_used") != []
    ):
        raise ValueError("external cohort authorization drifted")
    records = load_frozen_records(manifest_path, SUBSET)
    if len(records) != 8000:
        raise ValueError("external cohort query count drifted")
    corpus = _assert_pin(root, str(external["corpus"]), str(external["corpus_sha256"]))

    training = cast(Mapping[str, Any], config["training"])
    for key in ("probe_recipe", "comparison_contract", "primary_judge"):
        _assert_pin(root, str(training[key]), str(training[f"{key}_sha256"]))
    recipe = ProbeRecipe.from_dict(_load(root / str(training["probe_recipe"])))
    runtime = ProbeRecipe.from_dict(
        asdict(recipe)
        | {"seed": 42, "max_steps": training["max_steps"], "batch_size": training["batch_size"]}
    )
    if (
        runtime.max_length != training["max_length"]
        or runtime.negatives_per_example != training["negatives_per_example"]
        or runtime.max_steps
        * runtime.batch_size
        * runtime.max_length
        * (2 + runtime.negatives_per_example)
        != training["token_count"]
    ):
        raise ValueError("runtime probe recipe does not match frozen token budget")

    arm_evidence: dict[str, Any] = {}
    passage_sets: dict[str, set[str]] = {}
    for role in ("control", "variant"):
        arm = cast(Mapping[str, Any], cast(Mapping[str, Any], config["arms"])[role])
        train_input = _assert_pin(root, str(arm["train_input"]), str(arm["train_input_sha256"]))
        identity = _arm_training_identity(train_input)
        if identity != (3072, 768, 4, 3072):
            raise ValueError(f"{role} pilot training input budget drifted")
        passage_sets[role] = {str(row["source_passage_id"]) for row in read_records(train_input)}
        reused = cast(Mapping[str, Any], arm["reused_seed42"])
        source_run = root / str(reused["source_run"])
        train_summary = source_run / "train_summary.json"
        if _sha256(train_summary) != str(reused["train_summary_sha256"]):
            raise ValueError(f"{role} seed-42 training summary drifted")
        if _artifact_fingerprint(source_run / "model") != str(reused["model_fingerprint"]):
            raise ValueError(f"{role} seed-42 model drifted")
        staged = _seed42_destination(config, root, role)
        staged_ok = (
            staged.is_dir()
            and _sha256(staged / "train_summary.json") == str(reused["train_summary_sha256"])
            and _artifact_fingerprint(staged / "model") == str(reused["model_fingerprint"])
        )
        if require_staged_seed42 and not staged_ok:
            raise ValueError(f"{role} seed-42 reuse has not been staged")
        arm_evidence[role] = {
            "train_input_sha256": _sha256(train_input),
            "seed42_train_summary_sha256": _sha256(train_summary),
            "seed42_model_fingerprint": _artifact_fingerprint(source_run / "model"),
            "seed42_staged": staged_ok,
        }
    if passage_sets["control"] != passage_sets["variant"]:
        raise ValueError("matched arms no longer use identical passage IDs")

    resources = cast(Mapping[str, Any], config["resources"])
    free = shutil.disk_usage(root).free
    if free < int(resources["minimum_free_disk_bytes"]):
        raise ValueError("insufficient disk for crash-safe external confirm")
    return {
        "schema_version": 1,
        "contract": CONTRACT,
        "status": "verified",
        "config_sha256": _sha256(config_path),
        "external_query_count": len(records),
        "external_corpus_sha256": _sha256(corpus),
        "external_corpus_document_count": int(external["corpus_document_count"]),
        "arms": arm_evidence,
        "training_seeds": [42, 43, 44],
        "reused_without_retraining": [42],
        "newly_trained": [43, 44],
        "interval": "paired_query_percentile_two_sided_97_5",
        "primary_threshold": 0.01,
        "operator_command": cast(Mapping[str, Any], config["execution"])["operator_command"],
        "evaluation_encode_batch_size": cast(Mapping[str, Any], config["execution"])[
            "evaluation_encode_batch_size"
        ],
        "execution_amendment_sha256": str(amendment["sha256"]),
        "expensive_run_started": False,
        "four_point_five_b_full_authorized": False,
        "final_tests_used": [],
        "free_disk_bytes": free,
    }


def stage_reused_seed42(config_path: Path) -> dict[str, Any]:
    """Atomically stage only completed training state; never copy old evaluation results."""
    config = _load(config_path)
    preflight_trivia_confirm(config_path, require_staged_seed42=False)
    root = _root(config_path)
    staged_roles: dict[str, str] = {}
    for role in ("control", "variant"):
        arm = cast(Mapping[str, Any], cast(Mapping[str, Any], config["arms"])[role])
        reused = cast(Mapping[str, Any], arm["reused_seed42"])
        source = root / str(reused["source_run"])
        destination = _seed42_destination(config, root, role)
        if destination.exists():
            staged_roles[role] = "already_staged"
            continue
        temporary = destination.with_name(destination.name + ".staging")
        if temporary.exists():
            raise FileExistsError(f"stale seed-42 staging directory: {temporary}")
        temporary.mkdir(parents=True)
        try:
            shutil.copy2(source / "train_summary.json", temporary / "train_summary.json")
            shutil.copy2(source / "negative_audit.jsonl", temporary / "negative_audit.jsonl")
            os.symlink((source / "model").resolve(), temporary / "model", target_is_directory=True)
            write_json(
                temporary / "reused_seed42.json",
                {
                    "schema_version": 1,
                    "status": "training_reused_without_retraining",
                    "source_run": str(source.relative_to(root)),
                    "train_summary_sha256": _sha256(source / "train_summary.json"),
                    "model_fingerprint": _artifact_fingerprint(source / "model"),
                    "old_evaluation_artifacts_copied": False,
                    "final_tests_used": [],
                },
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temporary, destination)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        staged_roles[role] = "staged"
    verified = preflight_trivia_confirm(config_path, require_staged_seed42=True)
    return {"status": "seed42_staged", "roles": staged_roles, "preflight": verified}


def _seed_metric_rows(path: Path, expected_ids: set[str]) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for row in read_records(path):
        query_id = str(row["example_id"])
        if query_id in rows:
            raise ValueError("duplicate query ID in retrieval output")
        values = {metric: float(row[metric]) for metric in EXPECTED_METRICS}
        if any(not math.isfinite(value) for value in values.values()):
            raise ValueError("non-finite retrieval metric")
        rows[query_id] = values
    if set(rows) != expected_ids:
        raise ValueError("retrieval output does not cover the frozen external query IDs")
    return rows


def _paired_97_5(
    control: Mapping[str, float], variant: Mapping[str, float], *, samples: int, seed: int
) -> dict[str, float | int]:
    ids = sorted(control)
    if not ids or set(ids) != set(variant):
        raise ValueError("paired 97.5% bootstrap requires identical non-empty query IDs")
    differences = np.asarray([variant[key] - control[key] for key in ids], dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(seed))
    estimates = np.empty(samples, dtype=np.float64)
    chunk = 128
    for start in range(0, samples, chunk):
        count = min(chunk, samples - start)
        indices = rng.integers(0, len(ids), size=(count, len(ids)))
        estimates[start : start + count] = differences[indices].mean(axis=1)
    low, high = np.quantile(estimates, [0.0125, 0.9875])
    return {
        "query_count": len(ids),
        "bootstrap_samples": samples,
        "seed": seed,
        "difference": float(differences.mean()),
        "ci97_5_low": float(low),
        "ci97_5_high": float(high),
        "variant_win_fraction": float(np.mean(estimates > 0.0)),
    }


def compare_trivia_confirm(config_path: Path) -> dict[str, Any]:
    """Aggregate fixed seeds, bootstrap queries at 97.5%, and apply frozen gates."""
    preflight = preflight_trivia_confirm(config_path, require_staged_seed42=True)
    config = _load(config_path)
    root = _root(config_path)
    external = cast(Mapping[str, Any], config["external_development"])
    manifest = json.loads((root / str(external["manifest"])).read_text(encoding="utf-8"))
    ids_path = root / str(manifest["sets"][SUBSET]["id_path"])
    expected_ids = {str(row["id"]) for row in read_records(ids_path)}
    execution = cast(Mapping[str, Any], config["execution"])
    run_root = root / str(execution["output_root"])
    by_arm: dict[str, dict[int, dict[str, dict[str, float]]]] = {}
    seed_means: dict[str, dict[str, dict[str, float]]] = {}
    for role in ("control", "variant"):
        arm = cast(Mapping[str, Any], cast(Mapping[str, Any], config["arms"])[role])
        runs = cast(Mapping[str, Any], arm["runs"])
        by_arm[role] = {}
        seed_means[role] = {}
        for seed in (42, 43, 44):
            run_dir = run_root / str(runs[str(seed)])
            result_path = run_dir / "result.json"
            per_query_path = run_dir / "corpus_retrieval_per_query.jsonl"
            if not result_path.is_file() or not per_query_path.is_file():
                raise ValueError(f"incomplete confirm run: {role} seed {seed}")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            retrieval = cast(Mapping[str, Any], result.get("corpus_retrieval", {}))
            if (
                retrieval.get("status") != "measured"
                or retrieval.get("query_count") != 8000
                or retrieval.get("corpus_sha256") != external["corpus_sha256"]
            ):
                raise ValueError(f"confirm retrieval identity drifted: {role} seed {seed}")
            rows = _seed_metric_rows(per_query_path, expected_ids)
            by_arm[role][seed] = rows
            seed_means[role][str(seed)] = {
                metric: fmean(row[metric] for row in rows.values()) for metric in EXPECTED_METRICS
            }
    averaged: dict[str, dict[str, dict[str, float]]] = {"control": {}, "variant": {}}
    for role in ("control", "variant"):
        averaged[role] = {
            query_id: {
                metric: fmean(by_arm[role][seed][query_id][metric] for seed in (42, 43, 44))
                for metric in EXPECTED_METRICS
            }
            for query_id in sorted(expected_ids)
        }
    evaluation = cast(Mapping[str, Any], config["evaluation"])
    comparisons = {
        metric: _paired_97_5(
            {query_id: row[metric] for query_id, row in averaged["control"].items()},
            {query_id: row[metric] for query_id, row in averaged["variant"].items()},
            samples=int(evaluation["bootstrap_samples"]),
            seed=int(evaluation["bootstrap_seed"]),
        )
        for metric in EXPECTED_METRICS
    }
    seed_dispersion: dict[str, Any] = {}
    for role in ("control", "variant"):
        seed_dispersion[role] = {}
        for metric in EXPECTED_METRICS:
            values = [seed_means[role][str(seed)][metric] for seed in (42, 43, 44)]
            seed_dispersion[role][metric] = {
                "mean": fmean(values),
                "sample_std": stdev(values),
                "minimum": min(values),
                "maximum": max(values),
                "range": max(values) - min(values),
            }
    primary_passed = float(comparisons["corpus_ndcg_at_10"]["ci97_5_low"]) >= float(
        evaluation["practical_effect_threshold"]
    )
    guardrails = cast(Mapping[str, Any], evaluation["guardrails"])
    guardrail_results = {
        metric: {
            "ci97_5_low": comparisons[metric]["ci97_5_low"],
            "minimum": cast(Mapping[str, Any], rule)["lower_bound_minimum"],
            "passed": float(comparisons[metric]["ci97_5_low"])
            >= float(cast(Mapping[str, Any], rule)["lower_bound_minimum"]),
        }
        for metric, rule in guardrails.items()
    }
    passed = primary_passed and all(bool(result["passed"]) for result in guardrail_results.values())
    summary = {
        "schema_version": 1,
        "contract": CONTRACT,
        "status": "external_dev_confirm_complete",
        "decision": "eligible_for_finalist_freeze_review" if passed else "not_confirmed",
        "preflight": preflight,
        "paired_query_bootstrap": {
            "interval": "two_sided_97_5_percentile",
            "rng": "numpy.random.PCG64",
            "fixed_seed_aggregation": "per_query_mean_before_query_resampling",
            "metrics": comparisons,
        },
        "seed_means": seed_means,
        "seed_dispersion": seed_dispersion,
        "primary_gate": {
            "metric": "corpus_ndcg_at_10",
            "lower_bound_minimum": 0.01,
            "passed": primary_passed,
        },
        "guardrails": guardrail_results,
        "selection_claim": (
            "external_dev_confirm_passed_pending_finalist_freeze_review" if passed else None
        ),
        "retained_for_finalist_freeze": passed,
        "task06_or_task09_promotion_authorized": False,
        "four_point_five_b_full_authorized": False,
        "final_tests_used": [],
    }
    measurement_root = root / str(execution["measurement_root"])
    measurement_root.mkdir(parents=True, exist_ok=True)
    write_json(measurement_root / "summary.json", summary)
    return summary
