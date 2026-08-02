"""Prospective, dev-only validation of the preregistered D01b 1.5B selector."""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any, cast

import yaml

from doc2query.config import load_config
from doc2query.evaluation.bootstrap import paired_bootstrap
from doc2query.evaluation.d01_campaign import D01B_PROSPECTIVE_COHORT_CONTRACT
from doc2query.evaluation.d01_pipeline import _artifact_fingerprint
from doc2query.evaluation.d01_usefulness import (
    D01UsefulnessContract,
    _compact_candidates,
    _difficulty_report,
    _load_natural_scores,
    _load_or_encode,
    _select_groups,
    _selection_report,
)
from doc2query.evaluation.datasets import load_frozen_records
from doc2query.evaluation.retrieval import percentile
from doc2query.utils.records import read_records, write_json

PROSPECTIVE_CONTRACT = "task05-d01b-prospective-1.5b-v1"
PREREGISTERED_COHORT_CONTRACT = "task05-d01b-prospective-cohort-v1"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_json(temporary, value)
    os.replace(temporary, path)


def _positive(record: Mapping[str, Any]) -> Mapping[str, Any]:
    positives = record.get("positives")
    if not isinstance(positives, list) or not positives:
        raise ValueError("prospective record has no positive")
    return cast(Mapping[str, Any], sorted(positives, key=lambda item: str(item["doc_id"]))[0])


def _load_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected mapping: {path}")
    return payload


def _assert_file_pin(root: Path, section: Mapping[str, Any], *, label: str) -> Path:
    path = root / str(section.get("path", section.get("config", "")))
    expected = section.get("sha256")
    if not path.is_file() or (expected is not None and _file_sha256(path) != str(expected)):
        raise ValueError(f"prospective {label} fingerprint mismatch")
    return path


def load_prospective_contract(path: Path) -> tuple[dict[str, Any], Path, str]:
    """Load and validate immutable preregistration pins without opening any output."""
    payload = _load_mapping(path)
    if (
        payload.get("schema_version") != 1
        or payload.get("contract") != PROSPECTIVE_CONTRACT
        or payload.get("status") != "preregistered_before_generation"
        or payload.get("final_tests_used") != []
    ):
        raise ValueError("invalid prospective D01b contract")
    root = next(
        (parent for parent in path.resolve().parents if (parent / "AGENTS.md").is_file()),
        None,
    )
    if root is None:
        raise ValueError("cannot resolve repository root")
    _assert_file_pin(root, cast(Mapping[str, Any], payload["adr"]), label="ADR")
    selector = cast(Mapping[str, Any], payload["selector"])
    _assert_file_pin(root, cast(Mapping[str, Any], selector["implementation"]), label="selector")
    _assert_file_pin(
        root,
        cast(Mapping[str, Any], selector["retrospective_contract"]),
        label="selector contract",
    )
    if str(selector.get("frozen_commit")) != "2164822":
        raise ValueError("prospective selector commit drifted")
    cohort = cast(Mapping[str, Any], payload["cohort"])
    preregistered = root / str(cohort["manifest"])
    prereg_payload = json.loads(preregistered.read_text(encoding="utf-8"))
    if (
        prereg_payload.get("contract") != PREREGISTERED_COHORT_CONTRACT
        or prereg_payload.get("final_tests_used") != []
    ):
        raise ValueError("invalid preregistered cohort manifest")
    return payload, root, _file_sha256(path)


def preflight_prospective(path: Path) -> dict[str, Any]:
    """Fail closed on any model, adapter, judge, corpus, or preregistration drift."""
    payload, root, contract_sha = load_prospective_contract(path)
    cohort = cast(Mapping[str, Any], payload["cohort"])
    source = root / str(cohort["source_records"])
    if _file_sha256(source) != str(cohort["source_records_sha256"]):
        raise ValueError("prospective source dev fingerprint mismatch")
    natural = cast(Mapping[str, Any], payload["natural_primary_scores"])
    _assert_file_pin(root, natural, label="natural primary scores")
    arms = cast(Mapping[str, Any], payload["arms"])
    observed_arms: dict[str, Any] = {}
    for role in ("baseline", "controlled"):
        arm = cast(Mapping[str, Any], arms[role])
        config_path = root / str(arm["generation_config"])
        if _file_sha256(config_path) != str(arm["generation_config_sha256"]):
            raise ValueError(f"{role} generation config drifted")
        config = load_config(config_path)
        if config.run.experiment_id != arm["generation_experiment_id"]:
            raise ValueError(f"{role} prospective experiment identity drifted")
        adapter = root / str(arm["adapter"])
        if _artifact_fingerprint(adapter) != str(arm["adapter_sha256"]):
            raise ValueError(f"{role} adapter drifted")
        training = root / str(arm["training_manifest"])
        if _file_sha256(training) != str(arm["training_manifest_sha256"]):
            raise ValueError(f"{role} training manifest drifted")
        observed_arms[role] = {
            "experiment_id": config.run.experiment_id,
            "config_sha256": _file_sha256(config_path),
            "adapter_sha256": _artifact_fingerprint(adapter),
        }
    scoring = cast(Mapping[str, Any], payload["scoring"])
    for role in ("primary", "shadow"):
        pin = cast(Mapping[str, Any], scoring[role])
        judge = _load_mapping(root / str(pin["config"]))
        for field in ("name_or_path", "revision", "trust_remote_code"):
            if judge.get(field) != pin.get(field):
                raise ValueError(f"{role} judge {field} drifted")
    corpus = cast(Mapping[str, Any], scoring["corpus_index"])
    corpus_manifest = _load_mapping(root / str(corpus["path"]) / "manifest.json")
    if corpus_manifest.get("index_fingerprint") != corpus.get("fingerprint"):
        raise ValueError("prospective corpus index drifted")
    return {
        "schema_version": 1,
        "contract": PROSPECTIVE_CONTRACT,
        "status": "verified",
        "contract_sha256": contract_sha,
        "selector_commit": "2164822",
        "arms": observed_arms,
        "cohort_selected_count": int(cohort["selected_count"]),
        "final_tests_used": [],
    }


def _selected_cohort(
    contract_path: Path,
) -> tuple[dict[str, Any], Path, list[dict[str, Any]], list[int], dict[str, Any]]:
    payload, root, contract_sha = load_prospective_contract(contract_path)
    cohort = cast(Mapping[str, Any], payload["cohort"])
    records = load_frozen_records(
        root / str(cohort["source_frozen_manifest"]), str(cohort["source_subset"])
    )
    excluded_path = (
        root / "data/processed/v1/evaluation/task04-v1" / f"{cohort['exclude_subset']}.ids.jsonl"
    )
    excluded = {str(row["id"]) for row in read_records(excluded_path)}
    minimum = int(cohort["minimum_hard_negatives"])
    seed = int(cohort["selection_seed"])
    eligible: list[tuple[str, str, int, dict[str, Any]]] = []
    for index, record in enumerate(records):
        example_id = str(record["example_id"])
        negatives = record.get("hard_negatives")
        if example_id not in excluded and isinstance(negatives, list) and len(negatives) >= minimum:
            digest = hashlib.sha256(f"{seed}:{example_id}".encode()).hexdigest()
            eligible.append((digest, example_id, index, record))
    eligible.sort(key=lambda item: (item[0], item[1]))
    selected_rows = eligible[: int(cohort["selected_count"])]
    selected = [item[3] for item in selected_rows]
    source_indices = [item[2] for item in selected_rows]
    example_ids = [item[1] for item in selected_rows]
    id_sha = hashlib.sha256(("\n".join(example_ids) + "\n").encode()).hexdigest()
    record_digest = hashlib.sha256()
    metadata_rows = []
    group_ids = []
    for record, source_index in zip(selected, source_indices, strict=True):
        positive = _positive(record)
        negatives = sorted(str(item["doc_id"]) for item in record["hard_negatives"])
        record_digest.update(_canonical_bytes(record))
        record_digest.update(b"\n")
        group_ids.append(f"{record['example_id']}::{positive['doc_id']}")
        metadata_rows.append(
            {
                "source_index": source_index,
                "example_id": str(record["example_id"]),
                "positive_doc_id": str(positive["doc_id"]),
                "hard_negative_doc_ids": negatives,
                "hard_negative_count": len(negatives),
            }
        )
    metadata_sha = hashlib.sha256(
        b"".join(_canonical_bytes(row) + b"\n" for row in metadata_rows)
    ).hexdigest()
    checks = {
        "eligible_count": len(eligible),
        "selected_id_list_sha256": id_sha,
        "selected_metadata_records_sha256": metadata_sha,
        "selected_source_records_sha256": record_digest.hexdigest(),
        "intersection_with_excluded_subset": len(set(example_ids) & excluded),
    }
    preregistered = json.loads((root / str(cohort["manifest"])).read_text(encoding="utf-8"))
    preregistered_identity = cast(Mapping[str, Any], preregistered["cohort_identity"])
    expected = {
        "eligible_count": int(cohort["eligible_count"]),
        "selected_id_list_sha256": str(cohort["selected_id_list_sha256"]),
        "selected_source_records_sha256": str(cohort["selected_records_sha256"]),
        "intersection_with_excluded_subset": int(cohort["intersection_with_excluded_subset"]),
        "selected_metadata_records_sha256": str(
            preregistered_identity["selected_metadata_records_sha256"]
        ),
    }
    for field, value in expected.items():
        if checks[field] != value:
            raise ValueError(f"prospective cohort {field} drifted")
    selection_policy = {
        "policy": "sha256_unseen_dev_min_hn_v1",
        "selection_seed": seed,
        "minimum_hard_negatives": minimum,
        "selected_count": len(selected),
        "selected_id_list_sha256": id_sha,
        "selected_source_records_sha256": record_digest.hexdigest(),
        "quality_metrics_used": [],
        "final_tests_used": [],
    }
    materialized = {
        "schema_version": 1,
        "contract": D01B_PROSPECTIVE_COHORT_CONTRACT,
        "status": "materialized_before_generation",
        "preregistration_contract_sha256": contract_sha,
        "source_subset": str(cohort["source_subset"]),
        "excluded_subset": str(cohort["exclude_subset"]),
        "selected_example_ids": example_ids,
        "selected_group_ids": group_ids,
        "selected_group_ids_sha256": hashlib.sha256("\n".join(group_ids).encode()).hexdigest(),
        "selected_source_indices_sha256": _canonical_sha256(source_indices),
        "selection_policy": selection_policy,
        "selection_policy_fingerprint": _canonical_sha256(selection_policy),
        "cohort_checks": checks,
        "quality_metrics_used": [],
        "final_tests_used": [],
    }
    return payload, root, selected, source_indices, materialized


def prepare_prospective_cohort(contract_path: Path, output_path: Path) -> dict[str, Any]:
    """Materialize only IDs and fingerprints; never materialize passage text."""
    _payload, _root, _records, _indices, manifest = _selected_cohort(contract_path)
    if output_path.is_file():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError("existing prospective cohort manifest drifted")
        return cast(dict[str, Any], existing)
    _atomic_json(output_path, manifest)
    return manifest


def validate_materialized_cohort(contract_path: Path, output_path: Path) -> dict[str, Any]:
    _payload, _root, _records, _indices, expected = _selected_cohort(contract_path)
    if not output_path.is_file():
        raise FileNotFoundError("run prepare-cohort before expensive phases")
    observed = json.loads(output_path.read_text(encoding="utf-8"))
    if observed != expected:
        raise ValueError("materialized prospective cohort identity mismatch")
    return cast(dict[str, Any], observed)


def assert_exact_k_summary(path: Path, *, groups: int = 2000, k: int = 4) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if (
        summary.get("status") != "measured"
        or summary.get("final_tests_used") != []
        or int(summary.get("source_passage_count", -1)) != groups
        or int(summary.get("generation_count", -1)) != groups * k
        or int(summary.get("exhausted_groups", -1)) != 0
        or int(summary.get("target_queries_per_passage", -1)) != k
    ):
        raise ValueError("prospective generation failed exact-K completeness")
    return cast(dict[str, Any], summary)


def assert_scoring_summary(path: Path, *, rows: int = 8000) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    judges = summary.get("judges", {})
    corpus = summary.get("protocols", {}).get("corpus_retrieval", {})
    if (
        summary.get("status") != "measured"
        or summary.get("final_tests_used") != []
        or int(summary.get("generation_count", -1)) != rows
        or judges.get("primary_status") != "measured"
        or judges.get("shadow_status") != "measured"
        or corpus.get("status") != "measured"
    ):
        raise ValueError("prospective scoring is incomplete")
    return cast(dict[str, Any], summary)


def _copy_risk_bootstrap(
    candidates: Sequence[Any], selected: Mapping[str, Sequence[Any]], *, samples: int, seed: int
) -> dict[str, float | int]:
    by_group: dict[str, list[Any]] = defaultdict(list)
    for item in candidates:
        by_group[str(item.group_id)].append(item)
    anchors = {
        group_id: [item for item in items if item.role == "baseline"]
        for group_id, items in by_group.items()
    }
    left = {
        group_id: fmean(float(item.copy_risk) for item in items)
        for group_id, items in anchors.items()
    }
    right = {
        group_id: fmean(float(item.copy_risk) for item in items)
        for group_id, items in selected.items()
    }
    return paired_bootstrap(left, right, samples=samples, seed=seed)


def evaluate_prospective_gates(
    paired: Mapping[str, Mapping[str, Any]], gate_config: Mapping[str, Any]
) -> tuple[dict[str, Any], bool]:
    """Apply only the preregistered CI rules to hybrid-minus-anchor metrics."""
    aliases = {"format_valid_rate": "format_valid", "copy_risk_rate": "copy_risk_rate"}
    results: dict[str, Any] = {}
    for configured_name, raw_rule in gate_config.items():
        rule = cast(Mapping[str, Any], raw_rule)
        metric = aliases.get(str(configured_name), str(configured_name))
        measured = paired.get(metric)
        if measured is None:
            results[str(configured_name)] = {"status": "failed", "reason": "not_measured"}
            continue
        if rule.get("direction") == "lower":
            threshold = float(rule["maximum_upper_ci"])
            passed = float(measured["ci95_high"]) <= threshold
            criterion = f"ci95_high <= {threshold}"
        else:
            margin = float(rule["noninferiority_margin"])
            passed = float(measured["ci95_low"]) >= -margin
            criterion = f"ci95_low >= {-margin}"
        results[str(configured_name)] = {
            "status": "passed" if passed else "failed",
            "criterion": criterion,
            "bootstrap": dict(measured),
        }
    return results, bool(results) and all(item["status"] == "passed" for item in results.values())


def _write_selected(
    path: Path,
    selected: Mapping[str, Sequence[Any]],
    objectives: Mapping[str, Mapping[str, float]],
    *,
    identity_sha256: str,
    probe_authorized: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for group_id in sorted(selected):
            for rank, item in enumerate(selected[group_id]):
                row = {
                    "contract": PROSPECTIVE_CONTRACT,
                    "status": "prospective_selection_complete",
                    "selection_identity_sha256": identity_sha256,
                    "selection_rank": rank,
                    "evaluation_group_id": group_id,
                    "evaluation_id": item.evaluation_id,
                    "candidate_identity": item.identity,
                    "role": item.role,
                    "experiment_id": item.experiment_id,
                    "generated": item.text,
                    "requested_form": item.requested_form,
                    "requested_intent": item.requested_intent,
                    "natural_margin": item.natural_margin,
                    "synthetic_margin": item.metrics["pool_margin"],
                    "margin_excess": item.margin_excess,
                    "copy_risk": item.copy_risk,
                    "group_objective": dict(objectives[group_id]),
                    "promotion_eligible": False,
                    "probe_materialization_authorized": probe_authorized,
                    "final_tests_used": [],
                }
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def select_compare_prospective(
    contract_path: Path,
    *,
    cohort_manifest_path: Path,
    baseline_rows_path: Path,
    controlled_rows_path: Path,
    output_json: Path,
    output_markdown: Path,
    output_selected: Path,
    semantic_cache_dir: Path,
    semantic_device: str = "cuda",
    semantic_encoder: Any | None = None,
) -> dict[str, Any]:
    """Run the frozen selector, reserved-shadow comparison and preregistered gates."""
    payload, root, contract_sha = load_prospective_contract(contract_path)
    cohort = validate_materialized_cohort(contract_path, cohort_manifest_path)
    selector_pin = cast(Mapping[str, Any], payload["selector"])
    retrospective_path = root / str(
        cast(Mapping[str, Any], selector_pin["retrospective_contract"])["path"]
    )
    selector_contract = D01UsefulnessContract.load(retrospective_path)
    natural = _load_natural_scores(
        selector_contract.natural_scores_path,
        str(selector_contract.payload["natural_primary_scores"]["judge"]),
    )
    thresholds = cast(Mapping[str, Any], payload["copy_risk"])
    baseline = _compact_candidates(
        baseline_rows_path,
        role="baseline",
        natural_scores=natural,
        copy_thresholds=thresholds,
    )
    controlled = _compact_candidates(
        controlled_rows_path,
        role="controlled",
        natural_scores=natural,
        copy_thresholds=thresholds,
    )
    candidates = [*baseline, *controlled]
    baseline_groups = {item.group_id for item in baseline}
    if (
        baseline_groups != {item.group_id for item in controlled}
        or len(baseline_groups) != int(payload["cohort"]["selected_count"])
        or len(baseline) != 4 * len(baseline_groups)
        or len(controlled) != len(baseline)
    ):
        raise ValueError("prospective selector requires matched exact four-of-four arms")
    natural_by_group: dict[str, float] = {}
    for item in baseline:
        previous = natural_by_group.setdefault(item.group_id, item.natural_margin)
        if abs(previous - item.natural_margin) > 1e-12:
            raise ValueError("natural margin differs within a prospective group")
    margins = [natural_by_group[key] for key in sorted(natural_by_group)]
    q25, q75 = percentile(margins, 0.25), percentile(margins, 0.75)
    if q25 is None or q75 is None:
        raise RuntimeError("cannot calibrate prospective natural-margin scale")
    margin_scale = max(1e-6, q75 - q25)
    embeddings, cache = _load_or_encode(
        candidates,
        contract=selector_contract,
        cache_dir=semantic_cache_dir,
        device=semantic_device,
        encoder=semantic_encoder,
    )
    weights = cast(Mapping[str, Any], selector_pin["objective_weights"])
    selected, objectives, changed = _select_groups(
        candidates, embeddings, margin_scale=margin_scale, weights=weights
    )
    evaluation = cast(Mapping[str, Any], payload["evaluation"])
    samples, seed = int(evaluation["bootstrap_samples"]), int(evaluation["bootstrap_seed"])
    selection = _selection_report(
        candidates, selected, embeddings, bootstrap_samples=samples, bootstrap_seed=seed
    )
    paired = cast(
        dict[str, Mapping[str, Any]],
        selection["paired_group_bootstrap_selected_minus_anchor"],
    )
    paired["copy_risk_rate"] = _copy_risk_bootstrap(
        candidates, selected, samples=samples, seed=seed
    )
    gates, passed = evaluate_prospective_gates(paired, cast(Mapping[str, Any], evaluation["gates"]))
    input_identity = {
        "contract_sha256": contract_sha,
        "cohort_manifest_sha256": _file_sha256(cohort_manifest_path),
        "baseline_rows_sha256": _file_sha256(baseline_rows_path),
        "controlled_rows_sha256": _file_sha256(controlled_rows_path),
        "selector_implementation_sha256": selector_pin["implementation"]["sha256"],
        "selector_contract_sha256": selector_pin["retrospective_contract"]["sha256"],
        "final_tests_used": [],
    }
    identity_sha = _canonical_sha256(input_identity)
    _write_selected(
        output_selected,
        selected,
        objectives,
        identity_sha256=identity_sha,
        probe_authorized=passed,
    )
    report = {
        "schema_version": 1,
        "contract": PROSPECTIVE_CONTRACT,
        "status": "prospective_complete",
        "decision": "authorize_equal_budget_probe_inputs" if passed else "stop",
        "identity": {**input_identity, "identity_sha256": identity_sha},
        "cohort": {
            "group_count": len(baseline_groups),
            "selection_policy_fingerprint": cohort["selection_policy_fingerprint"],
        },
        "candidate_count": len(candidates),
        "selected_count": sum(len(items) for items in selected.values()),
        "changed_group_count": changed,
        "natural_margin_scale_iqr": margin_scale,
        "difficulty_diagnostic": _difficulty_report(candidates),
        "semantic_cache": cache,
        "selection": selection,
        "preregistered_gates": gates,
        "all_preregistered_gates_passed": passed,
        "selection_uses_shadow": False,
        "primary_bm25_improvement_may_be_construction_induced": True,
        "promotion_eligible": False,
        "probe_materialization_authorized": passed,
        "selected_rows": str(output_selected),
        "selected_rows_sha256": _file_sha256(output_selected),
        "four_point_five_b_authorized": False,
        "four_point_five_b_requires_separate_decision": True,
        "final_tests_used": [],
    }
    _atomic_json(output_json, report)
    lines = [
        "# Prospective D01b 1.5B validation",
        "",
        f"- Status: `{report['status']}`",
        f"- Decision: `{report['decision']}`",
        f"- Passage groups: `{len(baseline_groups)}`",
        f"- Controlled selected rate: `{float(selection['controlled_selected_rate']):.6f}`",
        "- Shadow used by selector: `false`",
        "- Final tests used: `[]`",
        "",
        "## Preregistered gates",
        "",
        "| Metric | Status | Difference | 95% CI |",
        "|---|---|---:|---:|",
    ]
    for metric, gate in gates.items():
        bootstrap = gate.get("bootstrap", {})
        difference = float(bootstrap.get("difference", float("nan")))
        low = float(bootstrap.get("ci95_low", float("nan")))
        high = float(bootstrap.get("ci95_high", float("nan")))
        lines.append(
            f"| {metric} | {gate['status']} | {difference:.6f} | [{low:.6f}, {high:.6f}] |"
        )
    lines.extend(
        [
            "",
            "Primary/BM25 changes may partly follow from selector construction; reserved shadow "
            "is the independent judge guardrail. This report never authorizes a final-test "
            "opening.",
        ]
    )
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
