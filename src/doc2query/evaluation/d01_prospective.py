"""Prospective, dev-only validation of the preregistered D01b 1.5B selector."""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
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
from doc2query.evaluation.embedder_probe import ProbeRecipe
from doc2query.evaluation.retrieval import percentile
from doc2query.evaluation.statistical_contract import build_budget_manifest
from doc2query.generation.deduplicate import query_key
from doc2query.utils.records import read_records, write_json

PROSPECTIVE_CONTRACTS = frozenset(
    {
        "task05-d01b-prospective-1.5b-v1",
        "task05-d01b-prospective-1.5b-v2",
        "task05-d01b-prospective-1.5b-v3",
        "task05-d01b-scale-interaction-4.5b-pilot-v1",
    }
)
PREREGISTERED_COHORT_CONTRACTS = frozenset(
    {
        "task05-d01b-prospective-cohort-v1",
        "task05-d01b-prospective-cohort-v2",
        "task05-d01b-prospective-cohort-v3",
        "task05-d01b-prospective-cohort-v4-scale-pilot",
    }
)
PROSPECTIVE_PROBE_INPUT_CONTRACT = "task05-d01b-prospective-probe-inputs-v1"


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


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
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
        or payload.get("contract") not in PROSPECTIVE_CONTRACTS
        or payload.get("status")
        not in {"preregistered_before_generation", "preregistered_before_pilot"}
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
    expected_manifest_sha = cohort.get("manifest_sha256")
    if expected_manifest_sha is not None and _file_sha256(preregistered) != str(
        expected_manifest_sha
    ):
        raise ValueError("prospective cohort preregistration fingerprint mismatch")
    prereg_payload = json.loads(preregistered.read_text(encoding="utf-8"))
    if (
        prereg_payload.get("contract") not in PREREGISTERED_COHORT_CONTRACTS
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
    decoding = cast(Mapping[str, Any], arms["decoding"])
    observed_arms: dict[str, Any] = {}
    for role in ("baseline", "controlled"):
        arm = cast(Mapping[str, Any], arms[role])
        config_path = root / str(arm["generation_config"])
        if _file_sha256(config_path) != str(arm["generation_config_sha256"]):
            raise ValueError(f"{role} generation config drifted")
        config = load_config(config_path)
        if config.run.experiment_id != arm["generation_experiment_id"]:
            raise ValueError(f"{role} prospective experiment identity drifted")
        if arm.get("preserve_duplicate_slots") is not None and bool(
            arm["preserve_duplicate_slots"]
        ) != bool(config.generation.preserve_duplicate_slots):
            raise ValueError(f"{role} prospective duplicate policy drifted")
        expected_decoding = {
            "seed": config.run.seed,
            "do_sample": config.generation.do_sample,
            "temperature": config.generation.temperature,
            "top_p": config.generation.top_p,
            "max_new_tokens": config.generation.max_new_tokens,
            "max_attempts_per_query": config.generation.max_attempts_per_query,
            "queries_per_arm_per_passage": config.generation.target_query_count,
        }
        if any(decoding.get(field) != value for field, value in expected_decoding.items()):
            raise ValueError(f"{role} prospective decoding contract drifted")
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
        "contract": payload["contract"],
        "status": "verified",
        "contract_sha256": contract_sha,
        "selector_commit": "2164822",
        "arms": observed_arms,
        "cohort_selected_count": int(cohort["selected_count"]),
        "final_tests_used": [],
    }


def _selection_rows(
    records: Sequence[dict[str, Any]],
    *,
    excluded: set[str],
    minimum_hard_negatives: int,
    seed: int,
) -> list[tuple[str, str, int, dict[str, Any]]]:
    rows: list[tuple[str, str, int, dict[str, Any]]] = []
    for index, record in enumerate(records):
        example_id = str(record["example_id"])
        negatives = record.get("hard_negatives")
        if (
            example_id not in excluded
            and isinstance(negatives, list)
            and len(negatives) >= minimum_hard_negatives
        ):
            digest = hashlib.sha256(f"{seed}:{example_id}".encode()).hexdigest()
            rows.append((digest, example_id, index, record))
    rows.sort(key=lambda item: (item[0], item[1]))
    return rows


def _id_list_sha256(ids: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()


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
    subset_excluded = {str(row["id"]) for row in read_records(excluded_path)}
    minimum = int(cohort["minimum_hard_negatives"])
    seed = int(cohort["selection_seed"])
    prior_excluded: set[str] = set()
    prior_identities: list[dict[str, Any]] = []
    for raw_prior in cohort.get("prior_cohort_exclusions", []):
        prior = cast(Mapping[str, Any], raw_prior)
        manifest_path = root / str(prior["manifest"])
        if _file_sha256(manifest_path) != str(prior["manifest_sha256"]):
            raise ValueError("prospective prior cohort manifest fingerprint mismatch")
        prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        prior_identity = cast(Mapping[str, Any], prior_manifest["cohort_identity"])
        prior_minimum = int(prior_identity["minimum_hard_negatives"])
        prior_rows = _selection_rows(
            records,
            excluded=subset_excluded | prior_excluded,
            minimum_hard_negatives=prior_minimum,
            seed=int(prior["selection_seed"]),
        )[: int(prior["selected_count"])]
        prior_ids = [item[1] for item in prior_rows]
        prior_sha = _id_list_sha256(prior_ids)
        if prior_sha != str(prior["selected_id_list_sha256"]):
            raise ValueError("prospective prior cohort reconstruction drifted")
        prior_excluded.update(prior_ids)
        prior_identities.append(
            {
                "manifest_sha256": str(prior["manifest_sha256"]),
                "selected_id_list_sha256": prior_sha,
                "selected_count": len(prior_ids),
            }
        )
    excluded = subset_excluded | prior_excluded
    available = [record for record in records if str(record["example_id"]) not in excluded]
    eligible_source_order = [
        record
        for record in available
        if isinstance(record.get("hard_negatives"), list)
        and len(record["hard_negatives"]) >= minimum
    ]
    eligible = _selection_rows(
        records,
        excluded=excluded,
        minimum_hard_negatives=minimum,
        seed=seed,
    )
    selected_rows = eligible[: int(cohort["selected_count"])]
    selected = [item[3] for item in selected_rows]
    source_indices = [item[2] for item in selected_rows]
    example_ids = [item[1] for item in selected_rows]
    id_sha = _id_list_sha256(example_ids)
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
        "available_after_exclusions": len(available),
        "eligible_count": len(eligible),
        "remaining_source_order_id_list_sha256": _id_list_sha256(
            [str(record["example_id"]) for record in available]
        ),
        "remaining_eligible_source_order_id_list_sha256": _id_list_sha256(
            [str(record["example_id"]) for record in eligible_source_order]
        ),
        "selected_id_list_sha256": id_sha,
        "selected_group_ids_sha256": hashlib.sha256("\n".join(group_ids).encode()).hexdigest(),
        "selected_source_indices_sha256": _canonical_sha256(source_indices),
        "selected_metadata_records_sha256": metadata_sha,
        "selected_source_records_sha256": record_digest.hexdigest(),
        "intersection_with_excluded_subset": len(set(example_ids) & subset_excluded),
        "intersection_with_prior_cohorts": len(set(example_ids) & prior_excluded),
    }
    if not prior_identities:
        checks = {
            field: checks[field]
            for field in (
                "eligible_count",
                "selected_id_list_sha256",
                "selected_metadata_records_sha256",
                "selected_source_records_sha256",
                "intersection_with_excluded_subset",
            )
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
    if prior_identities:
        expected.update(
            {
                "available_after_exclusions": int(cohort["available_after_exclusions"]),
                "remaining_source_order_id_list_sha256": str(
                    preregistered_identity["remaining_source_order_id_list_sha256"]
                ),
                "remaining_eligible_source_order_id_list_sha256": str(
                    preregistered_identity["remaining_eligible_source_order_id_list_sha256"]
                ),
                "selected_group_ids_sha256": str(
                    preregistered_identity["selected_group_ids_sha256"]
                ),
                "selected_source_indices_sha256": str(
                    preregistered_identity["selected_source_indices_sha256"]
                ),
                "intersection_with_prior_cohorts": int(cohort["intersection_with_prior_cohorts"]),
            }
        )
    for field, value in expected.items():
        if checks[field] != value:
            raise ValueError(f"prospective cohort {field} drifted")
    selection_policy: dict[str, Any] = {
        "policy": f"sha256_unseen_dev_min_hn_v{len(prior_identities) + 1}",
        "selection_seed": seed,
        "minimum_hard_negatives": minimum,
        "selected_count": len(selected),
        "selected_id_list_sha256": id_sha,
        "selected_source_records_sha256": record_digest.hexdigest(),
        "quality_metrics_used": [],
        "final_tests_used": [],
    }
    if prior_identities:
        selection_policy["prior_cohort_exclusions"] = prior_identities
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


def _duplicate_rate(items: Sequence[Any]) -> float:
    if not items:
        raise ValueError("cannot measure duplicate rate for an empty candidate group")
    return 1.0 - len({query_key(str(item.text)) for item in items}) / len(items)


def _duplicate_rate_bootstrap(
    candidates: Sequence[Any], selected: Mapping[str, Sequence[Any]], *, samples: int, seed: int
) -> dict[str, float | int]:
    by_group: dict[str, list[Any]] = defaultdict(list)
    for item in candidates:
        by_group[str(item.group_id)].append(item)
    anchors = {
        group_id: [item for item in items if item.role == "baseline"]
        for group_id, items in by_group.items()
    }
    left = {group_id: _duplicate_rate(items) for group_id, items in anchors.items()}
    right = {group_id: _duplicate_rate(items) for group_id, items in selected.items()}
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
    contract: str,
    identity_sha256: str,
    probe_authorized: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for group_id in sorted(selected):
            for rank, item in enumerate(selected[group_id]):
                row = {
                    "contract": contract,
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
    paired["duplicate_rate"] = _duplicate_rate_bootstrap(
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
        contract=str(payload["contract"]),
        identity_sha256=identity_sha,
        probe_authorized=passed,
    )
    report = {
        "schema_version": 1,
        "contract": payload["contract"],
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
    heading = (
        "# Prospective D01b 4.5B scale-interaction pilot"
        if str(payload["contract"]) == "task05-d01b-scale-interaction-4.5b-pilot-v1"
        else "# Prospective D01b 1.5B validation"
    )
    lines = [
        heading,
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


def materialize_prospective_probe_inputs(
    contract_path: Path,
    *,
    report_path: Path,
    selected_rows_path: Path,
    baseline_rows_path: Path,
    controlled_rows_path: Path,
    probe_recipe_path: Path,
    baseline_output: Path,
    hybrid_output: Path,
    manifest_output: Path,
) -> dict[str, Any]:
    """Materialize the authorized equal-budget pair inputs without starting training."""
    contract, _root, contract_sha = load_prospective_contract(contract_path)
    contract_name = str(contract["contract"])
    supported = {
        "task05-d01b-prospective-1.5b-v3",
        "task05-d01b-scale-interaction-4.5b-pilot-v1",
    }
    if contract_name not in supported:
        raise ValueError("probe inputs require an authorized prospective D01b contract")
    scale_pilot = contract_name == "task05-d01b-scale-interaction-4.5b-pilot-v1"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    required_report = {
        "contract": contract["contract"],
        "status": "prospective_complete",
        "decision": "authorize_equal_budget_probe_inputs",
        "all_preregistered_gates_passed": True,
        "probe_materialization_authorized": True,
        "final_tests_used": [],
    }
    if any(report.get(key) != value for key, value in required_report.items()):
        raise ValueError("prospective report does not authorize probe input materialization")
    if report.get("four_point_five_b_authorized") is not False:
        raise ValueError("prospective report must keep full 4.5B authorization closed")
    identity = cast(Mapping[str, Any], report.get("identity", {}))
    if identity.get("contract_sha256") != contract_sha:
        raise ValueError("prospective report contract identity drifted")
    if report.get("selected_rows_sha256") != _file_sha256(selected_rows_path):
        raise ValueError("prospective selected rows fingerprint drifted")
    expected_scored_hashes = {
        "baseline": identity.get("baseline_rows_sha256"),
        "controlled": identity.get("controlled_rows_sha256"),
    }
    scored_paths = {"baseline": baseline_rows_path, "controlled": controlled_rows_path}
    if any(
        _file_sha256(scored_paths[role]) != expected_scored_hashes[role] for role in scored_paths
    ):
        raise ValueError("prospective scored rows fingerprint drifted")

    recipe_raw = yaml.safe_load(probe_recipe_path.read_text(encoding="utf-8"))
    if not isinstance(recipe_raw, Mapping):
        raise ValueError("probe recipe must be a mapping")
    recipe = ProbeRecipe.from_dict(recipe_raw)
    if (
        recipe.negative_recipe.strategy != "hn0_filter"
        or recipe.negative_recipe.false_negative_policy != "drop"
    ):
        raise ValueError("prospective probe inputs require frozen HN0+filter/drop")
    calibration = recipe.negative_recipe.load_calibration()
    if calibration is None:
        raise ValueError("prospective probe inputs require the pinned dev calibration")
    primary_pin = cast(Mapping[str, Any], cast(Mapping[str, Any], contract["scoring"])["primary"])
    if calibration.primary_judge_name != primary_pin.get(
        "name_or_path"
    ) or calibration.primary_judge_revision != primary_pin.get("revision"):
        raise ValueError("probe calibration does not match prospective primary judge")

    scored: dict[tuple[str, str], dict[str, Any]] = {}
    by_role: dict[str, list[dict[str, Any]]] = {}
    for role, path in scored_paths.items():
        rows = list(read_records(path))
        if len(rows) != int(report["selected_count"]):
            raise ValueError(f"prospective {role} arm is not exact equal budget")
        by_role[role] = rows
        for row in rows:
            evaluation_id = str(row.get("evaluation_id", ""))
            key = (role, evaluation_id)
            if not evaluation_id or key in scored:
                raise ValueError("prospective scoring identities are not unique")
            if row.get("final_tests_used") != []:
                raise ValueError("probe materialization forbids final-test provenance")
            scored[key] = row

    selected = list(read_records(selected_rows_path))
    if len(selected) != int(report["selected_count"]):
        raise ValueError("prospective selected arm is not equal budget")
    selected_ids: list[tuple[str, str]] = []
    expected_selection_identity = identity.get("identity_sha256")
    for row in selected:
        evaluation_id = str(row.get("evaluation_id", ""))
        role = str(row.get("role", ""))
        key = (role, evaluation_id)
        if (
            key not in scored
            or row.get("selection_identity_sha256") != expected_selection_identity
            or row.get("probe_materialization_authorized") is not True
            or row.get("final_tests_used") != []
        ):
            raise ValueError("selected row is not authorized by the prospective report")
        selected_ids.append(key)
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("prospective selected identities are not unique")

    def materialize(
        rows: Sequence[Mapping[str, Any]], generator_id: str
    ) -> tuple[list[dict[str, Any]], set[str]]:
        output: list[dict[str, Any]] = []
        groups: dict[str, int] = defaultdict(int)
        ineligible_groups: set[str] = set()
        for row in rows:
            negatives = row.get("hard_negatives")
            scores = row.get("primary_negative_scores")
            positive = row.get("positive")
            if (
                not isinstance(negatives, list)
                or not isinstance(scores, list)
                or len(negatives) != len(scores)
                or not isinstance(positive, Mapping)
            ):
                raise ValueError("scored row lacks aligned positive/negative evidence")
            retained = [
                negative
                for negative, score in zip(negatives, scores, strict=True)
                if float(score) < calibration.threshold
            ]
            group_id = str(row["evaluation_group_id"])
            if not retained:
                ineligible_groups.add(group_id)
            groups[group_id] += 1
            pair_id = f"{row.get('experiment_id', generator_id)}:{row['evaluation_id']}"
            output.append(
                {
                    "pair_id": pair_id,
                    "example_id": pair_id,
                    "query": str(row["generated"]),
                    "generated": str(row["generated"]),
                    "mode": "deterministic",
                    "candidate_index": 0,
                    "positive": dict(positive),
                    "positives": [dict(positive)],
                    "hard_negatives": retained,
                    "source_example_id": str(row["example_id"]),
                    "source_passage_id": str(row["doc_id"]),
                    "generator_experiment_id": generator_id,
                    "_materialization_group_id": group_id,
                }
            )
        expected_groups = int(cast(Mapping[str, Any], report["cohort"])["group_count"])
        if len(groups) != expected_groups or set(groups.values()) != {4}:
            raise ValueError("probe inputs require exactly four queries per passage")
        return output, ineligible_groups

    baseline_materialized, baseline_ineligible = materialize(
        by_role["baseline"], str(cast(Mapping[str, Any], contract["arms"])["baseline"]["id"])
    )
    hybrid_materialized, hybrid_ineligible = materialize(
        [scored[key] for key in selected_ids],
        (
            "D01B-SCALE-PILOT-HYBRID-4.5B-S42"
            if scale_pilot
            else "D01B-PROSPECTIVE-V3-HYBRID-1.5B-S42"
        ),
    )
    ineligible_groups = baseline_ineligible | hybrid_ineligible
    group_to_passage: dict[str, str] = {}
    for row in [*baseline_materialized, *hybrid_materialized]:
        group_id = str(row["_materialization_group_id"])
        passage_id = str(row["source_passage_id"])
        previous = group_to_passage.setdefault(group_id, passage_id)
        if previous != passage_id:
            raise ValueError("prospective group maps to multiple positive passages")
    seen_passages: set[str] = set()
    duplicate_passage_groups: set[str] = set()
    for group_id in sorted(group_to_passage):
        if group_id in ineligible_groups:
            continue
        passage_id = group_to_passage[group_id]
        if passage_id in seen_passages:
            duplicate_passage_groups.add(group_id)
        else:
            seen_passages.add(passage_id)
    excluded_groups = ineligible_groups | duplicate_passage_groups
    target_groups = int(
        cast(Mapping[str, Any], contract.get("probe", {})).get(
            "input_passages", len(group_to_passage) - len(excluded_groups)
        )
    )
    eligible_group_ids = [
        group_id for group_id in sorted(group_to_passage) if group_id not in excluded_groups
    ]
    if len(eligible_group_ids) < target_groups:
        raise ValueError("prospective probe cohort is smaller than the frozen passage budget")
    selected_group_ids = set(eligible_group_ids[:target_groups])
    excluded_groups |= set(group_to_passage) - selected_group_ids
    baseline_materialized = [
        row
        for row in baseline_materialized
        if str(row["_materialization_group_id"]) not in excluded_groups
    ]
    hybrid_materialized = [
        row
        for row in hybrid_materialized
        if str(row["_materialization_group_id"]) not in excluded_groups
    ]
    for rows in (baseline_materialized, hybrid_materialized):
        rows.sort(key=lambda row: (str(row["_materialization_group_id"]), str(row["pair_id"])))
        for row in rows:
            del row["_materialization_group_id"]
    if len(baseline_materialized) != len(hybrid_materialized):
        raise ValueError("prospective probe arms have unequal pair budgets")
    if not baseline_materialized:
        raise ValueError("dual-arm HN0+filter/drop intersection is empty")
    _atomic_jsonl(baseline_output, baseline_materialized)
    _atomic_jsonl(hybrid_output, hybrid_materialized)
    pair_count = len(baseline_materialized)
    source_group_count = int(cast(Mapping[str, Any], report["cohort"])["group_count"])
    group_count = source_group_count - len(excluded_groups)
    if pair_count != group_count * 4:
        raise ValueError("common eligible probe cohort is not uniform exact K=4")
    for rows in (baseline_materialized, hybrid_materialized):
        passage_counts: dict[str, int] = defaultdict(int)
        for row in rows:
            passage_counts[str(row["source_passage_id"])] += 1
        if len(passage_counts) != group_count or set(passage_counts.values()) != {4}:
            raise ValueError("probe input has duplicate passages or non-uniform K")
    probe_section = cast(Mapping[str, Any], contract.get("probe", {}))
    budget_steps = int(probe_section.get("max_steps", recipe.max_steps))
    budget_batch = int(probe_section.get("batch_size", recipe.batch_size))
    budget_length = int(probe_section.get("max_length", recipe.max_length))
    training_budget = build_budget_manifest(
        token_count=(
            budget_steps * budget_batch * budget_length * (2 + recipe.negatives_per_example)
        ),
        pair_count=pair_count,
        unique_passage_count=group_count,
        queries_per_passage=4,
    )
    manifest = {
        "schema_version": 1,
        "contract": PROSPECTIVE_PROBE_INPUT_CONTRACT,
        "status": "materialized_and_cpu_validated",
        "source_contract": contract["contract"],
        "source_contract_sha256": contract_sha,
        "source_report": str(report_path),
        "source_report_sha256": _file_sha256(report_path),
        "selection_identity_sha256": expected_selection_identity,
        "selection_policy": "frozen_best_four_of_eight_hybrid_vs_all_four_observed_anchor",
        "negative_recipe": recipe.negative_recipe.manifest(calibration),
        "probe_recipe": asdict(recipe),
        "probe_recipe_fingerprint": recipe.fingerprint,
        "comparison_budget": training_budget,
        "common_eligibility": {
            "policy": "dual_arm_group_intersection_hn0_filter_drop",
            "source_group_count": source_group_count,
            "eligible_group_count": group_count,
            "dropped_group_count": len(excluded_groups),
            "hn_filter_dropped_group_count": len(ineligible_groups),
            "duplicate_passage_dropped_group_count": len(duplicate_passage_groups),
            "baseline_ineligible_group_count": len(baseline_ineligible),
            "hybrid_ineligible_group_count": len(hybrid_ineligible),
            "dropped_group_ids_sha256": _canonical_sha256(sorted(excluded_groups)),
        },
        "arms": {
            "baseline_w05": {
                "path": str(baseline_output),
                "sha256": _file_sha256(baseline_output),
                "pair_count": pair_count,
                "unique_passage_count": group_count,
                "queries_per_passage": 4,
            },
            "selected_hybrid": {
                "path": str(hybrid_output),
                "sha256": _file_sha256(hybrid_output),
                "pair_count": pair_count,
                "unique_passage_count": group_count,
                "queries_per_passage": 4,
            },
        },
        "training_started": False,
        "training_authorized": scale_pilot,
        "four_point_five_b_authorized": False,
        "four_point_five_b_full_authorized": False,
        "final_tests_used": [],
    }
    _atomic_json(manifest_output, manifest)
    return manifest
