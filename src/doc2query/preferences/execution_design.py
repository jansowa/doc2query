"""ID-only audit and fail-closed execution design for prospective Task 06 work."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from doc2query.preferences.hybrid_handoff import preflight_hybrid_handoff
from doc2query.preferences.llm_audit import load_llm_audit_config
from doc2query.utils.records import read_records, write_json

CONTRACT = "task06-candidate-execution-design-v1"
AUDIT_CONTRACT = "task06-id-only-cohort-audit-v1"
FORBIDDEN_SOURCE_MARKERS = ("trivia", "test", "final")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_ids(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(set(values)):
        digest.update(value.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping: {path}")
    return value


def _assert_non_test_path(path: Path) -> None:
    lowered = str(path).lower()
    if any(marker in lowered for marker in FORBIDDEN_SOURCE_MARKERS):
        raise ValueError(f"forbidden Task 06 source path: {path}")


def _training_pair_ids(rows: Sequence[Mapping[str, Any]], *, seed: int, maximum: int) -> set[str]:
    """Reproduce the quality-blind hash cap used by SFT after canonical validation."""
    ranked = sorted(
        (str(row["pair_id"]) for row in rows),
        key=lambda pair_id: hashlib.sha256(f"{seed}:{pair_id}".encode()).digest(),
    )
    return set(ranked[:maximum])


def run_id_only_cohort_audit(config_path: Path, output_path: Path) -> dict[str, Any]:
    """Count eligible train passage clusters without emitting IDs or reading test data."""
    config = _load_yaml(config_path)
    if (
        config.get("contract") != CONTRACT
        or config.get("status") != "owner_approved_design_pending_operator_command"
        or config.get("final_tests_used") != []
    ):
        raise ValueError("invalid Task 06 execution-design contract")
    root = config_path.resolve().parents[2]
    data = config.get("data")
    if not isinstance(data, dict):
        raise ValueError("missing data policy")
    source = root / str(data["source_train_pairs"])
    dedup = root / str(data["dedup_map"])
    split_manifest = root / str(data["split_manifest"])
    for path in (source, dedup, split_manifest):
        _assert_non_test_path(path.relative_to(root))
        if not path.is_file():
            raise FileNotFoundError(path)
    expected = data.get("sha256", {})
    if not isinstance(expected, dict):
        raise ValueError("data.sha256 must be a mapping")
    pinned_paths = (
        ("source_train_pairs", source),
        ("dedup_map", dedup),
        ("split_manifest", split_manifest),
    )
    for label, path in pinned_paths:
        if sha256_file(path) != expected.get(label):
            raise ValueError(f"pinned {label} drifted")

    split = json.loads(split_manifest.read_text(encoding="utf-8"))
    if split.get("positive_canonical_leakage") != 0:
        raise ValueError("frozen split manifest does not prove zero positive leakage")
    rows = list(read_records(source))
    allowed_fields = {"pair_id", "doc_id", "split"}
    identities = [{field: str(row.get(field, "")) for field in allowed_fields} for row in rows]
    if any(
        row["split"] != "train" or not row["pair_id"] or not row["doc_id"] for row in identities
    ):
        raise ValueError("Task 06 source must contain identified train pairs only")

    train_policy = config.get("adapter_training_exclusion")
    if not isinstance(train_policy, dict):
        raise ValueError("missing adapter training exclusion")
    trained_pairs = _training_pair_ids(
        identities,
        seed=int(train_policy["selection_seed"]),
        maximum=int(train_policy["max_pairs"]),
    )
    doc_ids = {row["doc_id"] for row in identities}
    trained_docs = {row["doc_id"] for row in identities if row["pair_id"] in trained_pairs}
    doc_to_cluster: dict[str, str] = {}
    for row in read_records(dedup):
        doc_id = str(row.get("doc_id", ""))
        if doc_id not in doc_ids:
            continue
        cluster_id = str(row.get("cluster_id", ""))
        if not cluster_id:
            raise ValueError(f"missing cluster for train document {doc_id}")
        previous = doc_to_cluster.setdefault(doc_id, cluster_id)
        if previous != cluster_id:
            raise ValueError(f"cluster identity drift for {doc_id}")
    missing = doc_ids - set(doc_to_cluster)
    if missing:
        raise ValueError(f"dedup map misses {len(missing)} train documents")
    trained_clusters = {doc_to_cluster[doc_id] for doc_id in trained_docs}
    eligible_docs = {doc_id for doc_id in doc_ids if doc_to_cluster[doc_id] not in trained_clusters}
    eligible_clusters = {doc_to_cluster[doc_id] for doc_id in eligible_docs}

    cohort_options = config.get("owner_decisions", {}).get("cohort_size", {}).get("options", {})
    if not isinstance(cohort_options, dict):
        raise ValueError("cohort options must be a mapping")
    option_capacity = {
        name: {
            "requested_passages": int(value),
            "available": len(eligible_docs) >= int(value),
        }
        for name, value in cohort_options.items()
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "contract": AUDIT_CONTRACT,
        "status": "audited_ids_only_pending_owner_decisions",
        "source": {
            "split": "train",
            "pair_count": len(identities),
            "unique_passage_count": len(doc_ids),
            "unique_cluster_count": len(set(doc_to_cluster.values())),
            "pair_ids_sha256": fingerprint_ids(row["pair_id"] for row in identities),
            "passage_ids_sha256": fingerprint_ids(doc_ids),
            "cluster_ids_sha256": fingerprint_ids(doc_to_cluster.values()),
            "artifact_sha256": sha256_file(source),
        },
        "exclusions": {
            "adapter_training_pair_count": len(trained_pairs),
            "adapter_training_passage_count": len(trained_docs),
            "adapter_training_cluster_count": len(trained_clusters),
            "adapter_training_pair_ids_sha256": fingerprint_ids(trained_pairs),
            "adapter_training_passage_ids_sha256": fingerprint_ids(trained_docs),
            "adapter_training_cluster_ids_sha256": fingerprint_ids(trained_clusters),
            "development_and_test": "excluded_by_train-only source and frozen cluster-safe split",
            "split_positive_canonical_leakage": 0,
            "task04_task05_active_cohorts": (
                "development cohorts excluded by train-only source; "
                "adapter train clusters excluded above"
            ),
            "corpus_background_membership": "not treated as active cohort membership",
            "triviaqa": "forbidden and not read",
        },
        "eligible": {
            "passage_count": len(eligible_docs),
            "cluster_count": len(eligible_clusters),
            "passage_ids_sha256": fingerprint_ids(eligible_docs),
            "cluster_ids_sha256": fingerprint_ids(eligible_clusters),
        },
        "option_capacity": option_capacity,
        "quality_fields_read": [],
        "raw_ids_emitted": False,
        "test_artifacts_read": False,
        "triviaqa_artifacts_read": False,
        "generation_started": False,
        "scoring_started": False,
        "final_tests_used": [],
    }
    if output_path.exists():
        raise FileExistsError(output_path)
    write_json(output_path, result)
    return result


def preflight_execution_design(config_path: Path, audit_path: Path) -> dict[str, Any]:
    """Validate the frozen design; remain blocked until every owner choice is explicit."""
    config = _load_yaml(config_path)
    if (
        config.get("contract") != CONTRACT
        or config.get("status") != "owner_approved_design_pending_operator_command"
        or config.get("final_tests_used") != []
    ):
        raise ValueError("invalid Task 06 execution-design contract")
    root = config_path.resolve().parents[2]
    handoff_path = root / str(config.get("handoff_config", ""))
    handoff = preflight_hybrid_handoff(handoff_path)
    if handoff.get("status") != "verified_ready_for_task06_execution_design_not_generation":
        raise ValueError("owner-approved Hybrid handoff is not verified")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("contract") != AUDIT_CONTRACT
        or audit.get("quality_fields_read") != []
        or audit.get("test_artifacts_read") is not False
        or audit.get("triviaqa_artifacts_read") is not False
        or audit.get("final_tests_used") != []
    ):
        raise ValueError("ID-only cohort audit is not safe")
    matrix = config.get("generation_matrix")
    if not isinstance(matrix, list) or len(matrix) != 8:
        raise ValueError("generation matrix must contain exactly eight requests")
    roles = [str(row.get("role")) for row in matrix if isinstance(row, dict)]
    if roles.count("w06_anchor") != 4 or roles.count("d01_controlled") != 4:
        raise ValueError("matrix must preserve four W06 and four D01 candidates")
    seeds = {int(row["seed"]) for row in matrix if isinstance(row, dict)}
    if len(seeds) < 2:
        raise ValueError("generation matrix requires at least two seeds")
    if any(
        row.get("uses_shadow_for_selection") is not False for row in matrix if isinstance(row, dict)
    ):
        raise ValueError("shadow judge must be reserved from selection")
    selector = config.get("safe_anchor_selector")
    if not isinstance(selector, dict) or selector.get("mutable") is not False:
        raise ValueError("safe-anchor selector must be immutable")
    scoring = config.get("scoring")
    if not isinstance(scoring, dict):
        raise ValueError("missing scoring policy")
    if scoring.get("shadow_role") != "independent_control_not_selection_signal":
        raise ValueError("shadow role drifted")
    if int(scoring.get("max_batch_size", 0)) > 8:
        raise ValueError("scoring batch exceeds machine safety cap")
    pins: list[tuple[Mapping[str, Any], str, str]] = []
    pins.extend(
        (selector, path_key, hash_key)
        for path_key, hash_key in (
            ("implementation", "implementation_sha256"),
            ("contract", "contract_sha256"),
        )
    )
    primary = scoring.get("primary")
    shadow = scoring.get("shadow")
    corpus = scoring.get("corpus_retrieval")
    if not all(isinstance(value, dict) for value in (primary, shadow, corpus)):
        raise ValueError("scoring components must be pinned mappings")
    assert isinstance(primary, dict) and isinstance(shadow, dict) and isinstance(corpus, dict)
    pins.extend(
        [
            (primary, "config", "config_sha256"),
            (shadow, "config", "config_sha256"),
            (corpus, "manifest", "manifest_sha256"),
        ]
    )
    for pin, path_key, hash_key in pins:
        path = root / str(pin[path_key])
        if not path.is_file() or sha256_file(path) != pin[hash_key]:
            raise ValueError(f"pinned execution-design input drifted: {pin[path_key]}")
    llm_audit = config.get("llm_pair_audit")
    if not isinstance(llm_audit, dict):
        raise ValueError("missing owner-approved dual-LLM audit")
    llm_config_path = root / str(llm_audit.get("config", ""))
    if not llm_config_path.is_file() or sha256_file(llm_config_path) != llm_audit.get(
        "config_sha256"
    ):
        raise ValueError("pinned dual-LLM audit config drifted")
    llm_config = load_llm_audit_config(llm_config_path)
    if (
        llm_audit.get("models") != ["openai/gpt-oss-120b", "qwen/qwen3.6-27b"]
        or llm_audit.get("ratings_per_pair") != 2
        or llm_audit.get("pair_count") != 500
        or llm_audit.get("human_evidence_claimed") is not False
        or llm_audit.get("safe_anchor_selection_signal") is not False
        or llm_audit.get("minimum_seconds_between_requests_global") != 4.0
        or llm_audit.get("parallel_groq_requests") is not False
        or llm_config.get("human_evidence_claimed") is not False
    ):
        raise ValueError("dual-LLM audit boundary drifted")
    decisions = config.get("owner_decisions")
    if not isinstance(decisions, dict):
        raise ValueError("missing owner decisions")
    unresolved = sorted(
        name
        for name, value in decisions.items()
        if not isinstance(value, dict) or value.get("selected") is None
    )
    execution = config.get("authorization")
    if not isinstance(execution, dict):
        raise ValueError("missing authorization boundary")
    authorized = all(
        execution.get(name) is True
        for name in ("generation_authorized", "scoring_authorized", "operator_command_authorized")
    )
    if unresolved:
        status = "blocked_pending_owner_decisions"
    elif authorized:
        status = "verified_ready_for_explicit_operator_command"
    else:
        status = "verified_design_pending_explicit_operator_command"
    return {
        "schema_version": 1,
        "contract": CONTRACT,
        "status": status,
        "config_sha256": sha256_file(config_path),
        "audit_sha256": sha256_file(audit_path),
        "handoff_config_sha256": sha256_file(handoff_path),
        "llm_audit_config_sha256": sha256_file(llm_config_path),
        "unresolved_owner_decisions": unresolved,
        "generation_matrix_size": len(matrix),
        "generation_seeds": sorted(seeds),
        "model_loading_performed": False,
        "generation_started": False,
        "scoring_started": False,
        "selection_started": False,
        "task07_training_started": False,
        "task09_started": False,
        "final_tests_used": [],
    }
