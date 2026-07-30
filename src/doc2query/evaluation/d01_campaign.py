"""Artifact audit and technical exact-K recovery for the post-D01 campaign."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from doc2query.config import load_config
from doc2query.evaluation.corpus import sha256_file
from doc2query.evaluation.d01_pipeline import (
    D01_GENERATION_CONTRACT,
    _artifact_fingerprint,
    _canonical_sha256,
    _controls,
    _file_sha256,
    _positive,
    evaluation_group_ids,
)
from doc2query.evaluation.datasets import evaluation_fingerprint, load_frozen_records
from doc2query.generation.deduplicate import query_key
from doc2query.utils.records import read_durable_jsonl_prefix, read_records, write_json

D01_AUDIT_CONTRACT = "task05-d01-artifact-audit-v2"
D01_RECOVERY_CONTRACT = "task05-d01-common-exact-k-v1"


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _require_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _arm_paths(raw: Mapping[str, Any]) -> dict[str, Path]:
    required = ("sft_summary", "adapter", "generation_config", "generations")
    missing = [field for field in required if not raw.get(field)]
    if missing:
        raise ValueError(f"D01 audit arm lacks paths: {missing}")
    generations = Path(str(raw["generations"]))
    return {
        "sft_summary": Path(str(raw["sft_summary"])),
        "adapter": Path(str(raw["adapter"])),
        "generation_config": Path(str(raw["generation_config"])),
        "generations": generations,
        "summary": Path(str(raw.get("generation_summary", f"{generations}.summary.json"))),
        "identity": Path(str(raw.get("generation_identity", f"{generations}.identity.json"))),
        "journal": Path(str(raw.get("generation_journal", f"{generations}.journal.jsonl"))),
    }


def _audit_arm(
    raw: Mapping[str, Any],
    *,
    records: Sequence[Mapping[str, Any]],
    frozen_manifest: Path,
    subset: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    arm_id = str(raw["id"])
    paths = _arm_paths(raw)
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(f"{arm_id}: missing {path}")
    sft = _require_mapping(paths["sft_summary"])
    summary = _require_mapping(paths["summary"])
    identity_file = _require_mapping(paths["identity"])
    identity = summary.get("identity")
    if not isinstance(identity, dict) or identity != identity_file:
        raise ValueError(f"{arm_id}: generation identity/summary mismatch")
    if summary.get("contract") != D01_GENERATION_CONTRACT or summary.get("status") != "measured":
        raise ValueError(f"{arm_id}: generation summary is not measured D01")
    if summary.get("final_tests_used") != [] or identity.get("final_tests_used") != []:
        raise ValueError(f"{arm_id}: final-test provenance is forbidden")
    config = load_config(paths["generation_config"])
    if config.run.experiment_id != summary.get("experiment_id"):
        raise ValueError(f"{arm_id}: generation config experiment mismatch")
    if _canonical_sha256(config.model_dump(mode="json")) != identity.get("resolved_config_sha256"):
        raise ValueError(f"{arm_id}: resolved generation config drift")
    if _file_sha256(paths["generation_config"]) != identity.get("config_file_sha256"):
        raise ValueError(f"{arm_id}: generation config file drift")
    cohort = identity.get("cohort", {})
    frozen_fingerprint = evaluation_fingerprint(frozen_manifest, subset)
    expected_ids = evaluation_group_ids(records)
    if cohort.get("subset") != subset or cohort.get("fingerprint") != frozen_fingerprint:
        raise ValueError(f"{arm_id}: frozen subset/fingerprint mismatch")
    if int(cohort.get("group_count", -1)) != len(records):
        raise ValueError(f"{arm_id}: frozen passage count mismatch")
    expected_ids_sha = hashlib.sha256("\n".join(expected_ids).encode()).hexdigest()
    if cohort.get("group_ids_sha256") != expected_ids_sha:
        raise ValueError(f"{arm_id}: frozen evaluation order mismatch")
    if sft.get("experiment_id") != str(raw["training_experiment_id"]):
        raise ValueError(f"{arm_id}: SFT experiment mismatch")
    if sft.get("adapter_path") != str(paths["adapter"]):
        raise ValueError(f"{arm_id}: SFT adapter path mismatch")
    if sft.get("model") != {
        "architecture": config.model.architecture,
        "name_or_path": config.model.name_or_path,
        "revision": config.model.revision,
        "trust_remote_code": config.model.trust_remote_code,
    }:
        raise ValueError(f"{arm_id}: SFT/generation model provenance mismatch")
    adapter_sha = _artifact_fingerprint(paths["adapter"])
    if identity.get("adapter", {}).get("artifact_sha256") != adapter_sha:
        raise ValueError(f"{arm_id}: adapter fingerprint mismatch")

    groups = read_durable_jsonl_prefix(paths["journal"])
    final_rows = list(read_records(paths["generations"]))
    if len(groups) != len(records):
        raise ValueError(f"{arm_id}: journal does not cover the full frozen cohort")
    if [str(group.get("evaluation_group_id")) for group in groups] != expected_ids:
        raise ValueError(f"{arm_id}: journal evaluation IDs/order mismatch")
    flattened = [query for group in groups for query in group.get("queries", [])]
    if flattened != final_rows:
        raise ValueError(f"{arm_id}: final JSONL differs from the durable journal")
    evaluation_ids = [str(row.get("evaluation_id", "")) for row in final_rows]
    if not all(evaluation_ids) or len(evaluation_ids) != len(set(evaluation_ids)):
        raise ValueError(f"{arm_id}: missing or duplicate evaluation IDs")

    complete_ids: list[str] = []
    exhausted_ids: list[str] = []
    observed = Counter[str]()
    for index, (record, group) in enumerate(zip(records, groups, strict=True)):
        group_id = expected_ids[index]
        queries = group.get("queries")
        stats = group.get("stats")
        if not isinstance(queries, list) or not isinstance(stats, Mapping):
            raise ValueError(f"{arm_id}: malformed journal group {group_id}")
        positive = _positive(record)
        controls = _controls(config, str(positive["text"]))
        next_control = 0
        seen: set[str] = set()
        for accepted_index, row in enumerate(queries):
            if row.get("evaluation_id") != f"{group_id}::candidate::{accepted_index}":
                raise ValueError(f"{arm_id}: evaluation ID mismatch in {group_id}")
            if row.get("example_id") != str(record["example_id"]):
                raise ValueError(f"{arm_id}: source query mismatch in {group_id}")
            if row.get("reference") != str(record["query"]) or row.get("positive") != positive:
                raise ValueError(
                    f"{arm_id}: passage/query differs from frozen record in {group_id}"
                )
            if row.get("hard_negatives") != record["hard_negatives"]:
                raise ValueError(f"{arm_id}: negatives differ from frozen record in {group_id}")
            if row.get("final_tests_used") != []:
                raise ValueError(f"{arm_id}: row has final-test provenance")
            key = query_key(str(row.get("generated", "")))
            if not key or key in seen:
                raise ValueError(f"{arm_id}: accepted output is empty/duplicate in {group_id}")
            seen.add(key)
            control_payload = row.get("control")
            while next_control < len(controls):
                expected_control = controls[next_control]
                expected_payload = (
                    expected_control.model_dump(mode="json")
                    if expected_control is not None
                    else None
                )
                if control_payload == expected_payload:
                    break
                next_control += 1
            if next_control == len(controls):
                raise ValueError(f"{arm_id}: control order mismatch in {group_id}")
            attempt = int(row.get("attempt", 0))
            ceiling = config.generation.max_attempts_per_query
            expected_seed = config.run.seed + index * 1000 + next_control * ceiling + attempt - 1
            if attempt not in range(1, ceiling + 1) or int(row.get("seed", -1)) != expected_seed:
                raise ValueError(f"{arm_id}: seed/attempt contract mismatch in {group_id}")
            next_control += 1
        is_complete = len(queries) == config.generation.target_query_count and not bool(
            stats.get("exhausted")
        )
        (complete_ids if is_complete else exhausted_ids).append(group_id)
        for field in ("attempts", "duplicate_outputs", "invalid_outputs"):
            observed[field] += int(stats.get(field, 0))
        observed["exhausted_groups"] += int(bool(stats.get("exhausted")))
    if len(final_rows) != int(summary.get("generation_count", -1)):
        raise ValueError(f"{arm_id}: generation count mismatch")
    for field in ("attempts", "duplicate_outputs", "invalid_outputs", "exhausted_groups"):
        if observed[field] != int(summary.get(field, -1)):
            raise ValueError(f"{arm_id}: summary {field} differs from journal")
    return (
        {
            "id": arm_id,
            "status": "verified_technical_artifacts",
            "sft_experiment_id": sft["experiment_id"],
            "sft_global_step": sft.get("global_step"),
            "dataset_fingerprint": sft.get("dataset_fingerprint"),
            "adapter_sha256": adapter_sha,
            "generation_identity_sha256": identity["identity_sha256"],
            "frozen_cohort_fingerprint": frozen_fingerprint,
            "source_passage_count": len(records),
            "generation_count": len(final_rows),
            "complete_group_count": len(complete_ids),
            "exhausted_group_count": len(exhausted_ids),
            "exhausted_group_ids": exhausted_ids,
            "observed_stats": dict(observed),
            "target_queries_per_passage": config.generation.target_query_count,
            "max_new_tokens": config.generation.max_new_tokens,
            "max_attempts_per_query": config.generation.max_attempts_per_query,
            "seed_contract": identity["seed_contract"],
            "final_tests_used": [],
            "quality_interpretation": "not_performed",
        },
        groups,
    )


def audit_d01_artifacts(
    *,
    frozen_manifest: Path,
    subset: str,
    arms: Sequence[Mapping[str, Any]],
    output_json: Path,
    output_markdown: Path,
) -> dict[str, Any]:
    """Verify completed D01 training/generation artifacts without scoring quality."""
    records = load_frozen_records(frozen_manifest, subset)
    reports = []
    for arm in arms:
        report, _groups = _audit_arm(
            arm, records=records, frozen_manifest=frozen_manifest, subset=subset
        )
        reports.append(report)
    result = {
        "schema_version": 1,
        "contract": D01_AUDIT_CONTRACT,
        "status": "verified",
        "scope": "technical_artifact_integrity_only",
        "frozen_manifest": str(frozen_manifest),
        "frozen_subset": subset,
        "frozen_cohort_fingerprint": evaluation_fingerprint(frozen_manifest, subset),
        "source_passage_count": len(records),
        "arms": reports,
        "final_tests_used": [],
        "quality_results": "not_measured",
    }
    write_json(output_json, result)
    lines = [
        "# Audyt techniczny artefaktów D01",
        "",
        "Nie wykonano ani nie zinterpretowano pomiarów jakościowych.",
        "",
        f"- Status: `{result['status']}`",
        f"- Frozen subset: `{subset}`",
        f"- Passage: `{len(records)}`",
        "- Final tests used: `[]`",
        "",
        "| Ramię | Adapter SHA-256 | Query | Complete | Exhausted | Invalid | Duplicate |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for arm in reports:
        stats = arm["observed_stats"]
        lines.append(
            f"| {arm['id']} | `{arm['adapter_sha256']}` | {arm['generation_count']} | "
            f"{arm['complete_group_count']} | {arm['exhausted_group_count']} | "
            f"{stats['invalid_outputs']} | {stats['duplicate_outputs']} |"
        )
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def prepare_common_exact_k_cohort(
    *,
    frozen_manifest: Path,
    subset: str,
    arms: Sequence[Mapping[str, Any]],
    output_dir: Path,
    target_k: int = 4,
) -> dict[str, Any]:
    """Freeze a quality-blind exact-K intersection and copy D01 rows into new artifacts."""
    if target_k != 4:
        raise ValueError("the post-D01 recovery contract requires exact K=4")
    records = load_frozen_records(frozen_manifest, subset)
    ordered_ids = evaluation_group_ids(records)
    audited: list[tuple[Mapping[str, Any], dict[str, Any], list[dict[str, Any]]]] = []
    for arm in arms:
        report, groups = _audit_arm(
            arm, records=records, frozen_manifest=frozen_manifest, subset=subset
        )
        audited.append((arm, report, groups))
    budget_contracts = {
        (
            report["target_queries_per_passage"],
            report["max_new_tokens"],
            report["max_attempts_per_query"],
            report["seed_contract"].get("base_seed"),
            report["seed_contract"].get("group_stride"),
            report["seed_contract"].get("attempt_stride"),
        )
        for _arm, report, _groups in audited
    }
    if budget_contracts != {(target_k, 64, 3, 42, 1000, 1)}:
        raise ValueError("all recovery arms must share the pinned K/retry/token/seed contract")
    complete_sets = [
        {
            str(group["evaluation_group_id"])
            for group in groups
            if len(group["queries"]) == target_k and not bool(group["stats"]["exhausted"])
        }
        for _arm, _report, groups in audited
    ]
    common = set.intersection(*complete_sets)
    selected_ids = [group_id for group_id in ordered_ids if group_id in common]
    if not selected_ids:
        raise ValueError("common exact-K cohort is empty")
    excluded = [group_id for group_id in ordered_ids if group_id not in common]
    selected_sha = hashlib.sha256("\n".join(selected_ids).encode()).hexdigest()
    source_positions = {group_id: index for index, group_id in enumerate(ordered_ids)}
    source_indices_sha = _canonical_sha256(
        [source_positions[group_id] for group_id in selected_ids]
    )
    selection_payload = {
        "policy": "common_exact_k_cohort",
        "selection_basis": "technical_completeness_only",
        "quality_metrics_used": [],
        "target_k": target_k,
        "source_frozen_cohort_fingerprint": evaluation_fingerprint(frozen_manifest, subset),
        "selected_group_ids_sha256": selected_sha,
        "selected_group_count": len(selected_ids),
    }
    selection_fingerprint = _canonical_sha256(selection_payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "common_exact_k_cohort.json"
    recovered: dict[str, Any] = {}
    for raw, audit, groups in audited:
        arm_id = str(raw["id"])
        source_summary = _require_mapping(_arm_paths(raw)["summary"])
        selected_groups = [group for group in groups if group["evaluation_group_id"] in common]
        copied_groups = deepcopy(selected_groups)
        identity = deepcopy(source_summary["identity"])
        identity["cohort"] = {
            **identity["cohort"],
            "group_count": len(selected_ids),
            "group_ids_sha256": selected_sha,
            "selection_policy": selection_payload,
            "selection_policy_fingerprint": selection_fingerprint,
            "source_indices_sha256": source_indices_sha,
        }
        identity["seed_contract"] = {
            **identity["seed_contract"],
            "index_basis": "original_frozen_subset_position",
        }
        identity.pop("identity_sha256", None)
        identity["identity_sha256"] = _canonical_sha256(identity)
        flattened: list[dict[str, Any]] = []
        selected_stats = Counter[str]()
        for group in copied_groups:
            for row in group["queries"]:
                row["source_generation_identity_sha256"] = row["generation_identity_sha256"]
                row["generation_identity_sha256"] = identity["identity_sha256"]
                row["comparison_selection_policy_fingerprint"] = selection_fingerprint
                flattened.append(row)
            for field in ("attempts", "duplicate_outputs", "invalid_outputs"):
                selected_stats[field] += int(group["stats"][field])
        arm_dir = output_dir / "recovered" / arm_id
        generations_path = arm_dir / "generations.exact_k4.jsonl"
        journal_path = generations_path.with_suffix(generations_path.suffix + ".journal.jsonl")
        _atomic_jsonl(generations_path, flattened)
        _atomic_jsonl(journal_path, copied_groups)
        summary = {
            **source_summary,
            "identity": identity,
            "source_passage_count": len(selected_ids),
            "generation_count": len(flattened),
            "attempts": selected_stats["attempts"],
            "invalid_outputs": selected_stats["invalid_outputs"],
            "duplicate_outputs": selected_stats["duplicate_outputs"],
            "exhausted_groups": 0,
            "effective_candidate_count_mean": float(target_k),
            "journal_path": str(journal_path),
            "output_path": str(generations_path),
            "recovery": {
                "contract": D01_RECOVERY_CONTRACT,
                "source_generations": str(_arm_paths(raw)["generations"]),
                "source_generations_sha256": _file_sha256(_arm_paths(raw)["generations"]),
                "source_summary_sha256": _file_sha256(_arm_paths(raw)["summary"]),
                "source_observed_stats": audit["observed_stats"],
                "selection_policy": selection_payload,
                "selection_policy_fingerprint": selection_fingerprint,
                "original_artifacts_preserved": True,
            },
            "final_tests_used": [],
        }
        write_json(
            generations_path.with_suffix(generations_path.suffix + ".identity.json"), identity
        )
        write_json(generations_path.with_suffix(generations_path.suffix + ".summary.json"), summary)
        recovered[arm_id] = {
            "generations": str(generations_path),
            "generation_summary": str(
                generations_path.with_suffix(generations_path.suffix + ".summary.json")
            ),
            "generation_identity_sha256": identity["identity_sha256"],
            "generation_count": len(flattened),
        }
    manifest = {
        "schema_version": 1,
        "contract": D01_RECOVERY_CONTRACT,
        "status": "materialized",
        "frozen_manifest": str(frozen_manifest),
        "frozen_subset": subset,
        "source_frozen_cohort_fingerprint": evaluation_fingerprint(frozen_manifest, subset),
        "source_group_count": len(ordered_ids),
        "target_queries_per_passage": target_k,
        "selected_group_count": len(selected_ids),
        "selected_group_ids": selected_ids,
        "selected_group_ids_sha256": selected_sha,
        "excluded_group_count": len(excluded),
        "excluded_groups": [
            {
                "evaluation_group_id": group_id,
                "reason": "not_exact_k_in_at_least_one_d01_arm",
                "incomplete_arms": [
                    str(audited[index][0]["id"])
                    for index, complete in enumerate(complete_sets)
                    if group_id not in complete
                ],
            }
            for group_id in excluded
        ],
        "selection_policy": selection_payload,
        "selection_policy_fingerprint": selection_fingerprint,
        "generation_budget": {
            "passage_count": len(selected_ids),
            "queries_per_passage": target_k,
            "pair_count": len(selected_ids) * target_k,
            "max_new_tokens": 64,
            "max_attempts_per_query": 3,
            "completion_token_ceiling": len(selected_ids) * target_k * 64 * 3,
        },
        "recovered_arms": recovered,
        "quality_metrics_used_for_selection": [],
        "original_artifacts_preserved": True,
        "final_tests_used": [],
    }
    write_json(manifest_path, manifest)
    return manifest


def load_common_cohort(
    records: Sequence[dict[str, Any]], cohort_manifest: Path
) -> tuple[list[dict[str, Any]], list[int], dict[str, Any]]:
    manifest = _require_mapping(cohort_manifest)
    if manifest.get("contract") != D01_RECOVERY_CONTRACT or manifest.get("final_tests_used") != []:
        raise ValueError("invalid common exact-K cohort manifest")
    expected = evaluation_group_ids(records)
    positions = {group_id: index for index, group_id in enumerate(expected)}
    selected_ids = manifest.get("selected_group_ids")
    if not isinstance(selected_ids, list) or not selected_ids:
        raise ValueError("common cohort has no selected group IDs")
    if any(str(group_id) not in positions for group_id in selected_ids):
        raise ValueError("common cohort contains a group outside frozen dev")
    selected = [records[positions[str(group_id)]] for group_id in selected_ids]
    source_indices = [positions[str(group_id)] for group_id in selected_ids]
    if evaluation_group_ids(selected) != [str(value) for value in selected_ids]:
        raise ValueError("common cohort order differs from its manifest")
    return selected, source_indices, manifest


def validate_baseline_provenance(
    *, config_path: Path, adapter_path: Path, training_manifest_path: Path
) -> dict[str, Any]:
    config = load_config(config_path)
    training = _require_mapping(training_manifest_path)
    resolved = training.get("config")
    if not isinstance(resolved, Mapping):
        raise ValueError("baseline training manifest lacks resolved config")
    trained_run = resolved.get("run", {})
    trained_model = resolved.get("model", {})
    trained_recipe = resolved.get("training", {})
    if "W06" in config.run.experiment_id:
        if trained_run.get("experiment_id") != "W06-4.5B-INSTRUCT-50K-8GB-BS8-L512":
            raise ValueError("matched W06 must use the real BS8 training run")
        if (
            int(config.training.per_device_train_batch_size) != 8
            or int(trained_recipe.get("per_device_train_batch_size", -1)) != 8
        ):
            raise ValueError("matched W06 config/training provenance is not BS8")
    if (
        trained_model.get("name_or_path") != config.model.name_or_path
        or trained_model.get("revision") != config.model.revision
    ):
        raise ValueError("baseline model/revision differs from training provenance")
    if not adapter_path.is_dir() or not (adapter_path / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(adapter_path)
    return {
        "status": "verified",
        "generation_experiment_id": config.run.experiment_id,
        "training_experiment_id": trained_run.get("experiment_id"),
        "training_batch_size": trained_recipe.get("per_device_train_batch_size"),
        "adapter_path": str(adapter_path),
        "adapter_sha256": _artifact_fingerprint(adapter_path),
        "model": trained_model,
        "final_tests_used": [],
    }


def validate_corpus_index(index_dir: Path, *, expected_fingerprint: str) -> dict[str, Any]:
    manifest_path = index_dir / "manifest.json"
    manifest = _require_mapping(manifest_path)
    if manifest.get("protocol") != "corpus_retrieval" or manifest.get("backend") != "bm25_sqlite":
        raise ValueError("post-D01 requires the frozen corpus_retrieval BM25 index")
    if manifest.get("index_fingerprint") != expected_fingerprint:
        raise ValueError("corpus index fingerprint differs from pinned probe provenance")
    database = index_dir / str(manifest["database_file"])
    if not database.is_file() or sha256_file(database) != manifest.get("database_sha256"):
        raise ValueError("corpus index database integrity mismatch")
    documents = Path(str(manifest["documents_path"]))
    if not documents.is_file() or sha256_file(documents) != manifest.get("documents_sha256"):
        raise ValueError("corpus documents integrity mismatch")
    if int(manifest.get("candidate_count", 0)) < 20:
        raise ValueError("corpus index is too small for round-trip@20")
    return {
        "status": "verified",
        "manifest_sha256": _file_sha256(manifest_path),
        "index_fingerprint": manifest["index_fingerprint"],
        "candidate_count": manifest["candidate_count"],
        "database_sha256": manifest["database_sha256"],
        "documents_sha256": manifest["documents_sha256"],
        "protocol": manifest["protocol"],
        "final_tests_used": [],
    }
