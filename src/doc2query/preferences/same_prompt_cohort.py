"""Quality-blind freeze of an additional same-prompt Task 06 cohort.

The cohort is selected from frozen train IDs only, disjoint by near-duplicate cluster
from the shared 50k SFT selection and from every prior Task 06 cohort.  No quality
field, judge score, or diversity-gate result influences the selection, no model is
loaded, and the frozen execution design is read but never modified.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import yaml

from doc2query.preferences.execution_design import (
    _training_pair_ids,
    fingerprint_ids,
    sha256_file,
)
from doc2query.utils.records import read_records, write_json

CONTRACT = "task06-same-prompt-preference-expansion-v2"
DESIGN_CONTRACT = "task06-candidate-execution-design-v1"
COHORT_STATUS = "materialized_after_quality_blind_id_freeze"
IDS_STATUS = "ids_frozen_before_text_materialization"
_FORBIDDEN_SOURCE_MARKERS = ("trivia", "test", "final")


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping: {path}")
    return value


def _assert_non_test_path(path: Path) -> None:
    lowered = str(path).lower()
    if any(marker in lowered for marker in _FORBIDDEN_SOURCE_MARKERS):
        raise ValueError(f"forbidden Task 06 source path: {path}")


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_expansion_config(path: Path) -> dict[str, Any]:
    config = _load_yaml(path)
    if config.get("contract") != CONTRACT:
        raise ValueError("invalid same-prompt expansion v2 contract")
    if config.get("status") != "frozen_ready_for_cohort_freeze":
        raise ValueError("same-prompt expansion v2 config is not frozen")
    if config.get("final_tests_used") != []:
        raise ValueError("same-prompt expansion v2 config declares final-test usage")
    authorization = config.get("authorization")
    if not isinstance(authorization, dict):
        raise ValueError("same-prompt expansion v2 authorization is missing")
    if authorization.get("cohort_freeze_authorized") is not True:
        raise ValueError("cohort freeze is not authorized")
    if authorization.get("tentative_pair_build_authorized") is not False:
        raise ValueError("cohort freeze must not authorize pair building")
    if authorization.get("final_tests_used") != []:
        raise ValueError("same-prompt expansion v2 authorization declares final-test usage")
    return config


def _excluded_clusters(
    root: Path, entries: Sequence[Mapping[str, Any]]
) -> tuple[set[str], list[str]]:
    clusters: set[str] = set()
    hashes: list[str] = []
    for entry in entries:
        path = root / str(entry["path"])
        if not path.is_file():
            raise ValueError(f"missing prior Task 06 ID manifest: {path}")
        digest = sha256_file(path)
        if digest != str(entry["sha256"]):
            raise ValueError(f"prior Task 06 ID manifest drifted: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("final_tests_used") != []:
            raise ValueError(f"invalid prior Task 06 ID manifest: {path}")
        if payload.get("status") != IDS_STATUS:
            raise ValueError(f"prior Task 06 ID manifest is not a frozen ID freeze: {path}")
        records = payload.get("records")
        if not isinstance(records, list) or not records:
            raise ValueError(f"prior Task 06 ID manifest omits records: {path}")
        clusters.update(str(row["cluster_id"]) for row in records)
        hashes.append(digest)
    return clusters, hashes


def _cohort_partition(cohort: Mapping[str, Any]) -> tuple[int, int] | None:
    """Read the optional disjoint-partition selector; absent means the whole pool."""
    raw = cohort.get("partition")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("cohort.partition must be a mapping")
    unexpected = sorted(set(raw) - {"index", "count"})
    if unexpected:
        raise ValueError(f"unsupported cohort.partition fields: {unexpected}")
    index = int(raw["index"])
    count = int(raw["count"])
    if count < 2 or not 0 <= index < count:
        raise ValueError("cohort.partition requires count >= 2 and 0 <= index < count")
    return index, count


def _in_partition(cluster_id: str, *, seed: int, partition: tuple[int, int] | None) -> bool:
    """Assign each near-duplicate cluster to exactly one partition, deterministically."""
    if partition is None:
        return True
    index, count = partition
    digest = hashlib.sha256(f"{seed}:partition:{cluster_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % count == index


def freeze_same_prompt_expansion_cohort(config_path: Path, output_dir: Path) -> dict[str, Any]:
    """Freeze a new cluster-disjoint same-prompt cohort without generating anything."""
    config = _load_expansion_config(config_path)
    root = config_path.resolve().parents[2]
    design_policy = cast(Mapping[str, Any], config["design"])
    design_path = root / str(design_policy["config"])
    if sha256_file(design_path) != str(design_policy["config_sha256"]):
        raise ValueError("frozen Task 06 execution design drifted")
    design = _load_yaml(design_path)
    if design.get("contract") != DESIGN_CONTRACT or design.get("final_tests_used") != []:
        raise ValueError("invalid Task 06 execution design")

    cohort = cast(Mapping[str, Any], config["cohort"])
    data = cast(Mapping[str, Any], design["data"])
    pinned = cast(Mapping[str, Any], data["sha256"])
    pairs_path = root / str(data["source_train_pairs"])
    dedup_path = root / str(data["dedup_map"])
    split_manifest_path = root / str(data["split_manifest"])
    source_path = root / str(cohort["source_records"])
    for path in (pairs_path, dedup_path, split_manifest_path, source_path):
        _assert_non_test_path(path.relative_to(root))
        if not path.is_file():
            raise FileNotFoundError(path)
    for label, path in (
        ("source_train_pairs", pairs_path),
        ("dedup_map", dedup_path),
        ("split_manifest", split_manifest_path),
    ):
        if sha256_file(path) != pinned.get(label):
            raise ValueError(f"pinned {label} drifted")
    if sha256_file(source_path) != str(cohort["source_records_sha256"]):
        raise ValueError("pinned canonical train records drifted")
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    if split_manifest.get("positive_canonical_leakage") != 0:
        raise ValueError("frozen split manifest does not prove zero positive leakage")

    id_manifest_path = output_dir / "cohort.ids.json"
    records_path = output_dir / "cohort.records.jsonl"
    manifest_path = output_dir / "cohort.manifest.json"
    if manifest_path.is_file() and records_path.is_file() and id_manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(existing, dict):
            raise ValueError("Task 06 cohort manifest must be a mapping")
        return existing
    if records_path.exists() or manifest_path.exists():
        raise FileExistsError("incomplete same-prompt expansion v2 cohort state exists")

    identities: list[dict[str, Any]] = [
        {
            "pair_id": str(row["pair_id"]),
            "example_id": str(row["example_id"]),
            "doc_id": str(row["doc_id"]),
            "negative_doc_ids": [str(value) for value in row.get("negative_doc_ids", [])],
            "split": str(row["split"]),
        }
        for row in read_records(pairs_path)
    ]
    if any(row["split"] != "train" for row in identities):
        raise ValueError("Task 06 source must contain train pairs only")
    training = cast(Mapping[str, Any], design["adapter_training_exclusion"])
    trained_pairs = _training_pair_ids(
        identities, seed=int(training["selection_seed"]), maximum=int(training["max_pairs"])
    )
    wanted_docs = {row["doc_id"] for row in identities}
    doc_to_cluster = {
        str(row["doc_id"]): str(row["cluster_id"])
        for row in read_records(dedup_path)
        if str(row.get("doc_id", "")) in wanted_docs
    }
    missing_clusters = wanted_docs - set(doc_to_cluster)
    if missing_clusters:
        raise ValueError(f"dedup map misses {len(missing_clusters)} train documents")
    trained_clusters = {
        doc_to_cluster[row["doc_id"]] for row in identities if row["pair_id"] in trained_pairs
    }
    if cohort.get("exclude_adapter_training_clusters") is not True:
        raise ValueError("adapter training clusters must be excluded")
    prior_clusters, prior_hashes = _excluded_clusters(
        root, cast(Sequence[Mapping[str, Any]], cohort["exclude_prior_cohort_ids"])
    )

    passage_count = int(cohort["passage_count"])
    minimum_negatives = int(cohort["min_hard_negatives"])
    seed = int(cohort["selection_seed"])
    partition = _cohort_partition(cohort)
    eligible = [
        {**row, "cluster_id": doc_to_cluster[row["doc_id"]]}
        for row in identities
        if len(row["negative_doc_ids"]) >= minimum_negatives
        and doc_to_cluster[row["doc_id"]] not in trained_clusters
        and doc_to_cluster[row["doc_id"]] not in prior_clusters
        and _in_partition(doc_to_cluster[row["doc_id"]], seed=seed, partition=partition)
    ]
    eligible.sort(
        key=lambda row: hashlib.sha256(
            f"{seed}:{row['cluster_id']}:{row['pair_id']}".encode()
        ).digest()
    )
    selected: list[dict[str, Any]] = []
    seen_clusters: set[str] = set()
    seen_examples: set[str] = set()
    for row in eligible:
        if row["cluster_id"] in seen_clusters or row["example_id"] in seen_examples:
            continue
        selected.append(row)
        seen_clusters.add(row["cluster_id"])
        seen_examples.add(row["example_id"])
        if len(selected) == passage_count:
            break
    if len(selected) != passage_count:
        raise RuntimeError("insufficient legal cluster-unique same-prompt expansion records")
    if seen_clusters & prior_clusters or seen_clusters & trained_clusters:
        raise RuntimeError("selected cohort overlaps an excluded cluster set")

    id_payload: dict[str, Any] = {
        "schema_version": 1,
        "contract": CONTRACT,
        "stage": "expansion_v2",
        "status": IDS_STATUS,
        "selection_seed": seed,
        "selection_policy": str(cohort["selection"]),
        "records": [
            {key: row[key] for key in ("pair_id", "example_id", "doc_id", "cluster_id")}
            for row in selected
        ],
        "quality_fields_used": [],
        "eligible_pair_count": len(eligible),
        "excluded_adapter_training_cluster_count": len(trained_clusters),
        "excluded_prior_cluster_count": len(prior_clusters),
        "excluded_prior_ids_sha256": prior_hashes,
        "final_tests_used": [],
    }
    if partition is not None:
        id_payload["partition"] = {"index": partition[0], "count": partition[1]}
    id_payload["fingerprint"] = _canonical_sha256(id_payload)
    if id_manifest_path.is_file():
        if json.loads(id_manifest_path.read_text(encoding="utf-8")) != id_payload:
            raise ValueError("previously frozen same-prompt expansion v2 IDs drifted")
    else:
        write_json(id_manifest_path, id_payload)

    selected_example_ids = {str(row["example_id"]) for row in selected}
    by_example = {
        str(row["example_id"]): row
        for row in read_records(source_path)
        if str(row["example_id"]) in selected_example_ids
    }
    if set(by_example) != selected_example_ids:
        raise ValueError("frozen same-prompt expansion IDs are missing from canonical records")
    materialized: list[dict[str, Any]] = []
    for item in selected:
        source = by_example[item["example_id"]]
        positives = [
            value for value in source["positives"] if str(value["doc_id"]) == item["doc_id"]
        ]
        negative_order = {value: index for index, value in enumerate(item["negative_doc_ids"])}
        negatives = sorted(
            (value for value in source["hard_negatives"] if str(value["doc_id"]) in negative_order),
            key=lambda value: negative_order[str(value["doc_id"])],
        )
        if len(positives) != 1 or len(negatives) < minimum_negatives:
            raise ValueError(f"cannot materialize selected pair {item['pair_id']}")
        materialized.append(
            {
                "example_id": item["pair_id"],
                "source_example_id": item["example_id"],
                "pair_id": item["pair_id"],
                "cluster_id": item["cluster_id"],
                "query": str(source["query"]),
                "positives": positives,
                "hard_negatives": negatives,
                "metadata": {
                    **dict(source.get("metadata", {})),
                    "task06_same_prompt_expansion_v2": True,
                },
                "split": "train",
            }
        )
    _write_jsonl_atomic(records_path, materialized)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "contract": CONTRACT,
        "stage": "expansion_v2",
        "status": COHORT_STATUS,
        "record_count": len(materialized),
        "cluster_count": len(seen_clusters),
        "eligible_pair_count": len(eligible),
        "ids_path": str(id_manifest_path),
        "ids_sha256": sha256_file(id_manifest_path),
        "ids_fingerprint": str(id_payload["fingerprint"]),
        "records_path": str(records_path),
        "records_sha256": sha256_file(records_path),
        "cluster_ids_sha256": fingerprint_ids(sorted(seen_clusters)),
        "config_sha256": sha256_file(config_path),
        "design_sha256": sha256_file(design_path),
        "source_train_pairs_sha256": sha256_file(pairs_path),
        "source_records_sha256": sha256_file(source_path),
        "quality_fields_used_for_selection": [],
        "excluded_adapter_training_cluster_count": len(trained_clusters),
        "excluded_prior_cluster_count": len(prior_clusters),
        "excluded_prior_ids_sha256": prior_hashes,
        "prior_cluster_overlap_count": 0,
        "generation_started": False,
        "scoring_started": False,
        "diversity_gate_applied": False,
        "pairs_built": False,
        "model_loading_performed": False,
        "final_tests_used": [],
    }
    if partition is not None:
        manifest["partition"] = {"index": partition[0], "count": partition[1]}
    write_json(manifest_path, manifest)
    return manifest
