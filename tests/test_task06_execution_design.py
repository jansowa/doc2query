from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml
from conftest import require_local_artifacts

from doc2query.preferences import execution_design
from doc2query.preferences.execution_design import (
    fingerprint_ids,
    preflight_execution_design,
    run_id_only_cohort_audit,
    sha256_file,
)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_id_only_audit_excludes_entire_adapter_training_clusters(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    config_path = root / "configs" / "preferences" / "design.yaml"
    source = root / "data" / "train.jsonl"
    dedup = root / "data" / "dedup.jsonl"
    split = root / "data" / "split.json"
    rows = [
        {"pair_id": "p1", "doc_id": "d1", "split": "train", "query": "must-not-read"},
        {"pair_id": "p2", "doc_id": "d2", "split": "train", "query": "must-not-read"},
        {"pair_id": "p3", "doc_id": "d3", "split": "train", "query": "must-not-read"},
    ]
    _write_jsonl(source, rows)
    _write_jsonl(
        dedup,
        [
            {"doc_id": "d1", "cluster_id": "c1"},
            {"doc_id": "d2", "cluster_id": "c1"},
            {"doc_id": "d3", "cluster_id": "c3"},
        ],
    )
    split.parent.mkdir(parents=True, exist_ok=True)
    split.write_text('{"positive_canonical_leakage": 0}\n', encoding="utf-8")
    config = {
        "contract": execution_design.CONTRACT,
        "status": "owner_approved_design_pending_operator_command",
        "final_tests_used": [],
        "data": {
            "source_train_pairs": "data/train.jsonl",
            "dedup_map": "data/dedup.jsonl",
            "split_manifest": "data/split.json",
            "sha256": {
                "source_train_pairs": sha256_file(source),
                "dedup_map": sha256_file(dedup),
                "split_manifest": sha256_file(split),
            },
        },
        "adapter_training_exclusion": {"selection_seed": 42, "max_pairs": 1},
        "owner_decisions": {"cohort_size": {"options": {"one": 1}}},
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = root / "reports" / "audit.json"

    result = run_id_only_cohort_audit(config_path, output)

    assert result["quality_fields_read"] == []
    assert result["raw_ids_emitted"] is False
    assert result["test_artifacts_read"] is False
    assert result["exclusions"]["adapter_training_cluster_count"] == 1
    # One selected pair excludes both documents in its near-duplicate cluster.
    assert result["eligible"]["passage_count"] in {1, 2}
    assert result["eligible"]["cluster_count"] == 1
    assert "d1" not in json.dumps(result)
    assert result["eligible"]["passage_ids_sha256"] == fingerprint_ids(
        {"d3"} if result["eligible"]["passage_count"] == 1 else {"d1", "d2"}
    )


def test_real_execution_design_preflight_is_fail_closed() -> None:
    require_local_artifacts()
    root = Path(__file__).resolve().parents[1]
    result = preflight_execution_design(
        root / "configs/preferences/task06_candidate_execution_design_v1.yaml",
        root / "reports/measurements/task06/candidate_execution_design_v1/id_only_audit.json",
    )

    assert result["status"] == "verified_design_pending_explicit_operator_command"
    assert result["unresolved_owner_decisions"] == []
    assert result["generation_matrix_size"] == 8
    assert result["generation_seeds"] == [6101, 6102]
    assert result["model_loading_performed"] is False
    assert result["final_tests_used"] == []
