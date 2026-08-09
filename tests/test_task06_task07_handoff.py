from __future__ import annotations

import builtins
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import doc2query.preferences.handoff as handoff_module
from doc2query.preferences.handoff import package_task07_inputs
from doc2query.training.dpo import (
    canonical_fingerprint,
    file_sha256,
    ordered_ids_fingerprint,
    validate_dpo_dataset,
)
from doc2query.utils.records import read_records as raw_read_records

HEX_A = "a" * 64
HEX_B = "b" * 64
HEX_C = "c" * 64
HEX_D = "d" * 64


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _provenance() -> dict[str, Any]:
    return {
        "dataset_id": "frozen-task06-fixture",
        "dataset_fingerprint": HEX_A,
        "selection_policy_id": "frozen-selection-v1",
        "selection_policy_fingerprint": HEX_B,
    }


def _preference(index: int, split: str) -> dict[str, Any]:
    suffix = f"{split}-{index}"
    return {
        "preference_id": f"pref-{suffix}",
        "prompt": f"Pasaż {suffix}\nZapytanie:",
        "chosen": f"Dobre pytanie {suffix}?",
        "rejected": f"Słabsze pytanie {suffix}?",
        "score_margin": 0.5,
        "chosen_candidate_id": f"chosen-{suffix}",
        "rejected_candidate_id": f"rejected-{suffix}",
        "passage_id": f"passage-{suffix}",
        "passage_cluster_id": f"cluster-{suffix}",
        "split": split,
        "provenance": _provenance(),
    }


def _continued(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "preference_id": row["preference_id"],
        "prompt": row["prompt"],
        "completion": row["chosen"],
        "candidate_id": row["chosen_candidate_id"],
        "passage_id": row["passage_id"],
        "passage_cluster_id": row["passage_cluster_id"],
        "split": row["split"],
        "provenance": row["provenance"],
    }


def _weight(row: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "preference_id": row["preference_id"],
        "split": row["split"],
        "sample_weight": 1.0 + index / 10,
        **_provenance(),
        "weight_policy_id": "frozen-weight-policy-v1",
        "weight_policy_fingerprint": HEX_C,
    }


@dataclass
class HandoffFixture:
    root: Path
    preferences_train: list[dict[str, Any]]
    preferences_dev: list[dict[str, Any]]
    continued_train: list[dict[str, Any]]
    continued_dev: list[dict[str, Any]]
    weights: list[dict[str, Any]]

    @property
    def preference_train_path(self) -> Path:
        return self.root / "source_preference_train.jsonl"

    @property
    def preference_dev_path(self) -> Path:
        return self.root / "source_preference_dev.jsonl"

    @property
    def continued_train_path(self) -> Path:
        return self.root / "source_continued_train.jsonl"

    @property
    def continued_dev_path(self) -> Path:
        return self.root / "source_continued_dev.jsonl"

    @property
    def weight_records_path(self) -> Path:
        return self.root / "weight_assignments.jsonl"

    @property
    def weight_manifest_path(self) -> Path:
        return self.root / "weight_manifest.json"

    def materialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        _write_jsonl(self.preference_train_path, self.preferences_train)
        _write_jsonl(self.preference_dev_path, self.preferences_dev)
        _write_jsonl(self.continued_train_path, self.continued_train)
        _write_jsonl(self.continued_dev_path, self.continued_dev)
        _write_jsonl(self.weight_records_path, self.weights)
        payload: dict[str, Any] = {
            "contract": "task06-score-weight-assignments-for-task07-v1",
            "records": {
                "path": self.weight_records_path.name,
                "sha256": file_sha256(self.weight_records_path),
                "record_count": len(self.weights),
            },
            **_provenance(),
            "weight_policy_id": "frozen-weight-policy-v1",
            "weight_policy_fingerprint": HEX_C,
            "ordered_preference_ids_fingerprint": ordered_ids_fingerprint(
                [row["preference_id"] for row in self.weights]
            ),
            "final_tests_used": [],
        }
        payload["artifact_fingerprint"] = canonical_fingerprint(payload)
        _write_json(self.weight_manifest_path, payload)

    def kwargs(self, output_dir: Path) -> dict[str, Path]:
        return {
            "preference_train_path": self.preference_train_path,
            "preference_dev_path": self.preference_dev_path,
            "continued_sft_train_path": self.continued_train_path,
            "continued_sft_dev_path": self.continued_dev_path,
            "weight_manifest_path": self.weight_manifest_path,
            "weight_records_path": self.weight_records_path,
            "output_dir": output_dir,
        }


@pytest.fixture
def handoff_files(tmp_path: Path) -> HandoffFixture:
    train = [_preference(0, "train"), _preference(1, "train")]
    dev = [_preference(0, "dev")]
    all_rows = train + dev
    fixture = HandoffFixture(
        root=tmp_path / "inputs",
        preferences_train=train,
        preferences_dev=dev,
        continued_train=[_continued(row) for row in train],
        continued_dev=[_continued(row) for row in dev],
        weights=[_weight(row, index) for index, row in enumerate(all_rows)],
    )
    fixture.materialize()
    return fixture


def _validation_kwargs(output: Path) -> dict[str, Path]:
    return {
        "task06_manifest_path": output / "manifest.json",
        "preference_train_path": output / "preference_train.jsonl",
        "preference_dev_path": output / "preference_dev.jsonl",
        "continued_sft_train_path": output / "continued_sft_train.jsonl",
        "continued_sft_dev_path": output / "continued_sft_dev.jsonl",
        "weighted_sft_train_path": output / "weighted_sft_train.jsonl",
        "weighted_sft_dev_path": output / "weighted_sft_dev.jsonl",
    }


def test_packaging_is_deterministic_and_matches_existing_validator(
    handoff_files: HandoffFixture, tmp_path: Path
) -> None:
    first = tmp_path / "package-one"
    second = tmp_path / "package-two"
    manifest = package_task07_inputs(**handoff_files.kwargs(first))
    second_manifest = package_task07_inputs(**handoff_files.kwargs(second))
    assert manifest.manifest_fingerprint == second_manifest.manifest_fingerprint
    assert manifest.contract == "task06-preference-data-for-task07-v1"
    assert manifest.automatic_thresholds_created is False
    assert manifest.relabeling_performed is False
    assert manifest.final_tests_used == []
    assert sorted(path.name for path in first.iterdir()) == sorted(
        path.name for path in second.iterdir()
    )
    for path in first.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes()

    dataset = validate_dpo_dataset(**_validation_kwargs(first))
    assert len(dataset.preference_train) == len(dataset.continued_sft_train) == 2
    assert len(dataset.preference_dev) == len(dataset.weighted_sft_dev) == 1
    assert [row.preference_id for row in dataset.weighted_sft_train] == [
        row.preference_id for row in dataset.preference_train
    ]
    source = handoff_files.preferences_train[0]
    packaged = dataset.preference_train[0]
    assert (packaged.prompt, packaged.chosen, packaged.rejected) == (
        source["prompt"],
        source["chosen"],
        source["rejected"],
    )
    assert (packaged.chosen_candidate_id, packaged.rejected_candidate_id) == (
        source["chosen_candidate_id"],
        source["rejected_candidate_id"],
    )


@pytest.mark.parametrize("fault", ["missing", "orphan", "duplicate"])
def test_preference_id_coverage_fails_closed(
    handoff_files: HandoffFixture, tmp_path: Path, fault: str
) -> None:
    if fault == "missing":
        handoff_files.continued_train.pop()
    elif fault == "orphan":
        handoff_files.continued_train[-1]["preference_id"] = "orphan-pref"
    else:
        duplicate = copy.deepcopy(handoff_files.preferences_train[0])
        duplicate["passage_id"] = "duplicate-passage"
        duplicate["passage_cluster_id"] = "duplicate-cluster"
        handoff_files.preferences_dev = [duplicate]
    handoff_files.materialize()
    with pytest.raises(ValueError, match=r"preference_id|coverage mismatch"):
        package_task07_inputs(**handoff_files.kwargs(tmp_path / "output"))


@pytest.mark.parametrize("fault", ["missing", "orphan", "duplicate"])
def test_weight_assignment_coverage_fails_closed(
    handoff_files: HandoffFixture, tmp_path: Path, fault: str
) -> None:
    if fault == "missing":
        handoff_files.weights.pop()
    elif fault == "orphan":
        handoff_files.weights[-1]["preference_id"] = "orphan-pref"
    else:
        handoff_files.weights.append(copy.deepcopy(handoff_files.weights[0]))
    handoff_files.materialize()
    match = "duplicate preference_id" if fault == "duplicate" else "coverage mismatch"
    with pytest.raises(ValueError, match=match):
        package_task07_inputs(**handoff_files.kwargs(tmp_path / "output"))


@pytest.mark.parametrize("weight", [0.0, -0.1, float("nan"), float("inf"), -float("inf")])
def test_weight_must_be_positive_and_finite(
    handoff_files: HandoffFixture, tmp_path: Path, weight: float
) -> None:
    handoff_files.weights[0]["sample_weight"] = weight
    handoff_files.materialize()
    with pytest.raises(ValidationError, match="sample_weight"):
        package_task07_inputs(**handoff_files.kwargs(tmp_path / "output"))


@pytest.mark.parametrize("drift", ["dataset", "selection", "weight_policy"])
def test_weight_provenance_drift_fails_closed(
    handoff_files: HandoffFixture, tmp_path: Path, drift: str
) -> None:
    if drift == "dataset":
        handoff_files.weights[0]["dataset_fingerprint"] = HEX_D
    elif drift == "selection":
        handoff_files.weights[0]["selection_policy_fingerprint"] = HEX_D
    else:
        handoff_files.weights[0]["weight_policy_fingerprint"] = HEX_D
    handoff_files.materialize()
    with pytest.raises(ValueError, match="provenance drift"):
        package_task07_inputs(**handoff_files.kwargs(tmp_path / "output"))


def test_weight_manifest_dataset_and_selection_drift_fails_closed(
    handoff_files: HandoffFixture, tmp_path: Path
) -> None:
    manifest = json.loads(handoff_files.weight_manifest_path.read_text(encoding="utf-8"))
    manifest["selection_policy_fingerprint"] = HEX_D
    manifest["artifact_fingerprint"] = canonical_fingerprint(
        {key: value for key, value in manifest.items() if key != "artifact_fingerprint"}
    )
    _write_json(handoff_files.weight_manifest_path, manifest)
    with pytest.raises(ValueError, match="weight manifest"):
        package_task07_inputs(**handoff_files.kwargs(tmp_path / "output"))


def test_weight_artifact_hash_and_order_fingerprint_are_enforced(
    handoff_files: HandoffFixture, tmp_path: Path
) -> None:
    handoff_files.weight_records_path.write_text(
        handoff_files.weight_records_path.read_text(encoding="utf-8") + " \n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sha256"):
        package_task07_inputs(**handoff_files.kwargs(tmp_path / "hash-output"))

    handoff_files.materialize()
    manifest = json.loads(handoff_files.weight_manifest_path.read_text(encoding="utf-8"))
    manifest["ordered_preference_ids_fingerprint"] = HEX_D
    manifest["artifact_fingerprint"] = canonical_fingerprint(
        {key: value for key, value in manifest.items() if key != "artifact_fingerprint"}
    )
    _write_json(handoff_files.weight_manifest_path, manifest)
    with pytest.raises(ValueError, match="ordered-ID fingerprint"):
        package_task07_inputs(**handoff_files.kwargs(tmp_path / "order-output"))


def test_continued_completion_must_equal_chosen(
    handoff_files: HandoffFixture, tmp_path: Path
) -> None:
    handoff_files.continued_train[0]["completion"] = "Inne pytanie"
    handoff_files.materialize()
    with pytest.raises(ValueError, match="differs from chosen"):
        package_task07_inputs(**handoff_files.kwargs(tmp_path / "output"))


@pytest.mark.parametrize("field", ["passage_id", "passage_cluster_id"])
def test_train_dev_passage_and_cluster_leakage_fails_closed(
    handoff_files: HandoffFixture, tmp_path: Path, field: str
) -> None:
    leaked = handoff_files.preferences_train[0][field]
    handoff_files.preferences_dev[0][field] = leaked
    handoff_files.continued_dev[0][field] = leaked
    handoff_files.materialize()
    with pytest.raises(ValueError, match="leakage"):
        package_task07_inputs(**handoff_files.kwargs(tmp_path / "output"))


def test_test_split_is_rejected_and_separate_test_artifact_is_never_read(
    handoff_files: HandoffFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    test_artifact = handoff_files.root / "preference_test.jsonl"
    test_artifact.write_text("this is deliberately not JSON\n", encoding="utf-8")
    opened: list[Path] = []

    def recording_reader(path: Path) -> Any:
        opened.append(path)
        return raw_read_records(path)

    monkeypatch.setattr(handoff_module, "read_records", recording_reader)
    package_task07_inputs(**handoff_files.kwargs(tmp_path / "valid-output"))
    assert test_artifact not in opened

    handoff_files.preferences_train[0]["split"] = "test"
    handoff_files.continued_train[0]["split"] = "test"
    handoff_files.weights[0]["split"] = "test"
    handoff_files.materialize()
    with pytest.raises((ValueError, ValidationError), match=r"test|literal"):
        package_task07_inputs(**handoff_files.kwargs(tmp_path / "rejected-output"))


def test_existing_output_is_not_overwritten(handoff_files: HandoffFixture, tmp_path: Path) -> None:
    output = tmp_path / "output"
    package_task07_inputs(**handoff_files.kwargs(output))
    snapshot = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(FileExistsError, match="already exists"):
        package_task07_inputs(**handoff_files.kwargs(output))
    assert snapshot == {path.name: path.read_bytes() for path in output.iterdir()}


def test_failure_removes_staging_and_never_imports_model_stack(
    handoff_files: HandoffFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_import = builtins.__import__
    forbidden = {"torch", "transformers", "tokenizers", "trl", "peft", "bitsandbytes"}

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".", maxsplit=1)[0] in forbidden:
            raise AssertionError(f"model or training dependency imported: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    package_task07_inputs(**handoff_files.kwargs(tmp_path / "model-free-output"))

    original_write = handoff_module._write_jsonl
    calls = 0

    def interrupted_write(path: Path, rows: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic interrupted write")
        original_write(path, rows)

    monkeypatch.setattr(handoff_module, "_write_jsonl", interrupted_write)
    output = tmp_path / "atomic-output"
    with pytest.raises(RuntimeError, match="interrupted write"):
        package_task07_inputs(**handoff_files.kwargs(output))
    assert not output.exists()
    assert list(tmp_path.glob(".atomic-output.staging-*")) == []
