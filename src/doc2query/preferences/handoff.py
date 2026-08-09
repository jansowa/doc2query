"""Fail-closed, model-free packaging of frozen Task 06 data for Task 07."""

from __future__ import annotations

import json
import math
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, TypeVar

from pydantic import Field, field_validator

from doc2query.schemas import StrictModel
from doc2query.training.dpo import (
    SHA256_PATTERN,
    ContinuedSFTRecord,
    FileArtifact,
    PreferenceRecord,
    ScoreWeightedContinuedSFTRecord,
    Task06PreferenceManifest,
    Task06SelectionProvenance,
    canonical_fingerprint,
    file_sha256,
    ordered_ids_fingerprint,
    validate_dpo_dataset,
)
from doc2query.utils.records import JsonlWriter, read_records, write_json

_RecordT = TypeVar("_RecordT", PreferenceRecord, ContinuedSFTRecord)


class WeightAssignment(StrictModel):
    """A frozen external weight assignment; this module never computes it."""

    preference_id: str = Field(min_length=1)
    split: Literal["train", "dev"]
    sample_weight: float = Field(gt=0.0)
    dataset_id: str = Field(min_length=1)
    dataset_fingerprint: str = Field(pattern=SHA256_PATTERN)
    selection_policy_id: str = Field(min_length=1)
    selection_policy_fingerprint: str = Field(pattern=SHA256_PATTERN)
    weight_policy_id: str = Field(min_length=1)
    weight_policy_fingerprint: str = Field(pattern=SHA256_PATTERN)

    @field_validator("sample_weight")
    @classmethod
    def weight_is_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("sample_weight must be finite")
        return value


class WeightAssignmentManifest(StrictModel):
    contract: Literal["task06-score-weight-assignments-for-task07-v1"]
    records: FileArtifact
    dataset_id: str = Field(min_length=1)
    dataset_fingerprint: str = Field(pattern=SHA256_PATTERN)
    selection_policy_id: str = Field(min_length=1)
    selection_policy_fingerprint: str = Field(pattern=SHA256_PATTERN)
    weight_policy_id: str = Field(min_length=1)
    weight_policy_fingerprint: str = Field(pattern=SHA256_PATTERN)
    ordered_preference_ids_fingerprint: str = Field(pattern=SHA256_PATTERN)
    final_tests_used: list[str] = Field(max_length=0)
    artifact_fingerprint: str = Field(pattern=SHA256_PATTERN)


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _artifact_fingerprint(payload: Mapping[str, Any], field: str) -> str:
    return canonical_fingerprint({key: value for key, value in payload.items() if key != field})


def _resolve_artifact(manifest_path: Path, artifact: FileArtifact) -> Path:
    declared = Path(artifact.path)
    return declared if declared.is_absolute() else manifest_path.parent / declared


def _load_records(path: Path, model: type[_RecordT], label: str) -> list[_RecordT]:
    rows: list[_RecordT] = []
    for position, raw in enumerate(read_records(path)):
        row = model.model_validate(raw)
        for field in (
            "preference_id",
            "prompt",
            "passage_id",
            "passage_cluster_id",
            "split",
        ):
            if raw.get(field) != getattr(row, field):
                raise ValueError(f"{label}[{position}]: {field} is not canonical and exact")
        exact_fields: tuple[str, ...]
        if isinstance(row, PreferenceRecord):
            exact_fields = ("chosen", "rejected", "chosen_candidate_id", "rejected_candidate_id")
        else:
            exact_fields = ("completion", "candidate_id")
        for field in exact_fields:
            if raw.get(field) != getattr(row, field):
                raise ValueError(f"{label}[{position}]: {field} is not canonical and exact")
        rows.append(row)
    return rows


def _unique_ids(rows: Sequence[Any], label: str) -> dict[str, Any]:
    by_id: dict[str, Any] = {}
    for row in rows:
        if row.preference_id in by_id:
            raise ValueError(f"duplicate preference_id in {label}: {row.preference_id}")
        by_id[row.preference_id] = row
    return by_id


def _validate_source_controls(
    preferences: Sequence[PreferenceRecord],
    controls: Sequence[ContinuedSFTRecord],
    split: str,
) -> list[ContinuedSFTRecord]:
    preference_by_id = _unique_ids(preferences, f"preference {split}")
    control_by_id = _unique_ids(controls, f"continued SFT {split}")
    missing = sorted(set(preference_by_id) - set(control_by_id))
    orphan = sorted(set(control_by_id) - set(preference_by_id))
    if missing or orphan:
        raise ValueError(
            f"continued SFT {split} coverage mismatch: missing={missing}, orphan={orphan}"
        )
    expected_order = [row.preference_id for row in preferences]
    actual_order = [row.preference_id for row in controls]
    if actual_order != expected_order:
        raise ValueError(f"continued SFT {split} order differs from preference data")
    for preference, control in zip(preferences, controls, strict=True):
        expected = (
            preference.prompt,
            preference.chosen,
            preference.chosen_candidate_id,
            preference.passage_id,
            preference.passage_cluster_id,
            preference.split,
            preference.provenance,
        )
        actual = (
            control.prompt,
            control.completion,
            control.candidate_id,
            control.passage_id,
            control.passage_cluster_id,
            control.split,
            control.provenance,
        )
        if actual != expected:
            raise ValueError(
                f"continued SFT completion or identity differs from chosen: "
                f"{preference.preference_id}"
            )
    return list(controls)


def _validate_preferences(
    train: Sequence[PreferenceRecord], dev: Sequence[PreferenceRecord]
) -> Task06SelectionProvenance:
    if not train or not dev:
        raise ValueError("preference train and dev must both be non-empty")
    all_rows = list(train) + list(dev)
    _unique_ids(all_rows, "preference train/dev")
    provenance = all_rows[0].provenance
    for expected_split, rows in (("train", train), ("dev", dev)):
        for row in rows:
            if row.split == "test":
                raise ValueError("test split is absolutely forbidden")
            if row.split != expected_split:
                raise ValueError(f"record appears in the wrong {expected_split} artifact")
            if row.provenance != provenance:
                raise ValueError("dataset or selection provenance drift in preference data")
    train_passages = {row.passage_id for row in train}
    dev_passages = {row.passage_id for row in dev}
    if overlap := sorted(train_passages & dev_passages):
        raise ValueError(f"passage leakage between train and dev: {overlap}")
    train_clusters = {row.passage_cluster_id for row in train}
    dev_clusters = {row.passage_cluster_id for row in dev}
    if overlap := sorted(train_clusters & dev_clusters):
        raise ValueError(f"cluster leakage between train and dev: {overlap}")
    return provenance


def _load_weight_assignments(
    manifest_path: Path,
    records_path: Path,
    preferences: Sequence[PreferenceRecord],
    provenance: Task06SelectionProvenance,
) -> tuple[WeightAssignmentManifest, list[WeightAssignment]]:
    raw_manifest = _load_json_object(manifest_path)
    manifest = WeightAssignmentManifest.model_validate(raw_manifest)
    if manifest.artifact_fingerprint != _artifact_fingerprint(raw_manifest, "artifact_fingerprint"):
        raise ValueError("weight artifact fingerprint mismatch")
    declared = _resolve_artifact(manifest_path, manifest.records).resolve()
    if declared != records_path.resolve():
        raise ValueError("weight records path differs from weight manifest")
    if file_sha256(records_path) != manifest.records.sha256:
        raise ValueError("weight records sha256 differs from weight manifest")
    raw_rows = list(read_records(records_path))
    if len(raw_rows) != manifest.records.record_count:
        raise ValueError("weight assignment record count differs from weight manifest")
    assignments = [WeightAssignment.model_validate(row) for row in raw_rows]
    assignment_by_id = _unique_ids(assignments, "weight assignments")
    expected_ids = [row.preference_id for row in preferences]
    actual_ids = [row.preference_id for row in assignments]
    missing = sorted(set(expected_ids) - set(assignment_by_id))
    orphan = sorted(set(assignment_by_id) - set(expected_ids))
    if missing or orphan:
        raise ValueError(f"weight assignment coverage mismatch: missing={missing}, orphan={orphan}")
    if actual_ids != expected_ids:
        raise ValueError("weight assignment order differs from frozen preference data")
    if manifest.ordered_preference_ids_fingerprint != ordered_ids_fingerprint(actual_ids):
        raise ValueError("weight assignment ordered-ID fingerprint mismatch")

    expected_selection = (
        provenance.dataset_id,
        provenance.dataset_fingerprint,
        provenance.selection_policy_id,
        provenance.selection_policy_fingerprint,
    )
    manifest_selection = (
        manifest.dataset_id,
        manifest.dataset_fingerprint,
        manifest.selection_policy_id,
        manifest.selection_policy_fingerprint,
    )
    if manifest_selection != expected_selection:
        raise ValueError("dataset or selection provenance drift in weight manifest")
    expected_weight = (manifest.weight_policy_id, manifest.weight_policy_fingerprint)
    for preference, assignment in zip(preferences, assignments, strict=True):
        assignment_selection = (
            assignment.dataset_id,
            assignment.dataset_fingerprint,
            assignment.selection_policy_id,
            assignment.selection_policy_fingerprint,
        )
        assignment_weight = (
            assignment.weight_policy_id,
            assignment.weight_policy_fingerprint,
        )
        if assignment_selection != expected_selection:
            raise ValueError(
                f"dataset or selection provenance drift in weight assignment: "
                f"{assignment.preference_id}"
            )
        if assignment_weight != expected_weight:
            raise ValueError(
                f"weight-policy provenance drift in assignment: {assignment.preference_id}"
            )
        if assignment.split != preference.split:
            raise ValueError(f"weight assignment split drift: {assignment.preference_id}")
    return manifest, assignments


def _write_jsonl(path: Path, rows: Sequence[StrictModel]) -> None:
    with JsonlWriter(path) as writer:
        for row in rows:
            writer.write(row.model_dump(mode="json"))


def package_task07_inputs(
    *,
    preference_train_path: Path,
    preference_dev_path: Path,
    continued_sft_train_path: Path,
    continued_sft_dev_path: Path,
    weight_manifest_path: Path,
    weight_records_path: Path,
    output_dir: Path,
) -> Task06PreferenceManifest:
    """Package already-frozen Task 06 rows without models, thresholds or relabeling."""
    if output_dir.exists():
        raise FileExistsError(f"Task 07 input output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"
    if staging.exists():  # pragma: no cover - UUID collision guard
        raise FileExistsError(f"staging output already exists: {staging}")

    try:
        preference_train = _load_records(
            preference_train_path, PreferenceRecord, "preference train"
        )
        preference_dev = _load_records(preference_dev_path, PreferenceRecord, "preference dev")
        provenance = _validate_preferences(preference_train, preference_dev)
        continued_train = _validate_source_controls(
            preference_train,
            _load_records(continued_sft_train_path, ContinuedSFTRecord, "continued SFT train"),
            "train",
        )
        continued_dev = _validate_source_controls(
            preference_dev,
            _load_records(continued_sft_dev_path, ContinuedSFTRecord, "continued SFT dev"),
            "dev",
        )
        all_preferences = preference_train + preference_dev
        weight_manifest, assignments = _load_weight_assignments(
            weight_manifest_path,
            weight_records_path,
            all_preferences,
            provenance,
        )
        weights = {row.preference_id: row for row in assignments}
        weighted_train = [
            ScoreWeightedContinuedSFTRecord(
                **row.model_dump(mode="json"),
                sample_weight=weights[row.preference_id].sample_weight,
                weight_policy_id=weight_manifest.weight_policy_id,
                weight_policy_fingerprint=weight_manifest.weight_policy_fingerprint,
            )
            for row in continued_train
        ]
        weighted_dev = [
            ScoreWeightedContinuedSFTRecord(
                **row.model_dump(mode="json"),
                sample_weight=weights[row.preference_id].sample_weight,
                weight_policy_id=weight_manifest.weight_policy_id,
                weight_policy_fingerprint=weight_manifest.weight_policy_fingerprint,
            )
            for row in continued_dev
        ]

        staging.mkdir()
        outputs: dict[str, tuple[Path, Sequence[StrictModel]]] = {
            "preference_train": (staging / "preference_train.jsonl", preference_train),
            "preference_dev": (staging / "preference_dev.jsonl", preference_dev),
            "continued_sft_train": (staging / "continued_sft_train.jsonl", continued_train),
            "continued_sft_dev": (staging / "continued_sft_dev.jsonl", continued_dev),
            "weighted_sft_train": (staging / "weighted_sft_train.jsonl", weighted_train),
            "weighted_sft_dev": (staging / "weighted_sft_dev.jsonl", weighted_dev),
        }
        for path, rows in outputs.values():
            _write_jsonl(path, rows)
        payload: dict[str, Any] = {
            "schema_version": 1,
            "contract": "task06-preference-data-for-task07-v1",
            "dataset_id": provenance.dataset_id,
            "dataset_fingerprint": provenance.dataset_fingerprint,
            "selection_policy_id": provenance.selection_policy_id,
            "selection_policy_fingerprint": provenance.selection_policy_fingerprint,
            "artifacts": {
                name: {
                    "path": path.name,
                    "sha256": file_sha256(path),
                    "record_count": len(rows),
                }
                for name, (path, rows) in outputs.items()
            },
            "automatic_thresholds_created": False,
            "relabeling_performed": False,
            "final_tests_used": [],
        }
        payload["manifest_fingerprint"] = canonical_fingerprint(payload)
        manifest = Task06PreferenceManifest.model_validate(payload)
        manifest_path = staging / "manifest.json"
        write_json(manifest_path, manifest.model_dump(mode="json"))
        validate_dpo_dataset(
            task06_manifest_path=manifest_path,
            preference_train_path=outputs["preference_train"][0],
            preference_dev_path=outputs["preference_dev"][0],
            continued_sft_train_path=outputs["continued_sft_train"][0],
            continued_sft_dev_path=outputs["continued_sft_dev"][0],
            weighted_sft_train_path=outputs["weighted_sft_train"][0],
            weighted_sft_dev_path=outputs["weighted_sft_dev"][0],
        )
        if output_dir.exists():
            raise FileExistsError(f"Task 07 input output already exists: {output_dir}")
        os.replace(staging, output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
