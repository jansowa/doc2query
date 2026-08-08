"""Model-free contracts, validation and planning for Task 07.

This module intentionally has no trainer, tokenizer, model or GPU dependencies.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, field_validator, model_validator

from doc2query.schemas import StrictModel
from doc2query.utils.records import read_records, write_json

SHA256_PATTERN = r"^[0-9a-f]{64}$"


def canonical_fingerprint(value: Any) -> str:
    """Hash JSON-compatible data with a stable canonical encoding."""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_ids_fingerprint(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def normalize_task06_query(text: str) -> str:
    """Use Task 06 exact-pair normalization: collapsed whitespace plus casefold."""
    return " ".join(text.strip().split()).casefold()


class Task06SelectionProvenance(StrictModel):
    dataset_id: str = Field(min_length=1)
    dataset_fingerprint: str = Field(pattern=SHA256_PATTERN)
    selection_policy_id: str = Field(min_length=1)
    selection_policy_fingerprint: str = Field(pattern=SHA256_PATTERN)


class PreferenceRecord(StrictModel):
    preference_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    chosen: str = Field(min_length=1)
    rejected: str = Field(min_length=1)
    score_margin: float
    chosen_candidate_id: str = Field(min_length=1)
    rejected_candidate_id: str = Field(min_length=1)
    passage_id: str = Field(min_length=1)
    passage_cluster_id: str = Field(min_length=1)
    split: Literal["train", "dev", "test"]
    provenance: Task06SelectionProvenance

    @field_validator("prompt", "chosen", "rejected")
    @classmethod
    def text_is_non_empty(cls, value: str) -> str:
        normalized = " ".join(value.strip().split())
        if not normalized:
            raise ValueError("text fields must be non-empty")
        return normalized

    @field_validator("score_margin")
    @classmethod
    def margin_is_positive_and_finite(cls, value: float) -> float:
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("score_margin must be positive and finite")
        return value

    @model_validator(mode="after")
    def pair_is_distinct(self) -> PreferenceRecord:
        if self.chosen_candidate_id == self.rejected_candidate_id:
            raise ValueError("chosen and rejected candidate IDs must differ")
        if normalize_task06_query(self.chosen) == normalize_task06_query(self.rejected):
            raise ValueError("chosen and rejected are identical after Task 06 normalization")
        return self


class ContinuedSFTRecord(StrictModel):
    preference_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    completion: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    passage_id: str = Field(min_length=1)
    passage_cluster_id: str = Field(min_length=1)
    split: Literal["train", "dev", "test"]
    provenance: Task06SelectionProvenance

    @field_validator("prompt", "completion")
    @classmethod
    def text_is_non_empty(cls, value: str) -> str:
        normalized = " ".join(value.strip().split())
        if not normalized:
            raise ValueError("text fields must be non-empty")
        return normalized


class ScoreWeightedContinuedSFTRecord(ContinuedSFTRecord):
    sample_weight: float = Field(gt=0.0)
    weight_policy_id: str = Field(min_length=1)
    weight_policy_fingerprint: str = Field(pattern=SHA256_PATTERN)

    @field_validator("sample_weight")
    @classmethod
    def weight_is_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("sample_weight must be finite")
        return value


class FileArtifact(StrictModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=SHA256_PATTERN)
    record_count: int = Field(ge=0)


class Task06PreferenceManifest(StrictModel):
    schema_version: int = Field(ge=1)
    contract: Literal["task06-preference-data-for-task07-v1"]
    dataset_id: str = Field(min_length=1)
    dataset_fingerprint: str = Field(pattern=SHA256_PATTERN)
    selection_policy_id: str = Field(min_length=1)
    selection_policy_fingerprint: str = Field(pattern=SHA256_PATTERN)
    artifacts: dict[str, FileArtifact]
    automatic_thresholds_created: Literal[False]
    relabeling_performed: Literal[False]
    final_tests_used: list[str] = Field(max_length=0)
    manifest_fingerprint: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def required_artifacts_exist(self) -> Task06PreferenceManifest:
        required = {
            "preference_train",
            "preference_dev",
            "continued_sft_train",
            "continued_sft_dev",
            "weighted_sft_train",
            "weighted_sft_dev",
        }
        if set(self.artifacts) != required:
            raise ValueError(f"Task 06 artifacts must be exactly {sorted(required)}")
        return self


class BaseModelIdentity(StrictModel):
    model_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    artifact_fingerprint: str = Field(pattern=SHA256_PATTERN)


class AdapterIdentity(StrictModel):
    adapter_id: str = Field(min_length=1)
    adapter_revision: str = Field(min_length=1)
    adapter_fingerprint: str = Field(pattern=SHA256_PATTERN)
    base_model_fingerprint: str = Field(pattern=SHA256_PATTERN)


class TokenizerIdentity(StrictModel):
    tokenizer_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    tokenizer_fingerprint: str = Field(pattern=SHA256_PATTERN)


class ModelStackIdentity(StrictModel):
    base_model: BaseModelIdentity
    sft_adapter: AdapterIdentity
    tokenizer: TokenizerIdentity

    @model_validator(mode="after")
    def adapter_matches_base(self) -> ModelStackIdentity:
        if self.sft_adapter.base_model_fingerprint != self.base_model.artifact_fingerprint:
            raise ValueError("SFT adapter does not identify the configured base model")
        return self


class TokenLengthEvidenceRecord(StrictModel):
    preference_id: str = Field(min_length=1)
    split: Literal["train", "dev"]
    prompt_tokens: int = Field(ge=1)
    chosen_tokens: int = Field(ge=1)
    rejected_tokens: int = Field(ge=1)
    prompt_chosen_tokens: int = Field(ge=2)
    prompt_rejected_tokens: int = Field(ge=2)

    @model_validator(mode="after")
    def totals_match(self) -> TokenLengthEvidenceRecord:
        if self.prompt_chosen_tokens != self.prompt_tokens + self.chosen_tokens:
            raise ValueError("prompt_chosen_tokens is inconsistent")
        if self.prompt_rejected_tokens != self.prompt_tokens + self.rejected_tokens:
            raise ValueError("prompt_rejected_tokens is inconsistent")
        return self


class TokenLengthEvidenceManifest(StrictModel):
    contract: Literal["task07-model-free-token-lengths-v1"]
    records: FileArtifact
    tokenizer: TokenizerIdentity
    dataset_fingerprint: str = Field(pattern=SHA256_PATTERN)
    ordered_preference_ids_fingerprint: str = Field(pattern=SHA256_PATTERN)
    model_loading_performed: Literal[False]
    final_tests_used: list[str] = Field(max_length=0)
    artifact_fingerprint: str = Field(pattern=SHA256_PATTERN)


class DPOArm(StrEnum):
    DPO = "dpo"
    CONTINUED_SFT = "continued_sft"
    SCORE_WEIGHTED_CONTINUED_SFT = "score_weighted_continued_sft"


class ArmBudget(StrictModel):
    arm: DPOArm
    cohort_fingerprint: str = Field(pattern=SHA256_PATTERN)
    seeds: list[int] = Field(min_length=1)
    target_token_budget: int = Field(ge=1)
    target_optimizer_steps: int = Field(ge=1)
    train_example_count: int = Field(ge=1)
    prompt_chosen_tokens_per_cohort: int = Field(ge=1)
    dpo_pair_tokens_per_cohort: int | None = Field(default=None, ge=1)
    weight_policy_id: str | None = None
    weight_policy_fingerprint: str | None = Field(default=None, pattern=SHA256_PATTERN)

    @field_validator("seeds")
    @classmethod
    def seeds_are_unique(cls, value: list[int]) -> list[int]:
        if value != sorted(set(value)):
            raise ValueError("seeds must be unique and sorted")
        return value

    @model_validator(mode="after")
    def arm_specific_fields(self) -> ArmBudget:
        weighted = self.arm == DPOArm.SCORE_WEIGHTED_CONTINUED_SFT
        if weighted != (self.weight_policy_id is not None):
            raise ValueError("only weighted SFT requires weight_policy_id")
        if weighted != (self.weight_policy_fingerprint is not None):
            raise ValueError("only weighted SFT requires weight_policy_fingerprint")
        if (self.arm == DPOArm.DPO) != (self.dpo_pair_tokens_per_cohort is not None):
            raise ValueError("only DPO records dpo_pair_tokens_per_cohort")
        return self


class DPOPlanConfig(StrictModel):
    plan_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    seeds: list[int] = Field(min_length=1)
    beta: float = Field(gt=0.0)
    loss_type: Literal["sigmoid"] = "sigmoid"
    learning_rate: float = Field(gt=0.0)
    max_length: int = Field(ge=64)
    max_prompt_length: int = Field(ge=1)
    target_token_budget: int = Field(ge=1)
    target_optimizer_steps: int = Field(ge=1)
    start_model: ModelStackIdentity
    reference_model: ModelStackIdentity
    weight_policy_id: str = Field(min_length=1)
    weight_policy_fingerprint: str = Field(pattern=SHA256_PATTERN)
    arms: list[DPOArm] = Field(
        default_factory=lambda: [
            DPOArm.DPO,
            DPOArm.CONTINUED_SFT,
            DPOArm.SCORE_WEIGHTED_CONTINUED_SFT,
        ]
    )

    @field_validator("seeds")
    @classmethod
    def seeds_are_unique(cls, value: list[int]) -> list[int]:
        if value != sorted(set(value)):
            raise ValueError("seeds must be unique and sorted")
        return value

    @model_validator(mode="after")
    def identities_and_arms_are_frozen(self) -> DPOPlanConfig:
        if self.arms != list(DPOArm):
            raise ValueError("the plan must contain DPO, continued SFT and weighted SFT")
        if self.reference_model != self.start_model:
            raise ValueError("reference model must exactly match the starting SFT model stack")
        if self.max_prompt_length >= self.max_length:
            raise ValueError("max_prompt_length must be smaller than max_length")
        return self


class DPOPlanManifest(StrictModel):
    contract: Literal["task07-dpo-plan-v1"]
    status: Literal["planned_not_trained"]
    plan_id: str = Field(min_length=1)
    plan_fingerprint: str = Field(pattern=SHA256_PATTERN)
    dataset_fingerprint: str = Field(pattern=SHA256_PATTERN)
    cohort_fingerprint: str = Field(pattern=SHA256_PATTERN)
    token_length_artifact_fingerprint: str = Field(pattern=SHA256_PATTERN)
    input_hashes: dict[str, str]
    start_model: ModelStackIdentity
    reference_model: ModelStackIdentity
    beta: float = Field(gt=0.0)
    loss_type: Literal["sigmoid"]
    learning_rate: float = Field(gt=0.0)
    max_length: int = Field(ge=64)
    max_prompt_length: int = Field(ge=1)
    arms: dict[DPOArm, ArmBudget]
    model_loading_performed: Literal[False]
    training_started: Literal[False]
    reference_logprobs_computed: Literal[False]
    final_tests_used: list[str] = Field(max_length=0)

    @model_validator(mode="after")
    def budgets_are_matched(self) -> DPOPlanManifest:
        if set(self.arms) != set(DPOArm):
            raise ValueError("all three mandatory arms must be present")
        values = list(self.arms.values())
        for key, arm in self.arms.items():
            if key != arm.arm:
                raise ValueError("arm mapping key does not match budget identity")
        matched = {
            (
                item.cohort_fingerprint,
                tuple(item.seeds),
                item.target_token_budget,
                item.target_optimizer_steps,
                item.train_example_count,
                item.prompt_chosen_tokens_per_cohort,
            )
            for item in values
        }
        if len(matched) != 1:
            raise ValueError("three-arm cohort, seeds or budgets are not matched")
        return self


class ReferenceLogprobRecord(StrictModel):
    preference_id: str = Field(min_length=1)
    position: int = Field(ge=0)
    chosen_logprob: float
    rejected_logprob: float

    @field_validator("chosen_logprob", "rejected_logprob")
    @classmethod
    def logprobs_are_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("reference logprobs must be finite")
        return value


class ReferenceLogprobManifest(StrictModel):
    contract: Literal["task07-precomputed-reference-logprobs-v1"]
    records: FileArtifact
    ordered_preference_ids_fingerprint: str = Field(pattern=SHA256_PATTERN)
    dataset_fingerprint: str = Field(pattern=SHA256_PATTERN)
    plan_fingerprint: str = Field(pattern=SHA256_PATTERN)
    reference_model: ModelStackIdentity
    tokenizer_fingerprint: str = Field(pattern=SHA256_PATTERN)
    artifact_fingerprint: str = Field(pattern=SHA256_PATTERN)
    final_tests_used: list[str] = Field(max_length=0)


class ValidatedDPODataset(StrictModel):
    preference_train: list[PreferenceRecord]
    preference_dev: list[PreferenceRecord]
    continued_sft_train: list[ContinuedSFTRecord]
    continued_sft_dev: list[ContinuedSFTRecord]
    weighted_sft_train: list[ScoreWeightedContinuedSFTRecord]
    weighted_sft_dev: list[ScoreWeightedContinuedSFTRecord]
    provenance: Task06SelectionProvenance
    input_hashes: dict[str, str]


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _manifest_payload_fingerprint(payload: Mapping[str, Any], field: str) -> str:
    return canonical_fingerprint({key: value for key, value in payload.items() if key != field})


def _resolve_artifact(manifest_path: Path, artifact: FileArtifact) -> Path:
    path = Path(artifact.path)
    return path if path.is_absolute() else manifest_path.parent / path


def _load_artifact_rows(
    manifest_path: Path, name: str, artifact: FileArtifact, expected_path: Path
) -> list[dict[str, Any]]:
    declared = _resolve_artifact(manifest_path, artifact).resolve()
    if declared != expected_path.resolve():
        raise ValueError(f"{name}: path differs from the Task 06 manifest")
    if file_sha256(expected_path) != artifact.sha256:
        raise ValueError(f"{name}: sha256 differs from the Task 06 manifest")
    rows = list(read_records(expected_path))
    if len(rows) != artifact.record_count:
        raise ValueError(f"{name}: record count differs from the Task 06 manifest")
    return rows


def _unique_by_id(rows: Sequence[Any], field: str, label: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for row in rows:
        identifier = str(getattr(row, field))
        if identifier in result:
            raise ValueError(f"duplicate {field} in {label}: {identifier}")
        result[identifier] = row
    return result


def _validate_control_rows(
    preferences: Sequence[PreferenceRecord],
    controls: Sequence[ContinuedSFTRecord],
    label: str,
) -> None:
    preference_by_id = _unique_by_id(preferences, "preference_id", f"{label} preferences")
    control_by_id = _unique_by_id(controls, "preference_id", label)
    missing = sorted(set(preference_by_id) - set(control_by_id))
    orphan = sorted(set(control_by_id) - set(preference_by_id))
    if missing or orphan:
        raise ValueError(f"{label} coverage mismatch: missing={missing}, orphan={orphan}")
    for preference_id, preference in preference_by_id.items():
        control = control_by_id[preference_id]
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
            raise ValueError(f"{label} record does not exactly match chosen: {preference_id}")


def validate_dpo_dataset(
    *,
    task06_manifest_path: Path,
    preference_train_path: Path,
    preference_dev_path: Path,
    continued_sft_train_path: Path,
    continued_sft_dev_path: Path,
    weighted_sft_train_path: Path,
    weighted_sft_dev_path: Path,
) -> ValidatedDPODataset:
    """Validate Task 06 inputs fail-closed without opening test data."""
    raw_manifest = _load_json(task06_manifest_path)
    manifest = Task06PreferenceManifest.model_validate(raw_manifest)
    expected_manifest_fp = _manifest_payload_fingerprint(raw_manifest, "manifest_fingerprint")
    if manifest.manifest_fingerprint != expected_manifest_fp:
        raise ValueError("Task 06 manifest fingerprint mismatch")
    expected_provenance = Task06SelectionProvenance(
        dataset_id=manifest.dataset_id,
        dataset_fingerprint=manifest.dataset_fingerprint,
        selection_policy_id=manifest.selection_policy_id,
        selection_policy_fingerprint=manifest.selection_policy_fingerprint,
    )

    paths = {
        "preference_train": preference_train_path,
        "preference_dev": preference_dev_path,
        "continued_sft_train": continued_sft_train_path,
        "continued_sft_dev": continued_sft_dev_path,
        "weighted_sft_train": weighted_sft_train_path,
        "weighted_sft_dev": weighted_sft_dev_path,
    }
    raw_rows = {
        name: _load_artifact_rows(task06_manifest_path, name, manifest.artifacts[name], path)
        for name, path in paths.items()
    }
    preferences = {
        split: [PreferenceRecord.model_validate(row) for row in raw_rows[f"preference_{split}"]]
        for split in ("train", "dev")
    }
    continued = {
        split: [
            ContinuedSFTRecord.model_validate(row) for row in raw_rows[f"continued_sft_{split}"]
        ]
        for split in ("train", "dev")
    }
    weighted = {
        split: [
            ScoreWeightedContinuedSFTRecord.model_validate(row)
            for row in raw_rows[f"weighted_sft_{split}"]
        ]
        for split in ("train", "dev")
    }

    all_preferences = preferences["train"] + preferences["dev"]
    _unique_by_id(all_preferences, "preference_id", "preference dataset")
    for expected_split in ("train", "dev"):
        split_rows = (
            preferences[expected_split] + continued[expected_split] + weighted[expected_split]
        )
        for row in split_rows:
            if row.split == "test":
                raise ValueError("test split is absolutely forbidden in a training plan")
            if row.split != expected_split:
                raise ValueError(f"record appears in the wrong {expected_split} artifact")
            if row.provenance != expected_provenance:
                raise ValueError("record provenance differs from Task 06 manifest")
        _validate_control_rows(
            preferences[expected_split], continued[expected_split], "continued SFT"
        )
        _validate_control_rows(
            preferences[expected_split], weighted[expected_split], "weighted SFT"
        )

    train_passages = {row.passage_id for row in preferences["train"]}
    dev_passages = {row.passage_id for row in preferences["dev"]}
    train_clusters = {row.passage_cluster_id for row in preferences["train"]}
    dev_clusters = {row.passage_cluster_id for row in preferences["dev"]}
    if overlap := sorted(train_passages & dev_passages):
        raise ValueError(f"passage leakage between train and dev: {overlap}")
    if overlap := sorted(train_clusters & dev_clusters):
        raise ValueError(f"cluster leakage between train and dev: {overlap}")

    return ValidatedDPODataset(
        preference_train=preferences["train"],
        preference_dev=preferences["dev"],
        continued_sft_train=continued["train"],
        continued_sft_dev=continued["dev"],
        weighted_sft_train=weighted["train"],
        weighted_sft_dev=weighted["dev"],
        provenance=expected_provenance,
        input_hashes={"task06_manifest": file_sha256(task06_manifest_path)}
        | {name: file_sha256(path) for name, path in paths.items()},
    )


def _load_structured(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a mapping")
    return value


def _validate_token_lengths(
    manifest_path: Path,
    records_path: Path,
    dataset: ValidatedDPODataset,
    tokenizer: TokenizerIdentity,
) -> tuple[TokenLengthEvidenceManifest, list[TokenLengthEvidenceRecord]]:
    raw = _load_json(manifest_path)
    manifest = TokenLengthEvidenceManifest.model_validate(raw)
    if manifest.artifact_fingerprint != _manifest_payload_fingerprint(raw, "artifact_fingerprint"):
        raise ValueError("token-length artifact fingerprint mismatch")
    if manifest.tokenizer != tokenizer:
        raise ValueError("tokenizer identity drift in token-length evidence")
    if manifest.dataset_fingerprint != dataset.provenance.dataset_fingerprint:
        raise ValueError("dataset fingerprint drift in token-length evidence")
    raw_rows = _load_artifact_rows(manifest_path, "token lengths", manifest.records, records_path)
    rows = [TokenLengthEvidenceRecord.model_validate(row) for row in raw_rows]
    _unique_by_id(rows, "preference_id", "token lengths")
    expected = [row.preference_id for row in dataset.preference_train + dataset.preference_dev]
    actual = [row.preference_id for row in rows]
    if actual != expected:
        raise ValueError("token-length evidence order or coverage differs from preference data")
    if manifest.ordered_preference_ids_fingerprint != ordered_ids_fingerprint(actual):
        raise ValueError("token-length ordered ID fingerprint mismatch")
    expected_splits = [row.split for row in dataset.preference_train + dataset.preference_dev]
    if [row.split for row in rows] != expected_splits:
        raise ValueError("token-length split order differs from preference data")
    return manifest, rows


def plan_dpo_controls(
    *,
    config_path: Path,
    token_length_manifest_path: Path,
    token_length_records_path: Path,
    output_path: Path,
    dataset: ValidatedDPODataset,
) -> DPOPlanManifest:
    """Plan three matched arms from frozen, pre-tokenized length evidence only."""
    if output_path.exists():
        raise FileExistsError(f"plan output already exists: {output_path}")
    config = DPOPlanConfig.model_validate(_load_structured(config_path))
    token_manifest, lengths = _validate_token_lengths(
        token_length_manifest_path, token_length_records_path, dataset, config.start_model.tokenizer
    )
    train_lengths = [row for row in lengths if row.split == "train"]
    if not train_lengths:
        raise ValueError("training cohort is empty")
    train_ids = [row.preference_id for row in dataset.preference_train]
    cohort_fingerprint = ordered_ids_fingerprint(train_ids)
    prompt_chosen = sum(row.prompt_chosen_tokens for row in train_lengths)
    dpo_pair = sum(row.prompt_chosen_tokens + row.prompt_rejected_tokens for row in train_lengths)

    def budget(
        arm: DPOArm,
        *,
        dpo_tokens: int | None = None,
        weight_policy_id: str | None = None,
        weight_policy_fingerprint: str | None = None,
    ) -> ArmBudget:
        return ArmBudget(
            arm=arm,
            cohort_fingerprint=cohort_fingerprint,
            seeds=config.seeds,
            target_token_budget=config.target_token_budget,
            target_optimizer_steps=config.target_optimizer_steps,
            train_example_count=len(train_ids),
            prompt_chosen_tokens_per_cohort=prompt_chosen,
            dpo_pair_tokens_per_cohort=dpo_tokens,
            weight_policy_id=weight_policy_id,
            weight_policy_fingerprint=weight_policy_fingerprint,
        )

    arms = {
        DPOArm.DPO: budget(DPOArm.DPO, dpo_tokens=dpo_pair),
        DPOArm.CONTINUED_SFT: budget(DPOArm.CONTINUED_SFT),
        DPOArm.SCORE_WEIGHTED_CONTINUED_SFT: budget(
            DPOArm.SCORE_WEIGHTED_CONTINUED_SFT,
            weight_policy_id=config.weight_policy_id,
            weight_policy_fingerprint=config.weight_policy_fingerprint,
        ),
    }
    payload: dict[str, Any] = {
        "contract": "task07-dpo-plan-v1",
        "status": "planned_not_trained",
        "plan_id": config.plan_id,
        "dataset_fingerprint": dataset.provenance.dataset_fingerprint,
        "cohort_fingerprint": cohort_fingerprint,
        "token_length_artifact_fingerprint": token_manifest.artifact_fingerprint,
        "input_hashes": dataset.input_hashes
        | {
            "config": file_sha256(config_path),
            "token_length_manifest": file_sha256(token_length_manifest_path),
            "token_length_records": file_sha256(token_length_records_path),
        },
        "start_model": config.start_model.model_dump(mode="json"),
        "reference_model": config.reference_model.model_dump(mode="json"),
        "beta": config.beta,
        "loss_type": config.loss_type,
        "learning_rate": config.learning_rate,
        "max_length": config.max_length,
        "max_prompt_length": config.max_prompt_length,
        "arms": {key.value: value.model_dump(mode="json") for key, value in arms.items()},
        "model_loading_performed": False,
        "training_started": False,
        "reference_logprobs_computed": False,
        "final_tests_used": [],
    }
    payload["plan_fingerprint"] = canonical_fingerprint(payload)
    manifest = DPOPlanManifest.model_validate(payload)
    write_json(output_path, manifest.model_dump(mode="json"))
    return manifest


def validate_reference_logprobs(
    *,
    records_path: Path,
    manifest_path: Path,
    plan_path: Path,
    dataset: ValidatedDPODataset,
) -> list[ReferenceLogprobRecord]:
    """Validate precomputed values and restart identity; never compute logprobs."""
    raw_plan = _load_json(plan_path)
    plan = DPOPlanManifest.model_validate(raw_plan)
    if plan.plan_fingerprint != _manifest_payload_fingerprint(raw_plan, "plan_fingerprint"):
        raise ValueError("plan fingerprint mismatch")
    raw_manifest = _load_json(manifest_path)
    manifest = ReferenceLogprobManifest.model_validate(raw_manifest)
    expected_artifact_fp = _manifest_payload_fingerprint(raw_manifest, "artifact_fingerprint")
    if manifest.artifact_fingerprint != expected_artifact_fp:
        raise ValueError("reference logprob artifact fingerprint mismatch")
    expected_identity = (
        plan.dataset_fingerprint,
        plan.plan_fingerprint,
        plan.reference_model,
        plan.reference_model.tokenizer.tokenizer_fingerprint,
    )
    actual_identity = (
        manifest.dataset_fingerprint,
        manifest.plan_fingerprint,
        manifest.reference_model,
        manifest.tokenizer_fingerprint,
    )
    if actual_identity != expected_identity:
        raise ValueError("reference logprobs belong to a different dataset, plan or model stack")
    if dataset.provenance.dataset_fingerprint != plan.dataset_fingerprint:
        raise ValueError("validated dataset differs from the plan")
    raw_rows = _load_artifact_rows(
        manifest_path, "reference logprobs", manifest.records, records_path
    )
    rows = [ReferenceLogprobRecord.model_validate(row) for row in raw_rows]
    by_id = _unique_by_id(rows, "preference_id", "reference logprobs")
    expected_ids = [row.preference_id for row in dataset.preference_train]
    actual_ids = [row.preference_id for row in rows]
    missing = sorted(set(expected_ids) - set(by_id))
    orphan = sorted(set(by_id) - set(expected_ids))
    if missing or orphan:
        raise ValueError(f"reference logprob coverage mismatch: missing={missing}, orphan={orphan}")
    if actual_ids != expected_ids or [row.position for row in rows] != list(range(len(rows))):
        raise ValueError("reference logprob order differs from the frozen training cohort")
    if manifest.ordered_preference_ids_fingerprint != ordered_ids_fingerprint(actual_ids):
        raise ValueError("reference logprob ordered ID fingerprint mismatch")
    return rows


def sigmoid_dpo_loss(
    policy_chosen_logprob: float,
    policy_rejected_logprob: float,
    reference_chosen_logprob: float,
    reference_rejected_logprob: float,
    beta: float,
) -> float:
    """Return scalar classical sigmoid DPO loss using stable softplus."""
    if (
        not all(
            math.isfinite(value)
            for value in (
                policy_chosen_logprob,
                policy_rejected_logprob,
                reference_chosen_logprob,
                reference_rejected_logprob,
                beta,
            )
        )
        or beta <= 0.0
    ):
        raise ValueError("DPO logprobs must be finite and beta must be positive")
    advantage = (policy_chosen_logprob - policy_rejected_logprob) - (
        reference_chosen_logprob - reference_rejected_logprob
    )
    value = -beta * advantage
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))
