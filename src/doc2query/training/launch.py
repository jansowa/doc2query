"""Fail-closed, model-free Task 07 launch-bundle preflight."""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from doc2query.schemas import StrictModel
from doc2query.training.dpo import (
    SHA256_PATTERN,
    ArmBudget,
    DPOArm,
    DPOPlanManifest,
    ModelStackIdentity,
    ValidatedDPODataset,
    canonical_fingerprint,
    file_sha256,
    ordered_ids_fingerprint,
    validate_dpo_dataset,
    validate_dpo_plan,
    validate_reference_logprobs,
    validate_token_length_evidence,
)
from doc2query.utils.records import write_json

_INPUT_NAMES = {
    "task06_manifest",
    "preference_train",
    "preference_dev",
    "continued_sft_train",
    "continued_sft_dev",
    "weighted_sft_train",
    "weighted_sft_dev",
    "dpo_plan",
    "token_length_manifest",
    "token_length_records",
    "reference_logprob_manifest",
    "reference_logprob_records",
}
_RECORD_INPUT_NAMES = {
    "preference_train",
    "preference_dev",
    "continued_sft_train",
    "continued_sft_dev",
    "weighted_sft_train",
    "weighted_sft_dev",
    "token_length_records",
    "reference_logprob_records",
}


class LaunchArm(StrictModel):
    """Validated inputs and frozen budget for one future model smoke arm."""

    arm: DPOArm
    train_input: Literal["preference_train", "continued_sft_train", "weighted_sft_train"]
    dev_input: Literal["preference_dev", "continued_sft_dev", "weighted_sft_dev"]
    budget: ArmBudget
    reference_logprobs_input: Literal["reference_logprob_records"] | None = None

    @model_validator(mode="after")
    def inputs_match_arm(self) -> LaunchArm:
        expected = {
            DPOArm.DPO: ("preference_train", "preference_dev", "reference_logprob_records"),
            DPOArm.CONTINUED_SFT: ("continued_sft_train", "continued_sft_dev", None),
            DPOArm.SCORE_WEIGHTED_CONTINUED_SFT: (
                "weighted_sft_train",
                "weighted_sft_dev",
                None,
            ),
        }[self.arm]
        if (self.train_input, self.dev_input, self.reference_logprobs_input) != expected:
            raise ValueError("launch inputs do not match arm identity")
        if self.budget.arm != self.arm:
            raise ValueError("launch arm budget identity mismatch")
        return self


class Task07LaunchManifest(StrictModel):
    """Versioned proof that frozen Task 07 inputs agree without loading models."""

    schema_version: Literal[1]
    contract: Literal["task07-model-free-launch-bundle-v1"]
    status: Literal["ready_for_model_smoke_not_trained"]
    plan_id: str = Field(min_length=1)
    plan_fingerprint: str = Field(pattern=SHA256_PATTERN)
    dataset_id: str = Field(min_length=1)
    dataset_fingerprint: str = Field(pattern=SHA256_PATTERN)
    selection_policy_id: str = Field(min_length=1)
    selection_policy_fingerprint: str = Field(pattern=SHA256_PATTERN)
    weight_policy_id: str = Field(min_length=1)
    weight_policy_fingerprint: str = Field(pattern=SHA256_PATTERN)
    cohort_fingerprint: str = Field(pattern=SHA256_PATTERN)
    start_model: ModelStackIdentity
    reference_model: ModelStackIdentity
    arms: dict[DPOArm, LaunchArm]
    input_hashes: dict[str, str]
    input_record_counts: dict[str, int]
    model_loading_performed: Literal[False]
    tokenizer_loading_performed: Literal[False]
    reference_logprobs_computed: Literal[False]
    training_started: Literal[False]
    evaluation_started: Literal[False]
    final_tests_used: list[str] = Field(max_length=0)
    bundle_fingerprint: str = Field(pattern=SHA256_PATTERN)

    @field_validator("input_hashes")
    @classmethod
    def hashes_are_complete(cls, value: dict[str, str]) -> dict[str, str]:
        if set(value) != _INPUT_NAMES:
            raise ValueError("launch input hashes must cover exactly all twelve inputs")
        if any(
            len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest)
            for digest in value.values()
        ):
            raise ValueError("launch input hashes must be lowercase SHA-256 values")
        return value

    @field_validator("input_record_counts")
    @classmethod
    def record_counts_are_complete(cls, value: dict[str, int]) -> dict[str, int]:
        if set(value) != _RECORD_INPUT_NAMES or any(count < 0 for count in value.values()):
            raise ValueError("launch record counts must cover exactly all eight record inputs")
        return value

    @model_validator(mode="after")
    def arms_and_identities_are_consistent(self) -> Task07LaunchManifest:
        if set(self.arms) != set(DPOArm):
            raise ValueError("launch bundle must contain exactly the three mandatory arms")
        if any(key != arm.arm for key, arm in self.arms.items()):
            raise ValueError("launch arm key differs from arm identity")
        if self.start_model != self.reference_model:
            raise ValueError("launch start/reference model stack and tokenizer must be identical")
        budgets = [arm.budget for arm in self.arms.values()]
        matched = {
            (
                budget.cohort_fingerprint,
                tuple(budget.seeds),
                budget.target_token_budget,
                budget.target_optimizer_steps,
                budget.train_example_count,
                budget.prompt_chosen_tokens_per_cohort,
            )
            for budget in budgets
        }
        if len(matched) != 1 or any(
            budget.cohort_fingerprint != self.cohort_fingerprint for budget in budgets
        ):
            raise ValueError("launch arm cohort, seeds or budgets are not matched")
        weighted = self.arms[DPOArm.SCORE_WEIGHTED_CONTINUED_SFT].budget
        if (weighted.weight_policy_id, weighted.weight_policy_fingerprint) != (
            self.weight_policy_id,
            self.weight_policy_fingerprint,
        ):
            raise ValueError("launch weight policy differs from weighted-SFT arm")
        payload = self.model_dump(mode="json")
        fingerprint = payload.pop("bundle_fingerprint")
        if fingerprint != canonical_fingerprint(payload):
            raise ValueError("launch bundle fingerprint mismatch")
        return self


def _validate_plan_inputs(
    plan: DPOPlanManifest,
    dataset: ValidatedDPODataset,
    token_manifest_hash: str,
    token_records_hash: str,
) -> None:
    expected = dataset.input_hashes | {
        "token_length_manifest": token_manifest_hash,
        "token_length_records": token_records_hash,
    }
    drift = {
        name: (plan.input_hashes.get(name), digest)
        for name, digest in expected.items()
        if plan.input_hashes.get(name) != digest
    }
    if drift:
        raise ValueError(f"plan input hash drift: {drift}")
    if plan.dataset_fingerprint != dataset.provenance.dataset_fingerprint:
        raise ValueError("dataset provenance differs between Task 06 data and plan")
    train_ids = [row.preference_id for row in dataset.preference_train]
    cohort_fingerprint = ordered_ids_fingerprint(train_ids)
    if plan.cohort_fingerprint != cohort_fingerprint:
        raise ValueError("plan cohort fingerprint differs from frozen training order")
    if any(arm.train_example_count != len(train_ids) for arm in plan.arms.values()):
        raise ValueError("plan train-example budget differs from frozen cohort")


def _validate_weight_policy(dataset: ValidatedDPODataset, plan: DPOPlanManifest) -> tuple[str, str]:
    weighted = dataset.weighted_sft_train + dataset.weighted_sft_dev
    policies = {(row.weight_policy_id, row.weight_policy_fingerprint) for row in weighted}
    if len(policies) != 1:
        raise ValueError("weighted-SFT weight policy is not unique")
    policy = policies.pop()
    plan_arm = plan.arms[DPOArm.SCORE_WEIGHTED_CONTINUED_SFT]
    plan_policy = (plan_arm.weight_policy_id, plan_arm.weight_policy_fingerprint)
    if policy != plan_policy:
        raise ValueError("weight policy drift between weighted-SFT data and plan")
    return policy


def _launch_arms(plan: DPOPlanManifest) -> dict[DPOArm, LaunchArm]:
    return {
        DPOArm.DPO: LaunchArm(
            arm=DPOArm.DPO,
            train_input="preference_train",
            dev_input="preference_dev",
            reference_logprobs_input="reference_logprob_records",
            budget=plan.arms[DPOArm.DPO],
        ),
        DPOArm.CONTINUED_SFT: LaunchArm(
            arm=DPOArm.CONTINUED_SFT,
            train_input="continued_sft_train",
            dev_input="continued_sft_dev",
            budget=plan.arms[DPOArm.CONTINUED_SFT],
        ),
        DPOArm.SCORE_WEIGHTED_CONTINUED_SFT: LaunchArm(
            arm=DPOArm.SCORE_WEIGHTED_CONTINUED_SFT,
            train_input="weighted_sft_train",
            dev_input="weighted_sft_dev",
            budget=plan.arms[DPOArm.SCORE_WEIGHTED_CONTINUED_SFT],
        ),
    }


def prepare_task07_launch(
    *,
    task06_manifest_path: Path,
    preference_train_path: Path,
    preference_dev_path: Path,
    continued_sft_train_path: Path,
    continued_sft_dev_path: Path,
    weighted_sft_train_path: Path,
    weighted_sft_dev_path: Path,
    plan_path: Path,
    token_length_manifest_path: Path,
    token_length_records_path: Path,
    reference_logprob_manifest_path: Path,
    reference_logprob_records_path: Path,
    output_dir: Path,
) -> Task07LaunchManifest:
    """Validate frozen inputs and atomically publish a model-free launch manifest."""
    if output_dir.exists():
        raise FileExistsError(f"Task 07 launch output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"
    if staging.exists():  # pragma: no cover - UUID collision guard
        raise FileExistsError(f"staging output already exists: {staging}")

    paths = {
        "task06_manifest": task06_manifest_path,
        "preference_train": preference_train_path,
        "preference_dev": preference_dev_path,
        "continued_sft_train": continued_sft_train_path,
        "continued_sft_dev": continued_sft_dev_path,
        "weighted_sft_train": weighted_sft_train_path,
        "weighted_sft_dev": weighted_sft_dev_path,
        "dpo_plan": plan_path,
        "token_length_manifest": token_length_manifest_path,
        "token_length_records": token_length_records_path,
        "reference_logprob_manifest": reference_logprob_manifest_path,
        "reference_logprob_records": reference_logprob_records_path,
    }
    try:
        dataset = validate_dpo_dataset(
            task06_manifest_path=task06_manifest_path,
            preference_train_path=preference_train_path,
            preference_dev_path=preference_dev_path,
            continued_sft_train_path=continued_sft_train_path,
            continued_sft_dev_path=continued_sft_dev_path,
            weighted_sft_train_path=weighted_sft_train_path,
            weighted_sft_dev_path=weighted_sft_dev_path,
        )
        plan = validate_dpo_plan(plan_path)
        weight_policy_id, weight_policy_fingerprint = _validate_weight_policy(dataset, plan)
        token_manifest, token_rows = validate_token_length_evidence(
            token_length_manifest_path,
            token_length_records_path,
            dataset,
            plan.start_model.tokenizer,
        )
        _validate_plan_inputs(
            plan,
            dataset,
            file_sha256(token_length_manifest_path),
            file_sha256(token_length_records_path),
        )
        if plan.token_length_artifact_fingerprint != token_manifest.artifact_fingerprint:
            raise ValueError("token-length artifact fingerprint differs from plan")
        reference_rows = validate_reference_logprobs(
            records_path=reference_logprob_records_path,
            manifest_path=reference_logprob_manifest_path,
            plan_path=plan_path,
            dataset=dataset,
        )
        input_hashes = {name: file_sha256(path) for name, path in paths.items()}
        input_record_counts = {
            "preference_train": len(dataset.preference_train),
            "preference_dev": len(dataset.preference_dev),
            "continued_sft_train": len(dataset.continued_sft_train),
            "continued_sft_dev": len(dataset.continued_sft_dev),
            "weighted_sft_train": len(dataset.weighted_sft_train),
            "weighted_sft_dev": len(dataset.weighted_sft_dev),
            "token_length_records": len(token_rows),
            "reference_logprob_records": len(reference_rows),
        }
        payload: dict[str, Any] = {
            "schema_version": 1,
            "contract": "task07-model-free-launch-bundle-v1",
            "status": "ready_for_model_smoke_not_trained",
            "plan_id": plan.plan_id,
            "plan_fingerprint": plan.plan_fingerprint,
            "dataset_id": dataset.provenance.dataset_id,
            "dataset_fingerprint": dataset.provenance.dataset_fingerprint,
            "selection_policy_id": dataset.provenance.selection_policy_id,
            "selection_policy_fingerprint": dataset.provenance.selection_policy_fingerprint,
            "weight_policy_id": weight_policy_id,
            "weight_policy_fingerprint": weight_policy_fingerprint,
            "cohort_fingerprint": plan.cohort_fingerprint,
            "start_model": plan.start_model.model_dump(mode="json"),
            "reference_model": plan.reference_model.model_dump(mode="json"),
            "arms": {
                arm.value: contract.model_dump(mode="json")
                for arm, contract in _launch_arms(plan).items()
            },
            "input_hashes": input_hashes,
            "input_record_counts": input_record_counts,
            "model_loading_performed": False,
            "tokenizer_loading_performed": False,
            "reference_logprobs_computed": False,
            "training_started": False,
            "evaluation_started": False,
            "final_tests_used": [],
        }
        payload["bundle_fingerprint"] = canonical_fingerprint(payload)
        manifest = Task07LaunchManifest.model_validate(payload)
        staging.mkdir()
        write_json(staging / "manifest.json", manifest.model_dump(mode="json"))
        if output_dir.exists():
            raise FileExistsError(f"Task 07 launch output already exists: {output_dir}")
        os.replace(staging, output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
