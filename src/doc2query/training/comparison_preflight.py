"""Model-free, fail-closed post-run evidence preflight for Task 07.

The module verifies externally produced run evidence for the three mandatory Task 07
arms.  It deliberately does not aggregate or compare metric values and cannot rank,
select, promote, train, or evaluate anything.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, field_validator, model_validator

from doc2query.evaluation.evidence_registry import (
    ConfidenceInterval,
    ConfigEvidence,
    EvidenceArtifact,
)
from doc2query.preferences.selection_preflight import (
    PreferenceSelectionPreflightBundleManifest,
)
from doc2query.schemas import StrictModel
from doc2query.training.dpo import (
    SHA256_PATTERN,
    ArmBudget,
    DPOArm,
    ModelStackIdentity,
    Task06PreferenceManifest,
    canonical_fingerprint,
    file_sha256,
    validate_dpo_dataset,
)
from doc2query.training.launch import Task07LaunchManifest
from doc2query.utils.records import read_records, write_json

PROTOCOL_CONTRACT = "task07-comparison-protocol-v1"
OUTCOME_CONTRACT = "task07-arm-outcome-evidence-v1"
PREFLIGHT_CONTRACT = "task07-comparison-preflight-v1"
BUNDLE_CONTRACT = "task07-comparison-preflight-bundle-v1"
PREFLIGHT_STATUS = "ready_for_future_task07_comparison_not_compared"

_FORBIDDEN_FINAL_TEST_MARKERS = (
    "final_test",
    "final-tests",
    "finaltests",
    "test_native_pl",
    "test_translated_msmarco_pl",
    "test_embedder",
    "test_intrinsic",
    "test_adversarial",
    "test_human_panel",
)


class EvidenceCategory(StrEnum):
    INTRINSIC = "intrinsic"
    SHADOW_INDEPENDENT = "shadow_independent"
    PROBE_EXTRINSIC = "probe_extrinsic"
    HUMAN = "human"
    COST = "cost"


class MetricDirection(StrEnum):
    MIN = "min"
    MAX = "max"


class GuardrailOperator(StrEnum):
    GREATER_OR_EQUAL = "ge"
    LESS_OR_EQUAL = "le"


class PinnedManifestIdentity(StrictModel):
    sha256: str = Field(pattern=SHA256_PATTERN)
    fingerprint: str = Field(pattern=SHA256_PATTERN)


class FrozenIdentity(StrictModel):
    identity_id: str = Field(min_length=1)
    fingerprint: str = Field(pattern=SHA256_PATTERN)


class RequiredMetric(StrictModel):
    name: str = Field(min_length=1)
    category: EvidenceCategory
    direction: MetricDirection
    unit: str = Field(min_length=1)
    definition_fingerprint: str = Field(pattern=SHA256_PATTERN)
    ci_definition_fingerprint: str = Field(pattern=SHA256_PATTERN)
    confidence_level: float = Field(gt=0.0, lt=1.0)
    minimum_sample_size: int = Field(ge=1)
    ci_required: Literal[True]
    sample_size_required: Literal[True]

    @property
    def key(self) -> str:
        return f"{self.category.value}:{self.name}"

    @property
    def definition(self) -> tuple[str, str, str, str, str, float]:
        return (
            self.name,
            self.category.value,
            self.direction.value,
            self.unit,
            self.definition_fingerprint,
            self.confidence_level,
        )


class PinnedGuardrail(StrictModel):
    guardrail_id: str = Field(min_length=1)
    metric_key: str = Field(min_length=1)
    operator: GuardrailOperator
    threshold: float
    definition_fingerprint: str = Field(pattern=SHA256_PATTERN)

    @field_validator("threshold")
    @classmethod
    def threshold_is_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("guardrail threshold must be explicitly supplied and finite")
        return value


class ComparisonArmProtocol(StrictModel):
    arm: DPOArm
    config_comparison_fingerprint: str = Field(pattern=SHA256_PATTERN)
    budget: ArmBudget

    @model_validator(mode="after")
    def identities_match(self) -> ComparisonArmProtocol:
        if self.budget.arm != self.arm:
            raise ValueError("comparison arm differs from its budget identity")
        return self


class Task07ComparisonProtocolManifest(StrictModel):
    """Externally frozen comparison requirements; never derived from outcomes."""

    schema_version: Literal[1]
    contract: Literal["task07-comparison-protocol-v1"]
    status: Literal["protocol_frozen_not_applied"]
    protocol_id: str = Field(min_length=1)
    protocol_version: str = Field(min_length=1)
    required_seeds: list[int] = Field(min_length=2)
    arms: dict[DPOArm, ComparisonArmProtocol]
    task06_handoff: PinnedManifestIdentity
    task06_selection_preflight: PinnedManifestIdentity
    task07_launch_bundle: PinnedManifestIdentity
    dataset: FrozenIdentity
    selection_policy: FrozenIdentity
    weight_policy: FrozenIdentity
    cohort: FrozenIdentity
    plan: FrozenIdentity
    model_stack: ModelStackIdentity
    model_stack_fingerprint: str = Field(pattern=SHA256_PATTERN)
    tokenizer_fingerprint: str = Field(pattern=SHA256_PATTERN)
    required_metrics: list[RequiredMetric] = Field(min_length=1)
    required_artifact_roles: dict[EvidenceCategory, list[str]]
    guardrails: list[PinnedGuardrail] = Field(min_length=1)
    final_tests_used: list[str] = Field(max_length=0)
    manifest_fingerprint: str = Field(pattern=SHA256_PATTERN)

    @field_validator("required_seeds")
    @classmethod
    def seeds_are_pinned_sorted_and_unique(cls, value: list[int]) -> list[int]:
        if value != sorted(set(value)):
            raise ValueError("required seeds must be sorted and unique")
        return value

    @field_validator("required_metrics")
    @classmethod
    def metrics_are_sorted_unique_and_cover_categories(
        cls, value: list[RequiredMetric]
    ) -> list[RequiredMetric]:
        keys = [metric.key for metric in value]
        if keys != sorted(set(keys)):
            raise ValueError("required metric keys must be sorted and unique")
        if {metric.category for metric in value} != set(EvidenceCategory):
            raise ValueError("required metrics must cover all five evidence categories")
        return value

    @field_validator("required_artifact_roles")
    @classmethod
    def artifact_roles_cover_all_categories(
        cls, value: dict[EvidenceCategory, list[str]]
    ) -> dict[EvidenceCategory, list[str]]:
        if set(value) != set(EvidenceCategory):
            raise ValueError("required artifact roles must cover all five evidence categories")
        if any(roles != sorted(set(roles)) or not roles for roles in value.values()):
            raise ValueError("artifact roles must be non-empty, sorted and unique")
        if any(not role.strip() for roles in value.values() for role in roles):
            raise ValueError("artifact roles must be non-empty")
        return value

    @model_validator(mode="after")
    def protocol_is_exact_and_self_fingerprinted(self) -> Task07ComparisonProtocolManifest:
        if set(self.arms) != set(DPOArm):
            raise ValueError("protocol must contain exactly the three mandatory arms")
        if any(key != arm.arm for key, arm in self.arms.items()):
            raise ValueError("protocol arm key differs from its declaration")
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
        if len(matched) != 1:
            raise ValueError("protocol arm budgets are not matched")
        if any(budget.seeds != self.required_seeds for budget in budgets):
            raise ValueError("protocol arm seeds differ from required seeds")
        if any(budget.cohort_fingerprint != self.cohort.fingerprint for budget in budgets):
            raise ValueError("protocol arm cohort differs from frozen cohort")
        expected_stack = canonical_fingerprint(self.model_stack.model_dump(mode="json"))
        if self.model_stack_fingerprint != expected_stack:
            raise ValueError("protocol model-stack fingerprint mismatch")
        if self.tokenizer_fingerprint != self.model_stack.tokenizer.tokenizer_fingerprint:
            raise ValueError("protocol tokenizer fingerprint mismatch")
        required_keys = {metric.key for metric in self.required_metrics}
        guardrail_ids = [guardrail.guardrail_id for guardrail in self.guardrails]
        if guardrail_ids != sorted(set(guardrail_ids)):
            raise ValueError("guardrail IDs must be sorted and unique")
        unknown = sorted(
            guardrail.metric_key
            for guardrail in self.guardrails
            if guardrail.metric_key not in required_keys
        )
        if unknown:
            raise ValueError(f"guardrails reference unknown required metrics: {unknown}")
        payload = self.model_dump(mode="json")
        fingerprint = payload.pop("manifest_fingerprint")
        if fingerprint != canonical_fingerprint(payload):
            raise ValueError("comparison protocol manifest fingerprint mismatch")
        return self


class OutcomeMetric(StrictModel):
    name: str = Field(min_length=1)
    category: EvidenceCategory
    direction: MetricDirection
    value: float
    unit: str = Field(min_length=1)
    definition_fingerprint: str = Field(pattern=SHA256_PATTERN)
    ci_definition_fingerprint: str = Field(pattern=SHA256_PATTERN)
    ci: ConfidenceInterval
    sample_size: int = Field(ge=1)

    @field_validator("value")
    @classmethod
    def value_is_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("outcome metric value must be finite")
        return value

    @property
    def key(self) -> str:
        return f"{self.category.value}:{self.name}"

    @property
    def definition(self) -> tuple[str, str, str, str, str, float]:
        return (
            self.name,
            self.category.value,
            self.direction.value,
            self.unit,
            self.definition_fingerprint,
            self.ci.confidence_level,
        )


class OutcomeEvidenceSection(StrictModel):
    category: EvidenceCategory
    artifacts: dict[str, EvidenceArtifact]
    metrics: list[OutcomeMetric] = Field(min_length=1)

    @model_validator(mode="after")
    def category_and_keys_are_consistent(self) -> OutcomeEvidenceSection:
        if not self.artifacts or any(not role.strip() for role in self.artifacts):
            raise ValueError("outcome evidence artifacts and roles must be non-empty")
        if any(metric.category != self.category for metric in self.metrics):
            raise ValueError("outcome metric category differs from evidence section")
        keys = [metric.key for metric in self.metrics]
        if keys != sorted(set(keys)):
            raise ValueError("outcome metric keys must be sorted and unique")
        return self


class Task07ArmOutcomeEvidenceManifest(StrictModel):
    """Evidence for one completed arm/seed run; values are never compared here."""

    schema_version: Literal[1]
    contract: Literal["task07-arm-outcome-evidence-v1"]
    run_id: str = Field(min_length=1)
    arm: DPOArm
    seed: int = Field(ge=0, le=2**32 - 1)
    run_status: Literal["completed", "failed", "interrupted"]
    producer_git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    config: ConfigEvidence
    task06_handoff_fingerprint: str = Field(pattern=SHA256_PATTERN)
    task06_selection_preflight_fingerprint: str = Field(pattern=SHA256_PATTERN)
    task07_launch_bundle_fingerprint: str = Field(pattern=SHA256_PATTERN)
    dataset: FrozenIdentity
    selection_policy: FrozenIdentity
    weight_policy: FrozenIdentity
    cohort: FrozenIdentity
    plan: FrozenIdentity
    model_stack: ModelStackIdentity
    model_stack_fingerprint: str = Field(pattern=SHA256_PATTERN)
    tokenizer_fingerprint: str = Field(pattern=SHA256_PATTERN)
    budget: ArmBudget
    evidence: dict[EvidenceCategory, OutcomeEvidenceSection]
    final_tests_used: list[str] = Field(max_length=0)
    manifest_fingerprint: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def identities_provenance_and_fingerprint_are_valid(
        self,
    ) -> Task07ArmOutcomeEvidenceManifest:
        if self.budget.arm != self.arm:
            raise ValueError("outcome arm differs from budget arm")
        if set(self.evidence) != set(EvidenceCategory):
            raise ValueError("outcome must contain all five separated evidence categories")
        if any(key != section.category for key, section in self.evidence.items()):
            raise ValueError("outcome evidence key differs from section category")
        if self.model_stack_fingerprint != canonical_fingerprint(
            self.model_stack.model_dump(mode="json")
        ):
            raise ValueError("outcome model-stack fingerprint mismatch")
        if self.tokenizer_fingerprint != self.model_stack.tokenizer.tokenizer_fingerprint:
            raise ValueError("outcome tokenizer fingerprint mismatch")
        artifacts = [
            self.config.artifact,
            *(
                artifact
                for section in self.evidence.values()
                for artifact in section.artifacts.values()
            ),
        ]
        if any(
            artifact.provenance.producer_git_commit != self.producer_git_commit
            for artifact in artifacts
        ):
            raise ValueError("outcome artifact provenance commit drift")
        if any(artifact.provenance.source_task != "Task 07" for artifact in artifacts):
            raise ValueError("outcome artifact source-task provenance drift")
        payload = self.model_dump(mode="json")
        fingerprint = payload.pop("manifest_fingerprint")
        if fingerprint != canonical_fingerprint(payload):
            raise ValueError("arm outcome manifest fingerprint mismatch")
        return self


class ComparisonPreflightArtifactSummary(StrictModel):
    path: Literal["comparison_preflight.json"]
    sha256: str = Field(pattern=SHA256_PATTERN)
    outcome_manifest_count: int = Field(ge=6)
    arm_count: Literal[3]
    seed_count: int = Field(ge=2)


class Task07ComparisonPreflightBundleManifest(StrictModel):
    schema_version: Literal[1]
    contract: Literal["task07-comparison-preflight-bundle-v1"]
    status: Literal["ready_for_future_task07_comparison_not_compared"]
    protocol_id: str = Field(min_length=1)
    protocol_fingerprint: str = Field(pattern=SHA256_PATTERN)
    preflight: ComparisonPreflightArtifactSummary
    comparison_started: Literal[False]
    selection_performed: Literal[False]
    promotion_performed: Literal[False]
    model_loading_performed: Literal[False]
    training_started: Literal[False]
    evaluation_started: Literal[False]
    final_tests_used: list[str] = Field(max_length=0)
    bundle_fingerprint: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def fingerprint_is_valid(self) -> Task07ComparisonPreflightBundleManifest:
        payload = self.model_dump(mode="json")
        fingerprint = payload.pop("bundle_fingerprint")
        if fingerprint != canonical_fingerprint(payload):
            raise ValueError("comparison preflight bundle fingerprint mismatch")
        return self


def _reject_final_test_path(path: Path) -> None:
    candidates = (path, path.resolve(strict=False))
    normalized = [
        "/".join(part.casefold().replace(" ", "_") for part in candidate.parts)
        for candidate in candidates
    ]
    if any(marker in value for marker in _FORBIDDEN_FINAL_TEST_MARKERS for value in normalized):
        raise ValueError(f"final-test path is forbidden and was not opened: {path}")


def _load_object(path: Path) -> dict[str, Any]:
    _reject_final_test_path(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _manifest_payload_fingerprint(payload: Mapping[str, Any], field: str) -> str:
    return canonical_fingerprint({key: value for key, value in payload.items() if key != field})


def _resolve_artifact(manifest_path: Path, artifact: EvidenceArtifact) -> Path:
    declared = Path(artifact.path)
    path = declared if declared.is_absolute() else manifest_path.parent / declared
    _reject_final_test_path(path)
    return path


def _record_count(path: Path, method: str) -> int:
    if method == "not_applicable":
        return 0
    if method == "jsonl":
        return sum(1 for _ in read_records(path))
    value = json.loads(path.read_text(encoding="utf-8"))
    if method == "single_json":
        if not isinstance(value, dict):
            raise ValueError(f"{path}: expected a single JSON object")
        return 1
    if method == "json_array":
        if not isinstance(value, list):
            raise ValueError(f"{path}: expected a JSON array")
        return len(value)
    raise ValueError(f"unsupported record count method: {method}")


def _verify_artifact(manifest_path: Path, artifact: EvidenceArtifact, role: str) -> Path:
    path = _resolve_artifact(manifest_path, artifact)
    if not path.is_file():
        raise ValueError(f"missing outcome artifact for {role}: {path}")
    if file_sha256(path) != artifact.sha256:
        raise ValueError(f"outcome artifact SHA-256 drift for {role}")
    if _record_count(path, artifact.record_count_method) != artifact.record_count:
        raise ValueError(f"outcome artifact record-count drift for {role}")
    return path


def _verify_config(
    manifest_path: Path,
    outcome: Task07ArmOutcomeEvidenceManifest,
    expected_comparison_fingerprint: str,
) -> None:
    path = _verify_artifact(manifest_path, outcome.config.artifact, "config")
    if outcome.config.format == "json":
        value = json.loads(path.read_text(encoding="utf-8"))
    else:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("outcome config must contain a mapping")
    payload = dict(value)
    if payload.get("seed") != outcome.seed or payload.get("arm") != outcome.arm.value:
        raise ValueError("outcome config arm or seed drift")
    if canonical_fingerprint(payload) != outcome.config.fingerprint:
        raise ValueError("outcome config fingerprint drift")
    comparison_payload = {key: item for key, item in payload.items() if key != "seed"}
    if canonical_fingerprint(comparison_payload) != outcome.config.comparison_fingerprint:
        raise ValueError("outcome config comparison fingerprint drift")
    if outcome.config.comparison_fingerprint != expected_comparison_fingerprint:
        raise ValueError("outcome config differs from frozen arm configuration")


def _load_protocol(path: Path) -> Task07ComparisonProtocolManifest:
    return Task07ComparisonProtocolManifest.model_validate(_load_object(path))


def _validate_task06_handoff(path: Path) -> tuple[Task06PreferenceManifest, str]:
    raw = _load_object(path)
    manifest = Task06PreferenceManifest.model_validate(raw)
    if manifest.manifest_fingerprint != _manifest_payload_fingerprint(raw, "manifest_fingerprint"):
        raise ValueError("Task 06 handoff manifest fingerprint mismatch")
    resolved: dict[str, Path] = {}
    for name, artifact in manifest.artifacts.items():
        declared = Path(artifact.path)
        artifact_path = declared if declared.is_absolute() else path.parent / declared
        _reject_final_test_path(artifact_path)
        resolved[name] = artifact_path
    validate_dpo_dataset(
        task06_manifest_path=path,
        preference_train_path=resolved["preference_train"],
        preference_dev_path=resolved["preference_dev"],
        continued_sft_train_path=resolved["continued_sft_train"],
        continued_sft_dev_path=resolved["continued_sft_dev"],
        weighted_sft_train_path=resolved["weighted_sft_train"],
        weighted_sft_dev_path=resolved["weighted_sft_dev"],
    )
    return manifest, file_sha256(path)


def _validate_selection_preflight(
    path: Path,
) -> tuple[PreferenceSelectionPreflightBundleManifest, str]:
    manifest = PreferenceSelectionPreflightBundleManifest.model_validate(_load_object(path))
    preflight_path = path.parent / manifest.preflight.path
    _reject_final_test_path(preflight_path)
    if not preflight_path.is_file() or file_sha256(preflight_path) != manifest.preflight.sha256:
        raise ValueError("Task 06 selection-preflight artifact SHA-256 drift")
    return manifest, file_sha256(path)


def _validate_launch(path: Path) -> tuple[Task07LaunchManifest, str]:
    return Task07LaunchManifest.model_validate(_load_object(path)), file_sha256(path)


def _validate_protocol_inputs(
    *,
    protocol: Task07ComparisonProtocolManifest,
    task06: Task06PreferenceManifest,
    task06_hash: str,
    selection: PreferenceSelectionPreflightBundleManifest,
    selection_hash: str,
    launch: Task07LaunchManifest,
    launch_hash: str,
) -> None:
    actual_manifests = {
        "Task 06 handoff": (task06_hash, task06.manifest_fingerprint),
        "Task 06 selection preflight": (selection_hash, selection.bundle_fingerprint),
        "Task 07 launch bundle": (launch_hash, launch.bundle_fingerprint),
    }
    expected_manifests = {
        "Task 06 handoff": protocol.task06_handoff,
        "Task 06 selection preflight": protocol.task06_selection_preflight,
        "Task 07 launch bundle": protocol.task07_launch_bundle,
    }
    for label, expected in expected_manifests.items():
        if actual_manifests[label] != (expected.sha256, expected.fingerprint):
            raise ValueError(f"{label} hash or fingerprint drift")
    if launch.input_hashes["task06_manifest"] != task06_hash:
        raise ValueError("Task 07 launch does not bind the supplied Task 06 handoff")
    expected_identity = (
        protocol.dataset.identity_id,
        protocol.dataset.fingerprint,
        protocol.selection_policy.identity_id,
        protocol.selection_policy.fingerprint,
        protocol.weight_policy.identity_id,
        protocol.weight_policy.fingerprint,
        protocol.cohort.fingerprint,
        protocol.plan.identity_id,
        protocol.plan.fingerprint,
    )
    actual_identity = (
        launch.dataset_id,
        launch.dataset_fingerprint,
        launch.selection_policy_id,
        launch.selection_policy_fingerprint,
        launch.weight_policy_id,
        launch.weight_policy_fingerprint,
        launch.cohort_fingerprint,
        launch.plan_id,
        launch.plan_fingerprint,
    )
    if actual_identity != expected_identity:
        raise ValueError("dataset, policy, cohort or plan drift against comparison protocol")
    if (task06.dataset_id, task06.dataset_fingerprint) != (
        protocol.dataset.identity_id,
        protocol.dataset.fingerprint,
    ):
        raise ValueError("Task 06 handoff dataset drift")
    if (task06.selection_policy_id, task06.selection_policy_fingerprint) != (
        protocol.selection_policy.identity_id,
        protocol.selection_policy.fingerprint,
    ):
        raise ValueError("Task 06 handoff selection-policy drift")
    if selection.policy_fingerprint != protocol.selection_policy.fingerprint:
        raise ValueError("Task 06 selection-preflight policy drift")
    if selection.context.dataset_fingerprint != protocol.dataset.fingerprint:
        raise ValueError("Task 06 selection-preflight dataset drift")
    if launch.start_model != protocol.model_stack or launch.reference_model != protocol.model_stack:
        raise ValueError("Task 07 launch model-stack drift")
    if protocol.model_stack_fingerprint != canonical_fingerprint(
        launch.start_model.model_dump(mode="json")
    ):
        raise ValueError("Task 07 launch model-stack fingerprint drift")
    if launch.start_model.tokenizer.tokenizer_fingerprint != protocol.tokenizer_fingerprint:
        raise ValueError("Task 07 launch tokenizer fingerprint drift")
    for arm in DPOArm:
        if launch.arms[arm].budget != protocol.arms[arm].budget:
            raise ValueError(f"Task 07 launch budget drift for arm {arm.value}")


def _load_outcome(path: Path) -> Task07ArmOutcomeEvidenceManifest:
    raw = _load_object(path)
    raw_evidence = raw.get("evidence")
    if not isinstance(raw_evidence, Mapping):
        raise ValueError("outcome evidence sections are missing")
    observed = {str(key) for key in raw_evidence}
    required = {category.value for category in EvidenceCategory}
    missing = sorted(required - observed)
    extra = sorted(observed - required)
    if missing or extra:
        raise ValueError(
            f"outcome evidence category coverage mismatch: missing={missing}, extra={extra}"
        )
    return Task07ArmOutcomeEvidenceManifest.model_validate(raw)


def _validate_outcome(
    path: Path,
    outcome: Task07ArmOutcomeEvidenceManifest,
    protocol: Task07ComparisonProtocolManifest,
) -> None:
    if outcome.run_status != "completed":
        raise ValueError(f"run is not completed for {outcome.arm.value}/{outcome.seed}")
    expected_arm = protocol.arms[outcome.arm]
    if (
        outcome.config.artifact.provenance.source_manifest_sha256
        != protocol.task07_launch_bundle.sha256
    ):
        raise ValueError("outcome config provenance source-manifest drift")
    _verify_config(path, outcome, expected_arm.config_comparison_fingerprint)
    expected_identity = (
        protocol.task06_handoff.fingerprint,
        protocol.task06_selection_preflight.fingerprint,
        protocol.task07_launch_bundle.fingerprint,
        protocol.dataset,
        protocol.selection_policy,
        protocol.weight_policy,
        protocol.cohort,
        protocol.plan,
        protocol.model_stack,
        protocol.model_stack_fingerprint,
        protocol.tokenizer_fingerprint,
        expected_arm.budget,
    )
    actual_identity = (
        outcome.task06_handoff_fingerprint,
        outcome.task06_selection_preflight_fingerprint,
        outcome.task07_launch_bundle_fingerprint,
        outcome.dataset,
        outcome.selection_policy,
        outcome.weight_policy,
        outcome.cohort,
        outcome.plan,
        outcome.model_stack,
        outcome.model_stack_fingerprint,
        outcome.tokenizer_fingerprint,
        outcome.budget,
    )
    if actual_identity != expected_identity:
        raise ValueError(
            f"dataset/cohort/plan/policy/model/tokenizer/budget drift for "
            f"{outcome.arm.value}/{outcome.seed}"
        )
    required_metrics = {metric.key: metric for metric in protocol.required_metrics}
    observed_metrics = {
        metric.key: metric for section in outcome.evidence.values() for metric in section.metrics
    }
    missing_metrics = sorted(set(required_metrics) - set(observed_metrics))
    extra_metrics = sorted(set(observed_metrics) - set(required_metrics))
    if missing_metrics or extra_metrics:
        raise ValueError(
            f"metric coverage mismatch for {outcome.arm.value}/{outcome.seed}: "
            f"missing={missing_metrics}, extra={extra_metrics}"
        )
    for key, requirement in required_metrics.items():
        metric = observed_metrics[key]
        if metric.definition != requirement.definition:
            raise ValueError(f"metric definition or CI drift for {key}")
        if metric.ci_definition_fingerprint != requirement.ci_definition_fingerprint:
            raise ValueError(f"CI definition drift for {key}")
        if metric.sample_size < requirement.minimum_sample_size:
            raise ValueError(f"sample size below pinned minimum for {key}")
    for category, required_roles in protocol.required_artifact_roles.items():
        section = outcome.evidence[category]
        missing_roles = sorted(set(required_roles) - set(section.artifacts))
        extra_roles = sorted(set(section.artifacts) - set(required_roles))
        if missing_roles or extra_roles:
            raise ValueError(
                f"{category.value} evidence artifact coverage mismatch: "
                f"missing={missing_roles}, extra={extra_roles}"
            )
        for role, artifact in sorted(section.artifacts.items()):
            if artifact.provenance.source_manifest_sha256 != protocol.task07_launch_bundle.sha256:
                raise ValueError(f"outcome artifact provenance source-manifest drift for {role}")
            _verify_artifact(path, artifact, f"{category.value}:{role}")


def prepare_task07_comparison_preflight(
    *,
    protocol_manifest_path: Path,
    task06_handoff_manifest_path: Path,
    task06_selection_preflight_manifest_path: Path,
    task07_launch_manifest_path: Path,
    outcome_manifest_paths: Sequence[Path],
    output_dir: Path,
) -> Task07ComparisonPreflightBundleManifest:
    """Validate complete comparable evidence and publish no experimental comparison."""
    all_input_paths = [
        protocol_manifest_path,
        task06_handoff_manifest_path,
        task06_selection_preflight_manifest_path,
        task07_launch_manifest_path,
        *outcome_manifest_paths,
    ]
    for path in all_input_paths:
        _reject_final_test_path(path)
    if output_dir.exists():
        raise FileExistsError(f"Task 07 comparison-preflight output already exists: {output_dir}")
    if not outcome_manifest_paths:
        raise ValueError("explicit Task 07 outcome manifests are required")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"
    try:
        protocol = _load_protocol(protocol_manifest_path)
        task06, task06_hash = _validate_task06_handoff(task06_handoff_manifest_path)
        selection, selection_hash = _validate_selection_preflight(
            task06_selection_preflight_manifest_path
        )
        launch, launch_hash = _validate_launch(task07_launch_manifest_path)
        _validate_protocol_inputs(
            protocol=protocol,
            task06=task06,
            task06_hash=task06_hash,
            selection=selection,
            selection_hash=selection_hash,
            launch=launch,
            launch_hash=launch_hash,
        )

        outcomes: dict[tuple[DPOArm, int], tuple[Task07ArmOutcomeEvidenceManifest, Path]] = {}
        for path in outcome_manifest_paths:
            outcome = _load_outcome(path)
            key = (outcome.arm, outcome.seed)
            if key in outcomes:
                raise ValueError(
                    f"duplicate arm/seed outcome evidence: {outcome.arm.value}/{outcome.seed}"
                )
            outcomes[key] = (outcome, path)
        expected = {(arm, seed) for arm in DPOArm for seed in protocol.required_seeds}
        observed = set(outcomes)
        missing = sorted((arm.value, seed) for arm, seed in expected - observed)
        extra = sorted((arm.value, seed) for arm, seed in observed - expected)
        if missing or extra:
            raise ValueError(f"arm x seed coverage mismatch: missing={missing}, extra={extra}")
        for key in sorted(outcomes, key=lambda item: (item[0].value, item[1])):
            outcome, path = outcomes[key]
            _validate_outcome(path, outcome, protocol)

        evidence_index = [
            {
                "arm": arm.value,
                "seed": seed,
                "run_id": outcomes[(arm, seed)][0].run_id,
                "run_status": outcomes[(arm, seed)][0].run_status,
                "manifest_sha256": file_sha256(outcomes[(arm, seed)][1]),
                "manifest_fingerprint": outcomes[(arm, seed)][0].manifest_fingerprint,
            }
            for arm, seed in sorted(expected, key=lambda item: (item[0].value, item[1]))
        ]
        preflight_payload: dict[str, Any] = {
            "schema_version": 1,
            "contract": PREFLIGHT_CONTRACT,
            "status": PREFLIGHT_STATUS,
            "protocol_id": protocol.protocol_id,
            "protocol_fingerprint": protocol.manifest_fingerprint,
            "input_manifest_hashes": {
                "protocol": file_sha256(protocol_manifest_path),
                "task06_handoff": task06_hash,
                "task06_selection_preflight": selection_hash,
                "task07_launch_bundle": launch_hash,
            },
            "evidence_index": evidence_index,
            "validated_arms": sorted(arm.value for arm in DPOArm),
            "validated_seeds": protocol.required_seeds,
            "required_metric_keys": sorted(metric.key for metric in protocol.required_metrics),
            "required_artifact_roles": {
                category.value: roles
                for category, roles in sorted(
                    protocol.required_artifact_roles.items(), key=lambda item: item[0].value
                )
            },
            "guardrail_ids_validated_not_applied": [
                guardrail.guardrail_id for guardrail in protocol.guardrails
            ],
            "comparison_started": False,
            "selection_performed": False,
            "promotion_performed": False,
            "model_loading_performed": False,
            "training_started": False,
            "evaluation_started": False,
            "final_tests_used": [],
        }
        staging.mkdir()
        preflight_path = staging / "comparison_preflight.json"
        write_json(preflight_path, preflight_payload)
        bundle_payload: dict[str, Any] = {
            "schema_version": 1,
            "contract": BUNDLE_CONTRACT,
            "status": PREFLIGHT_STATUS,
            "protocol_id": protocol.protocol_id,
            "protocol_fingerprint": protocol.manifest_fingerprint,
            "preflight": {
                "path": "comparison_preflight.json",
                "sha256": file_sha256(preflight_path),
                "outcome_manifest_count": len(outcomes),
                "arm_count": len(DPOArm),
                "seed_count": len(protocol.required_seeds),
            },
            "comparison_started": False,
            "selection_performed": False,
            "promotion_performed": False,
            "model_loading_performed": False,
            "training_started": False,
            "evaluation_started": False,
            "final_tests_used": [],
        }
        bundle_payload["bundle_fingerprint"] = canonical_fingerprint(bundle_payload)
        manifest = Task07ComparisonPreflightBundleManifest.model_validate(bundle_payload)
        write_json(staging / "manifest.json", manifest.model_dump(mode="json"))
        if output_dir.exists():
            raise FileExistsError(
                f"Task 07 comparison-preflight output already exists: {output_dir}"
            )
        os.replace(staging, output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
