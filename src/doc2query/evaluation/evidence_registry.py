"""Model-free, fail-closed evidence registry for the future Task 09 campaign.

The module consumes only explicitly supplied evidence manifests.  It never discovers
artifacts, opens final-test paths, loads models, evaluates runs or selects a winner.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from statistics import fmean
from typing import Any, Literal

import yaml
from pydantic import Field, field_validator, model_validator

from doc2query.evaluation.statistical_contract import build_budget_manifest
from doc2query.schemas import StrictModel
from doc2query.training.dpo import (
    SHA256_PATTERN,
    AdapterIdentity,
    BaseModelIdentity,
    TokenizerIdentity,
    canonical_fingerprint,
    file_sha256,
)
from doc2query.utils.records import write_json

EVIDENCE_CONTRACT = "task09-experiment-evidence-v1"
REGISTRY_CONTRACT = "task09-campaign-evidence-registry-v1"
REGISTRY_STATUS = "registry_ready_for_future_stage_review_no_selection"
INCOMPLETE_STATUS = "evidence_incomplete_not_ranked"
PARETO_STATUS = "pareto_front_computed_no_selection"

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


class MetricCategory(StrEnum):
    INTRINSIC = "intrinsic"
    PROBE_EXTRINSIC = "probe_extrinsic"
    HUMAN = "human"
    COST = "cost"


class MetricDirection(StrEnum):
    MIN = "min"
    MAX = "max"


class ConfidenceInterval(StrictModel):
    lower: float
    upper: float
    confidence_level: float = Field(gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def finite_and_ordered(self) -> ConfidenceInterval:
        if not all(math.isfinite(value) for value in (self.lower, self.upper)):
            raise ValueError("confidence interval bounds must be finite")
        if self.lower > self.upper:
            raise ValueError("confidence interval lower bound exceeds upper bound")
        return self


class MetricEvidence(StrictModel):
    name: str = Field(min_length=1)
    category: MetricCategory
    direction: MetricDirection
    value: float
    unit: str = Field(min_length=1)
    definition_fingerprint: str = Field(pattern=SHA256_PATTERN)
    ci: ConfidenceInterval | None = None
    sample_size: int | None = Field(default=None, ge=1)

    @field_validator("value")
    @classmethod
    def value_is_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("metric value must be finite")
        return value

    @property
    def key(self) -> str:
        return f"{self.category.value}:{self.name}"

    @property
    def definition(self) -> tuple[str, str, str, str, str]:
        return (
            self.name,
            self.category.value,
            self.direction.value,
            self.unit,
            self.definition_fingerprint,
        )


class MetricRequirement(StrictModel):
    name: str = Field(min_length=1)
    category: MetricCategory
    direction: MetricDirection
    unit: str = Field(min_length=1)
    definition_fingerprint: str = Field(pattern=SHA256_PATTERN)
    ci_required: bool = True
    sample_size_required: bool = True

    @property
    def key(self) -> str:
        return f"{self.category.value}:{self.name}"

    @property
    def definition(self) -> tuple[str, str, str, str, str]:
        return (
            self.name,
            self.category.value,
            self.direction.value,
            self.unit,
            self.definition_fingerprint,
        )


class ArtifactProvenance(StrictModel):
    source_task: str = Field(min_length=1)
    source_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    producer_git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")


class EvidenceArtifact(StrictModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=SHA256_PATTERN)
    record_count: int = Field(ge=0)
    record_count_method: Literal["jsonl", "json_array", "single_json", "not_applicable"]
    provenance: ArtifactProvenance
    artifact_fingerprint: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def descriptor_fingerprint_is_valid(self) -> EvidenceArtifact:
        payload = self.model_dump(mode="json")
        fingerprint = payload.pop("artifact_fingerprint")
        if fingerprint != canonical_fingerprint(payload):
            raise ValueError("artifact descriptor fingerprint mismatch")
        if self.record_count_method == "not_applicable" and self.record_count != 0:
            raise ValueError("not_applicable artifacts must declare record_count=0")
        if self.record_count_method == "single_json" and self.record_count != 1:
            raise ValueError("single_json artifacts must declare record_count=1")
        return self


class ConfigEvidence(StrictModel):
    artifact: EvidenceArtifact
    format: Literal["json", "yaml"]
    fingerprint: str = Field(pattern=SHA256_PATTERN)
    comparison_fingerprint: str = Field(pattern=SHA256_PATTERN)


class CampaignBudget(StrictModel):
    definition_version: Literal["task09-campaign-budget-v1"]
    token_count: int = Field(ge=1)
    optimizer_steps: int = Field(ge=0)
    pair_count: int = Field(ge=1)
    unique_passage_count: int = Field(ge=1)
    queries_per_passage: int = Field(ge=1)
    fingerprint: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def dimensions_and_fingerprint_are_valid(self) -> CampaignBudget:
        # Reuse Task 04's exact four-dimensional budget invariant.
        build_budget_manifest(
            token_count=self.token_count,
            pair_count=self.pair_count,
            unique_passage_count=self.unique_passage_count,
            queries_per_passage=self.queries_per_passage,
        )
        payload = self.model_dump(mode="json")
        fingerprint = payload.pop("fingerprint")
        if fingerprint != canonical_fingerprint(payload):
            raise ValueError("campaign budget fingerprint mismatch")
        return self


class EvidenceModelStack(StrictModel):
    base_model: BaseModelIdentity
    adapter: AdapterIdentity | None
    tokenizer: TokenizerIdentity
    fingerprint: str = Field(pattern=SHA256_PATTERN)
    comparison_fingerprint: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def identities_and_fingerprint_are_valid(self) -> EvidenceModelStack:
        if (
            self.adapter is not None
            and self.adapter.base_model_fingerprint != self.base_model.artifact_fingerprint
        ):
            raise ValueError("adapter does not identify the evidence base model")
        payload = self.model_dump(mode="json")
        comparison_fingerprint = payload.pop("comparison_fingerprint")
        fingerprint = payload.pop("fingerprint")
        if fingerprint != canonical_fingerprint(payload):
            raise ValueError("model stack fingerprint mismatch")
        comparison_payload = {
            "base_model": payload["base_model"],
            "adapter": (
                {
                    "adapter_id": payload["adapter"]["adapter_id"],
                    "base_model_fingerprint": payload["adapter"]["base_model_fingerprint"],
                }
                if payload["adapter"] is not None
                else None
            ),
            "tokenizer": payload["tokenizer"],
        }
        if comparison_fingerprint != canonical_fingerprint(comparison_payload):
            raise ValueError("model stack comparison fingerprint mismatch")
        return self


class ExperimentEvidenceManifest(StrictModel):
    """Versioned evidence for one experiment arm and one seed."""

    schema_version: Literal[1]
    contract: Literal["task09-experiment-evidence-v1"]
    experiment_id: str = Field(min_length=1)
    arm_id: str = Field(min_length=1)
    stage_id: str = Field(min_length=1)
    run_status: Literal["completed", "failed", "interrupted"]
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    config: ConfigEvidence
    dataset_id: str = Field(min_length=1)
    dataset_fingerprint: str = Field(pattern=SHA256_PATTERN)
    split_id: str = Field(min_length=1)
    split_fingerprint: str = Field(pattern=SHA256_PATTERN)
    cohort_id: str = Field(min_length=1)
    cohort_fingerprint: str = Field(pattern=SHA256_PATTERN)
    model_stack: EvidenceModelStack
    seed: int = Field(ge=0, le=2**32 - 1)
    budget: CampaignBudget
    probe_recipe_fingerprint: str = Field(pattern=SHA256_PATTERN)
    artifacts: dict[str, EvidenceArtifact]
    metrics: list[MetricEvidence]
    final_tests_used: list[str] = Field(max_length=0)
    manifest_fingerprint: str = Field(pattern=SHA256_PATTERN)

    @field_validator("artifacts")
    @classmethod
    def artifact_roles_are_non_empty(
        cls, value: dict[str, EvidenceArtifact]
    ) -> dict[str, EvidenceArtifact]:
        if any(not role.strip() for role in value):
            raise ValueError("artifact roles must be non-empty")
        return value

    @field_validator("metrics")
    @classmethod
    def metric_keys_are_unique(cls, value: list[MetricEvidence]) -> list[MetricEvidence]:
        keys = [metric.key for metric in value]
        if len(keys) != len(set(keys)):
            raise ValueError("metric category/name pairs must be unique")
        return value

    @model_validator(mode="after")
    def fingerprint_is_valid(self) -> ExperimentEvidenceManifest:
        payload = self.model_dump(mode="json")
        fingerprint = payload.pop("manifest_fingerprint")
        if fingerprint != canonical_fingerprint(payload):
            raise ValueError("experiment evidence manifest fingerprint mismatch")
        declared_artifacts = [self.config.artifact, *self.artifacts.values()]
        if any(
            artifact.provenance.producer_git_commit != self.git_commit
            for artifact in declared_artifacts
        ):
            raise ValueError("artifact provenance commit differs from evidence run commit")
        return self


class CampaignEvidenceRequirements(StrictModel):
    schema_version: Literal[1]
    contract: Literal["task09-evidence-requirements-v1"]
    required_seeds: list[int] = Field(min_length=1)
    required_metrics: list[MetricRequirement] = Field(min_length=1)
    required_artifact_roles: list[str] = Field(default_factory=list)
    require_human_evidence: bool = True

    @field_validator("required_seeds")
    @classmethod
    def seeds_are_sorted_and_unique(cls, value: list[int]) -> list[int]:
        if value != sorted(set(value)):
            raise ValueError("required seeds must be sorted and unique")
        return value

    @field_validator("required_metrics")
    @classmethod
    def metric_requirements_are_unique(
        cls, value: list[MetricRequirement]
    ) -> list[MetricRequirement]:
        keys = [metric.key for metric in value]
        if len(keys) != len(set(keys)):
            raise ValueError("required metric category/name pairs must be unique")
        return value

    @field_validator("required_artifact_roles")
    @classmethod
    def roles_are_sorted_and_unique(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)) or any(not role.strip() for role in value):
            raise ValueError("required artifact roles must be non-empty, sorted and unique")
        return value


class ParetoCandidate(StrictModel):
    arm_id: str = Field(min_length=1)
    comparison_fingerprint: str = Field(pattern=SHA256_PATTERN)
    values: dict[str, float]
    evidence_complete: bool

    @field_validator("values")
    @classmethod
    def values_are_finite(cls, value: dict[str, float]) -> dict[str, float]:
        if any(not math.isfinite(item) for item in value.values()):
            raise ValueError("Pareto metric values must be finite")
        return value


class RegistryArtifactSummary(StrictModel):
    path: Literal["registry.json"]
    sha256: str = Field(pattern=SHA256_PATTERN)
    evidence_manifest_count: int = Field(ge=1)
    arm_count: int = Field(ge=1)


class CampaignRegistryBundleManifest(StrictModel):
    """Immutable publication contract for a registry-only Task 09 bundle."""

    schema_version: Literal[1]
    contract: Literal["task09-campaign-evidence-registry-v1-bundle-v1"]
    status: Literal["registry_ready_for_future_stage_review_no_selection"]
    registry: RegistryArtifactSummary
    campaign_started: Literal[False]
    model_loading_performed: Literal[False]
    training_started: Literal[False]
    evaluation_started: Literal[False]
    selection_performed: Literal[False]
    final_tests_used: list[str] = Field(max_length=0)
    bundle_fingerprint: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def fingerprint_is_valid(self) -> CampaignRegistryBundleManifest:
        payload = self.model_dump(mode="json")
        fingerprint = payload.pop("bundle_fingerprint")
        if fingerprint != canonical_fingerprint(payload):
            raise ValueError("campaign registry bundle fingerprint mismatch")
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


def _resolve_artifact(manifest_path: Path, artifact: EvidenceArtifact) -> Path:
    candidate = Path(artifact.path)
    path = candidate if candidate.is_absolute() else manifest_path.parent / candidate
    _reject_final_test_path(path)
    return path


def _record_count(path: Path, method: str) -> int:
    if method == "not_applicable":
        return 0
    if method == "jsonl":
        count = 0
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: expected a JSON object")
                count += 1
        return count
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
        raise ValueError(f"declared artifact is missing for {role}: {path}")
    if file_sha256(path) != artifact.sha256:
        raise ValueError(f"declared artifact SHA-256 drift for {role}")
    if _record_count(path, artifact.record_count_method) != artifact.record_count:
        raise ValueError(f"declared artifact record-count drift for {role}")
    return path


def _verify_config(manifest_path: Path, config: ConfigEvidence, seed: int) -> None:
    path = _verify_artifact(manifest_path, config.artifact, "config")
    if config.format == "json":
        value = json.loads(path.read_text(encoding="utf-8"))
    else:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("evidence config must contain a mapping")
    config_payload = dict(value)
    if config_payload.get("seed") != seed:
        raise ValueError("config seed differs from evidence manifest seed")
    if canonical_fingerprint(config_payload) != config.fingerprint:
        raise ValueError("config fingerprint drift")
    comparison_payload = {key: item for key, item in config_payload.items() if key != "seed"}
    if canonical_fingerprint(comparison_payload) != config.comparison_fingerprint:
        raise ValueError("config comparison fingerprint drift")


def load_experiment_evidence(path: Path) -> ExperimentEvidenceManifest:
    """Load and verify one explicitly supplied manifest and all declared artifacts."""
    raw = _load_object(path)
    manifest = ExperimentEvidenceManifest.model_validate(raw)
    _verify_config(path, manifest.config, manifest.seed)
    for role, artifact in sorted(manifest.artifacts.items()):
        _verify_artifact(path, artifact, role)
    return manifest


def _metric_definition_map(
    manifest: ExperimentEvidenceManifest,
) -> dict[str, tuple[str, str, str, str, str]]:
    return {metric.key: metric.definition for metric in manifest.metrics}


def _seed_signature(manifest: ExperimentEvidenceManifest) -> dict[str, Any]:
    return {
        "stage_id": manifest.stage_id,
        "git_commit": manifest.git_commit,
        "config_comparison_fingerprint": manifest.config.comparison_fingerprint,
        "dataset_id": manifest.dataset_id,
        "dataset_fingerprint": manifest.dataset_fingerprint,
        "split_id": manifest.split_id,
        "split_fingerprint": manifest.split_fingerprint,
        "cohort_id": manifest.cohort_id,
        "cohort_fingerprint": manifest.cohort_fingerprint,
        "budget": manifest.budget.model_dump(mode="json"),
        "probe_recipe_fingerprint": manifest.probe_recipe_fingerprint,
        "model_stack_comparison_fingerprint": manifest.model_stack.comparison_fingerprint,
        "base_model": manifest.model_stack.base_model.model_dump(mode="json"),
        "adapter": (
            manifest.model_stack.adapter.model_dump(mode="json")
            if manifest.model_stack.adapter is not None
            else None
        ),
        "tokenizer": manifest.model_stack.tokenizer.model_dump(mode="json"),
        "metric_definitions": _metric_definition_map(manifest),
    }


def _comparison_signature(manifest: ExperimentEvidenceManifest) -> dict[str, Any]:
    """Fields that must match across deliberately different experimental arms."""
    return {
        "experiment_id": manifest.experiment_id,
        "stage_id": manifest.stage_id,
        "git_commit": manifest.git_commit,
        "dataset_id": manifest.dataset_id,
        "dataset_fingerprint": manifest.dataset_fingerprint,
        "split_id": manifest.split_id,
        "split_fingerprint": manifest.split_fingerprint,
        "cohort_id": manifest.cohort_id,
        "cohort_fingerprint": manifest.cohort_fingerprint,
        "budget": manifest.budget.model_dump(mode="json"),
        "probe_recipe_fingerprint": manifest.probe_recipe_fingerprint,
        "metric_definitions": _metric_definition_map(manifest),
    }


def _mapping_drift(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[str]:
    return sorted(key for key in set(left) | set(right) if left.get(key) != right.get(key))


def compute_pareto_front(
    candidates: Sequence[ParetoCandidate],
    metric_requirements: Sequence[MetricRequirement],
) -> dict[str, Any]:
    """Return a direction-aware Pareto front without scalarization or selection."""
    required = {metric.key: metric.direction for metric in metric_requirements}
    fingerprints = {candidate.comparison_fingerprint for candidate in candidates}
    incomplete = (
        not candidates
        or len(fingerprints) != 1
        or any(not candidate.evidence_complete for candidate in candidates)
        or any(set(candidate.values) != set(required) for candidate in candidates)
    )
    if incomplete:
        return {
            "status": INCOMPLETE_STATUS,
            "pareto_front_arm_ids": [],
            "dominated_arm_ids": [],
            "selection_performed": False,
        }

    def dominates(left: ParetoCandidate, right: ParetoCandidate) -> bool:
        weakly_better = True
        strictly_better = False
        for key, direction in required.items():
            left_value = left.values[key]
            right_value = right.values[key]
            if direction == MetricDirection.MAX:
                weakly_better &= left_value >= right_value
                strictly_better |= left_value > right_value
            else:
                weakly_better &= left_value <= right_value
                strictly_better |= left_value < right_value
        return weakly_better and strictly_better

    front = sorted(
        candidate.arm_id
        for candidate in candidates
        if not any(
            other.arm_id != candidate.arm_id and dominates(other, candidate) for other in candidates
        )
    )
    dominated = sorted(
        candidate.arm_id for candidate in candidates if candidate.arm_id not in front
    )
    return {
        "status": PARETO_STATUS,
        "pareto_front_arm_ids": front,
        "dominated_arm_ids": dominated,
        "selection_performed": False,
    }


def _arm_summary(
    manifests: Sequence[ExperimentEvidenceManifest],
    requirements: CampaignEvidenceRequirements,
) -> dict[str, Any]:
    first = manifests[0]
    issues: list[str] = []
    required_seed_set = set(requirements.required_seeds)
    observed_seeds = sorted(manifest.seed for manifest in manifests)
    missing_seeds = sorted(required_seed_set - set(observed_seeds))
    unexpected_seeds = sorted(set(observed_seeds) - required_seed_set)
    if missing_seeds:
        issues.append("missing_seeds")
    if unexpected_seeds:
        issues.append("unexpected_seeds")
    if any(manifest.run_status != "completed" for manifest in manifests):
        issues.append("run_not_completed")

    baseline_signature = _seed_signature(first)
    seed_drift: dict[str, list[str]] = {}
    for manifest in manifests[1:]:
        drift = _mapping_drift(baseline_signature, _seed_signature(manifest))
        if drift:
            seed_drift[str(manifest.seed)] = drift
    if seed_drift:
        issues.append("seed_comparability_drift")

    requirements_by_key = {metric.key: metric for metric in requirements.required_metrics}
    missing_metrics: dict[str, list[str]] = {}
    missing_ci: dict[str, list[str]] = {}
    missing_sample_size: dict[str, list[str]] = {}
    metric_definition_drift: dict[str, list[str]] = {}
    missing_artifacts: dict[str, list[str]] = {}
    missing_human_seeds: list[int] = []
    for manifest in manifests:
        metrics = {metric.key: metric for metric in manifest.metrics}
        missing_metrics[str(manifest.seed)] = sorted(set(requirements_by_key) - set(metrics))
        missing_ci[str(manifest.seed)] = sorted(
            key
            for key, requirement in requirements_by_key.items()
            if key in metrics and requirement.ci_required and metrics[key].ci is None
        )
        missing_sample_size[str(manifest.seed)] = sorted(
            key
            for key, requirement in requirements_by_key.items()
            if key in metrics
            and requirement.sample_size_required
            and metrics[key].sample_size is None
        )
        metric_definition_drift[str(manifest.seed)] = sorted(
            key
            for key, requirement in requirements_by_key.items()
            if key in metrics and metrics[key].definition != requirement.definition
        )
        missing_artifacts[str(manifest.seed)] = sorted(
            set(requirements.required_artifact_roles) - set(manifest.artifacts)
        )
        if requirements.require_human_evidence and not any(
            metric.category == MetricCategory.HUMAN for metric in manifest.metrics
        ):
            missing_human_seeds.append(manifest.seed)

    for label, report in (
        ("missing_metrics", missing_metrics),
        ("missing_ci", missing_ci),
        ("missing_sample_size", missing_sample_size),
        ("metric_definition_drift", metric_definition_drift),
        ("missing_artifacts", missing_artifacts),
    ):
        if any(report.values()):
            issues.append(label)
    if missing_human_seeds:
        issues.append("missing_human_evidence")

    evidence_complete = not issues
    metric_means: dict[str, float] = {}
    if evidence_complete:
        for key in sorted(requirements_by_key):
            metric_means[key] = fmean(
                next(metric.value for metric in manifest.metrics if metric.key == key)
                for manifest in manifests
            )

    comparison_signature = _comparison_signature(first)
    return {
        "experiment_id": first.experiment_id,
        "arm_id": first.arm_id,
        "stage_id": first.stage_id,
        "observed_seeds": observed_seeds,
        "missing_seeds": missing_seeds,
        "unexpected_seeds": unexpected_seeds,
        "seed_drift": seed_drift,
        "missing_metrics": missing_metrics,
        "missing_ci": missing_ci,
        "missing_sample_size": missing_sample_size,
        "metric_definition_drift": metric_definition_drift,
        "missing_human_evidence_seeds": sorted(missing_human_seeds),
        "missing_artifacts": missing_artifacts,
        "issues": sorted(set(issues)),
        "evidence_complete": evidence_complete,
        "metric_means": metric_means,
        "comparison_signature": comparison_signature,
        "comparison_fingerprint": canonical_fingerprint(comparison_signature),
        "model_stack_fingerprint": first.model_stack.fingerprint,
        "aggregation": "arithmetic_mean_across_comparable_required_seeds_only",
    }


def _build_registry_payload(
    manifests: Sequence[ExperimentEvidenceManifest],
    manifest_hashes: Mapping[tuple[str, str, int], str],
    requirements: CampaignEvidenceRequirements,
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[ExperimentEvidenceManifest]] = defaultdict(list)
    for manifest in manifests:
        grouped[(manifest.experiment_id, manifest.arm_id)].append(manifest)
    summaries = [
        _arm_summary(sorted(group, key=lambda manifest: manifest.seed), requirements)
        for _, group in sorted(grouped.items())
    ]

    stages: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        stages[(str(summary["experiment_id"]), str(summary["stage_id"]))].append(summary)

    stage_reviews: list[dict[str, Any]] = []
    for (experiment_id, stage_id), arms in sorted(stages.items()):
        arms = sorted(arms, key=lambda arm: str(arm["arm_id"]))
        baseline = arms[0]["comparison_signature"]
        cross_arm_drift = {
            str(arm["arm_id"]): _mapping_drift(baseline, arm["comparison_signature"])
            for arm in arms[1:]
            if _mapping_drift(baseline, arm["comparison_signature"])
        }
        candidates = [
            ParetoCandidate(
                arm_id=str(arm["arm_id"]),
                comparison_fingerprint=str(arm["comparison_fingerprint"]),
                values=dict(arm["metric_means"]),
                evidence_complete=bool(arm["evidence_complete"]) and not cross_arm_drift,
            )
            for arm in arms
        ]
        pareto = compute_pareto_front(candidates, requirements.required_metrics)
        stage_reviews.append(
            {
                "experiment_id": experiment_id,
                "stage_id": stage_id,
                "arm_ids": [arm["arm_id"] for arm in arms],
                "cross_arm_drift": cross_arm_drift,
                "pareto": pareto,
                "continue_stop_decisions": [],
                "selection_performed": False,
            }
        )

    evidence_index = [
        {
            "experiment_id": manifest.experiment_id,
            "arm_id": manifest.arm_id,
            "stage_id": manifest.stage_id,
            "seed": manifest.seed,
            "manifest_sha256": manifest_hashes[
                (manifest.experiment_id, manifest.arm_id, manifest.seed)
            ],
            "manifest_fingerprint": manifest.manifest_fingerprint,
        }
        for manifest in sorted(
            manifests,
            key=lambda item: (item.experiment_id, item.stage_id, item.arm_id, item.seed),
        )
    ]
    return {
        "schema_version": 1,
        "contract": REGISTRY_CONTRACT,
        "status": REGISTRY_STATUS,
        "requirements": requirements.model_dump(mode="json"),
        "requirements_fingerprint": canonical_fingerprint(requirements.model_dump(mode="json")),
        "evidence_index": evidence_index,
        "arm_summaries": summaries,
        "stage_reviews": stage_reviews,
        "campaign_started": False,
        "model_loading_performed": False,
        "training_started": False,
        "evaluation_started": False,
        "selection_performed": False,
        "final_tests_used": [],
    }


def build_campaign_evidence_registry(
    *,
    evidence_manifest_paths: Sequence[Path],
    requirements: CampaignEvidenceRequirements,
    output_dir: Path,
) -> dict[str, Any]:
    """Verify explicit evidence and atomically publish a deterministic registry bundle."""
    if output_dir.exists():
        raise FileExistsError(f"Task 09 registry output already exists: {output_dir}")
    if not evidence_manifest_paths:
        raise ValueError("at least one explicit evidence manifest is required")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"
    try:
        manifests: list[ExperimentEvidenceManifest] = []
        hashes: dict[tuple[str, str, int], str] = {}
        for path in evidence_manifest_paths:
            manifest = load_experiment_evidence(path)
            key = (manifest.experiment_id, manifest.arm_id, manifest.seed)
            if key in hashes:
                raise ValueError(f"duplicate experiment_id/arm_id/seed: {key}")
            hashes[key] = file_sha256(path)
            manifests.append(manifest)

        registry = _build_registry_payload(manifests, hashes, requirements)
        staging.mkdir()
        registry_path = staging / "registry.json"
        write_json(registry_path, registry)
        bundle_manifest: dict[str, Any] = {
            "schema_version": 1,
            "contract": f"{REGISTRY_CONTRACT}-bundle-v1",
            "status": REGISTRY_STATUS,
            "registry": {
                "path": "registry.json",
                "sha256": file_sha256(registry_path),
                "evidence_manifest_count": len(manifests),
                "arm_count": len(registry["arm_summaries"]),
            },
            "campaign_started": False,
            "model_loading_performed": False,
            "training_started": False,
            "evaluation_started": False,
            "selection_performed": False,
            "final_tests_used": [],
        }
        bundle_manifest["bundle_fingerprint"] = canonical_fingerprint(bundle_manifest)
        validated_bundle = CampaignRegistryBundleManifest.model_validate(bundle_manifest)
        serialized_bundle = validated_bundle.model_dump(mode="json")
        write_json(staging / "manifest.json", serialized_bundle)
        if output_dir.exists():
            raise FileExistsError(f"Task 09 registry output already exists: {output_dir}")
        os.replace(staging, output_dir)
        return serialized_bundle
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
