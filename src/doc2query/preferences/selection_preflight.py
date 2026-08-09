"""Model-free, fail-closed preflight for future Task 06 candidate selection.

This module validates already materialized evidence and externally frozen policy and
calibration artifacts.  It deliberately cannot score, rank, select, or build pairs.
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

from pydantic import Field, JsonValue, field_validator, model_validator

from doc2query.evaluation.evidence_registry import EvidenceArtifact
from doc2query.preferences.schemas import CandidateEvidenceBundle
from doc2query.schemas import StrictModel
from doc2query.training.dpo import (
    SHA256_PATTERN,
    canonical_fingerprint,
    file_sha256,
    ordered_ids_fingerprint,
)
from doc2query.utils.records import read_records, write_json

POLICY_CONTRACT = "task06-candidate-selection-policy-v1"
CALIBRATION_CONTRACT = "task06-component-calibration-evidence-v1"
HUMAN_CONTRACT = "task06-human-preference-calibration-evidence-v1"
PREFLIGHT_CONTRACT = "task06-preference-selection-preflight-bundle-v1"
PREFLIGHT_STATUS = "ready_for_future_preference_selection_not_selected"

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


class SelectionComponent(StrEnum):
    PRIMARY = "primary"
    SHADOW = "shadow"
    CORPUS_RETRIEVAL = "corpus_retrieval"
    LEXICAL_COPY = "lexical_copy"
    FOCUS = "focus"
    STYLE = "style"
    FORMAT = "format"


class MetricDirection(StrEnum):
    MIN = "min"
    MAX = "max"


_COMPONENT_FIELDS: dict[SelectionComponent, frozenset[str]] = {
    SelectionComponent.PRIMARY: frozenset(
        {"positive_score", "margin", "positive_rank", "best_sentence_score"}
    ),
    SelectionComponent.SHADOW: frozenset(
        {"positive_score", "margin", "positive_rank", "best_sentence_score"}
    ),
    SelectionComponent.CORPUS_RETRIEVAL: frozenset(
        {"source_rank", "reciprocal_rank", "recall_at_1", "recall_at_5", "ndcg_at_10"}
    ),
    SelectionComponent.LEXICAL_COPY: frozenset(
        {
            "content_lemma_jaccard",
            "content_lemma_precision",
            "content_lemma_recall",
            "longest_common_ngram",
            "longest_common_subsequence_ratio",
            "entity_preservation",
            "number_unit_preservation",
            "copy_risk",
        }
    ),
    SelectionComponent.FOCUS: frozenset({"focus_match", "confidence"}),
    SelectionComponent.STYLE: frozenset({"form_match", "intent_match", "confidence"}),
    SelectionComponent.FORMAT: frozenset(
        {
            "valid",
            "empty",
            "single_query",
            "has_meta_commentary",
            "too_long",
            "contains_answer",
        }
    ),
}


def _finite_json(value: JsonValue, *, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} contains a non-finite value")
    if isinstance(value, list):
        for item in value:
            _finite_json(item, label=label)
    elif isinstance(value, dict):
        for item in value.values():
            _finite_json(item, label=label)


class FrozenContext(StrictModel):
    dataset_id: str = Field(min_length=1)
    dataset_fingerprint: str = Field(pattern=SHA256_PATTERN)
    split_id: Literal["train", "dev"]
    split_fingerprint: str = Field(pattern=SHA256_PATTERN)
    cohort_id: str = Field(min_length=1)
    cohort_fingerprint: str = Field(pattern=SHA256_PATTERN)
    candidate_ids_fingerprint: str = Field(pattern=SHA256_PATTERN)
    candidate_count: int = Field(ge=1)


class NormalizationDefinition(StrictModel):
    method_id: str = Field(min_length=1)
    definition_fingerprint: str = Field(pattern=SHA256_PATTERN)
    parameters: dict[str, JsonValue]

    @field_validator("parameters")
    @classmethod
    def parameters_are_explicit_and_finite(
        cls, value: dict[str, JsonValue]
    ) -> dict[str, JsonValue]:
        if not value:
            raise ValueError("normalization parameters must be explicitly pinned")
        _finite_json(value, label="normalization parameters")
        return value


class MetricSelectionRule(StrictModel):
    metric_id: str = Field(min_length=1)
    source_field: str = Field(min_length=1)
    direction: MetricDirection
    normalization: NormalizationDefinition
    weight: float
    thresholds: dict[str, float]
    threshold_definition_fingerprint: str = Field(pattern=SHA256_PATTERN)
    calibration_fingerprint: str = Field(pattern=SHA256_PATTERN)

    @field_validator("weight")
    @classmethod
    def weight_is_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("metric weight must be finite")
        return value

    @field_validator("thresholds")
    @classmethod
    def thresholds_are_pinned_and_finite(cls, value: dict[str, float]) -> dict[str, float]:
        if not value or any(not key.strip() for key in value):
            raise ValueError("metric thresholds must be explicitly named and pinned")
        if any(not math.isfinite(item) for item in value.values()):
            raise ValueError("metric thresholds must be finite")
        return value


class ComponentSelectionPolicy(StrictModel):
    component: SelectionComponent
    evidence_definition_fingerprint: str = Field(pattern=SHA256_PATTERN)
    calibration_manifest_fingerprint: str = Field(pattern=SHA256_PATTERN)
    metrics: list[MetricSelectionRule] = Field(min_length=1)

    @model_validator(mode="after")
    def metric_fields_are_valid_and_unique(self) -> ComponentSelectionPolicy:
        metric_ids = [metric.metric_id for metric in self.metrics]
        if metric_ids != sorted(set(metric_ids)):
            raise ValueError("component metric IDs must be sorted and unique")
        invalid = sorted(
            metric.source_field
            for metric in self.metrics
            if metric.source_field not in _COMPONENT_FIELDS[self.component]
        )
        if invalid:
            raise ValueError(f"unsupported {self.component.value} metric fields: {invalid}")
        return self


class SelectionBandDefinition(StrictModel):
    definition_id: str = Field(min_length=1)
    definition_fingerprint: str = Field(pattern=SHA256_PATTERN)
    lower_quantile: float = Field(ge=0.0, le=1.0)
    upper_quantile: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def quantiles_are_ordered(self) -> SelectionBandDefinition:
        if not all(math.isfinite(value) for value in (self.lower_quantile, self.upper_quantile)):
            raise ValueError("selection band quantiles must be finite")
        if self.lower_quantile > self.upper_quantile:
            raise ValueError("selection band lower quantile exceeds upper quantile")
        return self


class PairingLimits(StrictModel):
    minimum_score_margin: float = Field(gt=0.0)
    margin_definition_fingerprint: str = Field(pattern=SHA256_PATTERN)
    near_miss: SelectionBandDefinition
    bottom: SelectionBandDefinition
    max_pairs_per_passage: int = Field(ge=1)
    max_rejected_per_chosen: int = Field(ge=1)

    @field_validator("minimum_score_margin")
    @classmethod
    def minimum_margin_is_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("minimum score margin must be finite")
        return value


class CandidateSelectionPolicyManifest(StrictModel):
    """Externally authored policy. Validation never derives its numeric values."""

    schema_version: Literal[1]
    contract: Literal["task06-candidate-selection-policy-v1"]
    policy_id: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)
    status: Literal["policy_frozen_not_applied"]
    candidate_evidence_bundle_sha256: str = Field(pattern=SHA256_PATTERN)
    candidate_evidence_bundle_fingerprint: str = Field(pattern=SHA256_PATTERN)
    candidate_evidence_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    context: FrozenContext
    components: dict[SelectionComponent, ComponentSelectionPolicy]
    pairing: PairingLimits
    human_calibration_manifest_fingerprint: str = Field(pattern=SHA256_PATTERN)
    producer_git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    final_tests_used: list[str] = Field(max_length=0)
    manifest_fingerprint: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def components_and_fingerprint_are_exact(self) -> CandidateSelectionPolicyManifest:
        if set(self.components) != set(SelectionComponent):
            raise ValueError("selection policy must cover exactly all required score components")
        for name, component in self.components.items():
            if name != component.component:
                raise ValueError("selection policy component key differs from its declaration")
        payload = self.model_dump(mode="json")
        fingerprint = payload.pop("manifest_fingerprint")
        if fingerprint != canonical_fingerprint(payload):
            raise ValueError("selection policy manifest fingerprint mismatch")
        return self


class CalibrationMetricDefinition(StrictModel):
    metric_id: str = Field(min_length=1)
    source_field: str = Field(min_length=1)
    direction: MetricDirection
    normalization_definition_fingerprint: str = Field(pattern=SHA256_PATTERN)
    threshold_definition_fingerprint: str = Field(pattern=SHA256_PATTERN)
    calibration_fingerprint: str = Field(pattern=SHA256_PATTERN)


class ComponentCalibrationEvidenceManifest(StrictModel):
    schema_version: Literal[1]
    contract: Literal["task06-component-calibration-evidence-v1"]
    calibration_id: str = Field(min_length=1)
    component: SelectionComponent
    context: FrozenContext
    calibration_cohort_id: str = Field(min_length=1)
    calibration_cohort_fingerprint: str = Field(pattern=SHA256_PATTERN)
    evidence_definition_fingerprint: str = Field(pattern=SHA256_PATTERN)
    metrics: list[CalibrationMetricDefinition] = Field(min_length=1)
    source_candidate_evidence_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    producer_git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    artifact: EvidenceArtifact
    final_tests_used: list[str] = Field(max_length=0)
    manifest_fingerprint: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def definitions_and_fingerprint_are_valid(self) -> ComponentCalibrationEvidenceManifest:
        metric_ids = [metric.metric_id for metric in self.metrics]
        if metric_ids != sorted(set(metric_ids)):
            raise ValueError("calibration metric IDs must be sorted and unique")
        invalid = sorted(
            metric.source_field
            for metric in self.metrics
            if metric.source_field not in _COMPONENT_FIELDS[self.component]
        )
        if invalid:
            raise ValueError(f"unsupported calibration metric fields: {invalid}")
        if self.artifact.provenance.producer_git_commit != self.producer_git_commit:
            raise ValueError("calibration artifact provenance commit drift")
        if self.artifact.provenance.source_task != "Task 06":
            raise ValueError("calibration artifact source-task provenance drift")
        if (
            self.artifact.provenance.source_manifest_sha256
            != self.source_candidate_evidence_manifest_sha256
        ):
            raise ValueError("calibration artifact source-manifest provenance drift")
        payload = self.model_dump(mode="json")
        fingerprint = payload.pop("manifest_fingerprint")
        if fingerprint != canonical_fingerprint(payload):
            raise ValueError("component calibration manifest fingerprint mismatch")
        return self


class ConfidenceInterval(StrictModel):
    lower: float
    upper: float
    confidence_level: float = Field(gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def bounds_are_finite_and_ordered(self) -> ConfidenceInterval:
        values = (self.lower, self.upper, self.confidence_level)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("human calibration confidence interval must be finite")
        if self.lower > self.upper:
            raise ValueError("human calibration CI lower bound exceeds upper bound")
        return self


class HumanAgreementEvidence(StrictModel):
    metric_id: str = Field(min_length=1)
    definition_fingerprint: str = Field(pattern=SHA256_PATTERN)
    value: float
    sample_size: int = Field(ge=1)
    ci: ConfidenceInterval

    @field_validator("value")
    @classmethod
    def value_is_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("human agreement must be finite")
        return value


class HumanPreferenceCalibrationEvidenceManifest(StrictModel):
    schema_version: Literal[1]
    contract: Literal["task06-human-preference-calibration-evidence-v1"]
    panel_id: str = Field(min_length=1)
    panel_version: str = Field(min_length=1)
    blinded: Literal[True]
    context: FrozenContext
    panel_cohort_fingerprint: str = Field(pattern=SHA256_PATTERN)
    annotator_protocol_fingerprint: str = Field(pattern=SHA256_PATTERN)
    criteria_definition_fingerprint: str = Field(pattern=SHA256_PATTERN)
    source_candidate_evidence_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    producer_git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    artifact: EvidenceArtifact
    sample_size: int = Field(ge=1)
    agreement: HumanAgreementEvidence
    final_tests_used: list[str] = Field(max_length=0)
    manifest_fingerprint: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def counts_and_fingerprint_are_valid(self) -> HumanPreferenceCalibrationEvidenceManifest:
        if self.artifact.record_count != self.sample_size:
            raise ValueError("human calibration sample size differs from artifact record count")
        if self.agreement.sample_size != self.sample_size:
            raise ValueError("human agreement sample size differs from panel sample size")
        if self.artifact.provenance.producer_git_commit != self.producer_git_commit:
            raise ValueError("human calibration artifact provenance commit drift")
        if self.artifact.provenance.source_task != "Task 06":
            raise ValueError("human calibration artifact source-task provenance drift")
        if (
            self.artifact.provenance.source_manifest_sha256
            != self.source_candidate_evidence_manifest_sha256
        ):
            raise ValueError("human calibration source-manifest provenance drift")
        payload = self.model_dump(mode="json")
        fingerprint = payload.pop("manifest_fingerprint")
        if fingerprint != canonical_fingerprint(payload):
            raise ValueError("human calibration manifest fingerprint mismatch")
        return self


class PreflightArtifactSummary(StrictModel):
    path: Literal["preflight.json"]
    sha256: str = Field(pattern=SHA256_PATTERN)
    candidate_count: int = Field(ge=1)
    component_count: Literal[7]


class PreferenceSelectionPreflightBundleManifest(StrictModel):
    schema_version: Literal[1]
    contract: Literal["task06-preference-selection-preflight-bundle-v1"]
    status: Literal["ready_for_future_preference_selection_not_selected"]
    policy_id: str = Field(min_length=1)
    policy_fingerprint: str = Field(pattern=SHA256_PATTERN)
    context: FrozenContext
    preflight: PreflightArtifactSummary
    generation_started: Literal[False]
    scoring_started: Literal[False]
    calibration_computed: Literal[False]
    selection_started: Literal[False]
    preferences_built: Literal[False]
    model_loading_performed: Literal[False]
    final_tests_used: list[str] = Field(max_length=0)
    bundle_fingerprint: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def fingerprint_is_valid(self) -> PreferenceSelectionPreflightBundleManifest:
        payload = self.model_dump(mode="json")
        fingerprint = payload.pop("bundle_fingerprint")
        if fingerprint != canonical_fingerprint(payload):
            raise ValueError("preflight bundle fingerprint mismatch")
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


def _record_count(path: Path, method: str) -> int:
    if method == "not_applicable":
        return 0
    if method == "jsonl":
        count = 0
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                if not isinstance(json.loads(line), dict):
                    raise ValueError(f"{path}:{line_number}: expected a JSON object")
                count += 1
        return count
    value = json.loads(path.read_text(encoding="utf-8"))
    if method == "single_json":
        return 1 if isinstance(value, dict) else -1
    if method == "json_array":
        return len(value) if isinstance(value, list) else -1
    raise ValueError(f"unsupported record count method: {method}")


def _verify_artifact(manifest_path: Path, artifact: EvidenceArtifact, role: str) -> None:
    declared = Path(artifact.path)
    path = declared if declared.is_absolute() else manifest_path.parent / declared
    _reject_final_test_path(path)
    if not path.is_file():
        raise ValueError(f"missing {role} artifact: {path}")
    if file_sha256(path) != artifact.sha256:
        raise ValueError(f"{role} artifact SHA-256 drift")
    if _record_count(path, artifact.record_count_method) != artifact.record_count:
        raise ValueError(f"{role} artifact record-count drift")


def _load_candidate_evidence(path: Path) -> list[CandidateEvidenceBundle]:
    _reject_final_test_path(path)
    bundles = [CandidateEvidenceBundle.model_validate(row) for row in read_records(path)]
    if not bundles:
        raise ValueError("candidate evidence bundle is empty")
    candidate_ids = [bundle.candidate.candidate_id for bundle in bundles]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate evidence contains duplicate candidate IDs")
    if any(
        forbidden in bundle.model_dump(mode="json")
        for bundle in bundles
        for forbidden in ("total_score", "rank", "chosen", "rejected")
    ):
        raise ValueError("candidate evidence unexpectedly contains selection output")
    return sorted(bundles, key=lambda bundle: bundle.candidate.candidate_id)


def _context_drift(expected: FrozenContext, actual: FrozenContext, label: str) -> None:
    expected_payload = expected.model_dump(mode="json")
    actual_payload = actual.model_dump(mode="json")
    drift = sorted(key for key in expected_payload if expected_payload[key] != actual_payload[key])
    if drift:
        raise ValueError(f"{label} dataset/split/cohort or candidate-ID drift: {drift}")


def _metric_signature_from_policy(
    rule: MetricSelectionRule,
) -> tuple[str, str, str, str, str, str]:
    return (
        rule.metric_id,
        rule.source_field,
        rule.direction.value,
        rule.normalization.definition_fingerprint,
        rule.threshold_definition_fingerprint,
        rule.calibration_fingerprint,
    )


def _metric_signature_from_calibration(
    rule: CalibrationMetricDefinition,
) -> tuple[str, str, str, str, str, str]:
    return (
        rule.metric_id,
        rule.source_field,
        rule.direction.value,
        rule.normalization_definition_fingerprint,
        rule.threshold_definition_fingerprint,
        rule.calibration_fingerprint,
    )


def _validate_evidence_manifest(
    *, manifest: Mapping[str, Any], manifest_path: Path, evidence_path: Path, count: int
) -> None:
    if manifest.get("status") != "evidence_assembled_not_ranked":
        raise ValueError("candidate evidence manifest status is not pre-selection")
    if manifest.get("final_tests_used") != []:
        raise ValueError("candidate evidence manifest must declare final_tests_used=[]")
    digest = file_sha256(evidence_path)
    if manifest.get("output_sha256") != digest or manifest.get("artifact_fingerprint") != digest:
        raise ValueError("candidate evidence bundle SHA-256/fingerprint drift")
    counts = manifest.get("counts")
    if not isinstance(counts, Mapping) or counts.get("complete") != count:
        raise ValueError("candidate evidence record-count drift")
    if any(counts.get(key) != 0 for key in ("missing", "orphan", "duplicate")):
        raise ValueError("candidate evidence manifest is not complete")
    if manifest.get("model_scoring_performed_by_assembler") is not False:
        raise ValueError("candidate evidence assembler execution provenance drift")
    _reject_final_test_path(manifest_path)


def prepare_preference_selection_preflight(
    *,
    candidate_evidence_path: Path,
    candidate_evidence_manifest_path: Path,
    policy_manifest_path: Path,
    calibration_manifest_paths: Sequence[Path],
    human_manifest_path: Path,
    output_dir: Path,
) -> PreferenceSelectionPreflightBundleManifest:
    """Validate external evidence and publish readiness without performing selection."""
    explicit_paths = [
        candidate_evidence_path,
        candidate_evidence_manifest_path,
        policy_manifest_path,
        *calibration_manifest_paths,
        human_manifest_path,
    ]
    for path in explicit_paths:
        _reject_final_test_path(path)
    required_single_inputs = {
        "candidate evidence": candidate_evidence_path,
        "candidate evidence manifest": candidate_evidence_manifest_path,
        "selection policy manifest": policy_manifest_path,
        "human calibration evidence": human_manifest_path,
    }
    missing_inputs = [label for label, path in required_single_inputs.items() if not path.is_file()]
    if missing_inputs:
        raise ValueError(f"missing explicit preflight inputs: {sorted(missing_inputs)}")
    if output_dir.exists():
        raise FileExistsError(f"Task 06 selection preflight output already exists: {output_dir}")
    if not calibration_manifest_paths:
        raise ValueError("missing calibration evidence manifests")
    missing_calibrations = [str(path) for path in calibration_manifest_paths if not path.is_file()]
    if missing_calibrations:
        raise ValueError(f"missing calibration evidence manifests: {sorted(missing_calibrations)}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"
    try:
        bundles = _load_candidate_evidence(candidate_evidence_path)
        evidence_manifest = _load_object(candidate_evidence_manifest_path)
        _validate_evidence_manifest(
            manifest=evidence_manifest,
            manifest_path=candidate_evidence_manifest_path,
            evidence_path=candidate_evidence_path,
            count=len(bundles),
        )
        policy = CandidateSelectionPolicyManifest.model_validate(_load_object(policy_manifest_path))
        evidence_sha256 = file_sha256(candidate_evidence_path)
        evidence_manifest_sha256 = file_sha256(candidate_evidence_manifest_path)
        if (
            policy.candidate_evidence_bundle_sha256 != evidence_sha256
            or policy.candidate_evidence_bundle_fingerprint != evidence_sha256
        ):
            raise ValueError("selection policy candidate evidence hash/fingerprint drift")
        if policy.candidate_evidence_manifest_sha256 != evidence_manifest_sha256:
            raise ValueError("selection policy candidate evidence manifest SHA-256 drift")

        candidate_ids = [bundle.candidate.candidate_id for bundle in bundles]
        observed_candidate_fingerprint = ordered_ids_fingerprint(candidate_ids)
        if policy.context.candidate_count != len(candidate_ids):
            raise ValueError("selection policy candidate record-count drift")
        if policy.context.candidate_ids_fingerprint != observed_candidate_fingerprint:
            raise ValueError("selection policy candidate-ID drift")
        observed_splits = {bundle.candidate.split for bundle in bundles}
        if observed_splits != {policy.context.split_id}:
            raise ValueError("selection policy split drift from candidate evidence")
        observed_split_fingerprint = canonical_fingerprint(
            {"split_id": policy.context.split_id, "candidate_ids": candidate_ids}
        )
        if policy.context.split_fingerprint != observed_split_fingerprint:
            raise ValueError("selection policy split fingerprint drift from candidate evidence")
        observed_cohort_fingerprint = canonical_fingerprint(
            [
                {
                    "candidate_id": bundle.candidate.candidate_id,
                    "passage_id": bundle.candidate.passage_id,
                    "passage_cluster_id": bundle.candidate.passage_cluster_id,
                }
                for bundle in bundles
            ]
        )
        if policy.context.cohort_fingerprint != observed_cohort_fingerprint:
            raise ValueError("selection policy cohort fingerprint drift from candidate evidence")

        calibrations: dict[SelectionComponent, ComponentCalibrationEvidenceManifest] = {}
        calibration_hashes: dict[str, str] = {}
        for path in calibration_manifest_paths:
            calibration = ComponentCalibrationEvidenceManifest.model_validate(_load_object(path))
            if calibration.component in calibrations:
                raise ValueError(f"duplicate calibration component: {calibration.component.value}")
            _context_drift(policy.context, calibration.context, calibration.component.value)
            if calibration.source_candidate_evidence_manifest_sha256 != evidence_manifest_sha256:
                raise ValueError(f"{calibration.component.value} calibration provenance drift")
            _verify_artifact(
                path, calibration.artifact, f"{calibration.component.value} calibration"
            )
            calibrations[calibration.component] = calibration
            calibration_hashes[calibration.component.value] = file_sha256(path)
        missing = sorted(
            component.value for component in set(SelectionComponent) - set(calibrations)
        )
        unexpected = sorted(
            component.value for component in set(calibrations) - set(SelectionComponent)
        )
        if missing or unexpected:
            raise ValueError(
                f"calibration component coverage mismatch: missing={missing}, orphan={unexpected}"
            )

        for component, component_policy in policy.components.items():
            calibration = calibrations[component]
            if (
                component_policy.calibration_manifest_fingerprint
                != calibration.manifest_fingerprint
            ):
                raise ValueError(f"{component.value} calibration fingerprint drift")
            if (
                component_policy.evidence_definition_fingerprint
                != calibration.evidence_definition_fingerprint
            ):
                raise ValueError(f"{component.value} evidence definition drift")
            policy_definitions = [
                _metric_signature_from_policy(rule) for rule in component_policy.metrics
            ]
            calibration_definitions = [
                _metric_signature_from_calibration(rule) for rule in calibration.metrics
            ]
            if policy_definitions != calibration_definitions:
                raise ValueError(f"{component.value} metric definitions are not comparable")

        human = HumanPreferenceCalibrationEvidenceManifest.model_validate(
            _load_object(human_manifest_path)
        )
        _context_drift(policy.context, human.context, "human calibration")
        if human.source_candidate_evidence_manifest_sha256 != evidence_manifest_sha256:
            raise ValueError("human calibration source provenance drift")
        _verify_artifact(human_manifest_path, human.artifact, "human calibration")
        if policy.human_calibration_manifest_fingerprint != human.manifest_fingerprint:
            raise ValueError("human calibration fingerprint/provenance drift")

        preflight_payload: dict[str, Any] = {
            "schema_version": 1,
            "contract": "task06-preference-selection-preflight-v1",
            "status": PREFLIGHT_STATUS,
            "policy_id": policy.policy_id,
            "policy_fingerprint": policy.manifest_fingerprint,
            "context": policy.context.model_dump(mode="json"),
            "inputs": {
                "candidate_evidence": {
                    "sha256": evidence_sha256,
                    "record_count": len(bundles),
                    "candidate_ids_fingerprint": observed_candidate_fingerprint,
                },
                "candidate_evidence_manifest_sha256": evidence_manifest_sha256,
                "policy_manifest_sha256": file_sha256(policy_manifest_path),
                "calibration_manifest_sha256": dict(sorted(calibration_hashes.items())),
                "human_manifest_sha256": file_sha256(human_manifest_path),
            },
            "component_coverage": [component.value for component in SelectionComponent],
            "human_calibration": {
                "panel_id": human.panel_id,
                "panel_cohort_fingerprint": human.panel_cohort_fingerprint,
                "sample_size": human.sample_size,
                "agreement": human.agreement.model_dump(mode="json"),
                "criteria_definition_fingerprint": human.criteria_definition_fingerprint,
                "annotator_protocol_fingerprint": human.annotator_protocol_fingerprint,
            },
            "selection_outputs": [],
            "generation_started": False,
            "scoring_started": False,
            "calibration_computed": False,
            "selection_started": False,
            "preferences_built": False,
            "model_loading_performed": False,
            "final_tests_used": [],
        }
        staging.mkdir()
        preflight_path = staging / "preflight.json"
        write_json(preflight_path, preflight_payload)
        bundle_payload: dict[str, Any] = {
            "schema_version": 1,
            "contract": PREFLIGHT_CONTRACT,
            "status": PREFLIGHT_STATUS,
            "policy_id": policy.policy_id,
            "policy_fingerprint": policy.manifest_fingerprint,
            "context": policy.context.model_dump(mode="json"),
            "preflight": {
                "path": "preflight.json",
                "sha256": file_sha256(preflight_path),
                "candidate_count": len(bundles),
                "component_count": len(SelectionComponent),
            },
            "generation_started": False,
            "scoring_started": False,
            "calibration_computed": False,
            "selection_started": False,
            "preferences_built": False,
            "model_loading_performed": False,
            "final_tests_used": [],
        }
        bundle_payload["bundle_fingerprint"] = canonical_fingerprint(bundle_payload)
        manifest = PreferenceSelectionPreflightBundleManifest.model_validate(bundle_payload)
        write_json(staging / "manifest.json", manifest.model_dump(mode="json"))
        if output_dir.exists():
            raise FileExistsError(
                f"Task 06 selection preflight output already exists: {output_dir}"
            )
        os.replace(staging, output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
