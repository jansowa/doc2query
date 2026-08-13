"""Fail-closed, quality-blind same-prompt diversity gate for Task 06.

The gate decides which same-prompt candidate groups may ever enter DPO pair building.
It reads generated candidate texts, controls, and decoding provenance only: judge
scores are never opened, candidates are never ranked, and no chosen/rejected pair is
emitted.  Thresholds must be frozen by a prospective ADR before any pair is read; this
module only applies an externally pinned policy.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import Field, model_validator

from doc2query.evaluation.diversity import pairwise_lemma_jaccards, self_bleu_score
from doc2query.evaluation.retrieval import distribution
from doc2query.preferences.build import normalized_query_jaccard
from doc2query.schemas import StrictModel
from doc2query.text.normalization import SimplePolishNormalizer
from doc2query.training.dpo import (
    SHA256_PATTERN,
    canonical_fingerprint,
    file_sha256,
    normalize_task06_query,
    ordered_ids_fingerprint,
)
from doc2query.utils.records import JsonlWriter, read_records, write_json

POLICY_CONTRACT = "task06-same-prompt-diversity-gate-policy-v1"
GATE_CONTRACT = "task06-same-prompt-diversity-gate-v1"
GATE_STATUS = "diversity_gate_applied_not_paired"
GENERATION_CONTRACT = "task06-same-prompt-preference-expansion-v1"
GENERATION_STATUS = "same_prompt_generation_complete"
EXACT_NORMALIZATION = "task06_whitespace_casefold"
LEMMA_BACKEND = SimplePolishNormalizer.cache_namespace

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
_FORBIDDEN_QUALITY_FIELDS = (
    "primary_margin",
    "primary_negative_scores",
    "primary_source_score",
    "shadow_score",
    "pool_margin",
    "pool_rank",
    "total_score",
    "chosen",
    "rejected",
)


class GateFailure(StrEnum):
    PROMPT_MISMATCH = "prompt_mismatch"
    UNEXPECTED_GROUP_SIZE = "unexpected_group_size"
    INSUFFICIENT_EFFECTIVE_CANDIDATES = "insufficient_effective_candidates"
    DUPLICATE_RATE_ABOVE_THRESHOLD = "duplicate_rate_above_threshold"
    SELF_BLEU_ABOVE_THRESHOLD = "self_bleu_above_threshold"
    NO_PAIRABLE_CANDIDATE_PAIR = "no_pairable_candidate_pair"


class GateNormalization(StrictModel):
    exact_duplicate: Literal["task06_whitespace_casefold"]
    lemma_backend: Literal["simple_pl:v1:nfkc:stopwords-v1"]
    near_duplicate_lemma_jaccard: float = Field(gt=0.0, le=1.0)


class GateGroupThresholds(StrictModel):
    require_exact_same_prompt: Literal[True]
    expected_candidates_per_group: int = Field(ge=2)
    min_effective_candidates: int = Field(ge=2)
    max_duplicate_rate: float = Field(ge=0.0, lt=1.0)
    max_effective_self_bleu: float = Field(gt=0.0, le=1.0)
    max_min_pairwise_query_jaccard: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def group_can_admit_a_pair(self) -> GateGroupThresholds:
        if self.min_effective_candidates > self.expected_candidates_per_group:
            raise ValueError("min_effective_candidates exceeds the frozen group size")
        return self


class SameGroupDiversityGatePolicy(StrictModel):
    schema_version: Literal[1]
    contract: Literal["task06-same-prompt-diversity-gate-policy-v1"]
    policy_id: str = Field(min_length=1)
    status: Literal["frozen_before_pair_read"]
    adr: str = Field(min_length=1)
    normalization: GateNormalization
    group: GateGroupThresholds
    final_tests_used: list[str] = Field(max_length=0)


class SameGroupDiversityVerdict(StrictModel):
    """Per-group eligibility with every measured quantity kept visible."""

    group_id: str = Field(min_length=1)
    example_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    split: Literal["train", "dev"]
    prompt_sha256: str = Field(pattern=SHA256_PATTERN)
    candidate_count: int = Field(ge=1)
    candidate_ids: list[str] = Field(min_length=1)
    distinct_normalized_count: int = Field(ge=1)
    duplicate_rate: float = Field(ge=0.0, le=1.0)
    effective_candidate_count: int = Field(ge=1)
    representative_candidate_ids: list[str] = Field(min_length=1)
    effective_cluster_sizes: list[int] = Field(min_length=1)
    effective_self_bleu: float | None = None
    min_pairwise_representative_query_jaccard: float | None = None
    mean_pairwise_representative_query_jaccard: float | None = None
    max_pairwise_lemma_jaccard: float | None = None
    distinct_temperature_count: int = Field(ge=1)
    distinct_top_p_count: int = Field(ge=1)
    distinct_seed_count: int = Field(ge=1)
    eligible: bool
    failure_reasons: list[str]

    @model_validator(mode="after")
    def verdict_is_consistent(self) -> SameGroupDiversityVerdict:
        if self.failure_reasons != sorted(set(self.failure_reasons)):
            raise ValueError("failure_reasons must be unique and sorted")
        if self.eligible != (not self.failure_reasons):
            raise ValueError("eligible must equal the absence of failure reasons")
        if self.effective_candidate_count != len(self.representative_candidate_ids):
            raise ValueError("representative count must equal effective candidate count")
        if sum(self.effective_cluster_sizes) != self.candidate_count:
            raise ValueError("cluster sizes must cover every candidate")
        return self


class GateArtifactSummary(StrictModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=SHA256_PATTERN)
    record_count: int = Field(ge=0)


class SameGroupDiversityGateManifest(StrictModel):
    schema_version: Literal[1]
    contract: Literal["task06-same-prompt-diversity-gate-v1"]
    status: Literal["diversity_gate_applied_not_paired"]
    policy_id: str = Field(min_length=1)
    policy_sha256: str = Field(pattern=SHA256_PATTERN)
    policy_fingerprint: str = Field(pattern=SHA256_PATTERN)
    generations_sha256: str = Field(pattern=SHA256_PATTERN)
    generation_identity_sha256: str = Field(pattern=SHA256_PATTERN)
    frozen_cohort_fingerprint: str = Field(pattern=SHA256_PATTERN)
    split: Literal["train", "dev"]
    group_count: int = Field(ge=1)
    candidate_count: int = Field(ge=1)
    eligible_group_count: int = Field(ge=0)
    rejected_group_count: int = Field(ge=0)
    group_ids_fingerprint: str = Field(pattern=SHA256_PATTERN)
    eligible_group_ids_fingerprint: str = Field(pattern=SHA256_PATTERN)
    verdicts: GateArtifactSummary
    report: GateArtifactSummary
    judge_scores_read: Literal[False]
    candidates_ranked: Literal[False]
    pairs_built: Literal[False]
    model_loading_performed: Literal[False]
    final_tests_used: list[str] = Field(max_length=0)
    manifest_fingerprint: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def fingerprint_and_counts_are_valid(self) -> SameGroupDiversityGateManifest:
        if self.eligible_group_count + self.rejected_group_count != self.group_count:
            raise ValueError("eligible and rejected group counts must cover every group")
        payload = self.model_dump(mode="json")
        fingerprint = payload.pop("manifest_fingerprint")
        if fingerprint != canonical_fingerprint(payload):
            raise ValueError("diversity gate manifest fingerprint mismatch")
        return self


def _reject_final_test_path(path: Path) -> None:
    candidates = (path, path.resolve(strict=False))
    normalized = [
        "/".join(part.casefold().replace(" ", "_") for part in candidate.parts)
        for candidate in candidates
    ]
    if any(marker in value for marker in _FORBIDDEN_FINAL_TEST_MARKERS for value in normalized):
        raise ValueError(f"final-test path is forbidden and was not opened: {path}")


def load_gate_policy(path: Path) -> SameGroupDiversityGatePolicy:
    """Load an externally frozen gate policy; this module never invents thresholds."""
    _reject_final_test_path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: gate policy must be a mapping")
    policy = SameGroupDiversityGatePolicy.model_validate(raw)
    if policy.normalization.exact_duplicate != EXACT_NORMALIZATION:
        raise ValueError("gate policy pins an unsupported exact-duplicate normalization")
    if policy.normalization.lemma_backend != LEMMA_BACKEND:
        raise ValueError("gate policy pins a lemma backend this build cannot reproduce")
    return policy


def _cluster_near_duplicates(queries: Sequence[str], *, threshold: float) -> list[int]:
    """Assign each candidate to a near-duplicate cluster keyed by first member index."""
    parents = list(range(len(queries)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    normalized = [normalize_task06_query(query) for query in queries]
    jaccards = pairwise_lemma_jaccards(queries)
    position = 0
    for left in range(len(queries)):
        for right in range(left + 1, len(queries)):
            similar = normalized[left] == normalized[right] or jaccards[position] >= threshold
            position += 1
            if similar:
                left_root, right_root = root(left), root(right)
                if left_root != right_root:
                    parents[max(left_root, right_root)] = min(left_root, right_root)
    return [root(index) for index in range(len(queries))]


def evaluate_group(
    rows: Sequence[Mapping[str, Any]], policy: SameGroupDiversityGatePolicy
) -> SameGroupDiversityVerdict:
    """Apply the frozen gate to one same-prompt group without reading quality fields."""
    if not rows:
        raise ValueError("a diversity gate group cannot be empty")
    ordered = sorted(rows, key=lambda row: (int(row["candidate_index"]), str(row["evaluation_id"])))
    queries = [str(row["generated"]) for row in ordered]
    candidate_ids = [str(row["evaluation_id"]) for row in ordered]
    failures: set[GateFailure] = set()

    prompts = {str(row["prompt"]) for row in ordered}
    prompt_hashes = {str(row["prompt_sha256"]) for row in ordered}
    if len(prompts) != 1 or len(prompt_hashes) != 1:
        failures.add(GateFailure.PROMPT_MISMATCH)
    if len(ordered) != policy.group.expected_candidates_per_group:
        failures.add(GateFailure.UNEXPECTED_GROUP_SIZE)

    distinct_normalized = {normalize_task06_query(query) for query in queries}
    duplicate_rate = 1.0 - len(distinct_normalized) / len(queries)
    clusters = _cluster_near_duplicates(
        queries, threshold=policy.normalization.near_duplicate_lemma_jaccard
    )
    cluster_order: list[int] = []
    for cluster in clusters:
        if cluster not in cluster_order:
            cluster_order.append(cluster)
    sizes = Counter(clusters)
    representatives = [queries[cluster] for cluster in cluster_order]
    representative_ids = [candidate_ids[cluster] for cluster in cluster_order]

    effective_self_bleu = self_bleu_score(representatives)
    representative_jaccards = [
        normalized_query_jaccard(representatives[left], representatives[right])
        for left in range(len(representatives))
        for right in range(left + 1, len(representatives))
    ]
    lemma_jaccards = pairwise_lemma_jaccards(queries)

    if len(representatives) < policy.group.min_effective_candidates:
        failures.add(GateFailure.INSUFFICIENT_EFFECTIVE_CANDIDATES)
    if duplicate_rate > policy.group.max_duplicate_rate:
        failures.add(GateFailure.DUPLICATE_RATE_ABOVE_THRESHOLD)
    if effective_self_bleu is None or effective_self_bleu > policy.group.max_effective_self_bleu:
        failures.add(GateFailure.SELF_BLEU_ABOVE_THRESHOLD)
    if (
        not representative_jaccards
        or min(representative_jaccards) > policy.group.max_min_pairwise_query_jaccard
    ):
        failures.add(GateFailure.NO_PAIRABLE_CANDIDATE_PAIR)

    configs = [dict(row.get("generation_config") or {}) for row in ordered]
    split = str((ordered[0].get("metadata") or {}).get("split", ""))
    if split not in {"train", "dev"}:
        raise ValueError(f"gate refuses split {split!r}")
    return SameGroupDiversityVerdict(
        group_id=str(ordered[0]["evaluation_group_id"]),
        example_id=str(ordered[0]["example_id"]),
        doc_id=str(ordered[0]["doc_id"]),
        split=cast(Literal["train", "dev"], split),
        prompt_sha256=str(ordered[0]["prompt_sha256"]),
        candidate_count=len(ordered),
        candidate_ids=candidate_ids,
        distinct_normalized_count=len(distinct_normalized),
        duplicate_rate=duplicate_rate,
        effective_candidate_count=len(representatives),
        representative_candidate_ids=representative_ids,
        effective_cluster_sizes=[sizes[cluster] for cluster in cluster_order],
        effective_self_bleu=effective_self_bleu,
        min_pairwise_representative_query_jaccard=(
            min(representative_jaccards) if representative_jaccards else None
        ),
        mean_pairwise_representative_query_jaccard=(
            sum(representative_jaccards) / len(representative_jaccards)
            if representative_jaccards
            else None
        ),
        max_pairwise_lemma_jaccard=max(lemma_jaccards) if lemma_jaccards else None,
        distinct_temperature_count=len({config.get("temperature") for config in configs}),
        distinct_top_p_count=len({config.get("top_p") for config in configs}),
        distinct_seed_count=len({row.get("seed") for row in ordered}),
        eligible=not failures,
        failure_reasons=sorted(failure.value for failure in failures),
    )


def _load_generations(
    generations_path: Path, summary_path: Path, identity_path: Path
) -> tuple[list[dict[str, Any]], str, str]:
    for path in (generations_path, summary_path, identity_path):
        _reject_final_test_path(path)
        if not path.is_file():
            raise ValueError(f"missing same-prompt generation input: {path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict) or not isinstance(identity, dict):
        raise ValueError("generation summary and identity must be mappings")
    if summary.get("contract") != GENERATION_CONTRACT or summary.get("status") != GENERATION_STATUS:
        raise ValueError("same-prompt generation is not complete under the frozen contract")
    if summary.get("final_tests_used") != [] or identity.get("final_tests_used") != []:
        raise ValueError("same-prompt generation provenance declares final-test usage")
    generations_sha256 = file_sha256(generations_path)
    if summary.get("output_sha256") != generations_sha256:
        raise ValueError("same-prompt generations SHA-256 drifted from its summary")
    if identity.get("exact_same_prompt_required") is not True:
        raise ValueError("generation identity does not pin the exact-same-prompt contract")
    identity_sha256 = str(identity["identity_sha256"])

    rows = list(read_records(generations_path))
    if len(rows) != int(summary["generation_count"]):
        raise ValueError("same-prompt generation record count drifted from its summary")
    seen: set[str] = set()
    for row in rows:
        if any(field in row for field in _FORBIDDEN_QUALITY_FIELDS):
            raise ValueError("generations unexpectedly contain judge scores or selection output")
        if row.get("final_tests_used") != []:
            raise ValueError("generation record declares final-test usage")
        if str(row.get("generation_identity_sha256")) != identity_sha256:
            raise ValueError("generation record identity drift")
        split = str((row.get("metadata") or {}).get("split", ""))
        if split not in {"train", "dev"}:
            raise ValueError(f"gate refuses split {split!r}")
        evaluation_id = str(row["evaluation_id"])
        if evaluation_id in seen:
            raise ValueError(f"duplicate evaluation_id: {evaluation_id}")
        seen.add(evaluation_id)
    return rows, generations_sha256, identity_sha256


def _metric_distributions(verdicts: Sequence[SameGroupDiversityVerdict]) -> dict[str, Any]:
    fields = (
        "duplicate_rate",
        "effective_candidate_count",
        "effective_self_bleu",
        "min_pairwise_representative_query_jaccard",
        "mean_pairwise_representative_query_jaccard",
        "max_pairwise_lemma_jaccard",
    )
    summary: dict[str, Any] = {"group_count": len(verdicts)}
    for field in fields:
        values = [
            float(value) for verdict in verdicts if (value := getattr(verdict, field)) is not None
        ]
        summary[field] = distribution(values)
    return summary


def apply_same_prompt_diversity_gate(
    *,
    generations_path: Path,
    generations_summary_path: Path,
    generations_identity_path: Path,
    policy_path: Path,
    output_dir: Path,
) -> SameGroupDiversityGateManifest:
    """Publish per-group eligibility for future pair building without building pairs."""
    policy = load_gate_policy(policy_path)
    rows, generations_sha256, identity_sha256 = _load_generations(
        generations_path, generations_summary_path, generations_identity_path
    )
    if output_dir.exists():
        raise FileExistsError(f"Task 06 diversity gate output already exists: {output_dir}")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["evaluation_group_id"]), []).append(row)
    cohort_fingerprints = {str(row["frozen_cohort_fingerprint"]) for row in rows}
    if len(cohort_fingerprints) != 1:
        raise ValueError("same-prompt generations mix frozen cohorts")
    splits = {str((row.get("metadata") or {}).get("split")) for row in rows}
    if len(splits) != 1:
        raise ValueError("same-prompt generations mix splits")

    verdicts = [evaluate_group(grouped[group_id], policy) for group_id in sorted(grouped)]
    eligible = [verdict for verdict in verdicts if verdict.eligible]
    failure_histogram = Counter(
        reason for verdict in verdicts for reason in verdict.failure_reasons
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"
    try:
        staging.mkdir()
        verdicts_path = staging / "group_verdicts.jsonl"
        with JsonlWriter(verdicts_path) as writer:
            for verdict in verdicts:
                writer.write(verdict.model_dump(mode="json"))
        report_payload: dict[str, Any] = {
            "schema_version": 1,
            "contract": GATE_CONTRACT,
            "status": GATE_STATUS,
            "policy_id": policy.policy_id,
            "policy": policy.model_dump(mode="json"),
            "inputs": {
                "generations_sha256": generations_sha256,
                "generations_summary_sha256": file_sha256(generations_summary_path),
                "generations_identity_sha256": identity_sha256,
                "policy_sha256": file_sha256(policy_path),
            },
            "split": sorted(splits)[0],
            "group_count": len(verdicts),
            "candidate_count": len(rows),
            "eligible_group_count": len(eligible),
            "rejected_group_count": len(verdicts) - len(eligible),
            "eligible_group_rate": len(eligible) / len(verdicts),
            "rejected_group_rate": (len(verdicts) - len(eligible)) / len(verdicts),
            "failure_reason_counts": dict(sorted(failure_histogram.items())),
            "all_groups": _metric_distributions(verdicts),
            "eligible_groups": _metric_distributions(eligible),
            "judge_scores_read": False,
            "candidates_ranked": False,
            "pairs_built": False,
            "model_loading_performed": False,
            "final_tests_used": [],
        }
        report_path = staging / "report.json"
        write_json(report_path, report_payload)
        manifest_payload: dict[str, Any] = {
            "schema_version": 1,
            "contract": GATE_CONTRACT,
            "status": GATE_STATUS,
            "policy_id": policy.policy_id,
            "policy_sha256": file_sha256(policy_path),
            "policy_fingerprint": canonical_fingerprint(policy.model_dump(mode="json")),
            "generations_sha256": generations_sha256,
            "generation_identity_sha256": identity_sha256,
            "frozen_cohort_fingerprint": sorted(cohort_fingerprints)[0],
            "split": sorted(splits)[0],
            "group_count": len(verdicts),
            "candidate_count": len(rows),
            "eligible_group_count": len(eligible),
            "rejected_group_count": len(verdicts) - len(eligible),
            "group_ids_fingerprint": ordered_ids_fingerprint(
                [verdict.group_id for verdict in verdicts]
            ),
            "eligible_group_ids_fingerprint": ordered_ids_fingerprint(
                [verdict.group_id for verdict in eligible]
            ),
            "verdicts": {
                "path": verdicts_path.name,
                "sha256": file_sha256(verdicts_path),
                "record_count": len(verdicts),
            },
            "report": {
                "path": report_path.name,
                "sha256": file_sha256(report_path),
                "record_count": 1,
            },
            "judge_scores_read": False,
            "candidates_ranked": False,
            "pairs_built": False,
            "model_loading_performed": False,
            "final_tests_used": [],
        }
        manifest_payload["manifest_fingerprint"] = canonical_fingerprint(manifest_payload)
        manifest = SameGroupDiversityGateManifest.model_validate(manifest_payload)
        write_json(staging / "manifest.json", manifest.model_dump(mode="json"))
        if output_dir.exists():
            raise FileExistsError(f"Task 06 diversity gate output already exists: {output_dir}")
        os.replace(staging, output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
