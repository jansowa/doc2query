"""Fail-closed builder of defect-anchored Task 06 preference pairs (policy v2).

Policy v1 ordered pairs by the primary reranker margin.  The completed dual-LLM audit
measured that this signal is mis-aimed: margin bands are flat against judge agreement
(0.693/0.730/0.700), agreement with the automatic order is 0.708-0.718 while the two
judges agree with each other 0.879, and consensus rises with how *defective* the rejected
side is (0.909 for a missing round trip vs 0.797 for margin alone).

So v2 builds each pair from a **named defect contrast** instead:

* **axis A** — answerability/grounding: the rejected side is judged unanswerable or fails
  the corpus round trip @100; the chosen side is clean *and* judged answerable;
* **axis B** — lexical easiness: both sides are judged answerable, but the rejected side
  sits above the frozen ``content_jaccard`` cut and the chosen side below the lower one;
* **axis C** — out of this release (V2-02 did not deliver its acceptance criterion).

``pool_margin`` survives only as a sanity condition on the chosen side and never orders
anything (``margin_used_for_ordering=false`` on every record).  Every threshold comes
from ``configs/preferences/task06_defect_pair_policy_v2.yaml``, frozen prospectively by
``reports/decisions/task06_defect_pair_policy_v2.md``; this module never invents,
calibrates or relaxes one.

Answerability verdicts are **read** from SHA-256 pinned journals of the accepted judge —
nothing here loads a model, runs a GPU, touches a final test split or authorizes DPO.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import Field, model_validator

from doc2query.evaluation.d01_usefulness import _copy_risk as copy_risk_flag
from doc2query.preferences.answerability_judge import judge_item_id
from doc2query.preferences.build import normalized_query_jaccard
from doc2query.preferences.diversity_gate import _reject_final_test_path
from doc2query.preferences.pair_policy import (
    CopyRiskPolicy,
    ExcludedSignal,
    FormatPolicy,
    _Candidate,
    _cluster_ids,
    _components,
    _format_admissible,
    _load_gate,
    _load_scoring,
)
from doc2query.schemas import StrictModel
from doc2query.training.dpo import (
    SHA256_PATTERN,
    canonical_fingerprint,
    file_sha256,
    normalize_task06_query,
    ordered_ids_fingerprint,
)
from doc2query.utils.records import (
    JsonlWriter,
    read_durable_jsonl_prefix,
    write_json,
)

POLICY_CONTRACT = "task06-defect-pair-policy-v2"
BUILD_CONTRACT = "task06-defect-pairs-v2"
BUILD_STATUS = "defect_pairs_built_not_audited"
SCORING_STATUS = "measured"
RELEASED_AXES = ("A", "B")


class PairFailureV2(StrEnum):
    GROUP_NOT_GATE_ELIGIBLE = "group_not_gate_eligible"
    NO_ADMISSIBLE_CHOSEN = "no_admissible_chosen"
    NO_AXIS_DEFECT_REJECTED = "no_axis_defect_rejected"
    NEAR_DUPLICATE_QUERY_PAIR = "near_duplicate_query_pair"


class PairingPolicyV2(StrictModel):
    max_pairs_per_group: Literal[1]
    require_exact_same_prompt: Literal[True]
    require_diversity_gate_eligible: Literal[True]
    restrict_to_gate_representatives: Literal[True]
    max_normalized_query_jaccard: float = Field(ge=0.0, le=1.0)
    axis_assignment: Literal["deterministic_hash_of_group_id"]
    axis_assignment_salt: str = Field(min_length=1)


class PrimaryPolicyV2(StrictModel):
    """Primary survives as a sanity check only; it may never order or tie-break."""

    judge: str = Field(min_length=1)
    signal: Literal["pool_margin"]
    role: Literal["chosen_side_sanity_only"]
    min_chosen_margin_exclusive: float
    used_for_ordering: Literal[False]
    used_for_tie_break: Literal[False]


class ShadowPolicyV2(StrictModel):
    judge: str = Field(min_length=1)
    role: Literal["recorded_only_never_selection_never_veto"]
    veto_on_margin_inversion: Literal[False]
    veto_on_rank_inversion: Literal[False]


class VerdictJournalPin(StrictModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=SHA256_PATTERN)
    record_count: int = Field(ge=1)


class AnswerabilityPolicy(StrictModel):
    judge_contract: Literal["task06-answerability-judge-v1"]
    adr: str = Field(min_length=1)
    join_key: Literal["judge_item_id"]
    verdict_journals: list[VerdictJournalPin] = Field(min_length=1)
    chosen_required_verdict: Literal["yes"]
    uncertain_blocks_chosen: Literal[True]
    uncertain_is_defect: Literal[False]
    missing_verdict_is_fatal: Literal[True]


class CorpusRoundTripPolicyV2(StrictModel):
    role: Literal["independent_filter"]
    chosen_required_field: str = Field(pattern=r"^corpus_round_trip_at_\d+$")
    axis_a_defect_field: str = Field(pattern=r"^corpus_round_trip_at_\d+$")
    axis_b_rejected_required_field: str = Field(pattern=r"^corpus_round_trip_at_\d+$")


class EntityPreservationPolicy(StrictModel):
    """Kept for byte-identical continuity with the measured supply; inert here."""

    role: Literal["inherited_inert_check"]
    required_chosen_value: float
    measured_constant_on_these_cohorts: float
    claimed_as_hallucination_filter: Literal[False]


class FocusPolicyV2(StrictModel):
    role: Literal["reported_label_only"]
    reject_chosen_on_zero_focus_accuracy: Literal[False]
    abstention_penalized: Literal[False]


class AxisSpec(StrictModel):
    id: Literal["A", "B", "C"]
    name: str = Field(min_length=1)
    priority: int = Field(ge=1)
    in_release: bool = True
    reason: str | None = None
    chosen_requires_judge_yes: bool | None = None
    rejected_requires_judge_yes: bool | None = None
    rejected_requires_round_trip_100: bool | None = None
    overlap_signal: str | None = None
    rejected_min_overlap: float | None = None
    chosen_max_overlap: float | None = None
    rejected_defects: list[str] | None = None

    @model_validator(mode="after")
    def released_axes_are_fully_specified(self) -> AxisSpec:
        if not self.in_release:
            if not self.reason:
                raise ValueError(f"axis {self.id} is out of release without a stated reason")
            return self
        if self.chosen_requires_judge_yes is not True:
            raise ValueError(f"axis {self.id} must require an answerable chosen side")
        if not self.rejected_defects:
            raise ValueError(f"axis {self.id} must name at least one rejected defect")
        if self.id == "B":
            if self.overlap_signal != "content_jaccard":
                raise ValueError("axis B must cut on content_jaccard")
            if self.rejected_min_overlap is None or self.chosen_max_overlap is None:
                raise ValueError("axis B needs both overlap cuts")
            if not self.chosen_max_overlap < self.rejected_min_overlap:
                raise ValueError("axis B cuts must leave a positive overlap gap")
            if self.rejected_requires_judge_yes is not True:
                raise ValueError("axis B rejected must stay answerable, or it collapses into A")
            if self.rejected_requires_round_trip_100 is not True:
                raise ValueError("axis B rejected must keep its corpus round trip")
        return self


class TieBreakPolicy(StrictModel):
    variant: Literal["divpo"]
    signal: Literal["mean_normalized_query_jaccard_to_group_representatives"]
    chosen_selection: Literal["minimum"]
    rejected_selection: Literal["maximum"]
    deterministic_fallback: list[str] = Field(min_length=2)


class ConstructedRejectedPolicy(StrictModel):
    enabled: Literal[False]
    max_share: float = Field(ge=0.0, le=0.0)


class AuditSamplePolicyV2(StrictModel):
    target_pair_count: int = Field(ge=1)
    axis_quotas: dict[str, int]
    axis_quota_fallback: Literal["reallocate_unused_quota_to_other_axis"]
    seed: int = Field(ge=0)
    strata: list[str] = Field(min_length=1)
    allocation: Literal["proportional_largest_remainder"]
    ordering: Literal["pair_id"]
    orientation: Literal["deterministic_counterbalanced_committed_before_review"]
    output_dir: str = Field(min_length=1)

    @model_validator(mode="after")
    def quotas_and_strata_are_consistent(self) -> AuditSamplePolicyV2:
        if sorted(self.axis_quotas) != sorted(RELEASED_AXES):
            raise ValueError("axis quotas must cover exactly the released axes")
        if sum(self.axis_quotas.values()) != self.target_pair_count:
            raise ValueError("axis quotas must sum to the target pair count")
        if "axis" not in self.strata:
            raise ValueError("the v2 audit sample must stratify by axis")
        if any("margin" in stratum for stratum in self.strata):
            raise ValueError("margin must not be a stratification dimension in v2")
        return self


class DefectPairPolicy(StrictModel):
    schema_version: Literal[1]
    contract: Literal["task06-defect-pair-policy-v2"]
    policy_id: str = Field(min_length=1)
    status: Literal["frozen_before_pair_read"]
    adr: str = Field(min_length=1)
    pairing: PairingPolicyV2
    primary: PrimaryPolicyV2
    shadow: ShadowPolicyV2
    answerability: AnswerabilityPolicy
    corpus_round_trip: CorpusRoundTripPolicyV2
    format: FormatPolicy
    copy_risk: CopyRiskPolicy
    entity_preservation: EntityPreservationPolicy
    focus: FocusPolicyV2
    axes: list[AxisSpec] = Field(min_length=3)
    tie_break: TieBreakPolicy
    constructed_rejected: ConstructedRejectedPolicy
    excluded_signals: list[ExcludedSignal] = Field(min_length=1)
    audit_sample: AuditSamplePolicyV2
    authorized_cohorts: list[str] = Field(min_length=1)
    final_tests_used: list[str] = Field(max_length=0)

    @model_validator(mode="after")
    def released_axes_and_exclusions_are_frozen(self) -> DefectPairPolicy:
        released = tuple(axis.id for axis in self.axes if axis.in_release)
        if released != RELEASED_AXES:
            raise ValueError(f"policy v2 releases exactly axes {RELEASED_AXES}, not {released}")
        excluded = {row.name for row in self.excluded_signals}
        for required in ("pool_margin_as_ordering_key", "total_score"):
            if required not in excluded:
                raise ValueError(f"{required} must stay excluded from the v2 pair policy")
        return self

    def axis(self, axis_id: str) -> AxisSpec:
        for axis in self.axes:
            if axis.id == axis_id:
                return axis
        raise KeyError(axis_id)


class DefectPair(StrictModel):
    """One frozen defect-anchored pair; every component of both sides stays visible."""

    pair_id: str = Field(min_length=1)
    cohort_id: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    example_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    passage_cluster_id: str = Field(min_length=1)
    split: Literal["train"]
    prompt: str = Field(min_length=1)
    prompt_sha256: str = Field(pattern=SHA256_PATTERN)
    passage: str = Field(min_length=1)
    axis: Literal["A", "B"]
    axis_preference_order: list[str] = Field(min_length=2, max_length=2)
    chosen_candidate_id: str = Field(min_length=1)
    rejected_candidate_id: str = Field(min_length=1)
    chosen: str = Field(min_length=1)
    rejected: str = Field(min_length=1)
    chosen_verdict: Literal["yes"]
    rejected_verdict: Literal["yes", "no", "uncertain"]
    chosen_components: dict[str, Any]
    rejected_components: dict[str, Any]
    rejected_defect_labels: list[str] = Field(min_length=1)
    normalized_query_jaccard: float = Field(ge=0.0, le=1.0)
    chosen_group_distinctness: float = Field(ge=0.0, le=1.0)
    rejected_group_typicality: float = Field(ge=0.0, le=1.0)
    # Zapisany do analizy, NIGDY nie użyty do porządkowania ani tie-breaku;
    # w v2 może być ujemny, bo margines nie wybiera strony.
    primary_margin_delta: float
    margin_used_for_ordering: Literal[False]
    constructed_rejected: Literal[False]
    requested_form: str = Field(min_length=1)
    requested_intent: str = Field(min_length=1)
    policy_id: str = Field(min_length=1)
    final_tests_used: list[str] = Field(max_length=0)

    @model_validator(mode="after")
    def pair_is_distinct_and_labelled(self) -> DefectPair:
        if self.chosen_candidate_id == self.rejected_candidate_id:
            raise ValueError("chosen and rejected candidate IDs must differ")
        if normalize_task06_query(self.chosen) == normalize_task06_query(self.rejected):
            raise ValueError("chosen and rejected are identical after Task 06 normalization")
        if self.rejected_defect_labels != sorted(set(self.rejected_defect_labels)):
            raise ValueError("rejected_defect_labels must be unique and sorted")
        if self.axis == "B" and self.rejected_verdict != "yes":
            raise ValueError("an axis B rejected side must stay answerable")
        if list(self.axis_preference_order) != sorted(RELEASED_AXES) and list(
            self.axis_preference_order
        ) != sorted(RELEASED_AXES, reverse=True):
            raise ValueError("axis preference order must be a permutation of the released axes")
        return self


class AxisAttempt(StrictModel):
    axis: Literal["A", "B"]
    admissible_chosen_count: int = Field(ge=0)
    admissible_rejected_count: int = Field(ge=0)
    failure_reason: str | None = None


class DefectPairGroupOutcome(StrictModel):
    group_id: str = Field(min_length=1)
    gate_eligible: bool
    representative_count: int = Field(ge=0)
    axis_preference_order: list[str]
    attempts: list[AxisAttempt]
    paired: bool
    pair_id: str | None = None
    axis: str | None = None
    failure_reasons: list[str]

    @model_validator(mode="after")
    def outcome_is_consistent(self) -> DefectPairGroupOutcome:
        if self.failure_reasons != sorted(set(self.failure_reasons)):
            raise ValueError("failure_reasons must be unique and sorted")
        if self.paired != (self.pair_id is not None) or self.paired != (self.axis is not None):
            raise ValueError("a paired group must carry both its pair_id and its axis")
        if self.paired and self.failure_reasons:
            raise ValueError("a paired group must not report failure reasons")
        if not self.paired and not self.failure_reasons:
            raise ValueError("an unpaired group must state why")
        return self


class PairArtifact(StrictModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=SHA256_PATTERN)
    record_count: int = Field(ge=0)


class DefectPairManifest(StrictModel):
    schema_version: Literal[1]
    contract: Literal["task06-defect-pairs-v2"]
    status: Literal["defect_pairs_built_not_audited"]
    cohort_id: str = Field(min_length=1)
    policy_id: str = Field(min_length=1)
    policy_sha256: str = Field(pattern=SHA256_PATTERN)
    policy_fingerprint: str = Field(pattern=SHA256_PATTERN)
    scoring_sha256: str = Field(pattern=SHA256_PATTERN)
    scoring_summary_sha256: str = Field(pattern=SHA256_PATTERN)
    cohort_records_sha256: str = Field(pattern=SHA256_PATTERN)
    diversity_gate_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    diversity_gate_verdicts_sha256: str = Field(pattern=SHA256_PATTERN)
    generation_identity_sha256: str = Field(pattern=SHA256_PATTERN)
    frozen_cohort_fingerprint: str = Field(pattern=SHA256_PATTERN)
    answerability_judge_contract: str = Field(min_length=1)
    verdict_journal_sha256: list[str] = Field(min_length=1)
    verdicts_loaded: int = Field(ge=1)
    candidates_without_verdict: Literal[0]
    primary_judge: str = Field(min_length=1)
    shadow_judge: str = Field(min_length=1)
    split: Literal["train"]
    group_count: int = Field(ge=1)
    gate_eligible_group_count: int = Field(ge=0)
    candidate_count: int = Field(ge=1)
    pair_count: int = Field(ge=0)
    axis_pair_counts: dict[str, int]
    pair_ids_fingerprint: str = Field(pattern=SHA256_PATTERN)
    pairs: PairArtifact
    group_outcomes: PairArtifact
    report: PairArtifact
    margin_used_for_ordering: Literal[False]
    shadow_used_for_selection: Literal[False]
    shadow_used_for_veto: Literal[False]
    total_score_computed: Literal[False]
    thresholds_calibrated_here: Literal[False]
    constructed_rejected_share: float = Field(ge=0.0, le=0.0)
    audit_completed: Literal[False]
    task07_training_authorized: Literal[False]
    final_tests_used: list[str] = Field(max_length=0)
    manifest_fingerprint: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def counts_and_fingerprint_are_valid(self) -> DefectPairManifest:
        if self.pair_count > self.gate_eligible_group_count:
            raise ValueError("a group can contribute at most one pair")
        if sum(self.axis_pair_counts.values()) != self.pair_count:
            raise ValueError("axis pair counts must sum to the pair count")
        payload = self.model_dump(mode="json")
        fingerprint = payload.pop("manifest_fingerprint")
        if fingerprint != canonical_fingerprint(payload):
            raise ValueError("defect pair manifest fingerprint mismatch")
        return self


def load_defect_pair_policy(path: Path) -> DefectPairPolicy:
    """Load the externally frozen v2 policy; nothing is derived or relaxed here."""
    _reject_final_test_path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: defect pair policy must be a mapping")
    return DefectPairPolicy.model_validate(raw)


def axis_preference_order(group_id: str, policy: DefectPairPolicy) -> tuple[str, str]:
    """Assign the axis order deterministically from the group hash, never from a counter."""
    digest = hashlib.sha256(f"{policy.pairing.axis_assignment_salt}\0{group_id}".encode())
    if int(digest.hexdigest()[:16], 16) % 2 == 0:
        return ("A", "B")
    return ("B", "A")


@dataclass(frozen=True)
class CertifiedCandidate:
    """One gate representative joined with its frozen answerability verdict."""

    candidate: _Candidate
    verdict: str
    mean_group_jaccard: float

    @property
    def candidate_id(self) -> str:
        return self.candidate.candidate_id

    @property
    def query(self) -> str:
        return self.candidate.query


def _clean_chosen(candidate: _Candidate, policy: DefectPairPolicy) -> bool:
    """The frozen cleanliness contract, byte-identical with the measured axis-A supply."""
    if not _format_admissible(candidate):
        return False
    if candidate.number(policy.corpus_round_trip.chosen_required_field) < 1.0:
        return False
    if candidate.number("entity_preservation") < policy.entity_preservation.required_chosen_value:
        return False
    if candidate.number("pool_margin") <= policy.primary.min_chosen_margin_exclusive:
        return False
    return not copy_risk_flag(candidate.row, policy.copy_risk.thresholds())


def chosen_admissible(
    certified: CertifiedCandidate, policy: DefectPairPolicy, axis_id: str
) -> bool:
    """`uncertain` and `no` both block the chosen role; only `yes` opens it."""
    if certified.verdict != policy.answerability.chosen_required_verdict:
        return False
    if not _clean_chosen(certified.candidate, policy):
        return False
    axis = policy.axis(axis_id)
    if axis.chosen_max_overlap is not None:
        return certified.candidate.number("content_jaccard") <= axis.chosen_max_overlap
    return True


def rejected_admissible(
    certified: CertifiedCandidate, policy: DefectPairPolicy, axis_id: str
) -> bool:
    """A rejected side needs a *named* defect; `uncertain` is never one."""
    if not _format_admissible(certified.candidate):
        return False
    round_trip = policy.corpus_round_trip
    if axis_id == "A":
        return (
            certified.verdict == "no"
            or certified.candidate.number(round_trip.axis_a_defect_field) < 1.0
        )
    axis = policy.axis(axis_id)
    if certified.verdict != "yes":
        return False
    if certified.candidate.number(round_trip.axis_b_rejected_required_field) < 1.0:
        return False
    assert axis.rejected_min_overlap is not None  # gwarantowane walidatorem polityki
    return certified.candidate.number("content_jaccard") >= axis.rejected_min_overlap


def _defect_labels(
    chosen: CertifiedCandidate,
    rejected: CertifiedCandidate,
    policy: DefectPairPolicy,
    axis_id: str,
) -> list[str]:
    """Reported-only labels; they never influence which pair is built."""
    labels: set[str] = set()
    if rejected.verdict == "no":
        labels.add("judge_unanswerable")
    if rejected.candidate.number(policy.corpus_round_trip.axis_a_defect_field) < 1.0:
        labels.add("weak_corpus_round_trip")
    axis = policy.axis(axis_id)
    if (
        axis.rejected_min_overlap is not None
        and rejected.candidate.number("content_jaccard") >= axis.rejected_min_overlap
    ):
        labels.add("high_lexical_overlap")
    if copy_risk_flag(rejected.candidate.row, policy.copy_risk.thresholds()):
        labels.add("copy_risk")
    if rejected.candidate.value("corpus_possibly_ambiguous_query") is True:
        labels.add("possible_ambiguous_query")
    focus = rejected.candidate.value("focus_accuracy")
    if focus is not None and float(focus) == 0.0:
        labels.add("wrong_focus")
    if rejected.candidate.number("shadow_pool_margin") < chosen.candidate.number(
        "shadow_pool_margin"
    ):
        labels.add("shadow_agrees")
    if rejected.candidate.value("judge_rank_disagreement") is True:
        labels.add("judge_rank_disagreement")
    if rejected.candidate.number("pool_margin") < chosen.candidate.number("pool_margin"):
        labels.add("lower_primary_margin")
    return sorted(labels)


def _pair_id(cohort_id: str, chosen: CertifiedCandidate, rejected: CertifiedCandidate) -> str:
    return canonical_fingerprint(
        {
            "contract": BUILD_CONTRACT,
            "cohort_id": cohort_id,
            "chosen_candidate_id": chosen.candidate_id,
            "rejected_candidate_id": rejected.candidate_id,
        }
    )[:32]


def _mean_group_jaccard(query: str, others: Sequence[str]) -> float:
    if not others:
        return 0.0
    return sum(normalized_query_jaccard(query, other) for other in others) / len(others)


def certify_group(
    candidates: Sequence[_Candidate], verdicts: Mapping[str, str]
) -> list[CertifiedCandidate]:
    """Join representatives with their verdicts and precompute the DivPO tie-break signal."""
    queries = [candidate.query for candidate in candidates]
    certified: list[CertifiedCandidate] = []
    for index, candidate in enumerate(candidates):
        passage = str(cast(Mapping[str, Any], candidate.row["positive"])["text"])
        item_id = judge_item_id(candidate.query, passage)
        verdict = verdicts.get(item_id)
        if verdict is None:
            raise ValueError(
                f"candidate {candidate.candidate_id} has no answerability verdict; "
                "the v2 builder refuses to guess one"
            )
        others = queries[:index] + queries[index + 1 :]
        certified.append(
            CertifiedCandidate(
                candidate=candidate,
                verdict=verdict,
                mean_group_jaccard=_mean_group_jaccard(candidate.query, others),
            )
        )
    return certified


def _tie_break_key(certified: CertifiedCandidate) -> tuple[int, str]:
    return (certified.candidate.candidate_index, certified.candidate_id)


def build_group_pair(
    certified: Sequence[CertifiedCandidate],
    *,
    cohort_id: str,
    group_id: str,
    gate_eligible: bool,
    passage_cluster_id: str,
    policy: DefectPairPolicy,
) -> tuple[DefectPair | None, DefectPairGroupOutcome]:
    """Apply the frozen v2 policy to one same-prompt group and emit at most one pair."""
    order = axis_preference_order(group_id, policy)
    if not gate_eligible:
        return None, DefectPairGroupOutcome(
            group_id=group_id,
            gate_eligible=False,
            representative_count=0,
            axis_preference_order=list(order),
            attempts=[],
            paired=False,
            failure_reasons=[PairFailureV2.GROUP_NOT_GATE_ELIGIBLE.value],
        )
    prompts = {str(row.candidate.row["prompt_sha256"]) for row in certified}
    if len(prompts) != 1:
        raise ValueError(f"group {group_id} does not share one prompt hash")

    attempts: list[AxisAttempt] = []
    max_jaccard = policy.pairing.max_normalized_query_jaccard
    for axis_id in order:
        chosen_pool = [row for row in certified if chosen_admissible(row, policy, axis_id)]
        rejected_pool = [row for row in certified if rejected_admissible(row, policy, axis_id)]
        # DivPO: chosen najbardziej odrębny w grupie, rejected najbardziej typowy.
        chosen_pool.sort(key=lambda row: (row.mean_group_jaccard, *_tie_break_key(row)))
        rejected_pool.sort(key=lambda row: (-row.mean_group_jaccard, *_tie_break_key(row)))
        if not chosen_pool:
            attempts.append(
                AxisAttempt(
                    axis=cast(Literal["A", "B"], axis_id),
                    admissible_chosen_count=0,
                    admissible_rejected_count=len(rejected_pool),
                    failure_reason=PairFailureV2.NO_ADMISSIBLE_CHOSEN.value,
                )
            )
            continue
        chosen = chosen_pool[0]
        others = [row for row in rejected_pool if row.candidate_id != chosen.candidate_id]
        if not others:
            attempts.append(
                AxisAttempt(
                    axis=cast(Literal["A", "B"], axis_id),
                    admissible_chosen_count=len(chosen_pool),
                    admissible_rejected_count=len(rejected_pool),
                    failure_reason=PairFailureV2.NO_AXIS_DEFECT_REJECTED.value,
                )
            )
            continue
        rejected = next(
            (
                row
                for row in others
                if normalized_query_jaccard(chosen.query, row.query) <= max_jaccard
            ),
            None,
        )
        if rejected is None:
            attempts.append(
                AxisAttempt(
                    axis=cast(Literal["A", "B"], axis_id),
                    admissible_chosen_count=len(chosen_pool),
                    admissible_rejected_count=len(rejected_pool),
                    failure_reason=PairFailureV2.NEAR_DUPLICATE_QUERY_PAIR.value,
                )
            )
            continue
        attempts.append(
            AxisAttempt(
                axis=cast(Literal["A", "B"], axis_id),
                admissible_chosen_count=len(chosen_pool),
                admissible_rejected_count=len(rejected_pool),
            )
        )
        pair_id = _pair_id(cohort_id, chosen, rejected)
        positive = cast(Mapping[str, Any], chosen.candidate.row["positive"])
        pair = DefectPair(
            pair_id=pair_id,
            cohort_id=cohort_id,
            group_id=group_id,
            example_id=str(chosen.candidate.row["example_id"]),
            doc_id=str(chosen.candidate.row["doc_id"]),
            passage_cluster_id=passage_cluster_id,
            split="train",
            prompt=str(chosen.candidate.row["prompt"]),
            prompt_sha256=str(chosen.candidate.row["prompt_sha256"]),
            passage=str(positive["text"]),
            axis=cast(Literal["A", "B"], axis_id),
            axis_preference_order=list(order),
            chosen_candidate_id=chosen.candidate_id,
            rejected_candidate_id=rejected.candidate_id,
            chosen=chosen.query,
            rejected=rejected.query,
            chosen_verdict="yes",
            rejected_verdict=cast(Literal["yes", "no", "uncertain"], rejected.verdict),
            chosen_components=_components(chosen.candidate),
            rejected_components=_components(rejected.candidate),
            rejected_defect_labels=_defect_labels(chosen, rejected, policy, axis_id),
            normalized_query_jaccard=normalized_query_jaccard(chosen.query, rejected.query),
            chosen_group_distinctness=chosen.mean_group_jaccard,
            rejected_group_typicality=rejected.mean_group_jaccard,
            primary_margin_delta=(
                chosen.candidate.number("pool_margin") - rejected.candidate.number("pool_margin")
            ),
            margin_used_for_ordering=False,
            constructed_rejected=False,
            requested_form=str(chosen.candidate.row["requested_form"]),
            requested_intent=str(chosen.candidate.row["requested_intent"]),
            policy_id=policy.policy_id,
            final_tests_used=[],
        )
        return pair, DefectPairGroupOutcome(
            group_id=group_id,
            gate_eligible=True,
            representative_count=len(certified),
            axis_preference_order=list(order),
            attempts=attempts,
            paired=True,
            pair_id=pair_id,
            axis=axis_id,
            failure_reasons=[],
        )

    reasons = sorted({attempt.failure_reason for attempt in attempts if attempt.failure_reason})
    return None, DefectPairGroupOutcome(
        group_id=group_id,
        gate_eligible=True,
        representative_count=len(certified),
        axis_preference_order=list(order),
        attempts=attempts,
        paired=False,
        failure_reasons=reasons,
    )


def load_pinned_verdicts(
    policy: DefectPairPolicy, journal_paths: Iterable[Path]
) -> tuple[dict[str, str], list[str]]:
    """Read judge journals whose SHA-256 the policy pins; refuse anything unpinned."""
    pins = {pin.sha256: pin for pin in policy.answerability.verdict_journals}
    verdicts: dict[str, str] = {}
    used: list[str] = []
    for path in journal_paths:
        _reject_final_test_path(path)
        if not path.is_file():
            raise ValueError(f"missing answerability verdict journal: {path}")
        digest = file_sha256(path)
        pin = pins.get(digest)
        if pin is None:
            raise ValueError(f"{path}: verdict journal is not pinned by {policy.policy_id}")
        count = 0
        for event in read_durable_jsonl_prefix(path):
            if event.get("event") != "verdict":
                continue
            item_id = str(event["item_id"])
            verdict = str(event["verdict"])
            previous = verdicts.get(item_id)
            if previous is not None and previous != verdict:
                raise ValueError(f"journals disagree on {item_id}: {previous} vs {verdict}")
            verdicts[item_id] = verdict
            count += 1
        if count != pin.record_count:
            raise ValueError(
                f"{path}: journal holds {count} verdicts, the policy pins {pin.record_count}"
            )
        used.append(digest)
    if not used:
        raise ValueError("the v2 builder needs at least one pinned verdict journal")
    return verdicts, sorted(used)


def _quantiles(values: Sequence[float]) -> dict[str, float | int | None]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "min": None, "p50": None, "max": None}

    def at(fraction: float) -> float:
        return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p25": at(0.25),
        "p50": at(0.50),
        "p75": at(0.75),
        "max": ordered[-1],
    }


def _report(
    pairs: Sequence[DefectPair],
    outcomes: Sequence[DefectPairGroupOutcome],
    certified: Mapping[str, Sequence[CertifiedCandidate]],
    policy: DefectPairPolicy,
    *,
    cohort_id: str,
) -> dict[str, Any]:
    eligible = [row for row in outcomes if row.gate_eligible]
    failures = Counter(reason for row in outcomes for reason in row.failure_reasons)
    per_axis_failures: Counter[str] = Counter()
    for row in outcomes:
        for attempt in row.attempts:
            if attempt.failure_reason:
                per_axis_failures[f"{attempt.axis}:{attempt.failure_reason}"] += 1
    labels = Counter(label for pair in pairs for label in pair.rejected_defect_labels)
    axis_counts = Counter(pair.axis for pair in pairs)
    preferred_axis = Counter(row.axis_preference_order[0] for row in eligible)
    fallback_pairs = sum(
        1 for pair in pairs if pair.axis != pair.axis_preference_order[0]
    )
    verdicts = Counter(row.verdict for members in certified.values() for row in members)
    return {
        "schema_version": 1,
        "contract": BUILD_CONTRACT,
        "status": BUILD_STATUS,
        "cohort_id": cohort_id,
        "policy_id": policy.policy_id,
        "policy": policy.model_dump(mode="json"),
        "group_count": len(outcomes),
        "gate_eligible_group_count": len(eligible),
        "pair_count": len(pairs),
        "pair_rate_among_gate_eligible": (len(pairs) / len(eligible)) if eligible else None,
        "axis_pair_counts": dict(sorted(axis_counts.items())),
        "preferred_axis_counts": dict(sorted(preferred_axis.items())),
        "pairs_built_on_fallback_axis": fallback_pairs,
        "candidate_verdict_counts": dict(sorted(verdicts.items())),
        "failure_reason_counts": dict(sorted(failures.items())),
        "per_axis_failure_counts": dict(sorted(per_axis_failures.items())),
        "rejected_defect_label_counts": dict(sorted(labels.items())),
        "requested_form_counts": dict(
            sorted(Counter(pair.requested_form for pair in pairs).items())
        ),
        "requested_intent_counts": dict(
            sorted(Counter(pair.requested_intent for pair in pairs).items())
        ),
        "chosen_content_jaccard": {
            axis: _quantiles(
                [
                    float(pair.chosen_components["content_jaccard"])
                    for pair in pairs
                    if pair.axis == axis
                ]
            )
            for axis in RELEASED_AXES
        },
        "rejected_content_jaccard": {
            axis: _quantiles(
                [
                    float(pair.rejected_components["content_jaccard"])
                    for pair in pairs
                    if pair.axis == axis
                ]
            )
            for axis in RELEASED_AXES
        },
        "primary_margin_delta": _quantiles([pair.primary_margin_delta for pair in pairs]),
        "primary_margin_delta_negative_share": (
            sum(1 for pair in pairs if pair.primary_margin_delta < 0.0) / len(pairs)
            if pairs
            else None
        ),
        "margin_used_for_ordering": False,
        "shadow_used_for_selection": False,
        "shadow_used_for_veto": False,
        "total_score_computed": False,
        "thresholds_calibrated_here": False,
        "constructed_rejected_share": 0.0,
        "audit_completed": False,
        "task07_training_authorized": False,
        "final_tests_used": [],
    }


def build_defect_pairs(
    *,
    cohort_dir: Path,
    policy_path: Path,
    journal_paths: Sequence[Path],
    output_dir: Path | None = None,
) -> DefectPairManifest:
    """Build at most one defect-anchored pair per same-prompt group of one frozen cohort."""
    policy = load_defect_pair_policy(policy_path)
    cohort_id = cohort_dir.name
    if cohort_id not in policy.authorized_cohorts:
        raise ValueError(
            f"cohort {cohort_id} is not authorized for pair building by {policy.policy_id}"
        )
    scoring_dir = cohort_dir / "d01_controlled" / "scoring"
    scoring_path = scoring_dir / "per_generation.jsonl"
    summary_path = scoring_dir / "summary.json"
    gate_dir = cohort_dir / "diversity_gate"
    cohort_records_path = cohort_dir / "cohort.records.jsonl"
    target = output_dir if output_dir is not None else cohort_dir / "defect_pairs_v2"
    if target.exists():
        raise FileExistsError(f"Task 06 defect pair output already exists: {target}")

    verdicts, journal_hashes = load_pinned_verdicts(policy, journal_paths)
    gate, gate_verdicts = _load_gate(gate_dir)
    rows = _load_scoring(scoring_path, summary_path, gate)
    clusters = _cluster_ids(cohort_records_path)

    judges = {(str(row["primary_judge"]), str(row["shadow_judge"])) for row in rows}
    if judges != {(policy.primary.judge, policy.shadow.judge)}:
        raise ValueError("scored candidates were not judged by the judges the policy pins")

    representatives = {
        group_id: [str(value) for value in verdict["representative_candidate_ids"]]
        for group_id, verdict in gate_verdicts.items()
    }
    grouped: dict[str, dict[str, _Candidate]] = {}
    for row in rows:
        group_id = str(row["evaluation_group_id"])
        grouped.setdefault(group_id, {})[str(row["evaluation_id"])] = _Candidate(
            candidate_id=str(row["evaluation_id"]),
            candidate_index=int(row["candidate_index"]),
            query=str(row["generated"]),
            row=row,
        )
    if set(grouped) != set(gate_verdicts):
        raise ValueError("scored groups and diversity gate groups do not cover each other")

    pairs: list[DefectPair] = []
    outcomes: list[DefectPairGroupOutcome] = []
    certified_by_group: dict[str, list[CertifiedCandidate]] = {}
    for group_id in sorted(grouped):
        verdict_row = gate_verdicts[group_id]
        by_id = grouped[group_id]
        example_id = str(next(iter(by_id.values())).row["example_id"])
        if example_id not in clusters:
            raise ValueError(f"group {group_id} is missing from the frozen cohort records")
        eligible = bool(verdict_row["eligible"])
        certified: list[CertifiedCandidate] = []
        if eligible:
            missing = [value for value in representatives[group_id] if value not in by_id]
            if missing:
                raise ValueError(f"group {group_id} misses gate representatives: {sorted(missing)}")
            certified = certify_group(
                [by_id[value] for value in representatives[group_id]], verdicts
            )
            certified_by_group[group_id] = certified
        pair, outcome = build_group_pair(
            certified,
            cohort_id=cohort_id,
            group_id=group_id,
            gate_eligible=eligible,
            passage_cluster_id=clusters[example_id],
            policy=policy,
        )
        outcomes.append(outcome)
        if pair is not None:
            pairs.append(pair)

    clusters_used = [pair.passage_cluster_id for pair in pairs]
    if len(set(clusters_used)) != len(clusters_used):
        raise ValueError("pairs must not repeat a near-duplicate passage cluster")

    axis_counts = Counter(pair.axis for pair in pairs)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.staging-{uuid.uuid4().hex}"
    try:
        staging.mkdir()
        pairs_path = staging / "pairs.jsonl"
        with JsonlWriter(pairs_path) as writer:
            for pair in pairs:
                writer.write(pair.model_dump(mode="json"))
        outcomes_path = staging / "group_outcomes.jsonl"
        with JsonlWriter(outcomes_path) as writer:
            for outcome in outcomes:
                writer.write(outcome.model_dump(mode="json"))
        report_path = staging / "report.json"
        write_json(
            report_path,
            _report(pairs, outcomes, certified_by_group, policy, cohort_id=cohort_id),
        )
        payload: dict[str, Any] = {
            "schema_version": 1,
            "contract": BUILD_CONTRACT,
            "status": BUILD_STATUS,
            "cohort_id": cohort_id,
            "policy_id": policy.policy_id,
            "policy_sha256": file_sha256(policy_path),
            "policy_fingerprint": canonical_fingerprint(policy.model_dump(mode="json")),
            "scoring_sha256": file_sha256(scoring_path),
            "scoring_summary_sha256": file_sha256(summary_path),
            "cohort_records_sha256": file_sha256(cohort_records_path),
            "diversity_gate_manifest_sha256": file_sha256(gate_dir / "manifest.json"),
            "diversity_gate_verdicts_sha256": gate.verdicts.sha256,
            "generation_identity_sha256": gate.generation_identity_sha256,
            "frozen_cohort_fingerprint": gate.frozen_cohort_fingerprint,
            "answerability_judge_contract": policy.answerability.judge_contract,
            "verdict_journal_sha256": journal_hashes,
            "verdicts_loaded": len(verdicts),
            "candidates_without_verdict": 0,
            "primary_judge": policy.primary.judge,
            "shadow_judge": policy.shadow.judge,
            "split": "train",
            "group_count": len(outcomes),
            "gate_eligible_group_count": gate.eligible_group_count,
            "candidate_count": len(rows),
            "pair_count": len(pairs),
            "axis_pair_counts": dict(sorted(axis_counts.items())),
            "pair_ids_fingerprint": ordered_ids_fingerprint([pair.pair_id for pair in pairs]),
            "pairs": {
                "path": pairs_path.name,
                "sha256": file_sha256(pairs_path),
                "record_count": len(pairs),
            },
            "group_outcomes": {
                "path": outcomes_path.name,
                "sha256": file_sha256(outcomes_path),
                "record_count": len(outcomes),
            },
            "report": {
                "path": report_path.name,
                "sha256": file_sha256(report_path),
                "record_count": 1,
            },
            "margin_used_for_ordering": False,
            "shadow_used_for_selection": False,
            "shadow_used_for_veto": False,
            "total_score_computed": False,
            "thresholds_calibrated_here": False,
            "constructed_rejected_share": 0.0,
            "audit_completed": False,
            "task07_training_authorized": False,
            "final_tests_used": [],
        }
        payload["manifest_fingerprint"] = canonical_fingerprint(payload)
        manifest = DefectPairManifest.model_validate(payload)
        write_json(staging / "manifest.json", manifest.model_dump(mode="json"))
        if target.exists():
            raise FileExistsError(f"Task 06 defect pair output already exists: {target}")
        os.replace(staging, target)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_defect_pair_manifest(path: Path) -> DefectPairManifest:
    return DefectPairManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))


__all__ = [
    "BUILD_CONTRACT",
    "POLICY_CONTRACT",
    "CertifiedCandidate",
    "DefectPair",
    "DefectPairGroupOutcome",
    "DefectPairManifest",
    "DefectPairPolicy",
    "PairFailureV2",
    "axis_preference_order",
    "build_defect_pairs",
    "build_group_pair",
    "certify_group",
    "chosen_admissible",
    "load_defect_pair_manifest",
    "load_defect_pair_policy",
    "load_pinned_verdicts",
    "rejected_admissible",
]
