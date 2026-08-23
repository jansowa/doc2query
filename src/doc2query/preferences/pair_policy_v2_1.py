"""Fail-closed builder of single-axis Task 06 preference pairs (policy v2.1).

Policy v2.0 stood on two defect axes.  Its completed audit measured that only one of
them works: agreement of the judge consensus with the automatic order is 0.974 on axis A
(n=154) but 0.250 on axis B (n=16), axis B never delivered its quota (192 of 250) and it
contributed three quarters of all consensus contradictions while holding 38.4% of the
sample.  So v2.1 retires axis B — the suspicion falls on the *hypothesis* that higher
lexical overlap is a defect, not on where its ``content_jaccard`` cut sat — and stands
openly on one axis:

* ``rejected`` carries a named answerability/grounding defect (judge verdict ``no`` or a
  failed corpus round trip @100);
* ``chosen`` is clean under the frozen contract *and* judged answerable.

Everything else is deliberately the byte-identical axis-A contract of v2.0, so the pairs
this module builds remain comparable with the measured supply of 2 253 axis-A pairs.  The
frozen v2.0 module and its closed measurement are never touched by this file.

What v2.1 explicitly does **not** fix: lexical easiness, monotony, focus compliance,
naturalness.  A one-axis preference signal teaches one thing; that is a recorded
limitation of the release, not a simplification (see §3.1 of the ADR).

``pool_margin`` never orders anything (``margin_used_for_ordering=false``).  Answerability
verdicts are **read** from SHA-256 pinned journals of the accepted judge — nothing here
loads a model, runs a GPU, touches a final test split or authorizes DPO.
"""

from __future__ import annotations

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
from doc2query.preferences.pair_policy_v2 import (
    AnswerabilityPolicy,
    ConstructedRejectedPolicy,
    EntityPreservationPolicy,
    FocusPolicyV2,
    PrimaryPolicyV2,
    ShadowPolicyV2,
    TieBreakPolicy,
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

POLICY_CONTRACT = "task06-defect-pair-policy-v2-1"
BUILD_CONTRACT = "task06-defect-pairs-v2-1"
BUILD_STATUS = "defect_pairs_built_not_audited"
RELEASED_AXIS = "A"


class PairFailureV21(StrEnum):
    GROUP_NOT_GATE_ELIGIBLE = "group_not_gate_eligible"
    NO_ADMISSIBLE_CHOSEN = "no_admissible_chosen"
    NO_AXIS_DEFECT_REJECTED = "no_axis_defect_rejected"
    NEAR_DUPLICATE_QUERY_PAIR = "near_duplicate_query_pair"


class PairingPolicyV21(StrictModel):
    max_pairs_per_group: Literal[1]
    require_exact_same_prompt: Literal[True]
    require_diversity_gate_eligible: Literal[True]
    restrict_to_gate_representatives: Literal[True]
    max_normalized_query_jaccard: float = Field(ge=0.0, le=1.0)
    single_axis: Literal["A"]


class CorpusRoundTripPolicyV21(StrictModel):
    role: Literal["independent_filter"]
    chosen_required_field: str = Field(pattern=r"^corpus_round_trip_at_\d+$")
    axis_a_defect_field: str = Field(pattern=r"^corpus_round_trip_at_\d+$")


class AxisSpecV21(StrictModel):
    id: Literal["A"]
    name: str = Field(min_length=1)
    chosen_requires_judge_yes: Literal[True]
    rejected_defects: list[str] = Field(min_length=1)


class RetiredAxis(StrictModel):
    """A retired axis stays named, with its evidence, so it cannot return quietly."""

    id: Literal["B", "C"]
    name: str = Field(min_length=1)
    retired_by: str = Field(min_length=1)
    evidence: list[str] = Field(min_length=1)
    return_requires: str = Field(min_length=1)


class AuditSamplePolicyV21(StrictModel):
    target_pair_count: int = Field(ge=1)
    seed: int = Field(ge=0)
    strata: list[str] = Field(min_length=1)
    defect_label_priority: list[str] = Field(min_length=1)
    allocation: Literal["proportional_largest_remainder"]
    ordering: Literal["pair_id"]
    orientation: Literal["deterministic_counterbalanced_committed_before_review"]
    output_dir: str = Field(min_length=1)
    minimum_pair_count_to_start: int = Field(ge=1)

    @model_validator(mode="after")
    def strata_stay_free_of_margin_and_axis(self) -> AuditSamplePolicyV21:
        if any("margin" in stratum for stratum in self.strata):
            raise ValueError("margin must not be a stratification dimension")
        if "axis" in self.strata:
            raise ValueError("v2.1 has one axis, so axis cannot stratify the sample")
        if "rejected_defect_label" not in self.strata:
            raise ValueError("the v2.1 sample must stratify by the primary defect label")
        if len(set(self.defect_label_priority)) != len(self.defect_label_priority):
            raise ValueError("the defect label priority must not repeat a label")
        return self


class AnchorCellPolicy(StrictModel):
    """Gold-anchor calibration cell; it gates nothing, in either direction."""

    enabled: bool
    target_pair_count: int = Field(ge=1)
    minimum_pair_count: int = Field(ge=1)
    seed: int = Field(ge=0)
    gating_role: Literal["none_calibration_only"]
    natural_query_source: Literal["frozen_train_split_cohort_records"]
    relabels_natural_pairs: Literal[False]
    output_dir: str = Field(min_length=1)

    @model_validator(mode="after")
    def minimum_does_not_exceed_target(self) -> AnchorCellPolicy:
        if self.minimum_pair_count > self.target_pair_count:
            raise ValueError("the anchor cell minimum cannot exceed its target")
        return self


class GatePrediction(StrictModel):
    id: str = Field(min_length=1)
    quantity: str = Field(min_length=1)
    role: Literal["confirmatory", "guardrail", "reported_only"]
    threshold: float | None = None
    direction: Literal["at_most", "at_least"] | None = None
    per_judge: bool | None = None
    pass_requires: str | None = None
    fail_requires: str | None = None
    planning_assumption: float | list[float] | None = None
    power_at_n800: float | None = None
    confirmatory_power_at_n800: list[float] | None = None
    bootstrap_replicates: int | None = None
    bootstrap_seed: int | None = None

    @model_validator(mode="after")
    def deciding_predictions_state_their_rule(self) -> GatePrediction:
        if self.role == "reported_only":
            if self.threshold is not None:
                raise ValueError(f"{self.id}: a reported-only prediction carries no threshold")
            return self
        if self.threshold is None or self.direction is None:
            raise ValueError(f"{self.id}: a deciding prediction needs a threshold and direction")
        if self.role == "confirmatory" and not self.pass_requires:
            raise ValueError(f"{self.id}: a confirmatory prediction must state its pass rule")
        if self.role == "guardrail" and not self.fail_requires:
            raise ValueError(f"{self.id}: a guardrail must state what makes it fire")
        return self


class DecisionRule(StrictModel):
    """The interval rule that replaces v2.0's point thresholds; thresholds unchanged."""

    interval: Literal["clopper_pearson_exact"]
    alpha: float = Field(gt=0.0, lt=0.5)
    sided: Literal["one_sided"]
    verdicts: list[Literal["pass", "fail", "inconclusive"]] = Field(min_length=3, max_length=3)
    inconclusive_is_fail_closed: Literal[True]
    multiplicity_correction: Literal["none_intersection_union_test"]
    escalation_permitted: Literal[False]
    predictions: list[GatePrediction] = Field(min_length=4)

    @model_validator(mode="after")
    def gate_keeps_one_guardrail_and_three_confirmatory(self) -> DecisionRule:
        roles = Counter(row.role for row in self.predictions)
        if roles["confirmatory"] < 3:
            raise ValueError("the v2.1 gate needs at least three confirmatory predictions")
        if roles["guardrail"] != 1:
            raise ValueError("the v2.1 gate declares exactly one guardrail (P1)")
        return self


class DefectPairPolicyV21(StrictModel):
    schema_version: Literal[1]
    contract: Literal["task06-defect-pair-policy-v2-1"]
    policy_id: str = Field(min_length=1)
    status: Literal["frozen_before_pair_read"]
    adr: str = Field(min_length=1)
    supersedes_policy_id: str = Field(min_length=1)
    pairing: PairingPolicyV21
    # Modele współdzielone z v2.0: te części kontraktu są bajtowo niezmienione,
    # więc reużycie schematu jest jednocześnie testem tej niezmienności.
    primary: PrimaryPolicyV2
    shadow: ShadowPolicyV2
    answerability: AnswerabilityPolicy
    corpus_round_trip: CorpusRoundTripPolicyV21
    format: FormatPolicy
    copy_risk: CopyRiskPolicy
    entity_preservation: EntityPreservationPolicy
    focus: FocusPolicyV2
    axis: AxisSpecV21
    retired_axes: list[RetiredAxis] = Field(min_length=1)
    tie_break: TieBreakPolicy
    constructed_rejected: ConstructedRejectedPolicy
    excluded_signals: list[ExcludedSignal] = Field(min_length=1)
    audit_sample: AuditSamplePolicyV21
    anchor_cell: AnchorCellPolicy
    decision_rule: DecisionRule
    authorized_cohorts: list[str] = Field(min_length=1)
    final_tests_used: list[str] = Field(max_length=0)

    @model_validator(mode="after")
    def retirements_and_exclusions_are_frozen(self) -> DefectPairPolicyV21:
        retired = {row.id for row in self.retired_axes}
        for required in ("B", "C"):
            if required not in retired:
                raise ValueError(f"axis {required} must be explicitly retired in v2.1")
        excluded = {row.name for row in self.excluded_signals}
        for signal in (
            "pool_margin_as_ordering_key",
            "total_score",
            "content_jaccard_as_defect_signal",
        ):
            if signal not in excluded:
                raise ValueError(f"{signal} must stay excluded from the v2.1 pair policy")
        return self

    def defect_label_stratum(self, labels: Sequence[str]) -> str:
        """Reduce the reported label set to one frozen, single-valued stratum key."""
        for label in self.audit_sample.defect_label_priority:
            if label in labels:
                return label
        raise ValueError(f"no prioritized defect label among {sorted(labels)}")


class DefectPairV21(StrictModel):
    """One frozen single-axis pair; every component of both sides stays visible."""

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
    axis: Literal["A"]
    chosen_candidate_id: str = Field(min_length=1)
    rejected_candidate_id: str = Field(min_length=1)
    chosen: str = Field(min_length=1)
    rejected: str = Field(min_length=1)
    chosen_verdict: Literal["yes"]
    rejected_verdict: Literal["yes", "no"]
    chosen_components: dict[str, Any]
    rejected_components: dict[str, Any]
    rejected_defect_labels: list[str] = Field(min_length=1)
    rejected_defect_label: str = Field(min_length=1)
    normalized_query_jaccard: float = Field(ge=0.0, le=1.0)
    chosen_group_distinctness: float = Field(ge=0.0, le=1.0)
    rejected_group_typicality: float = Field(ge=0.0, le=1.0)
    # Zapisany do analizy, NIGDY nie użyty do porządkowania ani tie-breaku.
    primary_margin_delta: float
    margin_used_for_ordering: Literal[False]
    constructed_rejected: Literal[False]
    requested_form: str = Field(min_length=1)
    requested_intent: str = Field(min_length=1)
    policy_id: str = Field(min_length=1)
    final_tests_used: list[str] = Field(max_length=0)

    @model_validator(mode="after")
    def pair_is_distinct_and_labelled(self) -> DefectPairV21:
        if self.chosen_candidate_id == self.rejected_candidate_id:
            raise ValueError("chosen and rejected candidate IDs must differ")
        if normalize_task06_query(self.chosen) == normalize_task06_query(self.rejected):
            raise ValueError("chosen and rejected are identical after Task 06 normalization")
        if self.rejected_defect_labels != sorted(set(self.rejected_defect_labels)):
            raise ValueError("rejected_defect_labels must be unique and sorted")
        if self.rejected_defect_label not in self.rejected_defect_labels:
            raise ValueError("the stratum label must be one of the reported defect labels")
        axis_defects = {"judge_unanswerable", "weak_corpus_round_trip"}
        if not axis_defects & set(self.rejected_defect_labels):
            raise ValueError("an axis A rejected side must carry a named axis A defect")
        return self


class GroupOutcomeV21(StrictModel):
    group_id: str = Field(min_length=1)
    gate_eligible: bool
    representative_count: int = Field(ge=0)
    admissible_chosen_count: int = Field(ge=0)
    admissible_rejected_count: int = Field(ge=0)
    paired: bool
    pair_id: str | None = None
    failure_reasons: list[str]

    @model_validator(mode="after")
    def outcome_is_consistent(self) -> GroupOutcomeV21:
        if self.failure_reasons != sorted(set(self.failure_reasons)):
            raise ValueError("failure_reasons must be unique and sorted")
        if self.paired != (self.pair_id is not None):
            raise ValueError("a paired group must carry its pair_id")
        if self.paired and self.failure_reasons:
            raise ValueError("a paired group must not report failure reasons")
        if not self.paired and not self.failure_reasons:
            raise ValueError("an unpaired group must state why")
        return self


class PairArtifactV21(StrictModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=SHA256_PATTERN)
    record_count: int = Field(ge=0)


class DefectPairManifestV21(StrictModel):
    schema_version: Literal[1]
    contract: Literal["task06-defect-pairs-v2-1"]
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
    axis: Literal["A"]
    retired_axes: list[str] = Field(min_length=1)
    group_count: int = Field(ge=1)
    gate_eligible_group_count: int = Field(ge=0)
    candidate_count: int = Field(ge=1)
    pair_count: int = Field(ge=0)
    defect_label_counts: dict[str, int]
    pair_ids_fingerprint: str = Field(pattern=SHA256_PATTERN)
    pairs: PairArtifactV21
    group_outcomes: PairArtifactV21
    report: PairArtifactV21
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
    def counts_and_fingerprint_are_valid(self) -> DefectPairManifestV21:
        if self.pair_count > self.gate_eligible_group_count:
            raise ValueError("a group can contribute at most one pair")
        if sum(self.defect_label_counts.values()) != self.pair_count:
            raise ValueError("primary defect label counts must sum to the pair count")
        payload = self.model_dump(mode="json")
        fingerprint = payload.pop("manifest_fingerprint")
        if fingerprint != canonical_fingerprint(payload):
            raise ValueError("defect pair manifest fingerprint mismatch")
        return self


def load_defect_pair_policy_v2_1(path: Path) -> DefectPairPolicyV21:
    """Load the externally frozen v2.1 policy; nothing is derived or relaxed here."""
    _reject_final_test_path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: defect pair policy must be a mapping")
    return DefectPairPolicyV21.model_validate(raw)


@dataclass(frozen=True)
class CertifiedCandidateV21:
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


def _clean_chosen(candidate: _Candidate, policy: DefectPairPolicyV21) -> bool:
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


def chosen_admissible(certified: CertifiedCandidateV21, policy: DefectPairPolicyV21) -> bool:
    """`uncertain` and `no` both block the chosen role; only `yes` opens it."""
    if certified.verdict != policy.answerability.chosen_required_verdict:
        return False
    return _clean_chosen(certified.candidate, policy)


def rejected_admissible(certified: CertifiedCandidateV21, policy: DefectPairPolicyV21) -> bool:
    """A rejected side needs a *named* axis A defect; `uncertain` is never one."""
    if not _format_admissible(certified.candidate):
        return False
    if certified.verdict == "no":
        return True
    if certified.verdict != "yes":
        return False
    return certified.candidate.number(policy.corpus_round_trip.axis_a_defect_field) < 1.0


def defect_labels(
    chosen: CertifiedCandidateV21,
    rejected: CertifiedCandidateV21,
    policy: DefectPairPolicyV21,
) -> list[str]:
    """Reported-only labels; they never influence which pair is built.

    ``high_lexical_overlap`` is gone with axis B: ``content_jaccard`` stays in the
    recorded components, but v2.1 refuses to call it a defect.
    """
    labels: set[str] = set()
    if rejected.verdict == "no":
        labels.add("judge_unanswerable")
    if rejected.candidate.number(policy.corpus_round_trip.axis_a_defect_field) < 1.0:
        labels.add("weak_corpus_round_trip")
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


def _pair_id(
    cohort_id: str, chosen: CertifiedCandidateV21, rejected: CertifiedCandidateV21
) -> str:
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
) -> list[CertifiedCandidateV21]:
    """Join representatives with their verdicts and precompute the DivPO tie-break signal."""
    queries = [candidate.query for candidate in candidates]
    certified: list[CertifiedCandidateV21] = []
    for index, candidate in enumerate(candidates):
        passage = str(cast(Mapping[str, Any], candidate.row["positive"])["text"])
        item_id = judge_item_id(candidate.query, passage)
        verdict = verdicts.get(item_id)
        if verdict is None:
            raise ValueError(
                f"candidate {candidate.candidate_id} has no answerability verdict; "
                "the v2.1 builder refuses to guess one"
            )
        others = queries[:index] + queries[index + 1 :]
        certified.append(
            CertifiedCandidateV21(
                candidate=candidate,
                verdict=verdict,
                mean_group_jaccard=_mean_group_jaccard(candidate.query, others),
            )
        )
    return certified


def _tie_break_key(certified: CertifiedCandidateV21) -> tuple[int, str]:
    return (certified.candidate.candidate_index, certified.candidate_id)


def build_group_pair(
    certified: Sequence[CertifiedCandidateV21],
    *,
    cohort_id: str,
    group_id: str,
    gate_eligible: bool,
    passage_cluster_id: str,
    policy: DefectPairPolicyV21,
) -> tuple[DefectPairV21 | None, GroupOutcomeV21]:
    """Apply the frozen v2.1 policy to one same-prompt group and emit at most one pair."""
    if not gate_eligible:
        return None, GroupOutcomeV21(
            group_id=group_id,
            gate_eligible=False,
            representative_count=0,
            admissible_chosen_count=0,
            admissible_rejected_count=0,
            paired=False,
            failure_reasons=[PairFailureV21.GROUP_NOT_GATE_ELIGIBLE.value],
        )
    prompts = {str(row.candidate.row["prompt_sha256"]) for row in certified}
    if len(prompts) != 1:
        raise ValueError(f"group {group_id} does not share one prompt hash")

    chosen_pool = [row for row in certified if chosen_admissible(row, policy)]
    rejected_pool = [row for row in certified if rejected_admissible(row, policy)]
    # DivPO: chosen najbardziej odrębny w grupie, rejected najbardziej typowy.
    chosen_pool.sort(key=lambda row: (row.mean_group_jaccard, *_tie_break_key(row)))
    rejected_pool.sort(key=lambda row: (-row.mean_group_jaccard, *_tie_break_key(row)))

    def unpaired(reason: PairFailureV21) -> tuple[None, GroupOutcomeV21]:
        return None, GroupOutcomeV21(
            group_id=group_id,
            gate_eligible=True,
            representative_count=len(certified),
            admissible_chosen_count=len(chosen_pool),
            admissible_rejected_count=len(rejected_pool),
            paired=False,
            failure_reasons=[reason.value],
        )

    if not chosen_pool:
        return unpaired(PairFailureV21.NO_ADMISSIBLE_CHOSEN)
    chosen = chosen_pool[0]
    others = [row for row in rejected_pool if row.candidate_id != chosen.candidate_id]
    if not others:
        return unpaired(PairFailureV21.NO_AXIS_DEFECT_REJECTED)
    max_jaccard = policy.pairing.max_normalized_query_jaccard
    rejected = next(
        (row for row in others if normalized_query_jaccard(chosen.query, row.query) <= max_jaccard),
        None,
    )
    if rejected is None:
        return unpaired(PairFailureV21.NEAR_DUPLICATE_QUERY_PAIR)

    labels = defect_labels(chosen, rejected, policy)
    pair_id = _pair_id(cohort_id, chosen, rejected)
    positive = cast(Mapping[str, Any], chosen.candidate.row["positive"])
    pair = DefectPairV21(
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
        axis="A",
        chosen_candidate_id=chosen.candidate_id,
        rejected_candidate_id=rejected.candidate_id,
        chosen=chosen.query,
        rejected=rejected.query,
        chosen_verdict="yes",
        rejected_verdict=cast(Literal["yes", "no"], rejected.verdict),
        chosen_components=_components(chosen.candidate),
        rejected_components=_components(rejected.candidate),
        rejected_defect_labels=labels,
        rejected_defect_label=policy.defect_label_stratum(labels),
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
    return pair, GroupOutcomeV21(
        group_id=group_id,
        gate_eligible=True,
        representative_count=len(certified),
        admissible_chosen_count=len(chosen_pool),
        admissible_rejected_count=len(rejected_pool),
        paired=True,
        pair_id=pair_id,
        failure_reasons=[],
    )


def load_pinned_verdicts(
    policy: DefectPairPolicyV21, journal_paths: Iterable[Path]
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
        raise ValueError("the v2.1 builder needs at least one pinned verdict journal")
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
    pairs: Sequence[DefectPairV21],
    outcomes: Sequence[GroupOutcomeV21],
    certified: Mapping[str, Sequence[CertifiedCandidateV21]],
    policy: DefectPairPolicyV21,
    *,
    cohort_id: str,
) -> dict[str, Any]:
    eligible = [row for row in outcomes if row.gate_eligible]
    failures = Counter(reason for row in outcomes for reason in row.failure_reasons)
    labels = Counter(label for pair in pairs for label in pair.rejected_defect_labels)
    strata = Counter(pair.rejected_defect_label for pair in pairs)
    verdicts = Counter(row.verdict for members in certified.values() for row in members)
    return {
        "schema_version": 1,
        "contract": BUILD_CONTRACT,
        "status": BUILD_STATUS,
        "cohort_id": cohort_id,
        "policy_id": policy.policy_id,
        "policy": policy.model_dump(mode="json"),
        "axis": RELEASED_AXIS,
        "retired_axes": sorted(row.id for row in policy.retired_axes),
        "group_count": len(outcomes),
        "gate_eligible_group_count": len(eligible),
        "pair_count": len(pairs),
        "pair_rate_among_gate_eligible": (len(pairs) / len(eligible)) if eligible else None,
        "candidate_verdict_counts": dict(sorted(verdicts.items())),
        "failure_reason_counts": dict(sorted(failures.items())),
        "rejected_defect_label_counts": dict(sorted(labels.items())),
        "primary_defect_label_counts": dict(sorted(strata.items())),
        "requested_form_counts": dict(
            sorted(Counter(pair.requested_form for pair in pairs).items())
        ),
        "requested_intent_counts": dict(
            sorted(Counter(pair.requested_intent for pair in pairs).items())
        ),
        # content_jaccard zostaje raportowany, ale nie jest w v2.1 sygnałem defektu.
        "chosen_content_jaccard": _quantiles(
            [float(pair.chosen_components["content_jaccard"]) for pair in pairs]
        ),
        "rejected_content_jaccard": _quantiles(
            [float(pair.rejected_components["content_jaccard"]) for pair in pairs]
        ),
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


def build_defect_pairs_v2_1(
    *,
    cohort_dir: Path,
    policy_path: Path,
    journal_paths: Sequence[Path],
    output_dir: Path | None = None,
) -> DefectPairManifestV21:
    """Build at most one single-axis pair per same-prompt group of one frozen cohort."""
    policy = load_defect_pair_policy_v2_1(policy_path)
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
    target = output_dir if output_dir is not None else cohort_dir / "defect_pairs_v2_1"
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

    pairs: list[DefectPairV21] = []
    outcomes: list[GroupOutcomeV21] = []
    certified_by_group: dict[str, list[CertifiedCandidateV21]] = {}
    for group_id in sorted(grouped):
        verdict_row = gate_verdicts[group_id]
        by_id = grouped[group_id]
        example_id = str(next(iter(by_id.values())).row["example_id"])
        if example_id not in clusters:
            raise ValueError(f"group {group_id} is missing from the frozen cohort records")
        eligible = bool(verdict_row["eligible"])
        certified: list[CertifiedCandidateV21] = []
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
            "axis": RELEASED_AXIS,
            "retired_axes": sorted(row.id for row in policy.retired_axes),
            "group_count": len(outcomes),
            "gate_eligible_group_count": gate.eligible_group_count,
            "candidate_count": len(rows),
            "pair_count": len(pairs),
            "defect_label_counts": dict(
                sorted(Counter(pair.rejected_defect_label for pair in pairs).items())
            ),
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
        manifest = DefectPairManifestV21.model_validate(payload)
        write_json(staging / "manifest.json", manifest.model_dump(mode="json"))
        if target.exists():
            raise FileExistsError(f"Task 06 defect pair output already exists: {target}")
        os.replace(staging, target)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_defect_pair_manifest_v2_1(path: Path) -> DefectPairManifestV21:
    return DefectPairManifestV21.model_validate(json.loads(path.read_text(encoding="utf-8")))


__all__ = [
    "BUILD_CONTRACT",
    "POLICY_CONTRACT",
    "RELEASED_AXIS",
    "CertifiedCandidateV21",
    "DefectPairManifestV21",
    "DefectPairPolicyV21",
    "DefectPairV21",
    "GroupOutcomeV21",
    "PairFailureV21",
    "build_defect_pairs_v2_1",
    "build_group_pair",
    "certify_group",
    "chosen_admissible",
    "defect_labels",
    "load_defect_pair_manifest_v2_1",
    "load_defect_pair_policy_v2_1",
    "load_pinned_verdicts",
    "rejected_admissible",
]
