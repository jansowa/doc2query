"""Strict, model-free contracts for Task 06 artifacts."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import Field, JsonValue, field_validator, model_validator

from doc2query.schemas import FocusMode, QueryControl, QueryForm, QueryIntent, StrictModel


def _default_planning_splits() -> list[Literal["train", "dev"]]:
    return ["train"]


class CandidateScore(StrictModel):
    """All score components stay visible after composite ranking."""

    ground_score: float
    negative_margin: float
    corpus_round_trip: float = Field(ge=0.0, le=1.0)
    effective_candidate_count: int = Field(ge=0)
    possible_false_negative: bool
    overlap_reward: float
    focus_accuracy: float = Field(ge=0.0, le=1.0)
    style_accuracy: float = Field(ge=0.0, le=1.0)
    format_score: float = Field(ge=0.0, le=1.0)
    copy_penalty: float = Field(ge=0.0)
    answerability_flag: bool
    total_score: float


class CandidateProvenance(StrictModel):
    generator_id: str = Field(min_length=1)
    checkpoint_id: str = Field(min_length=1)
    checkpoint_fingerprint: str = Field(min_length=1)
    generation_config_fingerprint: str = Field(min_length=1)
    primary_judge_id: str = Field(min_length=1)
    primary_judge_revision: str = Field(min_length=1)
    shadow_judge_id: str = Field(min_length=1)
    shadow_judge_revision: str = Field(min_length=1)
    corpus_fingerprint: str = Field(min_length=1)
    scoring_config_fingerprint: str = Field(min_length=1)
    miner_policy_id: str | None = None
    miner_fingerprint: str | None = None


class ScoredCandidate(StrictModel):
    candidate_id: str = Field(min_length=1)
    passage_id: str = Field(min_length=1)
    passage_cluster_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    query: str = Field(min_length=1)
    split: Literal["train", "dev", "test"]
    controls: dict[str, Any] = Field(default_factory=dict)
    generation: dict[str, Any] = Field(default_factory=dict)
    scores: CandidateScore
    provenance: CandidateProvenance
    failure_types: list[str] = Field(default_factory=list)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        normalized = " ".join(value.strip().split())
        if not normalized or "\n" in value or "\r" in value:
            raise ValueError("query must be a non-empty single line")
        return normalized

    @field_validator("failure_types")
    @classmethod
    def unique_failure_types(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("failure_types must be unique and sorted")
        return value


class SelectionPolicy(StrictModel):
    strategy: Literal["top_vs_bottom", "top_vs_near_miss"] = "top_vs_near_miss"
    min_score_margin: float = Field(default=0.25, gt=0.0)
    max_pairs_per_passage: int = Field(default=1, ge=1)
    min_chosen_format_score: float = Field(default=1.0, ge=0.0, le=1.0)
    min_rejected_format_score: float = Field(default=1.0, ge=0.0, le=1.0)
    min_rejected_ground_score: float | None = None
    require_chosen_answerable: bool = True
    reject_possible_false_negative_chosen: bool = True
    max_normalized_query_jaccard: float = Field(default=0.85, ge=0.0, le=1.0)


class CandidateSet(StrictModel):
    set_id: str = Field(min_length=1)
    passage_id: str = Field(min_length=1)
    passage_cluster_id: str = Field(min_length=1)
    split: Literal["train", "dev", "test"]
    prompt: str = Field(min_length=1)
    chosen_candidate_id: str = Field(min_length=1)
    rejected_candidate_ids: list[str] = Field(min_length=1)
    score_margins: list[float] = Field(min_length=1)
    strategy: Literal["top_vs_bottom", "top_vs_near_miss"]

    @model_validator(mode="after")
    def aligned_rejections(self) -> CandidateSet:
        if len(self.rejected_candidate_ids) != len(self.score_margins):
            raise ValueError("rejected_candidate_ids and score_margins must align")
        if self.chosen_candidate_id in self.rejected_candidate_ids:
            raise ValueError("chosen candidate cannot also be rejected")
        if len(set(self.rejected_candidate_ids)) != len(self.rejected_candidate_ids):
            raise ValueError("rejected candidate IDs must be unique")
        return self


class CandidatePlanningConfig(StrictModel):
    """Model-independent axes for a future, separately authorized generation run."""

    plan_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    plan_seed: int = Field(default=42, ge=0, le=2**32 - 1)
    target_candidates_per_passage: int = Field(default=8, ge=4, le=8)
    forms: list[QueryForm] = Field(min_length=1)
    intents: list[QueryIntent] = Field(min_length=1)
    focus_modes: list[FocusMode] = Field(min_length=1)
    temperatures: list[float] = Field(min_length=2)
    seeds: list[int] = Field(min_length=2)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    max_new_tokens: int = Field(default=64, ge=1, le=96)
    allowed_splits: list[Literal["train", "dev"]] = Field(default_factory=_default_planning_splits)

    @field_validator("forms", "intents", "focus_modes", "temperatures", "seeds", "allowed_splits")
    @classmethod
    def axes_are_unique(cls, value: list[Any]) -> list[Any]:
        if len(value) != len(set(value)):
            raise ValueError("planning axes must contain unique values")
        return value

    @field_validator("temperatures")
    @classmethod
    def temperatures_are_valid(cls, value: list[float]) -> list[float]:
        if any(item <= 0.0 or item > 5.0 for item in value):
            raise ValueError("temperatures must be in (0, 5]")
        return value

    @field_validator("seeds")
    @classmethod
    def seeds_are_valid(cls, value: list[int]) -> list[int]:
        if any(item < 0 or item > 2**32 - 1 for item in value):
            raise ValueError("seeds must be between 0 and 2**32 - 1")
        return value


class CandidateGenerationRequest(StrictModel):
    request_id: str = Field(min_length=1)
    plan_id: str = Field(min_length=1)
    plan_fingerprint: str = Field(min_length=64, max_length=64)
    candidate_index: int = Field(ge=0, le=7)
    passage_id: str = Field(min_length=1)
    passage_cluster_id: str = Field(min_length=1)
    passage: str = Field(min_length=1)
    source_pair_ids: list[str] = Field(min_length=1)
    split: Literal["train", "dev"]
    prompt: str = Field(min_length=1)
    control: QueryControl
    temperature: float = Field(gt=0.0, le=5.0)
    top_p: float = Field(gt=0.0, le=1.0)
    max_new_tokens: int = Field(ge=1, le=96)
    seed: int = Field(ge=0, le=2**32 - 1)

    @field_validator("source_pair_ids")
    @classmethod
    def source_pairs_are_unique(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("source_pair_ids must be unique and sorted")
        return value


class DecodingParameters(StrictModel):
    """Complete decoding recipe, including model-specific JSON-safe options."""

    do_sample: bool
    temperature: float = Field(gt=0.0, le=5.0)
    top_p: float = Field(gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=0)
    typical_p: float | None = Field(default=None, gt=0.0, le=1.0)
    min_p: float | None = Field(default=None, ge=0.0, le=1.0)
    max_new_tokens: int = Field(ge=1, le=96)
    min_new_tokens: int = Field(default=0, ge=0, le=96)
    repetition_penalty: float = Field(default=1.0, gt=0.0)
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = Field(default=0, ge=0)
    num_beams: int = Field(default=1, ge=1)
    seed: int = Field(ge=0, le=2**32 - 1)
    stop_sequences: list[str] = Field(default_factory=list)
    implementation_parameters: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("stop_sequences")
    @classmethod
    def stop_sequences_are_unique(cls, value: list[str]) -> list[str]:
        if any(not item for item in value) or value != sorted(set(value)):
            raise ValueError("stop_sequences must be non-empty, unique and sorted")
        return value

    @model_validator(mode="after")
    def sampling_parameters_are_coherent(self) -> DecodingParameters:
        if self.min_new_tokens > self.max_new_tokens:
            raise ValueError("min_new_tokens cannot exceed max_new_tokens")
        return self


class GeneratorProvenance(StrictModel):
    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    checkpoint_id: str = Field(min_length=1)
    checkpoint_fingerprint: str = Field(min_length=1)
    adapter_id: str | None = None
    adapter_fingerprint: str | None = None
    plan_id: str = Field(min_length=1)
    plan_fingerprint: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    decoding: DecodingParameters

    @model_validator(mode="after")
    def adapter_identity_is_complete(self) -> GeneratorProvenance:
        if (self.adapter_id is None) != (self.adapter_fingerprint is None):
            raise ValueError("adapter_id and adapter_fingerprint must be provided together")
        return self


class TokenLogprob(StrictModel):
    token: str
    token_id: int = Field(ge=0)
    logprob: float

    @field_validator("logprob")
    @classmethod
    def logprob_is_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("logprob must be finite")
        return value


class GeneratedCandidate(StrictModel):
    """Unscored generator output tied to exactly one planned request."""

    candidate_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    plan_id: str = Field(min_length=1)
    plan_fingerprint: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    passage_id: str = Field(min_length=1)
    passage_cluster_id: str = Field(min_length=1)
    passage: str = Field(min_length=1)
    split: Literal["train", "dev"]
    prompt: str = Field(min_length=1)
    query: str = Field(min_length=1)
    control: QueryControl
    provenance: GeneratorProvenance
    token_logprobs: list[TokenLogprob] | None = None
    sequence_logprob: float | None = None
    attempt: int | None = Field(default=None, ge=1)
    format_valid: bool | None = None
    duplicate_within_request: bool | None = None
    duplicate_candidate_ids: list[str] = Field(default_factory=list)

    @field_validator("query")
    @classmethod
    def generated_query_is_single_line(cls, value: str) -> str:
        normalized = " ".join(value.strip().split())
        if not normalized or "\n" in value or "\r" in value:
            raise ValueError("query must be a non-empty single line")
        return normalized

    @field_validator("duplicate_candidate_ids")
    @classmethod
    def duplicate_ids_are_unique(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("duplicate_candidate_ids must be unique and sorted")
        return value

    @field_validator("sequence_logprob")
    @classmethod
    def sequence_logprob_is_finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("sequence_logprob must be finite")
        return value

    @model_validator(mode="after")
    def provenance_matches_candidate(self) -> GeneratedCandidate:
        if (self.plan_id, self.plan_fingerprint) != (
            self.provenance.plan_id,
            self.provenance.plan_fingerprint,
        ):
            raise ValueError("candidate and generator provenance plan identity differ")
        if self.candidate_id in self.duplicate_candidate_ids:
            raise ValueError("candidate cannot list itself as a duplicate")
        return self


class EvidenceIdentity(StrictModel):
    candidate_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    plan_id: str = Field(min_length=1)
    plan_fingerprint: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    passage_id: str = Field(min_length=1)
    passage_cluster_id: str = Field(min_length=1)
    passage: str = Field(min_length=1)
    split: Literal["train", "dev"]


class JudgeEvidence(EvidenceIdentity):
    judge_role: Literal["primary", "shadow"]
    judge_id: str = Field(min_length=1)
    judge_revision: str = Field(min_length=1)
    raw_score_scale_id: str = Field(min_length=1)
    positive_score: float
    max_negative_score: float
    margin: float
    positive_rank: int = Field(ge=1)
    candidate_count: int = Field(ge=1)
    best_sentence_score: float | None = None
    all_scores_close: bool
    scoring_config_fingerprint: str = Field(min_length=1)

    @field_validator("positive_score", "max_negative_score", "margin", "best_sentence_score")
    @classmethod
    def judge_scores_are_finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("judge scores must be finite")
        return value

    @model_validator(mode="after")
    def rank_fits_candidate_pool(self) -> JudgeEvidence:
        if self.positive_rank > self.candidate_count:
            raise ValueError("positive_rank cannot exceed candidate_count")
        return self


class PrimaryJudgeEvidence(JudgeEvidence):
    judge_role: Literal["primary"] = "primary"


class ShadowJudgeEvidence(JudgeEvidence):
    judge_role: Literal["shadow"] = "shadow"


class CorpusRetrievalEvidence(EvidenceIdentity):
    retriever_id: str = Field(min_length=1)
    retriever_revision: str = Field(min_length=1)
    corpus_fingerprint: str = Field(min_length=1)
    source_rank: int = Field(ge=1)
    candidate_count: int = Field(ge=1)
    reciprocal_rank: float = Field(ge=0.0, le=1.0)
    recall_at_1: bool
    recall_at_5: bool
    ndcg_at_10: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def retrieval_flags_match_rank(self) -> CorpusRetrievalEvidence:
        if self.source_rank > self.candidate_count:
            raise ValueError("source_rank cannot exceed candidate_count")
        if self.recall_at_1 != (self.source_rank <= 1):
            raise ValueError("recall_at_1 does not match source_rank")
        if self.recall_at_5 != (self.source_rank <= 5):
            raise ValueError("recall_at_5 does not match source_rank")
        expected_rr = 1.0 / self.source_rank
        if not math.isclose(self.reciprocal_rank, expected_rr, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("reciprocal_rank does not match source_rank")
        return self


class LexicalCopyEvidence(EvidenceIdentity):
    content_lemma_jaccard: float = Field(ge=0.0, le=1.0)
    content_lemma_precision: float = Field(ge=0.0, le=1.0)
    content_lemma_recall: float = Field(ge=0.0, le=1.0)
    longest_common_ngram: int = Field(ge=0)
    longest_common_subsequence_ratio: float = Field(ge=0.0, le=1.0)
    entity_preservation: float | None = Field(default=None, ge=0.0, le=1.0)
    number_unit_preservation: float | None = Field(default=None, ge=0.0, le=1.0)
    copy_risk: bool
    normalization_version: str = Field(min_length=1)


class FocusEvidence(EvidenceIdentity):
    requested_focus_mode: FocusMode
    requested_focus_bucket: Literal["beginning", "middle", "end"] | None = None
    requested_focus_sentence_id: int | None = Field(default=None, ge=0)
    assigned_focus_bucket: Literal["beginning", "middle", "end"] | None = None
    assigned_focus_sentence_id: int | None = Field(default=None, ge=0)
    focus_match: bool | None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    method_id: str = Field(min_length=1)


class StyleEvidence(EvidenceIdentity):
    requested_form: QueryForm
    requested_intent: QueryIntent
    predicted_form: QueryForm
    predicted_intent: QueryIntent
    form_match: bool
    intent_match: bool | None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    classifier_id: str = Field(min_length=1)


class FormatEvidence(EvidenceIdentity):
    valid: bool
    empty: bool
    single_query: bool
    has_meta_commentary: bool
    too_long: bool
    contains_answer: bool
    violation_codes: list[str] = Field(default_factory=list)
    validator_version: str = Field(min_length=1)

    @field_validator("violation_codes")
    @classmethod
    def violation_codes_are_unique(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("violation_codes must be unique and sorted")
        return value

    @model_validator(mode="after")
    def validity_matches_flags(self) -> FormatEvidence:
        has_violation = any(
            (
                self.empty,
                not self.single_query,
                self.has_meta_commentary,
                self.too_long,
                self.contains_answer,
                bool(self.violation_codes),
            )
        )
        if self.valid == has_violation:
            raise ValueError("format validity is inconsistent with violation flags")
        return self


class CandidateEvidenceBundle(StrictModel):
    """Complete pre-calibration evidence. Deliberately has no composite score."""

    contract_version: Literal["task06-candidate-evidence-v1"] = "task06-candidate-evidence-v1"
    candidate: GeneratedCandidate
    primary_judge: PrimaryJudgeEvidence
    shadow_judge: ShadowJudgeEvidence
    corpus_retrieval: CorpusRetrievalEvidence
    lexical_copy: LexicalCopyEvidence
    focus: FocusEvidence
    style: StyleEvidence
    format: FormatEvidence

    @model_validator(mode="after")
    def components_match_candidate_and_controls(self) -> CandidateEvidenceBundle:
        identity = (
            self.candidate.candidate_id,
            self.candidate.request_id,
            self.candidate.plan_id,
            self.candidate.plan_fingerprint,
            self.candidate.passage_id,
            self.candidate.passage_cluster_id,
            self.candidate.passage,
            self.candidate.split,
        )
        for component in (
            self.primary_judge,
            self.shadow_judge,
            self.corpus_retrieval,
            self.lexical_copy,
            self.focus,
            self.style,
            self.format,
        ):
            component_identity = (
                component.candidate_id,
                component.request_id,
                component.plan_id,
                component.plan_fingerprint,
                component.passage_id,
                component.passage_cluster_id,
                component.passage,
                component.split,
            )
            if component_identity != identity:
                raise ValueError("evidence component identity differs from candidate")
        control = self.candidate.control
        if (
            self.focus.requested_focus_mode,
            self.focus.requested_focus_bucket,
            self.focus.requested_focus_sentence_id,
        ) != (control.focus_mode, control.focus_bucket, control.focus_sentence_id):
            raise ValueError("focus evidence differs from candidate control")
        if (self.style.requested_form, self.style.requested_intent) != (
            control.form,
            control.intent,
        ):
            raise ValueError("style evidence differs from candidate control")
        if (
            self.candidate.format_valid is not None
            and self.candidate.format_valid != self.format.valid
        ):
            raise ValueError("format evidence differs from candidate flag")
        return self
