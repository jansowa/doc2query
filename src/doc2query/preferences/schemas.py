"""Strict, model-free contracts for Task 06 artifacts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

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
