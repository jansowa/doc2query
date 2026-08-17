"""Fail-closed builder of tentative Task 06 preference pairs.

The builder turns already-scored same-prompt candidate groups into at most one
``chosen``/``rejected`` pair per prompt.  Every threshold comes from an externally
frozen policy (``configs/preferences/task06_tentative_pair_policy_v1.yaml``, ADR
``reports/decisions/task06_tentative_pair_policy_v1.md``); this module never invents,
calibrates or relaxes one.

Signal contract, frozen by that ADR:

* the primary reranker margin (``pool_margin``) is the **only** ordering signal;
* the shadow reranker may only **veto** a finished pair, never select a candidate;
* corpus round-trip is an independent filter, not a ranking key;
* ``entity_preservation`` is excluded entirely: it is a hallucinated-entity detector,
  not a specificity signal.

Nothing here loads a model, touches a final test split or authorizes DPO training.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import Field, model_validator

from doc2query.evaluation.d01_usefulness import _copy_risk as copy_risk_flag
from doc2query.preferences.build import normalized_query_jaccard
from doc2query.preferences.diversity_gate import (
    SameGroupDiversityGateManifest,
    _reject_final_test_path,
)
from doc2query.schemas import StrictModel
from doc2query.training.dpo import (
    SHA256_PATTERN,
    canonical_fingerprint,
    file_sha256,
    normalize_task06_query,
    ordered_ids_fingerprint,
)
from doc2query.utils.records import JsonlWriter, read_records, write_json

POLICY_CONTRACT = "task06-tentative-pair-policy-v1"
BUILD_CONTRACT = "task06-tentative-pairs-v1"
BUILD_STATUS = "tentative_pairs_built_not_audited"
SCORING_STATUS = "measured"
LEAD_IN_GUARD_ID = "task06_lead_in_guard_v1"

# Domknięcie zmierzonej ślepej plamki `format_valid` (predykcja P4 korpusu
# walidacyjnego nagrody): wtrącenie „Oto ...” bez dwukropka przechodzi przez
# `_PREFIX` w evaluation/format.py, którego ten guard celowo NIE zmienia.
_LEAD_IN_GUARD = re.compile(r"^(?:oto|otóż)\b", re.IGNORECASE | re.UNICODE)


class PairFailure(StrEnum):
    GROUP_NOT_GATE_ELIGIBLE = "group_not_gate_eligible"
    NO_ADMISSIBLE_CHOSEN = "no_admissible_chosen"
    NO_ADMISSIBLE_REJECTED = "no_admissible_rejected"
    NO_CANDIDATE_BELOW_MARGIN_GAP = "no_candidate_below_margin_gap"
    NEAR_DUPLICATE_QUERY_PAIR = "near_duplicate_query_pair"
    SHADOW_VETO = "shadow_veto"


class PairingThresholds(StrictModel):
    strategy: Literal["top_vs_near_miss"]
    max_pairs_per_group: Literal[1]
    require_exact_same_prompt: Literal[True]
    require_diversity_gate_eligible: Literal[True]
    restrict_to_gate_representatives: Literal[True]
    max_normalized_query_jaccard: float = Field(ge=0.0, le=1.0)


class PrimaryPolicy(StrictModel):
    judge: str = Field(min_length=1)
    signal: Literal["pool_margin"]
    role: Literal["sole_pair_building_signal"]
    min_chosen_margin_exclusive: float
    min_margin_gap: float = Field(gt=0.0)


class ShadowPolicy(StrictModel):
    judge: str = Field(min_length=1)
    signal: Literal["shadow_pool_margin"]
    role: Literal["veto_only_never_selection"]
    veto_on_margin_inversion: Literal[True]
    veto_on_rank_inversion: Literal[True]


class CorpusRoundTripPolicy(StrictModel):
    role: Literal["independent_filter"]
    chosen_required_field: str = Field(pattern=r"^corpus_round_trip_at_\d+$")
    rejected_required_field: str = Field(pattern=r"^corpus_round_trip_at_\d+$")


class FormatPolicy(StrictModel):
    require_format_valid: Literal[True]
    forbid_prefix: Literal[True]
    forbid_metacomment: Literal[True]
    forbid_multiple_query: Literal[True]
    forbid_empty: Literal[True]
    supplementary_lead_in_guard: Literal["task06_lead_in_guard_v1"]


class CopyRiskPolicy(StrictModel):
    inherited_from: Literal["task05-d01-copy-semantic-quality-v1"]
    minimum_query_words: int = Field(ge=1)
    copy_density: float = Field(ge=0.0)
    normalized_lcs: float = Field(ge=0.0)
    longest_copied_ngram: float = Field(ge=0.0)
    query_to_passage_length_ratio: float = Field(gt=0.0)
    reject_chosen_on_copy_risk: Literal[True]
    reject_rejected_on_copy_risk: Literal[False]

    def thresholds(self) -> dict[str, Any]:
        return {
            "minimum_query_words": self.minimum_query_words,
            "copy_density": self.copy_density,
            "normalized_lcs": self.normalized_lcs,
            "longest_copied_ngram": self.longest_copied_ngram,
            "query_to_passage_length_ratio": self.query_to_passage_length_ratio,
        }


class FocusPolicy(StrictModel):
    role: Literal["weak_filter_only"]
    reject_chosen_on_zero_focus_accuracy: Literal[True]
    abstention_penalized: Literal[False]


class ExcludedSignal(StrictModel):
    name: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class AuditSamplePolicy(StrictModel):
    target_pair_count: int = Field(ge=1)
    seed: int = Field(ge=0)
    strata: list[str] = Field(min_length=1)
    primary_margin_gap_bands: list[list[float]] = Field(min_length=1)
    allocation: Literal["proportional_largest_remainder"]
    ordering: Literal["pair_id"]
    orientation: Literal["deterministic_counterbalanced_committed_before_review"]

    @model_validator(mode="after")
    def bands_are_contiguous(self) -> AuditSamplePolicy:
        for band in self.primary_margin_gap_bands:
            if len(band) != 2 or not band[0] < band[1]:
                raise ValueError("every gap band must be an increasing [low, high) pair")
        lows = [band[0] for band in self.primary_margin_gap_bands]
        highs = [band[1] for band in self.primary_margin_gap_bands]
        if lows != sorted(lows) or highs[:-1] != lows[1:]:
            raise ValueError("gap bands must be sorted and contiguous")
        return self


class TentativePairPolicy(StrictModel):
    schema_version: Literal[1]
    contract: Literal["task06-tentative-pair-policy-v1"]
    policy_id: str = Field(min_length=1)
    status: Literal["frozen_before_pair_read"]
    adr: str = Field(min_length=1)
    pairing: PairingThresholds
    primary: PrimaryPolicy
    shadow: ShadowPolicy
    corpus_round_trip: CorpusRoundTripPolicy
    format: FormatPolicy
    copy_risk: CopyRiskPolicy
    focus: FocusPolicy
    excluded_signals: list[ExcludedSignal] = Field(min_length=1)
    audit_sample: AuditSamplePolicy
    authorized_cohorts: list[str] = Field(min_length=1)
    final_tests_used: list[str] = Field(max_length=0)

    @model_validator(mode="after")
    def excludes_entity_preservation(self) -> TentativePairPolicy:
        if "entity_preservation" not in {row.name for row in self.excluded_signals}:
            raise ValueError("entity_preservation must stay excluded from the pair policy")
        return self


class TentativePair(StrictModel):
    """One frozen pair; every component that produced it stays visible."""

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
    chosen_candidate_id: str = Field(min_length=1)
    rejected_candidate_id: str = Field(min_length=1)
    chosen: str = Field(min_length=1)
    rejected: str = Field(min_length=1)
    primary_margin_gap: float = Field(gt=0.0)
    chosen_components: dict[str, Any]
    rejected_components: dict[str, Any]
    rejected_failure_types: list[str] = Field(min_length=1)
    normalized_query_jaccard: float = Field(ge=0.0, le=1.0)
    requested_form: str = Field(min_length=1)
    requested_intent: str = Field(min_length=1)
    strategy: Literal["top_vs_near_miss"]
    policy_id: str = Field(min_length=1)
    final_tests_used: list[str] = Field(max_length=0)

    @model_validator(mode="after")
    def pair_is_distinct(self) -> TentativePair:
        if self.chosen_candidate_id == self.rejected_candidate_id:
            raise ValueError("chosen and rejected candidate IDs must differ")
        if normalize_task06_query(self.chosen) == normalize_task06_query(self.rejected):
            raise ValueError("chosen and rejected are identical after Task 06 normalization")
        if self.rejected_failure_types != sorted(set(self.rejected_failure_types)):
            raise ValueError("rejected_failure_types must be unique and sorted")
        return self


class GroupPairOutcome(StrictModel):
    group_id: str = Field(min_length=1)
    gate_eligible: bool
    representative_count: int = Field(ge=0)
    admissible_chosen_count: int = Field(ge=0)
    admissible_rejected_count: int = Field(ge=0)
    paired: bool
    pair_id: str | None = None
    failure_reasons: list[str]

    @model_validator(mode="after")
    def outcome_is_consistent(self) -> GroupPairOutcome:
        if self.failure_reasons != sorted(set(self.failure_reasons)):
            raise ValueError("failure_reasons must be unique and sorted")
        if self.paired != (not self.failure_reasons):
            raise ValueError("paired must equal the absence of failure reasons")
        if self.paired != (self.pair_id is not None):
            raise ValueError("a paired group must carry its pair_id")
        return self


class PairArtifactSummary(StrictModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=SHA256_PATTERN)
    record_count: int = Field(ge=0)


class TentativePairManifest(StrictModel):
    schema_version: Literal[1]
    contract: Literal["task06-tentative-pairs-v1"]
    status: Literal["tentative_pairs_built_not_audited"]
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
    primary_judge: str = Field(min_length=1)
    shadow_judge: str = Field(min_length=1)
    split: Literal["train"]
    group_count: int = Field(ge=1)
    gate_eligible_group_count: int = Field(ge=0)
    candidate_count: int = Field(ge=1)
    pair_count: int = Field(ge=0)
    pair_ids_fingerprint: str = Field(pattern=SHA256_PATTERN)
    pairs: PairArtifactSummary
    group_outcomes: PairArtifactSummary
    report: PairArtifactSummary
    shadow_used_for_selection: Literal[False]
    total_score_computed: Literal[False]
    thresholds_calibrated_here: Literal[False]
    audit_completed: Literal[False]
    task07_training_authorized: Literal[False]
    final_tests_used: list[str] = Field(max_length=0)
    manifest_fingerprint: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def fingerprint_is_valid(self) -> TentativePairManifest:
        if self.pair_count > self.gate_eligible_group_count:
            raise ValueError("a group can contribute at most one pair")
        payload = self.model_dump(mode="json")
        fingerprint = payload.pop("manifest_fingerprint")
        if fingerprint != canonical_fingerprint(payload):
            raise ValueError("tentative pair manifest fingerprint mismatch")
        return self


def load_pair_policy(path: Path) -> TentativePairPolicy:
    """Load an externally frozen pair policy; thresholds are never derived here."""
    _reject_final_test_path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: pair policy must be a mapping")
    policy = TentativePairPolicy.model_validate(raw)
    if policy.format.supplementary_lead_in_guard != LEAD_IN_GUARD_ID:
        raise ValueError("pair policy pins a lead-in guard this build cannot reproduce")
    return policy


def violates_lead_in_guard(text: str) -> bool:
    """Frozen supplementary guard for the measured `format_valid` blind spot."""
    return bool(_LEAD_IN_GUARD.match(" ".join(text.strip().split())))


@dataclass(frozen=True)
class _Candidate:
    candidate_id: str
    candidate_index: int
    query: str
    row: Mapping[str, Any]

    def value(self, field: str) -> Any:
        return self.row.get(field)

    def number(self, field: str) -> float:
        return float(cast(float, self.row[field]))


def _components(candidate: _Candidate) -> dict[str, Any]:
    """Keep every component that entered or documented the decision."""
    fields = (
        "pool_margin",
        "pool_rank",
        "pool_positive_score",
        "shadow_pool_margin",
        "shadow_pool_rank",
        "shadow_score",
        "corpus_round_trip_at_5",
        "corpus_round_trip_at_20",
        "corpus_round_trip_at_100",
        "corpus_margin_to_best_nonpositive",
        "corpus_possibly_ambiguous_query",
        "format_valid",
        "has_prefix",
        "has_metacomment",
        "multiple_query",
        "empty",
        "copy_density",
        "normalized_lcs",
        "longest_copied_ngram",
        "content_jaccard",
        "natural_content_jaccard",
        "passage_recall",
        "query_precision",
        "word_length",
        "focus_accuracy",
        "form_accuracy",
        "intent_accuracy",
        "entity_preservation",
        "number_preservation",
        "judge_rank_disagreement",
        "predicted_focus_bucket",
        "requested_focus",
        "seed",
    )
    payload: dict[str, Any] = {
        field: candidate.value(field) for field in fields if field in candidate.row
    }
    payload["generation_config"] = candidate.value("generation_config")
    payload["control"] = candidate.value("control")
    payload["candidate_index"] = candidate.candidate_index
    return payload


def _format_admissible(candidate: _Candidate) -> bool:
    """Every format flag pinned by the policy is mandatory, hence no threshold argument."""
    return (
        candidate.value("format_valid") is True
        and candidate.value("has_prefix") is False
        and candidate.value("has_metacomment") is False
        and candidate.value("multiple_query") is False
        and candidate.value("empty") is False
        and not violates_lead_in_guard(candidate.query)
    )


def _chosen_admissible(candidate: _Candidate, policy: TentativePairPolicy) -> bool:
    if not _format_admissible(candidate):
        return False
    if candidate.number("pool_margin") <= policy.primary.min_chosen_margin_exclusive:
        return False
    if candidate.number(policy.corpus_round_trip.chosen_required_field) < 1.0:
        return False
    if copy_risk_flag(candidate.row, policy.copy_risk.thresholds()):
        return False
    focus = candidate.value("focus_accuracy")
    return not (focus is not None and float(focus) == 0.0)


def _rejected_admissible(candidate: _Candidate, policy: TentativePairPolicy) -> bool:
    if not _format_admissible(candidate):
        return False
    return candidate.number(policy.corpus_round_trip.rejected_required_field) >= 1.0


def _shadow_vetoes(chosen: _Candidate, rejected: _Candidate) -> bool:
    """Shadow may only invalidate a finished pair; it never picks a candidate."""
    margin_inverted = chosen.number("shadow_pool_margin") < rejected.number("shadow_pool_margin")
    rank_inverted = chosen.number("shadow_pool_rank") > rejected.number("shadow_pool_rank")
    return margin_inverted or rank_inverted


def _rejected_failure_types(
    chosen: _Candidate, rejected: _Candidate, policy: TentativePairPolicy
) -> list[str]:
    labels = {"lower_primary_margin"}
    if rejected.number("corpus_round_trip_at_20") < 1.0:
        labels.add("weak_corpus_round_trip")
    if rejected.value("corpus_possibly_ambiguous_query") is True:
        labels.add("possible_ambiguous_query")
    if copy_risk_flag(rejected.row, policy.copy_risk.thresholds()):
        labels.add("copy_risk")
    if rejected.number("content_jaccard") < chosen.number("content_jaccard"):
        labels.add("lower_content_jaccard_than_chosen")
    focus = rejected.value("focus_accuracy")
    if focus is not None and float(focus) == 0.0:
        labels.add("wrong_focus")
    if rejected.number("shadow_pool_margin") < chosen.number("shadow_pool_margin"):
        labels.add("shadow_agrees")
    if rejected.value("judge_rank_disagreement") is True:
        labels.add("judge_rank_disagreement")
    return sorted(labels)


def _pair_id(cohort_id: str, chosen: _Candidate, rejected: _Candidate) -> str:
    return canonical_fingerprint(
        {
            "cohort_id": cohort_id,
            "chosen_candidate_id": chosen.candidate_id,
            "rejected_candidate_id": rejected.candidate_id,
        }
    )[:32]


def build_group_pair(
    candidates: Sequence[_Candidate],
    *,
    cohort_id: str,
    group_id: str,
    gate_eligible: bool,
    representative_ids: Sequence[str],
    passage_cluster_id: str,
    policy: TentativePairPolicy,
) -> tuple[TentativePair | None, GroupPairOutcome]:
    """Apply the frozen policy to one same-prompt group and emit at most one pair."""
    if not gate_eligible:
        return None, GroupPairOutcome(
            group_id=group_id,
            gate_eligible=False,
            representative_count=0,
            admissible_chosen_count=0,
            admissible_rejected_count=0,
            paired=False,
            failure_reasons=[PairFailure.GROUP_NOT_GATE_ELIGIBLE.value],
        )
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    missing = [value for value in representative_ids if value not in by_id]
    if missing:
        raise ValueError(f"group {group_id} misses gate representatives: {sorted(missing)}")
    representatives = [by_id[value] for value in representative_ids]
    prompts = {str(candidate.row["prompt_sha256"]) for candidate in candidates}
    if len(prompts) != 1:
        raise ValueError(f"group {group_id} does not share one prompt hash")

    ordered = sorted(
        representatives,
        key=lambda candidate: (-candidate.number("pool_margin"), candidate.candidate_index),
    )
    admissible_chosen = [row for row in ordered if _chosen_admissible(row, policy)]
    admissible_rejected = [row for row in ordered if _rejected_admissible(row, policy)]

    def outcome(reasons: Sequence[str], pair_id: str | None = None) -> GroupPairOutcome:
        return GroupPairOutcome(
            group_id=group_id,
            gate_eligible=True,
            representative_count=len(representatives),
            admissible_chosen_count=len(admissible_chosen),
            admissible_rejected_count=len(admissible_rejected),
            paired=pair_id is not None,
            pair_id=pair_id,
            failure_reasons=sorted(set(reasons)),
        )

    if not admissible_chosen:
        return None, outcome([PairFailure.NO_ADMISSIBLE_CHOSEN.value])
    chosen = admissible_chosen[0]
    others = [row for row in admissible_rejected if row.candidate_id != chosen.candidate_id]
    if not others:
        return None, outcome([PairFailure.NO_ADMISSIBLE_REJECTED.value])
    gap_limit = chosen.number("pool_margin") - policy.primary.min_margin_gap
    below_gap = [row for row in others if row.number("pool_margin") <= gap_limit]
    if not below_gap:
        return None, outcome([PairFailure.NO_CANDIDATE_BELOW_MARGIN_GAP.value])
    pairable = [
        row
        for row in below_gap
        if normalized_query_jaccard(chosen.query, row.query)
        <= policy.pairing.max_normalized_query_jaccard
    ]
    if not pairable:
        return None, outcome([PairFailure.NEAR_DUPLICATE_QUERY_PAIR.value])
    rejected = pairable[0]
    if _shadow_vetoes(chosen, rejected):
        return None, outcome([PairFailure.SHADOW_VETO.value])

    pair_id = _pair_id(cohort_id, chosen, rejected)
    positive = cast(Mapping[str, Any], chosen.row["positive"])
    pair = TentativePair(
        pair_id=pair_id,
        cohort_id=cohort_id,
        group_id=group_id,
        example_id=str(chosen.row["example_id"]),
        doc_id=str(chosen.row["doc_id"]),
        passage_cluster_id=passage_cluster_id,
        split="train",
        prompt=str(chosen.row["prompt"]),
        prompt_sha256=str(chosen.row["prompt_sha256"]),
        passage=str(positive["text"]),
        chosen_candidate_id=chosen.candidate_id,
        rejected_candidate_id=rejected.candidate_id,
        chosen=chosen.query,
        rejected=rejected.query,
        primary_margin_gap=chosen.number("pool_margin") - rejected.number("pool_margin"),
        chosen_components=_components(chosen),
        rejected_components=_components(rejected),
        rejected_failure_types=_rejected_failure_types(chosen, rejected, policy),
        normalized_query_jaccard=normalized_query_jaccard(chosen.query, rejected.query),
        requested_form=str(chosen.row["requested_form"]),
        requested_intent=str(chosen.row["requested_intent"]),
        strategy=policy.pairing.strategy,
        policy_id=policy.policy_id,
        final_tests_used=[],
    )
    return pair, outcome([], pair_id)


def _load_gate(gate_dir: Path) -> tuple[SameGroupDiversityGateManifest, dict[str, dict[str, Any]]]:
    manifest_path = gate_dir / "manifest.json"
    verdicts_path = gate_dir / "group_verdicts.jsonl"
    for path in (manifest_path, verdicts_path):
        _reject_final_test_path(path)
        if not path.is_file():
            raise ValueError(f"missing diversity gate input: {path}")
    manifest = SameGroupDiversityGateManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    if file_sha256(verdicts_path) != manifest.verdicts.sha256:
        raise ValueError("diversity gate verdicts drifted from the gate manifest")
    verdicts = {str(row["group_id"]): row for row in read_records(verdicts_path)}
    if len(verdicts) != manifest.group_count:
        raise ValueError("diversity gate verdict count drifted from the gate manifest")
    return manifest, verdicts


def _load_scoring(
    scoring_path: Path, summary_path: Path, gate: SameGroupDiversityGateManifest
) -> list[dict[str, Any]]:
    for path in (scoring_path, summary_path):
        _reject_final_test_path(path)
        if not path.is_file():
            raise ValueError(f"missing candidate scoring input: {path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict) or summary.get("status") != SCORING_STATUS:
        raise ValueError("candidate scoring is not complete under its frozen contract")
    rows = list(read_records(scoring_path))
    if len(rows) != int(summary["generation_count"]) or len(rows) != gate.candidate_count:
        raise ValueError("scored candidate count drifted from its summary or from the gate")
    seen: set[str] = set()
    for row in rows:
        if row.get("final_tests_used") != []:
            raise ValueError("scored candidate declares final-test usage")
        if str(row.get("generation_identity_sha256")) != gate.generation_identity_sha256:
            raise ValueError("scored candidate identity drifted from the gate manifest")
        if str(row.get("frozen_cohort_fingerprint")) != gate.frozen_cohort_fingerprint:
            raise ValueError("scored candidate cohort fingerprint drifted from the gate manifest")
        if str((row.get("metadata") or {}).get("split")) != gate.split:
            raise ValueError("pair builder refuses a split the gate did not clear")
        candidate_id = str(row["evaluation_id"])
        if candidate_id in seen:
            raise ValueError(f"duplicate scored candidate: {candidate_id}")
        seen.add(candidate_id)
    return rows


def _cluster_ids(cohort_records_path: Path) -> dict[str, str]:
    _reject_final_test_path(cohort_records_path)
    if not cohort_records_path.is_file():
        raise ValueError(f"missing frozen cohort records: {cohort_records_path}")
    clusters: dict[str, str] = {}
    for row in read_records(cohort_records_path):
        clusters[str(row["example_id"])] = str(row["cluster_id"])
    if not clusters:
        raise ValueError("frozen cohort records are empty")
    return clusters


def _report(
    pairs: Sequence[TentativePair],
    outcomes: Sequence[GroupPairOutcome],
    policy: TentativePairPolicy,
    *,
    cohort_id: str,
) -> dict[str, Any]:
    eligible = [row for row in outcomes if row.gate_eligible]
    failures = Counter(reason for row in outcomes for reason in row.failure_reasons)
    failure_types = Counter(label for pair in pairs for label in pair.rejected_failure_types)
    forms = Counter(pair.requested_form for pair in pairs)
    intents = Counter(pair.requested_intent for pair in pairs)
    gaps = sorted(pair.primary_margin_gap for pair in pairs)
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
        "failure_reason_counts": dict(sorted(failures.items())),
        "rejected_failure_type_counts": dict(sorted(failure_types.items())),
        "requested_form_counts": dict(sorted(forms.items())),
        "requested_intent_counts": dict(sorted(intents.items())),
        "primary_margin_gap": {
            "count": len(gaps),
            "min": gaps[0] if gaps else None,
            "p25": gaps[len(gaps) // 4] if gaps else None,
            "p50": gaps[len(gaps) // 2] if gaps else None,
            "p75": gaps[(3 * len(gaps)) // 4] if gaps else None,
            "max": gaps[-1] if gaps else None,
        },
        "shadow_used_for_selection": False,
        "total_score_computed": False,
        "thresholds_calibrated_here": False,
        "audit_completed": False,
        "task07_training_authorized": False,
        "final_tests_used": [],
    }


def build_tentative_pairs(
    *,
    cohort_dir: Path,
    policy_path: Path,
    output_dir: Path | None = None,
) -> TentativePairManifest:
    """Build at most one tentative pair per same-prompt group of one frozen cohort."""
    policy = load_pair_policy(policy_path)
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
    target = output_dir if output_dir is not None else cohort_dir / "tentative_pairs"
    if target.exists():
        raise FileExistsError(f"Task 06 tentative pair output already exists: {target}")

    gate, verdicts = _load_gate(gate_dir)
    rows = _load_scoring(scoring_path, summary_path, gate)
    clusters = _cluster_ids(cohort_records_path)

    judges = {(str(row["primary_judge"]), str(row["shadow_judge"])) for row in rows}
    if judges != {(policy.primary.judge, policy.shadow.judge)}:
        raise ValueError("scored candidates were not judged by the judges the policy pins")

    grouped: dict[str, list[_Candidate]] = {}
    for row in rows:
        grouped.setdefault(str(row["evaluation_group_id"]), []).append(
            _Candidate(
                candidate_id=str(row["evaluation_id"]),
                candidate_index=int(row["candidate_index"]),
                query=str(row["generated"]),
                row=row,
            )
        )
    if set(grouped) != set(verdicts):
        raise ValueError("scored groups and diversity gate groups do not cover each other")

    pairs: list[TentativePair] = []
    outcomes: list[GroupPairOutcome] = []
    for group_id in sorted(grouped):
        verdict = verdicts[group_id]
        candidates = grouped[group_id]
        example_id = str(candidates[0].row["example_id"])
        if example_id not in clusters:
            raise ValueError(f"group {group_id} is missing from the frozen cohort records")
        pair, outcome = build_group_pair(
            candidates,
            cohort_id=cohort_id,
            group_id=group_id,
            gate_eligible=bool(verdict["eligible"]),
            representative_ids=[str(value) for value in verdict["representative_candidate_ids"]],
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
        write_json(report_path, _report(pairs, outcomes, policy, cohort_id=cohort_id))
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
            "primary_judge": policy.primary.judge,
            "shadow_judge": policy.shadow.judge,
            "split": "train",
            "group_count": len(outcomes),
            "gate_eligible_group_count": gate.eligible_group_count,
            "candidate_count": len(rows),
            "pair_count": len(pairs),
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
            "shadow_used_for_selection": False,
            "total_score_computed": False,
            "thresholds_calibrated_here": False,
            "audit_completed": False,
            "task07_training_authorized": False,
            "final_tests_used": [],
        }
        payload["manifest_fingerprint"] = canonical_fingerprint(payload)
        manifest = TentativePairManifest.model_validate(payload)
        write_json(staging / "manifest.json", manifest.model_dump(mode="json"))
        if target.exists():
            raise FileExistsError(f"Task 06 tentative pair output already exists: {target}")
        os.replace(staging, target)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
