from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

from doc2query.preferences.pair_audit_export import BLIND_FIELDS
from doc2query.preferences.pair_audit_export_v2 import (
    _effective_quotas,
    _sample,
)
from doc2query.preferences.pair_policy import _Candidate
from doc2query.preferences.pair_policy_v2 import (
    DefectPairPolicy,
    axis_preference_order,
    build_defect_pairs,
    build_group_pair,
    load_defect_pair_policy,
    load_pinned_verdicts,
)

POLICY_PATH = Path("configs/preferences/task06_defect_pair_policy_v2.yaml")
GROUP_ID = "task06-preference::1::7"
PASSAGE = (
    "Koronawirusy to duża rodzina wirusów wywołujących choroby układu oddechowego u ludzi "
    "i u zwierząt. Zakażenie przenosi się drogą kropelkową oraz przez kontakt z skażonymi "
    "powierzchniami, a typowe objawy obejmują gorączkę, suchy kaszel, ból gardła i "
    "uczucie duszności. Okres wylęgania wynosi zwykle od dwóch do czternastu dni."
)


def _row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "evaluation_id": "1::7::same-prompt::0",
        "evaluation_group_id": GROUP_ID,
        "example_id": "1::7",
        "doc_id": "7",
        "candidate_index": 0,
        "generated": "jakie objawy wywołuje koronawirus",
        "prompt": "Wygeneruj jedno polskie zapytanie wyszukiwawcze.",
        "prompt_sha256": "a" * 64,
        "positive": {"doc_id": "7", "text": PASSAGE},
        "metadata": {"split": "train"},
        "requested_form": "full_question",
        "requested_intent": "fact_lookup",
        "requested_focus": "beginning",
        "generation_config": {"seed": 1, "temperature": 0.7, "top_p": 0.95},
        "control": {"form": "full_question", "intent": "fact_lookup"},
        "seed": 1,
        "pool_margin": 5.0,
        "pool_rank": 1,
        "pool_positive_score": 9.0,
        "shadow_pool_margin": 4.0,
        "shadow_pool_rank": 1,
        "corpus_round_trip_at_5": 1.0,
        "corpus_round_trip_at_20": 1.0,
        "corpus_round_trip_at_100": 1.0,
        "corpus_possibly_ambiguous_query": False,
        "format_valid": True,
        "has_prefix": False,
        "has_metacomment": False,
        "multiple_query": False,
        "empty": False,
        "copy_density": 0.1,
        "normalized_lcs": 0.1,
        "longest_copied_ngram": 1,
        "content_jaccard": 0.03,
        "entity_preservation": 1.0,
        "word_length": 5,
        "focus_accuracy": None,
        "judge_rank_disagreement": False,
        "primary_judge": "sdadas/polish-reranker-roberta-v3",
        "shadow_judge": "BAAI/bge-reranker-v2-m3",
        "final_tests_used": [],
    }
    row.update(overrides)
    return row


def _candidate(index: int, query: str, **overrides: Any) -> _Candidate:
    row = _row(
        evaluation_id=f"1::7::same-prompt::{index}",
        candidate_index=index,
        generated=query,
        **overrides,
    )
    return _Candidate(
        candidate_id=str(row["evaluation_id"]),
        candidate_index=index,
        query=query,
        row=row,
    )


def _policy() -> DefectPairPolicy:
    return load_defect_pair_policy(POLICY_PATH)


def _certified(
    items: list[tuple[_Candidate, str]],
) -> list[Any]:
    """Build certified candidates without touching the judge-item hash join."""
    from doc2query.preferences.pair_policy_v2 import CertifiedCandidate, _mean_group_jaccard

    queries = [candidate.query for candidate, _ in items]
    return [
        CertifiedCandidate(
            candidate=candidate,
            verdict=verdict,
            mean_group_jaccard=_mean_group_jaccard(
                candidate.query, queries[:index] + queries[index + 1 :]
            ),
        )
        for index, (candidate, verdict) in enumerate(items)
    ]


def _pair(items: list[tuple[_Candidate, str]], **overrides: Any) -> tuple[Any, Any]:
    kwargs: dict[str, Any] = {
        "cohort_id": "same_prompt_expansion_v1",
        "group_id": GROUP_ID,
        "gate_eligible": True,
        "passage_cluster_id": "7",
        "policy": _policy(),
    }
    kwargs.update(overrides)
    return build_group_pair(_certified(items), **kwargs)


# --- zamrożony kontrakt polityki ------------------------------------------------


def test_frozen_policy_degrades_margin_and_releases_exactly_axes_a_and_b() -> None:
    policy = _policy()

    assert policy.primary.role == "chosen_side_sanity_only"
    assert policy.primary.used_for_ordering is False
    assert policy.primary.used_for_tie_break is False
    assert policy.shadow.veto_on_margin_inversion is False
    assert policy.shadow.veto_on_rank_inversion is False
    assert [axis.id for axis in policy.axes if axis.in_release] == ["A", "B"]
    assert policy.axis("C").in_release is False
    assert policy.axis("B").rejected_min_overlap == 0.0857
    assert policy.axis("B").chosen_max_overlap == 0.0556
    assert policy.tie_break.variant == "divpo"
    assert policy.constructed_rejected.enabled is False
    assert policy.entity_preservation.claimed_as_hallucination_filter is False
    assert policy.focus.role == "reported_label_only"
    assert policy.audit_sample.axis_quotas == {"A": 250, "B": 250}
    assert policy.audit_sample.seed == 20260820
    assert policy.authorized_cohorts == [
        "same_prompt_expansion_v1",
        "same_prompt_expansion_v2",
        "same_prompt_expansion_v3",
    ]
    assert policy.final_tests_used == []


def test_margin_may_not_return_as_a_stratification_dimension(tmp_path: Path) -> None:
    raw = yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))
    raw["audit_sample"]["strata"] = ["cohort_id", "axis", "primary_margin_gap_band"]
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")

    with pytest.raises(ValueError, match="margin must not be a stratification dimension"):
        load_defect_pair_policy(path)


def test_axis_b_cuts_must_leave_a_positive_overlap_gap(tmp_path: Path) -> None:
    raw = yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))
    for axis in raw["axes"]:
        if axis["id"] == "B":
            axis["chosen_max_overlap"] = 0.2
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")

    with pytest.raises(ValueError, match="positive overlap gap"):
        load_defect_pair_policy(path)


def test_axis_b_rejected_must_stay_answerable_or_the_axis_collapses_into_a(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))
    for axis in raw["axes"]:
        if axis["id"] == "B":
            axis["rejected_requires_judge_yes"] = False
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")

    with pytest.raises(ValueError, match="collapses into A"):
        load_defect_pair_policy(path)


def test_axis_quotas_must_sum_to_the_target(tmp_path: Path) -> None:
    raw = yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))
    raw["audit_sample"]["axis_quotas"] = {"A": 300, "B": 250}
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")

    with pytest.raises(ValueError, match="must sum to the target"):
        load_defect_pair_policy(path)


# --- przypisanie osi ------------------------------------------------------------


def test_axis_assignment_is_deterministic_and_a_permutation() -> None:
    policy = _policy()

    first = axis_preference_order("grupa-1", policy)
    assert first == axis_preference_order("grupa-1", policy)
    assert sorted(first) == ["A", "B"]

    seen = Counter(
        axis_preference_order(f"task06-preference::{index}", policy)[0]
        for index in range(2000)
    )
    # Hasz nie jest kwotą; wymagamy tylko, żeby nie degenerował się do jednej osi.
    assert 800 < seen["A"] < 1200
    assert seen["A"] + seen["B"] == 2000


def test_group_falls_back_to_the_other_axis_when_the_preferred_one_cannot_pair() -> None:
    policy = _policy()
    order = axis_preference_order(GROUP_ID, policy)
    items = [
        (_candidate(0, "jakie objawy wywołuje koronawirus"), "yes"),
        (_candidate(1, "ile dni trwa okres wylęgania zakażenia"), "yes"),
        (_candidate(2, "zupełnie inne pytanie o rozkład jazdy pociągów"), "no"),
    ]

    pair, outcome = _pair(items)

    assert pair is not None
    # Overlap wszystkich kandydatów jest poniżej cięcia osi B, więc parę da tylko oś A.
    assert pair.axis == "A"
    assert outcome.axis_preference_order == list(order)
    if order[0] == "B":
        assert outcome.attempts[0].failure_reason is not None


# --- oś A -----------------------------------------------------------------------


def test_axis_a_chosen_needs_both_cleanliness_and_a_yes_verdict() -> None:
    items = [
        (_candidate(0, "jakie objawy wywołuje koronawirus"), "no"),
        (_candidate(1, "ile dni trwa okres wylęgania zakażenia"), "uncertain"),
        (_candidate(2, "czym jest rodzina wirusów oddechowych", corpus_round_trip_at_20=0.0),
         "yes"),
    ]

    pair, outcome = _pair(items)

    assert pair is None
    assert outcome.failure_reasons == ["no_admissible_chosen"]


def test_uncertain_blocks_chosen_and_is_never_a_defect_on_its_own() -> None:
    items = [
        (_candidate(0, "jakie objawy wywołuje koronawirus"), "yes"),
        (_candidate(1, "ile dni trwa okres wylęgania zakażenia"), "uncertain"),
    ]

    pair, outcome = _pair(items)

    assert pair is None
    assert outcome.failure_reasons == ["no_axis_defect_rejected"]


def test_uncertain_with_a_failed_round_trip_is_a_defect_by_the_round_trip() -> None:
    items = [
        (_candidate(0, "jakie objawy wywołuje koronawirus"), "yes"),
        (
            _candidate(1, "ile dni trwa okres wylęgania zakażenia",
                       corpus_round_trip_at_100=0.0),
            "uncertain",
        ),
    ]

    pair, _ = _pair(items)

    assert pair is not None
    assert pair.axis == "A"
    assert pair.rejected_verdict == "uncertain"
    assert pair.rejected_defect_labels == ["weak_corpus_round_trip"]


def test_axis_a_rejected_accepts_an_unanswerable_verdict_with_a_full_round_trip() -> None:
    items = [
        (_candidate(0, "jakie objawy wywołuje koronawirus"), "yes"),
        (_candidate(1, "kto wynalazł mikroskop elektronowy"), "no"),
    ]

    pair, _ = _pair(items)

    assert pair is not None
    assert "judge_unanswerable" in pair.rejected_defect_labels


# --- oś B -----------------------------------------------------------------------


def _axis_b_items() -> list[tuple[_Candidate, str]]:
    return [
        (_candidate(0, "jakie objawy wywołuje koronawirus", content_jaccard=0.02), "yes"),
        (
            _candidate(
                1,
                "typowe objawy obejmują gorączkę suchy kaszel ból gardła duszności",
                content_jaccard=0.30,
            ),
            "yes",
        ),
    ]


def test_axis_b_pairs_a_low_overlap_chosen_against_a_high_overlap_rejected() -> None:
    pair, _ = _pair(_axis_b_items())

    assert pair is not None
    assert pair.axis == "B"
    assert pair.chosen_candidate_id == "1::7::same-prompt::0"
    assert "high_lexical_overlap" in pair.rejected_defect_labels
    assert pair.rejected_verdict == "yes"


def test_axis_b_refuses_an_unanswerable_rejected_side() -> None:
    items = _axis_b_items()
    items[1] = (items[1][0], "no")

    pair, _ = _pair(items)

    # Werdykt `no` czyni z kandydata defekt osi A, nie osi B — pary osi B nie ma.
    assert pair is not None
    assert pair.axis == "A"


def test_axis_b_chosen_must_sit_below_the_frozen_lower_cut() -> None:
    items = [
        (_candidate(0, "jakie objawy wywołuje koronawirus", content_jaccard=0.07), "yes"),
        (
            _candidate(
                1,
                "typowe objawy obejmują gorączkę suchy kaszel ból gardła duszności",
                content_jaccard=0.30,
            ),
            "yes",
        ),
    ]

    pair, outcome = _pair(items)

    assert pair is None
    assert "no_admissible_chosen" in outcome.failure_reasons


# --- tie-break DivPO i degradacja marginesu ------------------------------------


def test_divpo_tie_break_picks_the_most_distinct_chosen_not_the_highest_margin() -> None:
    items = [
        # Najwyższy margines, ale leksykalnie bliźniaczy z kandydatem 1.
        (_candidate(0, "jakie objawy wywołuje koronawirus", pool_margin=9.0), "yes"),
        (_candidate(1, "jakie objawy wywołuje zakażenie koronawirusem", pool_margin=8.0), "yes"),
        # Odrębny w grupie, choć o najniższym marginesie.
        (_candidate(2, "ile dni trwa okres wylęgania", pool_margin=0.5), "yes"),
        (_candidate(3, "kto wynalazł mikroskop elektronowy", pool_margin=4.0), "no"),
    ]

    pair, _ = _pair(items)

    assert pair is not None
    assert pair.chosen_candidate_id == "1::7::same-prompt::2"
    assert pair.chosen_group_distinctness < 0.3
    # Margines był mniejszy niż u odrzuconego — polityka go nie porządkuje.
    assert pair.primary_margin_delta < 0.0
    assert pair.margin_used_for_ordering is False


def test_divpo_tie_break_picks_the_most_typical_rejected() -> None:
    items = [
        (_candidate(0, "ile dni trwa okres wylęgania"), "yes"),
        # Dwa defektowe rejected: bliźniak kandydata 2 jest bardziej typowy.
        (_candidate(1, "kto wynalazł mikroskop elektronowy"), "no"),
        (_candidate(2, "kto wynalazł mikroskop optyczny"), "no"),
    ]

    pair, _ = _pair(items)

    assert pair is not None
    assert pair.rejected_candidate_id in {"1::7::same-prompt::1", "1::7::same-prompt::2"}
    assert pair.rejected_group_typicality >= pair.chosen_group_distinctness


def test_shadow_inversion_no_longer_vetoes_a_defect_anchored_pair() -> None:
    items = [
        (_candidate(0, "jakie objawy wywołuje koronawirus", shadow_pool_margin=0.1,
                    shadow_pool_rank=9), "yes"),
        (_candidate(1, "kto wynalazł mikroskop elektronowy", shadow_pool_margin=5.0,
                    shadow_pool_rank=1), "no"),
    ]

    pair, _ = _pair(items)

    assert pair is not None
    assert "shadow_agrees" not in pair.rejected_defect_labels


def test_zero_focus_accuracy_no_longer_blocks_chosen() -> None:
    items = [
        (_candidate(0, "jakie objawy wywołuje koronawirus", focus_accuracy=0.0), "yes"),
        (_candidate(1, "kto wynalazł mikroskop elektronowy"), "no"),
    ]

    pair, _ = _pair(items)

    assert pair is not None
    assert pair.chosen_candidate_id == "1::7::same-prompt::0"


def test_copy_risk_blocks_chosen_but_remains_a_legitimate_rejected_label() -> None:
    items = [
        (_candidate(0, "koronawirusy to duża rodzina wirusów wywołujących choroby",
                    copy_density=0.9, normalized_lcs=0.9), "yes"),
        (_candidate(1, "ile dni trwa okres wylęgania"), "yes"),
        (_candidate(2, "zakażenie przenosi się drogą kropelkową oraz przez kontakt",
                    copy_density=0.9, normalized_lcs=0.9, corpus_round_trip_at_100=0.0), "yes"),
    ]

    pair, _ = _pair(items)

    assert pair is not None
    assert pair.chosen_candidate_id == "1::7::same-prompt::1"
    assert "copy_risk" in pair.rejected_defect_labels


def test_lead_in_guard_still_blocks_both_roles() -> None:
    items = [
        (_candidate(0, "Oto jakie objawy wywołuje koronawirus"), "yes"),
        (_candidate(1, "Oto kto wynalazł mikroskop elektronowy"), "no"),
    ]

    pair, outcome = _pair(items)

    assert pair is None
    assert outcome.failure_reasons == ["no_admissible_chosen"]


def test_near_duplicate_queries_cannot_be_paired() -> None:
    items = [
        (_candidate(0, "jakie objawy wywołuje koronawirus"), "yes"),
        (_candidate(1, "jakie objawy wywołuje koronawirus?", corpus_round_trip_at_100=0.0),
         "yes"),
    ]

    pair, outcome = _pair(items)

    assert pair is None
    assert "near_duplicate_query_pair" in outcome.failure_reasons


def test_gate_ineligible_group_never_reaches_the_policy() -> None:
    pair, outcome = _pair(
        [(_candidate(0, "jakie objawy wywołuje koronawirus"), "yes")], gate_eligible=False
    )

    assert pair is None
    assert outcome.gate_eligible is False
    assert outcome.failure_reasons == ["group_not_gate_eligible"]
    assert outcome.attempts == []


def test_pair_record_keeps_both_sides_and_declares_no_training_shortcut() -> None:
    items = [
        (_candidate(0, "jakie objawy wywołuje koronawirus"), "yes"),
        (_candidate(1, "kto wynalazł mikroskop elektronowy"), "no"),
    ]

    pair, _ = _pair(items)

    assert pair is not None
    payload: Mapping[str, Any] = json.loads(pair.model_dump_json())
    assert payload["final_tests_used"] == []
    assert payload["constructed_rejected"] is False
    assert payload["margin_used_for_ordering"] is False
    assert payload["chosen_verdict"] == "yes"
    assert payload["passage"] == PASSAGE
    assert "total_score" not in payload
    assert "primary_margin_gap" not in payload
    for role in ("chosen_components", "rejected_components"):
        for field in ("pool_margin", "shadow_pool_margin", "content_jaccard"):
            assert field in payload[role]


# --- fail-closed wejścia --------------------------------------------------------


def test_unauthorized_cohort_is_refused_before_any_read(tmp_path: Path) -> None:
    cohort = tmp_path / "same_prompt_expansion_v7"
    cohort.mkdir()

    with pytest.raises(ValueError, match="not authorized for pair building"):
        build_defect_pairs(cohort_dir=cohort, policy_path=POLICY_PATH, journal_paths=[])


def test_existing_output_directory_is_never_overwritten(tmp_path: Path) -> None:
    cohort = tmp_path / "same_prompt_expansion_v1"
    cohort.mkdir()
    existing = tmp_path / "already_there"
    existing.mkdir()

    with pytest.raises(FileExistsError):
        build_defect_pairs(
            cohort_dir=cohort,
            policy_path=POLICY_PATH,
            journal_paths=[],
            output_dir=existing,
        )


def test_unpinned_verdict_journal_is_refused(tmp_path: Path) -> None:
    journal = tmp_path / "verdicts.jsonl"
    journal.write_text(
        json.dumps({"event": "verdict", "item_id": "abc", "verdict": "yes"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not pinned by"):
        load_pinned_verdicts(_policy(), [journal])


def test_candidate_without_a_verdict_stops_the_build() -> None:
    from doc2query.preferences.pair_policy_v2 import certify_group

    with pytest.raises(ValueError, match="no answerability verdict"):
        certify_group([_candidate(0, "jakie objawy wywołuje koronawirus")], {})


# --- kwoty osi w eksporcie ------------------------------------------------------


def _pair_row(pair_id: str, axis: str, form: str = "full_question") -> dict[str, Any]:
    return {
        "pair_id": pair_id,
        "cohort_id": "same_prompt_expansion_v1",
        "axis": axis,
        "requested_form": form,
    }


def test_axis_quota_is_balanced_when_both_axes_have_supply() -> None:
    policy = _policy()
    pairs = [_pair_row(f"a{index:04d}", "A") for index in range(400)]
    pairs += [_pair_row(f"b{index:04d}", "B") for index in range(400)]

    sampled, _, quotas = _sample(pairs, policy)

    assert len(sampled) == 500
    assert Counter(row["axis"] for row in sampled) == {"A": 250, "B": 250}
    assert all(row.shortfall == 0 for row in quotas)


def test_unused_axis_quota_is_reallocated_and_the_shortfall_is_reported() -> None:
    policy = _policy()
    populations = {"A": 400, "B": 100}

    effective = _effective_quotas(populations, policy)

    assert effective == {"A": 400, "B": 100}

    pairs = [_pair_row(f"a{index:04d}", "A") for index in range(400)]
    pairs += [_pair_row(f"b{index:04d}", "B") for index in range(100)]
    sampled, _, quotas = _sample(pairs, policy)

    assert Counter(row["axis"] for row in sampled) == {"A": 400, "B": 100}
    assert {row.axis: row.shortfall for row in quotas} == {"A": 0, "B": 0}


def test_a_genuine_shortfall_is_never_filled_by_relaxing_anything() -> None:
    policy = _policy()
    pairs = [_pair_row(f"a{index:04d}", "A") for index in range(120)]
    pairs += [_pair_row(f"b{index:04d}", "B") for index in range(80)]

    sampled, _, quotas = _sample(pairs, policy)

    assert len(sampled) == 200
    assert sum(row.effective_quota for row in quotas) == 200
    assert policy.audit_sample.target_pair_count - len(sampled) == 300


def test_sampling_is_deterministic_under_the_frozen_seed() -> None:
    policy = _policy()
    pairs = [_pair_row(f"a{index:04d}", "A") for index in range(700)]
    pairs += [_pair_row(f"b{index:04d}", "B") for index in range(700)]

    first = [row["pair_id"] for row in _sample(pairs, policy)[0]]
    second = [row["pair_id"] for row in _sample(list(reversed(pairs)), policy)[0]]

    assert first == second


def test_blind_field_set_is_inherited_from_the_frozen_v1_export() -> None:
    assert BLIND_FIELDS == (
        "audit_id",
        "passage",
        "query_a",
        "query_b",
        "orientation_commitment",
    )
