from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from doc2query.preferences.defect_inventory import (
    CandidateProfile,
    classify_candidate,
    load_inventory,
    run_inventory,
    summarize_supply,
)
from doc2query.preferences.pair_policy import load_pair_policy

POLICY_PATH = Path("configs/preferences/task06_tentative_pair_policy_v1_1.yaml")
PASSAGE = (
    "Koronawirusy to duża rodzina wirusów wywołujących choroby układu oddechowego u ludzi "
    "i u zwierząt. Zakażenie przenosi się drogą kropelkową, a typowe objawy obejmują "
    "gorączkę, suchy kaszel i duszność. Okres wylęgania wynosi od dwóch do czternastu dni."
)


def _row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "evaluation_id": "1::7::same-prompt::0",
        "evaluation_group_id": "task06-preference::1::7",
        "generated": "jakie objawy wywołuje koronawirus",
        "positive": {"doc_id": "7", "text": PASSAGE},
        "format_valid": True,
        "has_prefix": False,
        "has_metacomment": False,
        "multiple_query": False,
        "empty": False,
        "corpus_round_trip_at_20": 1.0,
        "corpus_round_trip_at_100": 1.0,
        "entity_preservation": 1.0,
        "pool_margin": 3.0,
        "content_jaccard": 0.25,
        "copy_density": 0.1,
        "normalized_lcs": 0.1,
        "longest_copied_ngram": 1,
        "word_length": 5,
        "focus_accuracy": 1.0,
    }
    row.update(overrides)
    return row


def _profile(index: int, query: str, **overrides: Any) -> CandidateProfile:
    row = _row(
        evaluation_id=f"1::7::same-prompt::{index}", generated=query, **overrides
    )
    return classify_candidate(row, load_pair_policy(POLICY_PATH))


def test_clean_candidate_is_classified_as_chosen_material() -> None:
    profile = _profile(0, "jakie objawy wywołuje koronawirus")

    assert profile.format_admissible
    assert profile.clean_chosen
    assert profile.answerable_proxy
    assert not profile.axis_a_defect


def test_hallucinated_entity_and_round_trip_failure_feed_axis_a() -> None:
    hallucinated = _profile(0, "kiedy powstała szczepionka pfizer", entity_preservation=0.5)
    # W danych rzeczywistych trafienie @20 implikuje trafienie @100 (top-20 ⊂ top-100).
    unretrievable = _profile(
        1, "co to jest wirus", corpus_round_trip_at_100=0.0, corpus_round_trip_at_20=0.0
    )

    assert hallucinated.axis_a_defect and hallucinated.entity_hallucination
    assert not hallucinated.answerable_proxy
    assert unretrievable.axis_a_defect and unretrievable.round_trip_100_fail
    assert not unretrievable.clean_chosen


def test_copy_risk_and_lead_in_block_the_clean_chosen_role() -> None:
    copying = _profile(
        0,
        "koronawirusy to duża rodzina wirusów wywołujących choroby",
        copy_density=0.9,
        normalized_lcs=0.9,
        content_jaccard=0.9,
    )
    lead_in = _profile(1, "Oto jakie objawy wywołuje koronawirus")

    assert not copying.clean_chosen
    assert copying.answerable_proxy  # kopiowanie nie odbiera odpowiadalności
    assert not lead_in.format_admissible
    assert not lead_in.axis_a_defect  # defekt formatu nie wchodzi do osi A


def test_focus_abstention_counts_neither_as_correct_nor_wrong() -> None:
    abstained = _profile(0, "jakie objawy wywołuje koronawirus", focus_accuracy=None)

    assert not abstained.focus_correct
    assert not abstained.focus_wrong


def _cohorts(members: list[CandidateProfile]) -> dict[str, dict[str, list[CandidateProfile]]]:
    return {"same_prompt_expansion_v1": {"task06-preference::1::7": members}}


def test_axis_a_supply_requires_a_pairable_clean_and_defect_candidate() -> None:
    policy = load_pair_policy(POLICY_PATH)
    clean = _profile(0, "jakie objawy wywołuje koronawirus")
    defect = _profile(1, "kiedy powstała szczepionka pfizer", entity_preservation=0.5)

    report = summarize_supply(_cohorts([clean, defect]), policy)

    pooled = report["pooled"]
    assert pooled["eligible_groups"] == 1
    assert pooled["groups_with_clean_chosen"] == 1
    assert pooled["axis_a_pairable_groups"] == 1
    assert pooled["groups_with_entity_hallucination"] == 1


def test_near_duplicate_defect_candidate_is_not_counted_as_supply() -> None:
    policy = load_pair_policy(POLICY_PATH)
    clean = _profile(0, "jakie objawy wywołuje koronawirus")
    near_duplicate = _profile(
        1,
        "jakie objawy wywołuje koronawirus?",
        corpus_round_trip_at_100=0.0,
        corpus_round_trip_at_20=0.0,
    )

    report = summarize_supply(_cohorts([clean, near_duplicate]), policy)

    assert report["pooled"].get("axis_a_pairable_groups", 0) == 0
    assert report["pooled"]["groups_with_round_trip_100_fail"] == 1


def test_axis_b_supply_is_reported_at_candidate_cuts_without_freezing_one() -> None:
    policy = load_pair_policy(POLICY_PATH)
    members = [
        _profile(0, "jakie objawy wywołuje koronawirus", content_jaccard=0.1),
        _profile(1, "czym różni się zakażenie od przeziębienia", content_jaccard=0.2),
        _profile(
            2,
            "koronawirusy rodzina wirusów choroby układu oddechowego",
            content_jaccard=0.9,
        ),
        _profile(3, "okres wylęgania wirusa w dniach", content_jaccard=0.4),
    ]

    report = summarize_supply(_cohorts(members), policy)

    assert set(report["overlap_cut_candidates"]) == {"p50", "p75", "p90"}
    assert report["pooled"]["axis_b_pairable_groups_high_p75"] == 1
    assert report["thresholds_frozen_here"] is False
    assert report["final_tests_used"] == []


def test_axis_c_preliminary_supply_carries_the_broken_labels_caveat() -> None:
    policy = load_pair_policy(POLICY_PATH)
    members = [
        _profile(0, "jakie objawy wywołuje koronawirus", focus_accuracy=1.0),
        _profile(1, "ile trwa okres wylęgania zakażenia", focus_accuracy=0.0),
    ]

    report = summarize_supply(_cohorts(members), policy)

    assert report["pooled"]["axis_c_preliminary_pairable_groups"] == 1
    assert "focus_v2" in report["axis_c_labels_caveat"]
    assert report["answerability_judge_pending"] is True


def test_inventory_report_round_trips_through_disk(tmp_path: Path) -> None:
    policy = load_pair_policy(POLICY_PATH)
    members = [
        _profile(0, "jakie objawy wywołuje koronawirus"),
        _profile(1, "kiedy powstała szczepionka pfizer", entity_preservation=0.5),
    ]
    report = summarize_supply(_cohorts(members), policy)
    path = tmp_path / "summary.json"
    path.write_text(
        __import__("json").dumps(report, ensure_ascii=False), encoding="utf-8"
    )

    loaded = load_inventory(path)

    assert loaded["contract"] == "task06-defect-supply-inventory-v1"
    assert loaded["status"] == "design_input_measured_no_pairs_built"


def test_run_inventory_refuses_an_empty_cohort_list(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one cohort"):
        run_inventory(
            cohort_dirs=[], policy_path=POLICY_PATH, output_path=tmp_path / "out.json"
        )
