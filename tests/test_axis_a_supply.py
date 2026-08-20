from __future__ import annotations

import json
from pathlib import Path

import pytest

from doc2query.preferences.axis_a_supply import (
    CertifiedCandidate,
    load_verdicts,
    summarize_cohort,
)
from doc2query.preferences.defect_inventory import CandidateProfile
from doc2query.preferences.pair_policy import load_pair_policy

POLICY = Path("configs/preferences/task06_tentative_pair_policy_v1_1.yaml")


def _profile(
    candidate_id: str,
    query: str,
    *,
    clean: bool = True,
    format_ok: bool = True,
    rt100_fail: bool = False,
) -> CandidateProfile:
    return CandidateProfile(
        candidate_id=candidate_id,
        query=query,
        format_admissible=format_ok,
        answerable_proxy=clean,
        clean_chosen=clean,
        axis_a_defect=rt100_fail,
        entity_hallucination=False,
        round_trip_100_fail=rt100_fail,
        focus_correct=False,
        focus_wrong=False,
        content_jaccard=0.1,
    )


def test_chosen_requires_both_cleanliness_and_a_yes_verdict() -> None:
    clean_yes = CertifiedCandidate(_profile("a", "pytanie a"), "yes")
    clean_no = CertifiedCandidate(_profile("b", "pytanie b"), "no")
    dirty_yes = CertifiedCandidate(_profile("c", "pytanie c", clean=False), "yes")
    unjudged = CertifiedCandidate(_profile("d", "pytanie d"), None)

    assert clean_yes.admissible_chosen is True
    assert clean_no.admissible_chosen is False
    assert dirty_yes.admissible_chosen is False
    assert unjudged.admissible_chosen is False


def test_uncertain_blocks_chosen_and_is_never_a_defect() -> None:
    """Wymóg z ADR sędziego: abstencja nie może produkować strony rejected."""
    candidate = CertifiedCandidate(_profile("a", "pytanie"), "uncertain")

    assert candidate.admissible_chosen is False
    assert candidate.axis_a_rejected is False


def test_axis_a_rejected_comes_from_a_named_defect_only() -> None:
    judged_no = CertifiedCandidate(_profile("a", "q", clean=False), "no")
    round_trip_miss = CertifiedCandidate(_profile("b", "q", clean=False, rt100_fail=True), "yes")
    healthy = CertifiedCandidate(_profile("c", "q"), "yes")
    malformed = CertifiedCandidate(_profile("d", "q", clean=False, format_ok=False), "no")

    assert judged_no.axis_a_rejected is True
    assert round_trip_miss.axis_a_rejected is True
    assert healthy.axis_a_rejected is False
    # Kandydat poza formatem nie jest defektem osi A - odpada wcześniej, na formacie.
    assert malformed.axis_a_rejected is False


def test_summary_counts_pairable_groups_and_the_counterfactual() -> None:
    policy = load_pair_policy(POLICY)
    groups = {
        # grupa pairable: czysty chosen z yes + kandydat z no
        "g1": [
            CertifiedCandidate(_profile("a", "jak zalozono rzym w roku"), "yes"),
            CertifiedCandidate(_profile("b", "co jadano w antycznej grecji", clean=False), "no"),
        ],
        # grupa, ktora traci chosen wylacznie przez filtr sedziego
        "g2": [
            CertifiedCandidate(_profile("c", "kiedy powstal akwedukt"), "no"),
            CertifiedCandidate(_profile("d", "gdzie stal forum romanum", clean=False), "no"),
        ],
        # grupa bez zadnego defektu -> brak strony rejected
        "g3": [
            CertifiedCandidate(_profile("e", "ile lat ma koloseum"), "yes"),
            CertifiedCandidate(_profile("f", "z czego zbudowano koloseum"), "yes"),
        ],
    }

    summary = summarize_cohort(groups, policy)
    counts = summary["counts"]

    assert counts["groups"] == 3
    assert counts["groups_with_certified_chosen"] == 2  # g1, g3
    assert counts["groups_with_axis_a_rejected"] == 2  # g1, g2
    assert counts["groups_with_both_sides"] == 1  # tylko g1
    assert counts["pairable_groups"] == 1
    # Kontrfaktycznie czysty chosen mialy g1, g2 i g3 - sedzia zabrał g2.
    assert counts["groups_with_clean_chosen_before_judge"] == 3
    assert summary["chosen_supply_kept_by_judge"] == pytest.approx(2 / 3)


def test_pairing_respects_the_frozen_diversity_constraint() -> None:
    policy = load_pair_policy(POLICY)
    identical = "jak wysokie jest koloseum w rzymie"
    groups = {
        "g1": [
            CertifiedCandidate(_profile("a", identical), "yes"),
            CertifiedCandidate(_profile("b", identical, clean=False), "no"),
        ]
    }

    summary = summarize_cohort(groups, policy)

    assert summary["counts"]["groups_with_both_sides"] == 1
    # Obie strony to to samo zapytanie, więc para nie przechodzi bramki różnorodności.
    assert summary["counts"].get("pairable_groups", 0) == 0


def test_load_verdicts_refuses_journals_that_contradict_each_other(tmp_path: Path) -> None:
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    first.write_text(
        json.dumps({"event": "verdict", "item_id": "x", "verdict": "yes"}) + "\n", encoding="utf-8"
    )
    second.write_text(
        json.dumps({"event": "verdict", "item_id": "x", "verdict": "no"}) + "\n", encoding="utf-8"
    )

    assert load_verdicts([first]) == {"x": "yes"}
    with pytest.raises(ValueError, match="disagree on x"):
        load_verdicts([first, second])


def test_load_verdicts_ignores_non_verdict_events(tmp_path: Path) -> None:
    journal = tmp_path / "j.jsonl"
    journal.write_text(
        "\n".join(
            [
                json.dumps({"event": "out_of_schema", "item_id": "x", "content": "{}"}),
                json.dumps({"event": "verdict", "item_id": "y", "verdict": "no"}),
                json.dumps({"event": "batch_usage", "batch_id": "b"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert load_verdicts([journal]) == {"y": "no"}
