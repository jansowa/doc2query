from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

from doc2query.preferences.pair_policy import (
    TentativePairPolicy,
    _Candidate,
    build_group_pair,
    build_tentative_pairs,
    load_pair_policy,
    violates_lead_in_guard,
)

POLICY_PATH = Path("configs/preferences/task06_tentative_pair_policy_v1.yaml")
PASSAGE = (
    "Koronawirusy to duża rodzina wirusów wywołujących choroby układu oddechowego u ludzi "
    "i u zwierząt. Zakażenie przenosi się drogą kropelkową oraz przez kontakt z skażonymi "
    "powierzchniami, a typowe objawy obejmują gorączkę, suchy kaszel, ból gardła i "
    "uczucie duszności. Okres wylęgania wynosi zwykle od dwóch do czternastu dni, a "
    "ciężki przebieg dotyczy przede wszystkim osób starszych oraz pacjentów z chorobami "
    "przewlekłymi układu krążenia i oddechowego."
)


def _row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "evaluation_id": "1::7::same-prompt::0",
        "evaluation_group_id": "task06-preference::1::7",
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
        "content_jaccard": 0.3,
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


def _policy() -> TentativePairPolicy:
    return load_pair_policy(POLICY_PATH)


def _pair(candidates: list[_Candidate], **overrides: Any) -> tuple[Any, Any]:
    kwargs: dict[str, Any] = {
        "cohort_id": "same_prompt_expansion_v1",
        "group_id": "task06-preference::1::7",
        "gate_eligible": True,
        "representative_ids": [row.candidate_id for row in candidates],
        "passage_cluster_id": "7",
        "policy": _policy(),
    }
    kwargs.update(overrides)
    return build_group_pair(candidates, **kwargs)


def test_frozen_policy_pins_primary_only_and_excludes_entity_preservation() -> None:
    policy = _policy()

    assert policy.primary.role == "sole_pair_building_signal"
    assert policy.primary.signal == "pool_margin"
    assert policy.primary.min_margin_gap == 1.0
    assert policy.shadow.role == "veto_only_never_selection"
    assert policy.pairing.strategy == "top_vs_near_miss"
    assert policy.pairing.max_pairs_per_group == 1
    assert "entity_preservation" in {row.name for row in policy.excluded_signals}
    assert policy.final_tests_used == []


def test_policy_without_entity_preservation_exclusion_is_refused(tmp_path: Path) -> None:
    raw = yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))
    raw["excluded_signals"] = [{"name": "total_score", "reason": "brak wag"}]
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")

    with pytest.raises(ValueError, match="entity_preservation"):
        load_pair_policy(path)


def test_lead_in_guard_covers_the_measured_format_valid_blind_spot() -> None:
    assert violates_lead_in_guard("Oto zapytanie o objawy koronawirusa")
    assert violates_lead_in_guard("otóż jakie są objawy koronawirusa")
    assert not violates_lead_in_guard("jakie objawy wywołuje koronawirus")
    assert not violates_lead_in_guard("otoczenie stacji kolejowej w Krakowie")


def test_top_vs_near_miss_takes_the_highest_candidate_below_the_frozen_gap() -> None:
    candidates = [
        _candidate(0, "jakie objawy wywołuje koronawirus", pool_margin=5.0),
        _candidate(1, "objawy zakażenia koronawirusem u dorosłych", pool_margin=4.5),
        _candidate(2, "czym jest rodzina wirusów układu oddechowego", pool_margin=3.8),
        _candidate(3, "co to wirus", pool_margin=1.0),
    ]

    pair, outcome = _pair(candidates)

    assert pair is not None
    assert outcome.paired is True
    assert pair.chosen_candidate_id == "1::7::same-prompt::0"
    # 4.5 jest tylko 0.5 poniżej top-u, więc near-miss to dopiero 3.8.
    assert pair.rejected_candidate_id == "1::7::same-prompt::2"
    assert pair.primary_margin_gap == pytest.approx(1.2)
    assert "lower_primary_margin" in pair.rejected_failure_types


def test_group_without_any_candidate_below_the_gap_produces_no_pair() -> None:
    candidates = [
        _candidate(0, "jakie objawy wywołuje koronawirus", pool_margin=5.0),
        _candidate(1, "objawy zakażenia koronawirusem u dorosłych", pool_margin=4.6),
    ]

    pair, outcome = _pair(candidates)

    assert pair is None
    assert outcome.failure_reasons == ["no_candidate_below_margin_gap"]


def test_shadow_inversion_vetoes_the_pair_instead_of_reselecting_a_rejected() -> None:
    candidates = [
        _candidate(0, "jakie objawy wywołuje koronawirus", pool_margin=5.0,
                   shadow_pool_margin=1.0),
        _candidate(1, "czym jest rodzina wirusów oddechowych", pool_margin=3.0,
                   shadow_pool_margin=2.0),
        _candidate(2, "co to wirus", pool_margin=1.5, shadow_pool_margin=0.1),
    ]

    pair, outcome = _pair(candidates)

    assert pair is None
    assert outcome.failure_reasons == ["shadow_veto"]


def test_shadow_rank_inversion_also_vetoes_the_pair() -> None:
    candidates = [
        _candidate(0, "jakie objawy wywołuje koronawirus", pool_margin=5.0, shadow_pool_rank=3),
        _candidate(1, "czym jest rodzina wirusów oddechowych", pool_margin=3.0, shadow_pool_rank=1),
    ]

    pair, outcome = _pair(candidates)

    assert pair is None
    assert outcome.failure_reasons == ["shadow_veto"]


def test_chosen_requires_positive_primary_margin_and_corpus_round_trip() -> None:
    candidates = [
        _candidate(0, "jakie objawy wywołuje koronawirus", pool_margin=5.0,
                   corpus_round_trip_at_20=0.0),
        _candidate(1, "czym jest rodzina wirusów oddechowych", pool_margin=-0.5),
    ]

    pair, outcome = _pair(candidates)

    assert pair is None
    assert outcome.failure_reasons == ["no_admissible_chosen"]
    assert outcome.admissible_chosen_count == 0


def test_rejected_keeps_minimal_relevance_but_may_miss_the_top20_round_trip() -> None:
    candidates = [
        _candidate(0, "jakie objawy wywołuje koronawirus", pool_margin=5.0),
        _candidate(1, "czym jest rodzina wirusów oddechowych", pool_margin=3.0,
                   corpus_round_trip_at_20=0.0),
        _candidate(2, "zupełnie inne zapytanie o pogodę", pool_margin=2.0,
                   corpus_round_trip_at_100=0.0),
    ]

    pair, outcome = _pair(candidates)

    assert pair is not None
    assert pair.rejected_candidate_id == "1::7::same-prompt::1"
    assert "weak_corpus_round_trip" in pair.rejected_failure_types
    assert outcome.admissible_rejected_count == 2


def test_focus_abstention_never_penalizes_but_zero_focus_blocks_chosen() -> None:
    abstaining = [
        _candidate(0, "jakie objawy wywołuje koronawirus", pool_margin=5.0, focus_accuracy=None),
        _candidate(1, "czym jest rodzina wirusów oddechowych", pool_margin=3.0),
    ]
    wrong_focus = [
        _candidate(0, "jakie objawy wywołuje koronawirus", pool_margin=5.0, focus_accuracy=0.0),
        _candidate(1, "czym jest rodzina wirusów oddechowych", pool_margin=3.0),
        _candidate(2, "ile dni trwa okres wylęgania zakażenia", pool_margin=1.5),
    ]

    assert _pair(abstaining)[0] is not None
    chosen_of_wrong_focus = _pair(wrong_focus)[0]
    assert chosen_of_wrong_focus is not None
    assert chosen_of_wrong_focus.chosen_candidate_id == "1::7::same-prompt::1"


def test_copy_risk_blocks_chosen_but_stays_a_legitimate_rejected_type() -> None:
    candidates = [
        _candidate(0, "jakie objawy wywołuje koronawirus", pool_margin=5.0,
                   copy_density=0.9, normalized_lcs=0.9),
        _candidate(1, "czym jest rodzina wirusów oddechowych", pool_margin=4.0),
        _candidate(2, "koronawirusy to rodzina wirusów wywołujących choroby", pool_margin=2.0,
                   copy_density=0.9, normalized_lcs=0.9),
    ]

    pair, _ = _pair(candidates)

    assert pair is not None
    assert pair.chosen_candidate_id == "1::7::same-prompt::1"
    assert pair.rejected_candidate_id == "1::7::same-prompt::2"
    assert "copy_risk" in pair.rejected_failure_types


def test_lead_in_prefix_blocks_both_roles() -> None:
    candidates = [
        _candidate(0, "Oto jakie objawy wywołuje koronawirus", pool_margin=5.0),
        _candidate(1, "Oto czym jest rodzina wirusów oddechowych", pool_margin=3.0),
    ]

    pair, outcome = _pair(candidates)

    assert pair is None
    assert outcome.failure_reasons == ["no_admissible_chosen"]


def test_near_duplicate_queries_cannot_be_paired() -> None:
    candidates = [
        _candidate(0, "jakie objawy wywołuje koronawirus", pool_margin=5.0),
        _candidate(1, "jakie objawy wywołuje koronawirus?", pool_margin=3.0),
    ]

    pair, outcome = _pair(candidates)

    assert pair is None
    assert outcome.failure_reasons == ["near_duplicate_query_pair"]


def test_gate_ineligible_group_never_reaches_the_policy() -> None:
    candidates = [
        _candidate(0, "jakie objawy wywołuje koronawirus", pool_margin=5.0),
        _candidate(1, "czym jest rodzina wirusów oddechowych", pool_margin=3.0),
    ]

    pair, outcome = _pair(candidates, gate_eligible=False)

    assert pair is None
    assert outcome.failure_reasons == ["group_not_gate_eligible"]
    assert outcome.gate_eligible is False


def test_only_gate_representatives_may_enter_a_pair() -> None:
    candidates = [
        _candidate(0, "jakie objawy wywołuje koronawirus", pool_margin=9.0),
        _candidate(1, "czym jest rodzina wirusów oddechowych", pool_margin=5.0),
        _candidate(2, "co to wirus", pool_margin=1.0),
    ]

    pair, _ = _pair(candidates, representative_ids=["1::7::same-prompt::1", "1::7::same-prompt::2"])

    assert pair is not None
    assert pair.chosen_candidate_id == "1::7::same-prompt::1"
    assert pair.rejected_candidate_id == "1::7::same-prompt::2"


def test_unauthorized_cohort_is_refused_before_any_read(tmp_path: Path) -> None:
    cohort = tmp_path / "same_prompt_expansion_v7"
    cohort.mkdir()

    with pytest.raises(ValueError, match="not authorized for pair building"):
        build_tentative_pairs(cohort_dir=cohort, policy_path=POLICY_PATH)


def test_existing_output_directory_is_never_overwritten(tmp_path: Path) -> None:
    cohort = tmp_path / "same_prompt_expansion_v1"
    cohort.mkdir()
    existing = tmp_path / "already_there"
    existing.mkdir()

    with pytest.raises(FileExistsError):
        build_tentative_pairs(
            cohort_dir=cohort, policy_path=POLICY_PATH, output_dir=existing
        )


def test_pair_record_keeps_every_component_and_declares_no_final_tests() -> None:
    candidates = [
        _candidate(0, "jakie objawy wywołuje koronawirus", pool_margin=5.0),
        _candidate(1, "czym jest rodzina wirusów oddechowych", pool_margin=3.0),
    ]

    pair, _ = _pair(candidates)

    assert pair is not None
    payload: Mapping[str, Any] = json.loads(pair.model_dump_json())
    assert payload["final_tests_used"] == []
    assert payload["strategy"] == "top_vs_near_miss"
    assert payload["passage"] == PASSAGE
    for role in ("chosen_components", "rejected_components"):
        for field in ("pool_margin", "shadow_pool_margin", "corpus_round_trip_at_20"):
            assert field in payload[role]
    assert "total_score" not in payload
