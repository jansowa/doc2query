from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from doc2query.preferences.pair_audit_export import BLIND_FIELDS
from doc2query.preferences.pair_audit_export_v2_1 import (
    EXPORT_CONTRACT,
    _sample,
)
from doc2query.preferences.pair_policy import _Candidate
from doc2query.preferences.pair_policy_v2_1 import (
    DefectPairPolicyV21,
    build_group_pair,
    chosen_admissible,
    defect_labels,
    load_defect_pair_policy_v2_1,
    rejected_admissible,
)

POLICY_PATH = Path("configs/preferences/task06_defect_pair_policy_v2_1.yaml")
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


def _policy() -> DefectPairPolicyV21:
    return load_defect_pair_policy_v2_1(POLICY_PATH)


def _certified(items: list[tuple[_Candidate, str]]) -> list[Any]:
    from doc2query.preferences.pair_policy_v2_1 import (
        CertifiedCandidateV21,
        _mean_group_jaccard,
    )

    queries = [candidate.query for candidate, _ in items]
    return [
        CertifiedCandidateV21(
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


def test_policy_releases_one_axis_and_retires_b_with_its_evidence() -> None:
    policy = _policy()
    assert policy.axis.id == "A"
    assert policy.pairing.single_axis == "A"
    retired = {row.id: row for row in policy.retired_axes}
    assert set(retired) == {"B", "C"}
    assert retired["B"].evidence, "oś B musi wypadać z zapisanym dowodem"
    assert "walidacja" in retired["B"].return_requires.lower()
    assert policy.primary.used_for_ordering is False
    assert policy.constructed_rejected.max_share == 0.0
    assert policy.final_tests_used == []


def test_content_jaccard_is_excluded_as_a_defect_signal() -> None:
    excluded = {row.name for row in _policy().excluded_signals}
    assert "content_jaccard_as_defect_signal" in excluded
    assert "pool_margin_as_ordering_key" in excluded


def test_decision_rule_keeps_thresholds_and_one_guardrail() -> None:
    rule = _policy().decision_rule
    assert rule.interval == "clopper_pearson_exact"
    assert rule.inconclusive_is_fail_closed is True
    assert rule.escalation_permitted is False
    by_id = {row.id: row for row in rule.predictions}
    # Progi identyczne z v2.0 — zmienia się wnioskowanie, nie próg.
    assert by_id["P1"].threshold == 0.05
    assert by_id["P1"].role == "guardrail"
    assert by_id["P1"].fail_requires == "lower_bound_above_threshold"
    assert by_id["P2"].threshold == 0.30
    assert by_id["P3"].threshold == 0.031
    assert by_id["P4_prime"].threshold == 0.20
    assert by_id["P5"].role == "reported_only"
    assert by_id["P5"].threshold is None


def test_audit_sample_is_powered_and_stratifies_by_defect_label() -> None:
    sample = _policy().audit_sample
    assert sample.target_pair_count == 800
    assert sample.strata == ["cohort_id", "rejected_defect_label", "requested_form"]
    assert "axis" not in sample.strata
    assert sample.minimum_pair_count_to_start == 500


def test_anchor_cell_is_declared_non_gating() -> None:
    anchor = _policy().anchor_cell
    assert anchor.gating_role == "none_calibration_only"
    assert anchor.relabels_natural_pairs is False
    assert anchor.target_pair_count == 300
    assert anchor.output_dir != _policy().audit_sample.output_dir


def test_defect_label_stratum_follows_the_frozen_priority() -> None:
    policy = _policy()
    assert (
        policy.defect_label_stratum(["shadow_agrees", "judge_unanswerable"])
        == "judge_unanswerable"
    )
    assert (
        policy.defect_label_stratum(["lower_primary_margin", "weak_corpus_round_trip"])
        == "weak_corpus_round_trip"
    )
    with pytest.raises(ValueError, match="no prioritized defect label"):
        policy.defect_label_stratum(["high_lexical_overlap"])


# --- budowa pary ---------------------------------------------------------------


def test_unanswerable_rejected_pairs_with_clean_chosen() -> None:
    pair, outcome = _pair(
        [
            (_candidate(0, "jakie objawy wywołuje koronawirus"), "yes"),
            (_candidate(1, "ile trwa okres wylęgania wirusa grypy sezonowej"), "no"),
        ]
    )
    assert pair is not None
    assert pair.axis == "A"
    assert pair.chosen_verdict == "yes"
    assert pair.rejected_verdict == "no"
    assert "judge_unanswerable" in pair.rejected_defect_labels
    assert pair.rejected_defect_label == "judge_unanswerable"
    assert pair.margin_used_for_ordering is False
    assert outcome.paired is True


def test_weak_round_trip_is_the_other_named_axis_a_defect() -> None:
    pair, _ = _pair(
        [
            (_candidate(0, "jakie objawy wywołuje koronawirus"), "yes"),
            (
                _candidate(
                    1,
                    "jak przenosi się zakażenie drogą kropelkową",
                    corpus_round_trip_at_100=0.0,
                ),
                "yes",
            ),
        ]
    )
    assert pair is not None
    assert pair.rejected_defect_labels == ["weak_corpus_round_trip"]
    assert pair.rejected_defect_label == "weak_corpus_round_trip"


def test_uncertain_blocks_chosen_and_is_never_a_defect() -> None:
    policy = _policy()
    certified = _certified([(_candidate(0, "jakie objawy wywołuje koronawirus"), "uncertain")])
    assert chosen_admissible(certified[0], policy) is False
    assert rejected_admissible(certified[0], policy) is False


def test_high_lexical_overlap_alone_no_longer_builds_a_pair() -> None:
    """Oś B wypadła: samo wysokie pokrycie leksykalne nie jest już defektem."""
    pair, outcome = _pair(
        [
            (_candidate(0, "jakie objawy wywołuje koronawirus", content_jaccard=0.01), "yes"),
            (
                _candidate(
                    1,
                    "jakie są typowe objawy zakażenia koronawirusem u ludzi",
                    content_jaccard=0.40,
                ),
                "yes",
            ),
        ]
    )
    assert pair is None
    assert outcome.failure_reasons == ["no_axis_defect_rejected"]


def test_defect_labels_never_mention_lexical_overlap() -> None:
    policy = _policy()
    chosen, rejected = _certified(
        [
            (_candidate(0, "jakie objawy wywołuje koronawirus", content_jaccard=0.01), "yes"),
            (_candidate(1, "ile trwa okres wylęgania grypy", content_jaccard=0.90), "no"),
        ]
    )
    labels = defect_labels(chosen, rejected, policy)
    assert "high_lexical_overlap" not in labels
    assert "judge_unanswerable" in labels


def test_divpo_tie_break_picks_the_least_typical_chosen_and_ignores_margin() -> None:
    items = [
        # Najwyższy margines, ale najbardziej typowy w grupie — DivPO go nie wybierze.
        (_candidate(0, "jakie objawy wywołuje koronawirus u dzieci", pool_margin=9.0), "yes"),
        (_candidate(1, "jakie objawy wywołuje koronawirus", pool_margin=1.0), "yes"),
        (_candidate(2, "ile dni trwa wylęganie", pool_margin=2.0), "no"),
    ]
    certified = _certified(items)
    admissible = {
        row.candidate_id: row.mean_group_jaccard
        for row in certified
        if chosen_admissible(row, _policy())
    }
    expected = min(admissible, key=lambda key: (admissible[key], key))
    pair, _ = _pair(items)
    assert pair is not None
    assert pair.chosen_candidate_id == expected
    assert pair.chosen_group_distinctness == pytest.approx(admissible[expected])
    # Margines nie porządkuje: wybrany chosen nie musi mieć najwyższego marginesu.
    assert pair.margin_used_for_ordering is False


def test_near_duplicate_pair_is_refused() -> None:
    pair, outcome = _pair(
        [
            (_candidate(0, "jakie objawy wywołuje koronawirus"), "yes"),
            (_candidate(1, "jakie objawy wywołuje koronawirus"), "no"),
        ]
    )
    assert pair is None
    assert outcome.failure_reasons == ["near_duplicate_query_pair"]


def test_gate_ineligible_group_is_reported_not_paired() -> None:
    pair, outcome = _pair(
        [(_candidate(0, "jakie objawy wywołuje koronawirus"), "yes")],
        gate_eligible=False,
    )
    assert pair is None
    assert outcome.failure_reasons == ["group_not_gate_eligible"]
    assert outcome.gate_eligible is False


# --- próbka audytowa ------------------------------------------------------------


def _pair_row(index: int, label: str, cohort: str = "same_prompt_expansion_v1") -> dict[str, Any]:
    return {
        "pair_id": f"{index:032d}",
        "cohort_id": cohort,
        "rejected_defect_label": label,
        "requested_form": "full_question",
        "axis": "A",
    }


def test_sample_is_deterministic_and_stratified_without_axis() -> None:
    policy = _policy()
    population = [
        _pair_row(index, "judge_unanswerable" if index % 3 else "weak_corpus_round_trip")
        for index in range(1200)
    ]
    first, strata = _sample(population, policy)
    second, _ = _sample(population, policy)
    assert len(first) == policy.audit_sample.target_pair_count
    assert [row["pair_id"] for row in first] == [row["pair_id"] for row in second]
    assert sum(row.allocated for row in strata) == len(first)
    assert {row.rejected_defect_label for row in strata} == {
        "judge_unanswerable",
        "weak_corpus_round_trip",
    }


def test_sample_never_exceeds_the_available_population() -> None:
    policy = _policy()
    population = [_pair_row(index, "judge_unanswerable") for index in range(600)]
    sampled, _ = _sample(population, policy)
    assert len(sampled) == 600


def test_blind_field_set_and_export_contract_stay_frozen() -> None:
    assert EXPORT_CONTRACT == "task06-defect-pair-audit-blind-export-v2-1"
    assert list(BLIND_FIELDS) == [
        "audit_id",
        "passage",
        "query_a",
        "query_b",
        "orientation_commitment",
    ]


def test_audit_reader_dispatches_v2_1_exports_to_the_defect_label_slice(
    tmp_path: Path,
) -> None:
    from doc2query.preferences.groq_pair_audit import _V2_1_ADAPTER, load_export_manifest

    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"contract": "task06-defect-pair-audit-blind-export-v2-1"}')
    with pytest.raises(Exception):  # noqa: B017 - niepełny manifest, liczy się dyspozytor
        load_export_manifest(manifest)
    assert _V2_1_ADAPTER.slice_field == "rejected_defect_label"
    assert _V2_1_ADAPTER.label_field == "rejected_defect_labels"


def test_groq_audit_config_v2_1_changes_only_the_sample_size() -> None:
    import json

    base: Mapping[str, Any] = json.loads(
        Path("configs/preferences/task06_groq_preference_audit_v1.json").read_text(
            encoding="utf-8"
        )
    )
    new: Mapping[str, Any] = json.loads(
        Path("configs/preferences/task06_groq_preference_audit_v2_1.json").read_text(
            encoding="utf-8"
        )
    )
    assert new["pair_count"] == 800
    for key in (
        "contract",
        "prompt_version",
        "models",
        "batch_size",
        "api",
        "limits_per_model",
        "retry",
        "quota_scheduler",
        "resume_policy",
        "required_outputs",
        "disagreement_policy",
        "assignment",
        "blind_order_policy",
        "role",
        "human_evidence_claimed",
    ):
        assert new[key] == base[key], f"kontrakt Groq zmienił się w polu {key}"
