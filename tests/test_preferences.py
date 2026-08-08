from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from doc2query.preferences.audit import export_blind_audit, import_blind_audit
from doc2query.preferences.build import build_preference_dataset, select_candidate_sets
from doc2query.preferences.schemas import ScoredCandidate, SelectionPolicy
from doc2query.utils.records import read_records


def _candidate(
    candidate_id: str,
    query: str,
    total: float,
    *,
    passage_id: str = "p1",
    cluster_id: str = "c1",
    split: str = "train",
    answerable: bool = True,
    ground: float = 0.7,
    failure_types: list[str] | None = None,
) -> ScoredCandidate:
    payload: dict[str, Any] = {
        "candidate_id": candidate_id,
        "passage_id": passage_id,
        "passage_cluster_id": cluster_id,
        "prompt": f"Pasaż {passage_id}\nZapytanie:",
        "query": query,
        "split": split,
        "controls": {"style": "full_question"},
        "generation": {"temperature": 0.7, "seed": 42},
        "scores": {
            "ground_score": ground,
            "negative_margin": 0.4,
            "corpus_round_trip": 1.0,
            "effective_candidate_count": 11,
            "possible_false_negative": False,
            "overlap_reward": 0.6,
            "focus_accuracy": 1.0,
            "style_accuracy": 1.0,
            "format_score": 1.0,
            "copy_penalty": 0.1,
            "answerability_flag": answerable,
            "total_score": total,
        },
        "provenance": {
            "generator_id": "generator-v1",
            "checkpoint_id": "checkpoint-v1",
            "checkpoint_fingerprint": "checkpoint-sha",
            "generation_config_fingerprint": "generation-sha",
            "primary_judge_id": "primary",
            "primary_judge_revision": "primary-revision",
            "shadow_judge_id": "shadow",
            "shadow_judge_revision": "shadow-revision",
            "corpus_fingerprint": "corpus-sha",
            "scoring_config_fingerprint": "scoring-sha",
            "miner_policy_id": "hn0-filter",
            "miner_fingerprint": "miner-sha",
        },
        "failure_types": sorted(failure_types or []),
    }
    return ScoredCandidate.model_validate(payload)


def test_selects_near_miss_with_margin_and_keeps_components() -> None:
    candidates = [
        _candidate("best", "Jak działa pompa ciepła?", 3.0),
        _candidate(
            "near",
            "Na jakiej zasadzie pracuje instalacja grzewcza?",
            2.6,
            failure_types=["focus_mismatch"],
        ),
        _candidate("bottom", "Co to jest ogrzewanie?", 0.5, failure_types=["too_general"]),
    ]
    selected, report = select_candidate_sets(
        candidates,
        SelectionPolicy(strategy="top_vs_near_miss", min_score_margin=0.25),
    )
    assert len(selected) == 1
    assert selected[0].chosen_candidate_id == "best"
    assert selected[0].rejected_candidate_ids == ["near"]
    assert selected[0].score_margins == pytest.approx([0.4])
    assert report["rejected_failure_types"] == {"focus_mismatch": 1}


def test_rejects_passage_cluster_leakage() -> None:
    candidates = [
        _candidate("train", "Pytanie treningowe", 2.0),
        _candidate(
            "dev",
            "Pytanie walidacyjne",
            1.0,
            passage_id="p2",
            cluster_id="c1",
            split="dev",
        ),
    ]
    with pytest.raises(ValueError, match="passage cluster c1 crosses splits"):
        select_candidate_sets(candidates, SelectionPolicy())


def test_builds_trl_and_continued_sft_splits(tmp_path: Path) -> None:
    candidates = [
        _candidate("best", "Jak działa pompa ciepła?", 3.0),
        _candidate("near", "Jak ogrzewa instalacja?", 2.5, failure_types=["copy_risk"]),
    ]
    selected, _ = select_candidate_sets(
        candidates, SelectionPolicy(min_score_margin=0.25, max_normalized_query_jaccard=1.0)
    )
    manifest = build_preference_dataset(candidates, selected, tmp_path, output_format="jsonl")
    rows = list(read_records(tmp_path / "train.jsonl"))
    controls = list(read_records(tmp_path / "continued_sft_train.jsonl"))
    assert rows[0]["prompt"] == "Pasaż p1\nZapytanie:"
    assert rows[0]["chosen"] == "Jak działa pompa ciepła?"
    assert rows[0]["rejected"] == "Jak ogrzewa instalacja?"
    assert set(rows[0]["chosen_scores"]) >= {"ground_score", "total_score", "copy_penalty"}
    assert controls[0]["completion"] == rows[0]["chosen"]
    assert controls[0]["raw_total_score"] == 3.0
    assert manifest["pair_counts"] == {"train": 1, "dev": 0, "test": 0}
    assert manifest["final_tests_used"] == []


def test_blind_audit_is_deterministic_and_unblinds(tmp_path: Path) -> None:
    preferences = [
        {
            "preference_id": f"pref-{index}",
            "prompt": "Pasaż\nZapytanie:",
            "chosen": f"Dobre pytanie {index}?",
            "rejected": f"Gorsze pytanie {index}?",
            "split": "dev",
            "rejected_failure_types": ["too_general"],
        }
        for index in range(5)
    ]
    first = tmp_path / "first"
    second = tmp_path / "second"
    export_blind_audit(preferences, first, sample_size=3, seed=7)
    export_blind_audit(preferences, second, sample_size=3, seed=7)
    assert (first / "blind_review_form.jsonl").read_bytes() == (
        second / "blind_review_form.jsonl"
    ).read_bytes()

    completed = list(read_records(first / "blind_review_form.jsonl"))
    key = list(read_records(first / "machine_key.jsonl"))
    automatic = {row["audit_id"]: row["automatic_chosen_option"] for row in key}
    for row in completed:
        row["human_preference"] = automatic[row["audit_id"]]
        row["reason"] = "better_grounded"
    summary = import_blind_audit(completed, key, tmp_path / "imported")
    assert summary["automatic_human_agreement"] == 1.0
    assert summary["counts"] == {"agree": 3}


def test_audit_import_fails_closed_on_missing_rows(tmp_path: Path) -> None:
    preferences = [
        {
            "preference_id": "pref-1",
            "prompt": "Pasaż",
            "chosen": "Dobre pytanie?",
            "rejected": "Gorsze pytanie?",
            "split": "dev",
        }
    ]
    export_blind_audit(preferences, tmp_path, sample_size=1, seed=1)
    key = list(read_records(tmp_path / "machine_key.jsonl"))
    with pytest.raises(ValueError, match="audit is incomplete"):
        import_blind_audit([], key, tmp_path / "imported")
