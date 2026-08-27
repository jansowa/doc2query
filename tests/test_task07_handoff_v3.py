from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from doc2query.preferences.handoff_v3 import (
    CONSTANT_SCORE_MARGIN,
    WEIGHT_POLICY_ID,
    WEIGHT_RANGE,
    _percentile_weights,
    build_handoff,
    dev_split_assignment,
    weight_policy,
)
from doc2query.utils.records import read_records


def _pair(index: int, *, margin: float = 5.0) -> dict[str, Any]:
    return {
        "pair_id": f"{index:032d}",
        "cohort_id": "same_prompt_expansion_v1",
        "group_id": f"g{index}",
        "example_id": f"e{index}",
        "doc_id": f"d{index}",
        "passage_cluster_id": f"cluster-{index}",
        "split": "train",
        "prompt": f"Wygeneruj zapytanie dla pasażu {index}.",
        "prompt_sha256": "a" * 64,
        "passage": f"pasaż {index}",
        "rejected_variant": "bottom",
        "chosen_candidate_id": f"c{index}",
        "rejected_candidate_id": f"r{index}",
        "chosen": f"jakie objawy numer {index}",
        "rejected": f"ile kosztuje bilet {index}",
        "chosen_components": {"pool_margin": margin},
        "rejected_components": {"pool_margin": margin - 1.0},
        "votes_for_chosen": 6,
        "position_flips": 0,
        "tournament_comparisons": 9,
        "chosen_pool_size": 3,
        "rejected_pool_size": 6,
        "normalized_query_jaccard": 0.1,
        "requested_form": "full_question",
        "requested_intent": "fact_lookup",
        "selector": "task06-judge-selected-pair-policy-v3",
        "validated_defect_scope": ["ungrounded", "copy_verbatim", "too_general"],
        "form_or_focus_compliance_claimed": False,
        "final_tests_used": [],
    }


def _pairs_file(tmp_path: Path, count: int = 60) -> Path:
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        "\n".join(json.dumps(_pair(index, margin=float(index))) for index in range(count)) + "\n",
        encoding="utf-8",
    )
    return path


def _adr(tmp_path: Path) -> Path:
    path = tmp_path / "adr.md"
    path.write_text("polityka v3", encoding="utf-8")
    return path


def test_score_margin_is_a_declared_constant() -> None:
    """v3 nie ma stopniowanego marginesu; pole nie może udawać informacji."""
    assert CONSTANT_SCORE_MARGIN == 1.0


def test_weight_policy_is_declared_as_control_only() -> None:
    policy = weight_policy()
    assert policy["weight_policy_id"] == WEIGHT_POLICY_ID
    assert policy["role"] == "control_arm_weighting_only_never_selection"
    assert policy["signal"] == "chosen_components.pool_margin"


def test_percentile_weights_span_the_frozen_range_and_share_ties() -> None:
    weights = _percentile_weights([1.0, 2.0, 3.0, 4.0, 5.0])
    assert weights[0] == pytest.approx(WEIGHT_RANGE[0])
    assert weights[-1] == pytest.approx(WEIGHT_RANGE[1])
    assert weights == sorted(weights)
    tied = _percentile_weights([7.0, 7.0, 9.0])
    assert tied[0] == tied[1], "remisy dzielą rangę, nie rozstrzyga ich kolejność"
    assert all(weight > 0 for weight in _percentile_weights([0.0, 0.0]))


def test_split_is_deterministic_and_seed_dependent() -> None:
    first = [dev_split_assignment(f"c{i}", seed=1, dev_every=10) for i in range(200)]
    again = [dev_split_assignment(f"c{i}", seed=1, dev_every=10) for i in range(200)]
    other = [dev_split_assignment(f"c{i}", seed=2, dev_every=10) for i in range(200)]
    assert first == again
    assert first != other
    assert 0 < sum(first) < 200


def test_handoff_produces_every_artifact_the_packager_expects(tmp_path: Path) -> None:
    summary = build_handoff(
        pairs_path=_pairs_file(tmp_path),
        output_dir=tmp_path / "out",
        selection_adr=_adr(tmp_path),
        seed=20260827,
        dev_every=5,
    )
    directory = tmp_path / "out"
    for name in (
        "preference_train.jsonl",
        "preference_dev.jsonl",
        "continued_sft_train.jsonl",
        "continued_sft_dev.jsonl",
        "weight_records.jsonl",
        "weight_manifest.json",
    ):
        assert (directory / name).is_file(), name
    assert summary["train_count"] + summary["dev_count"] == 60
    assert summary["score_margin_is_constant"] is True
    assert summary["task07_training_authorized"] is False


def test_continued_sft_mirrors_the_chosen_side_in_the_same_order(tmp_path: Path) -> None:
    build_handoff(
        pairs_path=_pairs_file(tmp_path),
        output_dir=tmp_path / "out",
        selection_adr=_adr(tmp_path),
        dev_every=5,
    )
    preferences = list(read_records(tmp_path / "out" / "preference_train.jsonl"))
    controls = list(read_records(tmp_path / "out" / "continued_sft_train.jsonl"))
    assert [row["preference_id"] for row in controls] == [
        row["preference_id"] for row in preferences
    ]
    for preference, control in zip(preferences, controls, strict=True):
        assert control["completion"] == preference["chosen"]
        assert control["candidate_id"] == preference["chosen_candidate_id"]


def test_weights_follow_train_then_dev_order_and_cover_every_pair(tmp_path: Path) -> None:
    """Packager odrzuca inną kolejność niż train+dev, więc pilnujemy jej tutaj."""
    build_handoff(
        pairs_path=_pairs_file(tmp_path),
        output_dir=tmp_path / "out",
        selection_adr=_adr(tmp_path),
        dev_every=5,
    )
    directory = tmp_path / "out"
    expected = [row["preference_id"] for row in read_records(directory / "preference_train.jsonl")]
    expected += [row["preference_id"] for row in read_records(directory / "preference_dev.jsonl")]
    weights = list(read_records(directory / "weight_records.jsonl"))
    assert [row["preference_id"] for row in weights] == expected
    assert all(row["sample_weight"] > 0 for row in weights)
    manifest = json.loads((directory / "weight_manifest.json").read_text(encoding="utf-8"))
    assert manifest["records"]["record_count"] == len(weights)
    assert manifest["weight_policy_id"] == WEIGHT_POLICY_ID


def test_train_and_dev_never_share_a_passage_cluster(tmp_path: Path) -> None:
    build_handoff(
        pairs_path=_pairs_file(tmp_path),
        output_dir=tmp_path / "out",
        selection_adr=_adr(tmp_path),
        dev_every=5,
    )
    directory = tmp_path / "out"
    train = {
        row["passage_cluster_id"]
        for row in read_records(directory / "preference_train.jsonl")
    }
    dev = {row["passage_cluster_id"] for row in read_records(directory / "preference_dev.jsonl")}
    assert not train & dev


def test_handoff_refuses_to_overwrite(tmp_path: Path) -> None:
    pairs = _pairs_file(tmp_path)
    build_handoff(
        pairs_path=pairs, output_dir=tmp_path / "out", selection_adr=_adr(tmp_path), dev_every=5
    )
    with pytest.raises(FileExistsError):
        build_handoff(
            pairs_path=pairs, output_dir=tmp_path / "out", selection_adr=_adr(tmp_path), dev_every=5
        )
