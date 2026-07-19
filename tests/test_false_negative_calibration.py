from __future__ import annotations

import json
from pathlib import Path

import pytest

from doc2query.reranker.base import FrozenRerankerConfig
from doc2query.reranker.false_negative_calibration import (
    QueryScoreGroup,
    build_calibration_artifact,
    select_query_macro_youden_threshold,
)


def test_query_macro_youden_selects_data_driven_conservative_boundary() -> None:
    groups = [
        QueryScoreGroup("q1", (0.9,), (0.8, 0.1)),
        QueryScoreGroup("q2", (0.7,), (0.6, 0.2)),
    ]
    point = select_query_macro_youden_threshold(groups)
    assert point.threshold == 0.7
    assert point.query_macro_true_positive_rate == 1.0
    assert point.query_macro_false_positive_rate == 0.25
    assert point.youden_j == 0.75


def test_calibration_artifact_is_dev_only_and_reproducible(tmp_path: Path) -> None:
    dataset = tmp_path / "dev.parquet"
    dataset.write_bytes(b"frozen dev")
    scores = tmp_path / "scores.jsonl"
    rows = [
        {
            "judge": "primary",
            "query_id": "q1",
            "positive_doc_id": "p1",
            "negative_doc_ids": ["n1", "n2"],
            "document_scores": [0.9, 0.4, 0.1],
        },
        {
            "judge": "primary",
            "query_id": "q2",
            "positive_doc_id": "p2",
            "negative_doc_ids": ["n3", "n4"],
            "document_scores": [0.8, 0.3, 0.2],
        },
    ]
    scores.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    judge = FrozenRerankerConfig(
        name_or_path="primary",
        revision="a" * 40,
        license="test",
        max_length=128,
    )
    first = build_calibration_artifact(
        scores_path=scores,
        fit_dataset_path=dataset,
        fit_split="dev",
        judge=judge,
        bootstrap_samples=20,
        bootstrap_seed=7,
    )
    second = build_calibration_artifact(
        scores_path=scores,
        fit_dataset_path=dataset,
        fit_split="dev",
        judge=judge,
        bootstrap_samples=20,
        bootstrap_seed=7,
    )
    assert first == second
    assert first["threshold"] == 0.8
    assert first["tests_used_for_threshold_tuning"] == []
    assert len(first["artifact_fingerprint"]) == 64
    with pytest.raises(ValueError, match="restricted to a dev split"):
        build_calibration_artifact(
            scores_path=scores,
            fit_dataset_path=dataset,
            fit_split="test_native_pl",
            judge=judge,
            bootstrap_samples=20,
        )


def test_calibration_group_schema_preserves_one_query_weight(tmp_path: Path) -> None:
    dataset = tmp_path / "dev.parquet"
    dataset.write_bytes(b"dev")
    scores = tmp_path / "scores.jsonl"
    scores.write_text(
        json.dumps(
            {
                "schema": "possible_false_negative_dev_scores_v1",
                "judge": "primary",
                "query_id": "q1",
                "positive_doc_ids": ["p1", "p2"],
                "negative_doc_ids": ["n1"],
                "positive_scores": [0.9, 0.8],
                "negative_scores": [0.1],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    artifact = build_calibration_artifact(
        scores_path=scores,
        fit_dataset_path=dataset,
        fit_split="dev",
        judge=FrozenRerankerConfig("primary", "a" * 40, "test"),
        bootstrap_samples=10,
    )
    assert artifact["counts"] == {
        "queries": 1,
        "positive_pairs": 2,
        "inherited_negative_pairs": 1,
    }
