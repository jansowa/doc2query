from __future__ import annotations

import json
from pathlib import Path

import pytest

from doc2query.preferences.planning import (
    iter_generation_requests,
    prepare_planning_passages,
    write_generation_plan,
)
from doc2query.preferences.schemas import CandidatePlanningConfig
from doc2query.utils.records import read_records


def _config() -> CandidatePlanningConfig:
    return CandidatePlanningConfig.model_validate(
        {
            "plan_id": "unit-plan-v1",
            "plan_seed": 17,
            "target_candidates_per_passage": 8,
            "forms": ["full_question", "keyword_query"],
            "intents": ["fact_lookup", "definition"],
            "focus_modes": ["bucket", "marked_sentence"],
            "temperatures": [0.3, 0.7, 1.0],
            "seeds": [42, 43],
            "top_p": 0.95,
            "max_new_tokens": 64,
            "allowed_splits": ["train", "dev"],
        }
    )


def _source(split: str = "train") -> list[dict[str, object]]:
    return [
        {
            "pair_id": "q1::d1",
            "doc_id": "d1",
            "passage": "Pompa ciepła pobiera energię z otoczenia. Następnie ogrzewa budynek.",
            "split": split,
        },
        {
            "pair_id": "q2::d1",
            "doc_id": "d1",
            "passage": "Pompa ciepła pobiera energię z otoczenia. Następnie ogrzewa budynek.",
            "split": split,
        },
    ]


def test_plan_is_deterministic_and_covers_axes() -> None:
    passages = prepare_planning_passages(
        _source(), [{"doc_id": "d1", "cluster_id": "cluster-1"}], _config()
    )
    first = list(iter_generation_requests(passages, _config()))
    second = list(iter_generation_requests(passages, _config()))
    assert [row.model_dump() for row in first] == [row.model_dump() for row in second]
    assert len(first) == 8
    assert [row.candidate_index for row in first] == list(range(8))
    assert len({row.request_id for row in first}) == 8
    assert {row.seed for row in first} == {42, 43}
    assert {row.temperature for row in first} == {0.3, 0.7, 1.0}
    assert {row.control.form.value for row in first} == {"full_question", "keyword_query"}
    assert first[0].source_pair_ids == ["q1::d1", "q2::d1"]
    assert all("Zapytanie:" in row.prompt for row in first)


def test_planner_rejects_test_rows() -> None:
    with pytest.raises(ValueError, match="must never consume test"):
        prepare_planning_passages(
            _source("test"), [{"doc_id": "d1", "cluster_id": "cluster-1"}], _config()
        )


def test_planner_rejects_cluster_crossing_splits() -> None:
    source: list[dict[str, object]] = [
        *_source()[:1],
        {
            "pair_id": "q3::d2",
            "doc_id": "d2",
            "passage": "Drugi dokument ma inną treść.",
            "split": "dev",
        },
    ]
    dedup = [
        {"doc_id": "d1", "cluster_id": "shared"},
        {"doc_id": "d2", "cluster_id": "shared"},
    ]
    with pytest.raises(ValueError, match="cluster shared crosses"):
        prepare_planning_passages(source, dedup, _config())


def test_write_plan_is_atomic_and_records_no_execution(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    dedup = tmp_path / "dedup.jsonl"
    config = tmp_path / "config.yaml"
    output = tmp_path / "requests.jsonl"
    manifest = tmp_path / "manifest.json"
    source.write_text("\n".join(json.dumps(row) for row in _source()) + "\n", encoding="utf-8")
    dedup.write_text('{"doc_id":"d1","cluster_id":"cluster-1"}\n', encoding="utf-8")
    config.write_text(
        """plan_id: unit-plan-v1
plan_seed: 17
target_candidates_per_passage: 4
forms: [full_question, keyword_query]
intents: [fact_lookup, definition]
focus_modes: [bucket, marked_sentence]
temperatures: [0.3, 0.7, 1.0]
seeds: [42, 43]
top_p: 0.95
max_new_tokens: 64
allowed_splits: [train]
""",
        encoding="utf-8",
    )
    summary = write_generation_plan(source, dedup, config, output, manifest)
    rows = list(read_records(output))
    assert len(rows) == 4
    assert summary["status"] == "planned_not_generated"
    assert summary["generation_started"] is False
    assert summary["scoring_started"] is False
    assert summary["final_tests_used"] == []
    assert len(summary["requests_sha256"]) == 64
    with pytest.raises(FileExistsError, match="already exist"):
        write_generation_plan(source, dedup, config, output, manifest)
