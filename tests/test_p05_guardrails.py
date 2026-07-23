from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from doc2query.evaluation.p04_decision import evaluate_p04_comparison
from doc2query.evaluation.p05_guardrails import build_dev_screen_report
from doc2query.evaluation.statistical_contract import StatisticalContract, build_budget_manifest


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_build_report_is_paired_and_consumable_by_p04(tmp_path: Path) -> None:
    contract = StatisticalContract.load(Path("configs/evaluation/comparison_contract_v1.yaml"))
    manifest = {
        "statistical_contract": contract.reference(),
        "comparison_budget": build_budget_manifest(
            token_count=100, pair_count=10, unique_passage_count=10, queries_per_passage=1
        ),
    }
    control_probe = tmp_path / "control_probe.jsonl"
    arm_probe = tmp_path / "arm_probe.jsonl"
    control_guard = tmp_path / "control_guard.jsonl"
    arm_guard = tmp_path / "arm_guard.jsonl"
    ids = ["a", "b", "c"]
    _write(control_probe, [{"example_id": key, "corpus_ndcg_at_10": 0.1} for key in ids])
    _write(arm_probe, [{"example_id": key, "corpus_ndcg_at_10": 0.12} for key in ids])
    shared = {
        "corpus_round_trip_at_20": 1.0,
        "sentence_level_source_hit": 1.0,
        "format_valid_rate": 1.0,
    }
    _write(control_guard, [{"example_id": key, **shared} for key in ids])
    _write(arm_guard, [{"example_id": key, **shared} for key in ids])
    report = build_dev_screen_report(
        arm_id="arm",
        control_id="control",
        arm_result=manifest,
        control_result=manifest,
        arm_per_query_path=arm_probe,
        control_per_query_path=control_probe,
        arm_guardrails_path=arm_guard,
        control_guardrails_path=control_guard,
        contract=contract,
    )
    decision = evaluate_p04_comparison(report, control_manifest=manifest, contract=contract)
    assert decision["status"] == "eligible"
    assert report["paired_query_bootstrap"]["query_count"] == 3


def test_build_report_rejects_unpaired_guardrails(tmp_path: Path) -> None:
    contract = StatisticalContract.load(Path("configs/evaluation/comparison_contract_v1.yaml"))
    manifest = {
        "statistical_contract": contract.reference(),
        "comparison_budget": build_budget_manifest(
            token_count=100, pair_count=10, unique_passage_count=10, queries_per_passage=1
        ),
    }
    paths = [tmp_path / f"{index}.jsonl" for index in range(4)]
    _write(paths[0], [{"example_id": "a", "corpus_ndcg_at_10": 0.1}])
    _write(paths[1], [{"example_id": "a", "corpus_ndcg_at_10": 0.1}])
    guard = {
        "corpus_round_trip_at_20": 1.0,
        "sentence_level_source_hit": 1.0,
        "format_valid_rate": 1.0,
    }
    _write(paths[2], [{"example_id": "a", **guard}])
    _write(paths[3], [{"example_id": "b", **guard}])
    with pytest.raises(ValueError, match="identical query IDs"):
        build_dev_screen_report(
            arm_id="arm",
            control_id="control",
            arm_result=manifest,
            control_result=manifest,
            arm_per_query_path=paths[0],
            control_per_query_path=paths[1],
            arm_guardrails_path=paths[2],
            control_guardrails_path=paths[3],
            contract=contract,
        )
