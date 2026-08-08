from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from doc2query.evaluation.p04_decision import evaluate_p04_comparison
from doc2query.evaluation.p05_guardrails import (
    build_dev_confirm_report,
    build_dev_screen_report,
)
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
    assert report["paired_query_bootstrap"]["rng"] == "numpy.random.PCG64"


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


def test_build_dev_confirm_report_separates_seed_variance_from_query_bootstrap(
    tmp_path: Path,
) -> None:
    contract = StatisticalContract.load(Path("configs/evaluation/comparison_contract_v1.yaml"))
    manifest = {
        "statistical_contract": contract.reference(),
        "comparison_budget": build_budget_manifest(
            token_count=400, pair_count=40, unique_passage_count=10, queries_per_passage=4
        ),
    }
    ids = ["a", "b", "c"]
    arm_paths: dict[int, Path] = {}
    control_paths: dict[int, Path] = {}
    arm_results: dict[int, dict[str, Any]] = {}
    control_results: dict[int, dict[str, Any]] = {}
    for offset, seed in enumerate((42, 43, 44)):
        control_path = tmp_path / f"control-{seed}.jsonl"
        arm_path = tmp_path / f"arm-{seed}.jsonl"
        _write(
            control_path,
            [
                {"example_id": key, "corpus_ndcg_at_10": 0.10 + offset * 0.005}
                for key in ids
            ],
        )
        _write(
            arm_path,
            [
                {"example_id": key, "corpus_ndcg_at_10": 0.13 + offset * 0.005}
                for key in ids
            ],
        )
        control_paths[seed] = control_path
        arm_paths[seed] = arm_path
        control_results[seed] = dict(manifest)
        arm_results[seed] = dict(manifest)

    guardrails = tmp_path / "guardrails.jsonl"
    shared = {
        "corpus_round_trip_at_20": 1.0,
        "sentence_level_source_hit": 1.0,
        "format_valid_rate": 1.0,
    }
    _write(guardrails, [{"example_id": key, **shared} for key in ids])

    report = build_dev_confirm_report(
        arm_id="arm-confirm",
        control_id="control-confirm",
        arm_results=arm_results,
        control_results=control_results,
        arm_per_query_paths=arm_paths,
        control_per_query_paths=control_paths,
        arm_guardrails_path=guardrails,
        control_guardrails_path=guardrails,
        contract=contract,
    )
    decision = evaluate_p04_comparison(
        report, control_manifest=control_results[42], contract=contract
    )
    assert decision["status"] == "eligible"
    assert [row["seed"] for row in report["per_seed"]] == [42, 43, 44]
    assert report["training_seed_summary"]["corpus_ndcg_at_10"]["sample_sd"] > 0
    assert report["paired_query_bootstrap"]["includes_training_seed_variance"] is False
    assert (
        report["paired_query_bootstrap"]["fixed_training_seed_aggregation"]
        == "per_query_mean_before_query_resampling"
    )
