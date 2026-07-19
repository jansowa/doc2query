from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from doc2query.evaluation.p03_sensitivity import (
    ARM_NAMES,
    assert_equal_budget,
    assert_no_test_ids,
    assert_sensitivity_compatible,
    common_cohort,
    generate_w05_queries,
    load_sensitivity_config,
    mock_smoke,
    preflight,
)
from doc2query.utils.records import read_records


def _records() -> list[dict[str, Any]]:
    return [
        {
            "example_id": f"train-{index}",
            "query": f"natural {index}",
            "metadata": {"split": "train"},
            "positives": [{"doc_id": f"p-{index}", "text": f"positive {index}"}],
            "hard_negatives": [{"doc_id": f"n-{index}", "text": f"negative {index}"}],
        }
        for index in range(4)
    ]


def _generation_config() -> dict[str, Any]:
    return {
        "generator": {
            "checkpoint": "runs/W05-1.5B-50K-8GB/checkpoint-3125",
            "model_name_or_path": "speakleash/Bielik-1.5B-v3",
            "revision": "4b25049621bf3952a1fc9314c89773102eda0333",
            "max_length": 32,
            "max_new_tokens": 8,
            "do_sample": False,
            "num_return_sequences": 1,
            "seed": 42,
        }
    }


def test_generation_resume_has_exactly_one_row_per_id(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    journal = tmp_path / "generation.sqlite"
    output = tmp_path / "generations.jsonl"
    with pytest.raises(InterruptedError):
        generate_w05_queries(
            _records(),
            raw_config=_generation_config(),
            cohort_fingerprint="f" * 64,
            journal_path=journal,
            output_path=output,
            mock=True,
            interrupt_after=2,
        )
    report = generate_w05_queries(
        _records(),
        raw_config=_generation_config(),
        cohort_fingerprint="f" * 64,
        journal_path=journal,
        output_path=output,
        mock=True,
    )
    rows = list(read_records(output))
    assert report["resumed_records"] == 2
    assert len(rows) == len({row["example_id"] for row in rows}) == 4
    assert all(row["mode"] == "deterministic" for row in rows)
    assert all(row["candidate_index"] == 0 for row in rows)
    assert all(row["revision"] == "4b25049621bf3952a1fc9314c89773102eda0333" for row in rows)
    assert all(row["fingerprint"] == rows[0]["fingerprint"] for row in rows)
    progress = capsys.readouterr().err
    assert "[P03 generation/resume]" in progress
    assert "4/4 (100.0%)" in progress
    assert "eta=" in progress


def test_final_test_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="frozen test IDs"):
        assert_no_test_ids(["train-1", "test-1"], {"test-1"})


def test_common_cohort_is_identical_and_in_frozen_order() -> None:
    base = [
        {
            "example_id": f"q-{index}",
            "query": f"query {index}",
            "positive": f"positive {index}",
            "negative": f"negative {index}",
        }
        for index in range(4)
    ]
    arms = {
        "hn0": base,
        "hn0_filter": [base[0], base[2], base[3]],
        "hn1_bm25": [base[3], base[2], base[0]],
    }
    rows, report = common_cohort(arms, ["q-0", "q-1", "q-2", "q-3"])
    assert report["ordered_example_ids"] == ["q-0", "q-2", "q-3"]
    assert report["drop_rate"] == 0.25
    for arm in ARM_NAMES:
        assert [row["example_id"] for row in rows[arm]] == ["q-0", "q-2", "q-3"]
        assert [row["query"] for row in rows[arm]] == ["query 0", "query 2", "query 3"]


def _budget() -> dict[str, Any]:
    return {
        "cohort_examples": 10,
        "seed": 42,
        "max_steps": 1000,
        "batch_size": 8,
        "max_length": 192,
        "sequences_per_example": 3,
        "padding": "max_length",
        "tokens_per_step": 4608,
        "total_padded_tokens": 4_608_000,
        "cohort_unpadded_tokens": {"negative": 123},
    }


def test_budget_contract_rejects_trajectory_drift_but_allows_negative_lengths() -> None:
    budgets = {arm: _budget() for arm in ARM_NAMES}
    budgets["hn1_bm25"]["cohort_unpadded_tokens"] = {"negative": 999}
    assert assert_equal_budget(budgets)["total_padded_tokens"] == 4_608_000
    budgets["hn1_bm25"]["max_steps"] = 999
    with pytest.raises(ValueError, match="max_steps"):
        assert_equal_budget(budgets)


def _contract(arm: str) -> dict[str, Any]:
    return {
        "contract_version": "p03-w05-sensitivity-v1",
        "diagnostic_scope": "W05 hard-negative recipe sensitivity; not generator comparison",
        "arm": arm,
        "hard_negative_strategy": arm,
        "probe_model_name_or_path": "probe",
        "probe_model_revision": "r" * 40,
        "probe_recipe_version": "probe-v1.1-p03",
        "tokenizer_name_or_path": "probe",
        "tokenizer_revision": "r" * 40,
        "max_length": 192,
        "batch_size": 8,
        "max_steps": 1000,
        "learning_rate": 2e-5,
        "warmup_ratio": 0.05,
        "seed": 42,
        "loss": "loss",
        "false_negative_policy": "drop",
        "calibration_artifact_fingerprint": "c" * 64,
        "primary_judge_name": "judge",
        "primary_judge_revision": "j" * 40,
        "possible_false_negative_threshold": 8.6,
        "bm25_index_fingerprint": "b" * 64 if arm == "hn1_bm25" else None,
        "cohort_fingerprint": "f" * 64,
        "cohort_count": 10,
        "generation_fingerprint": "g" * 64,
        "dev_fingerprint": "d" * 64,
        "budget": _budget(),
        "final_tests_used": [],
    }


def test_sensitivity_comparator_allows_only_negative_strategy_drift() -> None:
    assert assert_sensitivity_compatible(_contract("hn0"), _contract("hn1_bm25"))
    changed = json.loads(json.dumps(_contract("hn1_bm25")))
    changed["learning_rate"] = 1e-4
    with pytest.raises(ValueError, match="learning_rate"):
        assert_sensitivity_compatible(_contract("hn0"), changed)


def test_preflight_without_model_loading_and_mock_smoke(tmp_path: Path) -> None:
    root = Path.cwd()
    raw = load_sensitivity_config(Path("configs/evaluation/p03_w05_sensitivity.yaml"))
    report = preflight(raw, root, require_model_cache=False)
    assert report["scope"] == "task04-p03-only"
    assert report["final_tests_used"] == []
    smoke = mock_smoke(tmp_path / "smoke")
    assert smoke["status"] == "passed"
    assert smoke["generation_resume"]["resumed_records"] == 2


def test_runner_help_and_shell_mock_smoke() -> None:
    help_result = subprocess.run(
        ["bash", "scripts/run_p03_w05_sensitivity.sh", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert help_result.returncode == 0
    assert "--dry-run" in help_result.stdout
    smoke_result = subprocess.run(
        ["bash", "scripts/run_p03_w05_sensitivity.sh", "--smoke"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert smoke_result.returncode == 0, smoke_result.stderr
    payload = json.loads(smoke_result.stdout)
    assert payload["mock_only"] is True
