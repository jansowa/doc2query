from __future__ import annotations

import copy
import shutil
from pathlib import Path
from typing import Any

import pytest

from doc2query.evaluation.p04_decision import REQUIRED_METRICS, evaluate_p04_comparison
from doc2query.evaluation.statistical_contract import StatisticalContract, build_budget_manifest


def _contract() -> StatisticalContract:
    return StatisticalContract.load(Path("configs/evaluation/comparison_contract_v1.yaml"))


def _summary(values: list[float]) -> dict[str, float]:
    if len(values) == 1:
        sd = 0.0
    else:
        mean = sum(values) / len(values)
        sd = (sum((value - mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5
    return {
        "mean": sum(values) / len(values),
        "sample_sd": sd,
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
    }


def _manifest(contract: StatisticalContract) -> dict[str, Any]:
    return {
        "statistical_contract": contract.reference(),
        "comparison_budget": build_budget_manifest(
            token_count=4000,
            pair_count=400,
            unique_passage_count=400,
            queries_per_passage=1,
        ),
    }


def _report(stage: str = "dev_confirm") -> tuple[dict[str, Any], dict[str, Any]]:
    contract = _contract()
    manifest = _manifest(contract)
    seeds = [42] if stage == "dev_screen" else [42, 43, 44]
    per_seed: list[dict[str, Any]] = [
        {"seed": seed, "metrics": {metric: 0.5 + index * 0.01 for metric in REQUIRED_METRICS}}
        for index, seed in enumerate(seeds)
    ]
    seed_summary = {
        metric: _summary([float(row["metrics"][metric]) for row in per_seed])
        for metric in REQUIRED_METRICS
    }
    lows = {
        "corpus_ndcg_at_10": 0.011,
        "corpus_round_trip_at_20": -0.019,
        "sentence_level_source_hit": -0.019,
        "format_valid_rate": -0.004,
    }
    report = {
        **copy.deepcopy(manifest),
        "stage": stage,
        "evaluation_sets": ["dev_intrinsic"],
        "data_sources": ["dev_intrinsic"],
        "final_tests_used": [],
        "per_seed": per_seed,
        "training_seed_summary": seed_summary,
        "paired_query_bootstrap": {
            "unit": "query",
            "paired": True,
            "query_id_fingerprint": "q" * 64,
            "query_count": 100,
            "includes_training_seed_variance": False,
            "samples": 10_000,
            "seed": 20_260_721,
            "metrics": {
                metric: {"difference": low + 0.01, "ci_low": low, "ci_high": low + 0.02}
                for metric, low in lows.items()
            },
        },
    }
    return report, manifest


def test_eligible_requires_practical_effect_and_all_guardrails() -> None:
    report, control = _report()
    result = evaluate_p04_comparison(report, control_manifest=control, contract=_contract())
    assert result["status"] == "eligible"
    assert result["selection_claim"] is None


def test_non_inferior_only_and_guardrail_rejection() -> None:
    report, control = _report()
    report["paired_query_bootstrap"]["metrics"]["corpus_ndcg_at_10"] = {
        "difference": 0.005,
        "ci_low": 0.0,
        "ci_high": 0.01,
    }
    result = evaluate_p04_comparison(report, control_manifest=control, contract=_contract())
    assert result["status"] == "non_inferior_only"

    report, control = _report()
    report["paired_query_bootstrap"]["metrics"]["format_valid_rate"] = {
        "difference": -0.005,
        "ci_low": -0.006,
        "ci_high": -0.004,
    }
    result = evaluate_p04_comparison(report, control_manifest=control, contract=_contract())
    assert result["status"] == "rejected"


def test_missing_seed_ci_and_seed_variance_pooling_are_incomplete() -> None:
    report, control = _report()
    report["per_seed"].pop()
    report["paired_query_bootstrap"]["metrics"].pop("sentence_level_source_hit")
    report["paired_query_bootstrap"]["includes_training_seed_variance"] = True
    result = evaluate_p04_comparison(report, control_manifest=control, contract=_contract())
    assert result["status"] == "incomplete"
    assert any("seed set mismatch" in error for error in result["errors"])
    assert any("sentence_level_source_hit" in error for error in result["errors"])
    assert any("must not aggregate" in error for error in result["errors"])


def test_budget_adr_and_final_test_drift_fail_closed() -> None:
    report, control = _report()
    report["comparison_budget"]["token_count"] += 1
    report["statistical_contract"]["adr_fingerprint"] = "0" * 64
    report["data_sources"].append("test_native_pl")
    result = evaluate_p04_comparison(report, control_manifest=control, contract=_contract())
    assert result["status"] == "incomplete"
    assert any("token_count" in error for error in result["errors"])
    assert any("adr_fingerprint" in error for error in result["errors"])
    assert any("final test" in error for error in result["errors"])


def test_successive_halving_enforces_stage_seed_sets() -> None:
    report, control = _report("dev_screen")
    assert (
        evaluate_p04_comparison(report, control_manifest=control, contract=_contract())["status"]
        == "eligible"
    )
    report["per_seed"].append(copy.deepcopy(report["per_seed"][0]) | {"seed": 43})
    result = evaluate_p04_comparison(report, control_manifest=control, contract=_contract())
    assert result["status"] == "incomplete"


def test_contract_load_fails_when_pinned_adr_file_drifts(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    contract_path = root / "configs/evaluation/comparison_contract_v1.yaml"
    adr_path = root / "reports/decisions/task04_p04_statistical_budget_contract.md"
    contract_path.parent.mkdir(parents=True)
    adr_path.parent.mkdir(parents=True)
    (root / "AGENTS.md").write_text("fixture\n", encoding="utf-8")
    shutil.copyfile(Path("configs/evaluation/comparison_contract_v1.yaml"), contract_path)
    shutil.copyfile(Path("reports/decisions/task04_p04_statistical_budget_contract.md"), adr_path)
    adr_path.write_text(adr_path.read_text(encoding="utf-8") + "drift\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ADR fingerprint"):
        StatisticalContract.load(contract_path)
