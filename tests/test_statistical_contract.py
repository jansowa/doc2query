from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from doc2query.evaluation.statistical_contract import (
    StatisticalContract,
    assert_same_comparison_contract,
    build_budget_manifest,
)


def _manifest(contract: StatisticalContract) -> dict[str, Any]:
    return {
        "statistical_contract": contract.reference(),
        "comparison_budget": build_budget_manifest(
            token_count=1_000_000,
            pair_count=100,
            unique_passage_count=100,
            queries_per_passage=1,
        ),
    }


def test_p04_contract_loads_and_pins_adr() -> None:
    contract = StatisticalContract.load(Path("configs/evaluation/comparison_contract_v1.yaml"))
    reference = contract.reference()
    assert reference["contract_version"] == "task04-p04-v1"
    assert reference["adr_version"] == "1.0.0"
    assert len(reference["contract_fingerprint"]) == 64


def test_p04_budget_requires_exact_uniform_k() -> None:
    with pytest.raises(ValueError, match="pair_count"):
        build_budget_manifest(
            token_count=100,
            pair_count=5,
            unique_passage_count=3,
            queries_per_passage=2,
        )
    with pytest.raises(ValueError, match="positive integers"):
        build_budget_manifest(
            token_count=0,
            pair_count=1,
            unique_passage_count=1,
            queries_per_passage=1,
        )


def test_comparison_accepts_only_identical_p04_contract_and_budget() -> None:
    contract = StatisticalContract.load(Path("configs/evaluation/comparison_contract_v1.yaml"))
    manifest = _manifest(contract)
    result = assert_same_comparison_contract(manifest, copy.deepcopy(manifest))
    assert result["comparison_budget"]["pair_count"] == 100

    changed_budget = copy.deepcopy(manifest)
    changed_budget["comparison_budget"]["token_count"] += 1
    with pytest.raises(ValueError, match="token_count"):
        assert_same_comparison_contract(manifest, changed_budget)

    changed_contract = copy.deepcopy(manifest)
    changed_contract["statistical_contract"]["adr_version"] = "2.0.0"
    with pytest.raises(ValueError, match="adr_version"):
        assert_same_comparison_contract(manifest, changed_contract)


def test_comparison_fails_closed_when_p04_metadata_is_missing() -> None:
    with pytest.raises(ValueError, match="statistical_contract"):
        assert_same_comparison_contract({}, {})
