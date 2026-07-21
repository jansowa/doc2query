"""Versioned P-04 statistical and comparison-budget contract."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

CONTRACT_SCHEMA_VERSION = 1
BUDGET_DEFINITION_VERSION = "probe-budget-v1"
BUDGET_FIELDS = (
    "token_count",
    "pair_count",
    "unique_passage_count",
    "queries_per_passage",
)
CONTRACT_REFERENCE_FIELDS = (
    "contract_version",
    "contract_fingerprint",
    "adr_id",
    "adr_version",
)


def _fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _non_empty_string(value: Any, *, field: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"P-04 comparison contract requires {field}")
    return result


@dataclass(frozen=True)
class StatisticalContract:
    """Validated pre-registered contract and its immutable manifest reference."""

    payload: dict[str, Any]
    fingerprint: str

    @classmethod
    def load(cls, path: Path) -> StatisticalContract:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("P-04 comparison contract must be a YAML mapping")
        if raw.get("schema_version") != CONTRACT_SCHEMA_VERSION:
            raise ValueError("unsupported P-04 comparison contract schema")
        _non_empty_string(raw.get("contract_version"), field="contract_version")
        adr = raw.get("adr")
        if not isinstance(adr, Mapping):
            raise ValueError("P-04 comparison contract requires an ADR mapping")
        _non_empty_string(adr.get("id"), field="adr.id")
        _non_empty_string(adr.get("version"), field="adr.version")
        _non_empty_string(adr.get("path"), field="adr.path")
        primary = raw.get("primary_metric")
        if not isinstance(primary, Mapping) or primary.get("name") != "corpus_ndcg_at_10":
            raise ValueError("P-04 primary metric must be corpus_ndcg_at_10")
        if primary.get("dataset") != "test_native_pl" or primary.get("profile") != "full":
            raise ValueError("P-04 primary metric must use full test_native_pl")
        practical = raw.get("minimum_practical_effect")
        if not isinstance(practical, Mapping) or float(practical.get("absolute", 0.0)) <= 0:
            raise ValueError("P-04 requires a positive minimum practical effect")
        non_inferiority = raw.get("non_inferiority")
        if not isinstance(non_inferiority, Mapping) or set(non_inferiority) != {
            "grounding",
            "answerability",
            "format",
        }:
            raise ValueError(
                "P-04 requires grounding, answerability and format non-inferiority rules"
            )
        for name, rule in non_inferiority.items():
            if not isinstance(rule, Mapping) or float(rule.get("margin", 0.0)) <= 0:
                raise ValueError(f"P-04 non-inferiority rule {name} needs a positive margin")
        resampling = raw.get("resampling")
        if not isinstance(resampling, Mapping):
            raise ValueError("P-04 requires separate resampling rules")
        if resampling.get("training_variance_unit") != "independent_training_seed":
            raise ValueError("P-04 training variance must be reported across independent seeds")
        if resampling.get("query_uncertainty_unit") != "query":
            raise ValueError("P-04 bootstrap uncertainty must use query as its unit")
        halving = raw.get("successive_halving")
        stages = halving.get("stages") if isinstance(halving, Mapping) else None
        if not isinstance(stages, list) or len(stages) < 2:
            raise ValueError("P-04 requires at least two successive-halving stages")
        budget = raw.get("budget")
        if not isinstance(budget, Mapping):
            raise ValueError("P-04 requires a budget mapping")
        if budget.get("definition_version") != BUDGET_DEFINITION_VERSION:
            raise ValueError(f"P-04 budget definition must be {BUDGET_DEFINITION_VERSION!r}")
        if tuple(budget.get("required_equal_dimensions", ())) != BUDGET_FIELDS:
            raise ValueError("P-04 budget must require all four dimensions in canonical order")
        access = raw.get("data_access")
        if not isinstance(access, Mapping) or access.get("final_test_openings") != 1:
            raise ValueError("P-04 must permit exactly one final-test opening")
        return cls(payload=dict(raw), fingerprint=_fingerprint(raw))

    def reference(self) -> dict[str, Any]:
        adr = self.payload["adr"]
        assert isinstance(adr, Mapping)
        return {
            "contract_version": str(self.payload["contract_version"]),
            "contract_fingerprint": self.fingerprint,
            "adr_id": str(adr["id"]),
            "adr_version": str(adr["version"]),
            "adr_path": str(adr["path"]),
        }


def build_budget_manifest(
    *,
    token_count: int,
    pair_count: int,
    unique_passage_count: int,
    queries_per_passage: int,
) -> dict[str, Any]:
    """Build an exact, comparison-safe four-dimensional budget manifest."""
    values = {
        "token_count": token_count,
        "pair_count": pair_count,
        "unique_passage_count": unique_passage_count,
        "queries_per_passage": queries_per_passage,
    }
    if any(not isinstance(value, int) or value < 1 for value in values.values()):
        raise ValueError("P-04 budget dimensions must be positive integers")
    if pair_count != unique_passage_count * queries_per_passage:
        raise ValueError(
            "P-04 budget requires pair_count == unique_passage_count * queries_per_passage"
        )
    return {"definition_version": BUDGET_DEFINITION_VERSION, **values}


def assert_same_comparison_contract(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed on missing or different P-04 ADR and budget definitions."""
    left_contract = left.get("statistical_contract")
    right_contract = right.get("statistical_contract")
    if not isinstance(left_contract, Mapping) or not isinstance(right_contract, Mapping):
        raise ValueError("comparison requires a complete P-04 statistical_contract")
    contract_mismatches = [
        field
        for field in CONTRACT_REFERENCE_FIELDS
        if not left_contract.get(field) or left_contract.get(field) != right_contract.get(field)
    ]
    if contract_mismatches:
        raise ValueError(
            "comparison rejected due to different P-04 contract versions: "
            + ", ".join(contract_mismatches)
        )
    left_budget = left.get("comparison_budget")
    right_budget = right.get("comparison_budget")
    if not isinstance(left_budget, Mapping) or not isinstance(right_budget, Mapping):
        raise ValueError("comparison requires a complete P-04 comparison_budget")
    if (
        left_budget.get("definition_version") != BUDGET_DEFINITION_VERSION
        or right_budget.get("definition_version") != BUDGET_DEFINITION_VERSION
    ):
        raise ValueError("comparison rejected due to a different P-04 budget definition")
    budget_mismatches = [
        field
        for field in BUDGET_FIELDS
        if not isinstance(left_budget.get(field), int)
        or left_budget.get(field) != right_budget.get(field)
    ]
    if budget_mismatches:
        raise ValueError(
            "comparison rejected due to different P-04 budgets: " + ", ".join(budget_mismatches)
        )
    return {
        "statistical_contract": {
            field: left_contract[field] for field in CONTRACT_REFERENCE_FIELDS
        },
        "comparison_budget": {
            "definition_version": BUDGET_DEFINITION_VERSION,
            **{field: left_budget[field] for field in BUDGET_FIELDS},
        },
    }
