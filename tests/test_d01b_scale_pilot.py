from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml

from doc2query.evaluation.d01_prospective import assert_scoring_summary
from doc2query.evaluation.d01b_scale_pilot import (
    _assert_contract_shape,
    preflight_scale_pilot,
)

CONFIG = Path("configs/evaluation/d01b_scale_interaction_4_5b_pilot_v1.yaml")


def _config() -> dict[str, Any]:
    value = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("cohort", "selected_count"), 999, "cohort"),
        (("selector", "frozen_commit"), "new", "selector"),
        (("arms", "shared_model", "revision"), "main", "model"),
        (("arms", "decoding", "seed"), 43, "decoding"),
        (("arms", "decoding", "top_p"), 0.9, "decoding"),
        (("scoring", "shadow", "revision"), "main", "judge"),
        (("copy_risk", "copy_density"), 0.5, "copy-risk"),
        (("probe", "token_count"), 1, "probe"),
        (("probe", "model_revision"), "main", "probe model"),
        (("probe", "p04_thresholds", "corpus_ndcg_at_10"), 0.009, "P-04"),
        (("evaluation", "bootstrap_seed"), 1, "bootstrap"),
        (
            ("evaluation", "gates", "shadow_pool_recall_at_1", "noninferiority_margin"),
            0.03,
            "metrics",
        ),
        (("multiplicity", "dev_confirm_required_seeds"), [42, 43], "multiplicity"),
        (("resources", "minimum_free_disk_bytes"), 1, "resource"),
        (("authorization", "dev_confirm"), True, "authorization"),
        (("final_tests_used",), ["test_embedder"], "contract"),
    ],
)
def test_scale_pilot_contract_fails_closed_on_drift(
    path: tuple[str, ...], value: object, message: str
) -> None:
    config = deepcopy(_config())
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        _assert_contract_shape(config)


def test_scale_pilot_rejects_any_final_test_reference() -> None:
    config = deepcopy(_config())
    config["cohort"]["source_records"] = "data/processed/v1/test.parquet"
    with pytest.raises(ValueError, match="final-test"):
        _assert_contract_shape(config)


def test_real_scale_pilot_preflight_without_materialization() -> None:
    result = preflight_scale_pilot(CONFIG, require_materialized=False)
    assert result["status"] == "verified"
    assert result["generation_cohort_count"] == 1000
    assert result["evaluation_cohort_count"] == 2000
    assert result["generation_evaluation_overlap"] == 0
    assert result["four_point_five_b_full_authorized"] is False
    assert result["final_tests_used"] == []


def test_scale_pilot_scoring_summary_uses_frozen_4000_row_budget(tmp_path: Path) -> None:
    summary = {
        "status": "measured",
        "final_tests_used": [],
        "generation_count": 4000,
        "judges": {"primary_status": "measured", "shadow_status": "measured"},
        "protocols": {"corpus_retrieval": {"status": "measured"}},
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")

    assert assert_scoring_summary(path, rows=1000 * 4) == summary
    with pytest.raises(ValueError, match="scoring is incomplete"):
        assert_scoring_summary(path)
