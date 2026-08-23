from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import require_local_artifacts

from doc2query.evaluation.d01b_scale_confirm import (
    assess_confirm_feasibility,
    sensitivity_from_pilot_ci,
)

CONFIG = Path("configs/evaluation/d01b_scale_interaction_4_5b_pilot_v1.yaml")
SNAPSHOT = Path(
    "reports/measurements/task05/d01b_scale_interaction_4_5b_dev_confirm_feasibility_v1.json"
)


def test_real_id_only_audit_fails_closed_for_591_record_reserve() -> None:
    require_local_artifacts()
    result = assess_confirm_feasibility(CONFIG)
    assert result == json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    assert result["status"] == "blocked_insufficient_unseen_development"
    assert result["audit"]["legal_unseen_eligible_count"] == 591
    assert result["audit"]["intersection_with_any_excluded_or_seen_id"] == 0
    assert result["audit"]["record_text_fields_used"] == []
    assert result["audit"]["raw_ids_emitted"] is False
    assert result["sensitivity"]["required_queries_for_80pct_power"] == 3922
    assert result["confirm_config_frozen"] is False
    assert result["expensive_run_authorized"] is False
    assert result["final_tests_used"] == []


def test_sensitivity_uses_two_sided_97_5_interval() -> None:
    result = sensitivity_from_pilot_ci(
        pilot_difference=0.02073792007878962,
        pilot_ci95=(0.011055484860771694, 0.03017264616376007),
        pilot_queries=2000,
        confirm_queries=591,
        practical_effect_threshold=0.01,
    )
    assert result["projected_half_width_one_seed_variance"] == pytest.approx(0.020108814663528603)
    assert result["projected_ci_at_pilot_effect"][0] == pytest.approx(0.0006291054152610179)
    assert result["optimistic_three_independent_seed_lower_bound"] < 0.01
    assert result["required_queries_for_90pct_power"] == 5121
    assert result["seed_variance_estimable_from_one_seed_pilot"] is False


def test_sensitivity_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="query counts"):
        sensitivity_from_pilot_ci(
            pilot_difference=0.02,
            pilot_ci95=(0.01, 0.03),
            pilot_queries=0,
            confirm_queries=591,
            practical_effect_threshold=0.01,
        )
