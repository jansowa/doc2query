from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml
from conftest import require_local_artifacts

from doc2query.preferences.hybrid_handoff import _assert_shape, preflight_hybrid_handoff

CONFIG = Path("configs/preferences/d01b_hybrid_task06_handoff_v1.yaml")


def _config() -> dict[str, Any]:
    value = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("source_procedure", "candidate_count"), 7, "procedure"),
        (("source_procedure", "w06_anchor", "role"), "winner", "roles"),
        (("source_procedure", "selector", "anchor"), "none", "selector"),
        (("task07_start", "adapter_role"), "w06_anchor", "Task 07"),
        (("authorization", "task06_generation_authorized"), True, "authorization"),
        (("authorization", "task07_training_authorized"), True, "authorization"),
        (("next_stage_requirements", "minimum_generation_seeds"), 1, "requirements"),
        (("final_tests_used",), ["test"], "contract"),
    ],
)
def test_hybrid_handoff_contract_fails_closed(
    path: tuple[str, ...], value: object, message: str
) -> None:
    config = deepcopy(_config())
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        _assert_shape(config)


def test_real_hybrid_handoff_preflight_is_model_free_and_closed() -> None:
    require_local_artifacts()
    result = preflight_hybrid_handoff(CONFIG)
    assert result["status"] == "verified_ready_for_task06_execution_design_not_generation"
    assert result["candidate_pool"] == {"w06": 4, "d01_controlled": 4, "selected": 4}
    assert result["task07_start_adapter_role"] == "d01_controlled"
    assert result["model_loading_performed"] is False
    assert result["task06_generation_authorized"] is False
    assert result["task07_training_authorized"] is False
    assert result["final_tests_used"] == []
