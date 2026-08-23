from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml
from conftest import require_local_artifacts

from doc2query.evaluation.d01b_trivia_confirm import (
    _assert_contract_shape,
    _paired_97_5,
    preflight_trivia_confirm,
)

CONFIG = Path("configs/evaluation/d01b_scale_interaction_4_5b_trivia_dev_confirm_v1.yaml")
RUNNER = Path("scripts/run_task05_d01b_scale_interaction_4_5b_trivia_dev_confirm.sh")


def _config() -> dict[str, Any]:
    value = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("external_development", "query_count"), 591, "external"),
        (("external_development", "positive_filter"), ">= 23.50", "external"),
        (("training", "seeds"), [42, 43], "training"),
        (("training", "reused_without_retraining"), [], "training"),
        (("training", "token_count"), 1, "training"),
        (("evaluation", "interval_quantiles"), [0.025, 0.975], "97.5"),
        (("evaluation", "practical_effect_threshold"), 0.0, "97.5"),
        (("execution", "evaluation_encode_batch_size"), 32, "batch-8"),
        (("execution_amendment", "prior_evaluation_encode_batch_size"), 8, "batch-8"),
        (("authorization", "pilot_retraining"), True, "authorization"),
        (("authorization", "final_tests"), True, "authorization"),
        (("final_tests_used",), ["anything"], "contract"),
    ],
)
def test_trivia_confirm_contract_fails_closed(
    path: tuple[str, ...], value: object, message: str
) -> None:
    config = deepcopy(_config())
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        _assert_contract_shape(config)


def test_real_trivia_confirm_preflight_before_seed42_staging() -> None:
    require_local_artifacts()
    result = preflight_trivia_confirm(CONFIG, require_staged_seed42=False)
    assert result["status"] == "verified"
    assert result["external_query_count"] == 8000
    assert result["training_seeds"] == [42, 43, 44]
    assert result["reused_without_retraining"] == [42]
    assert result["newly_trained"] == [43, 44]
    assert result["interval"] == "paired_query_percentile_two_sided_97_5"
    assert result["primary_threshold"] == 0.01
    assert result["evaluation_encode_batch_size"] == 8
    assert result["expensive_run_started"] is False
    assert result["final_tests_used"] == []


def test_trivia_confirm_runner_uses_amended_encode_batch() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    assert "--evaluation-encode-batch-size 8" in runner
    assert "--evaluation-encode-batch-size 32" not in runner


def test_paired_97_5_bootstrap_is_deterministic_and_variant_minus_control() -> None:
    control = {f"q-{index}": 0.0 for index in range(20)}
    variant = {f"q-{index}": 0.02 for index in range(20)}
    first = _paired_97_5(control, variant, samples=200, seed=7)
    second = _paired_97_5(control, variant, samples=200, seed=7)
    assert first == second
    assert first["query_count"] == 20
    assert first["difference"] == pytest.approx(0.02)
    assert first["ci97_5_low"] == pytest.approx(0.02)
    assert first["ci97_5_high"] == pytest.approx(0.02)


def test_paired_97_5_rejects_mismatched_ids() -> None:
    with pytest.raises(ValueError, match="identical"):
        _paired_97_5({"a": 0.0}, {"b": 1.0}, samples=10, seed=1)
