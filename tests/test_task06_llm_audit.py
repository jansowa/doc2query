from __future__ import annotations

import json
from pathlib import Path

import pytest

from doc2query.preferences.llm_audit import (
    build_dual_llm_request_plan,
    load_llm_audit_config,
    plan_summary,
)


def test_owner_approved_groq_contract_pins_limits_and_models() -> None:
    config = load_llm_audit_config(Path("configs/preferences/task06_groq_preference_audit_v1.json"))

    assert config["limits_per_model"]["minimum_seconds_between_requests"] == 4.0
    assert config["quota_scheduler"]["global_request_serialization"] is True
    assert config["quota_scheduler"]["on_model_quota_exhausted"] == (
        "defer_model_and_switch_to_other"
    )
    assert config["quota_scheduler"]["when_both_models_deferred"] == (
        "stop_cleanly_incomplete_quota_deferred"
    )
    assert config["quota_scheduler"]["automatic_next_day_wakeup"] is False
    assert config["human_evidence_claimed"] is False


def test_dual_plan_assigns_every_blind_pair_to_both_models(tmp_path: Path) -> None:
    config_path = Path("configs/preferences/task06_groq_preference_audit_v1.json")
    config = load_llm_audit_config(config_path)
    blind = tmp_path / "blind.jsonl"
    with blind.open("w", encoding="utf-8") as handle:
        for index in range(500):
            handle.write(
                json.dumps(
                    {
                        "audit_id": f"audit-{index:04d}",
                        "passage": f"Pasaż {index}",
                        "query_a": f"Zapytanie A {index}",
                        "query_b": f"Zapytanie B {index}",
                        "orientation_commitment": f"commit-{index}",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    summary = plan_summary(build_dual_llm_request_plan(config, blind))

    assert summary["pair_count"] == 500
    assert summary["rating_count"] == 1000
    assert summary["by_model"] == {
        "openai/gpt-oss-120b": {"requests": 250, "ratings": 500},
        "qwen/qwen3.6-27b": {"requests": 250, "ratings": 500},
    }
    assert summary["human_evidence_claimed"] is False


def test_blind_plan_rejects_leaked_automatic_preference(tmp_path: Path) -> None:
    config = load_llm_audit_config(Path("configs/preferences/task06_groq_preference_audit_v1.json"))
    config["pair_count"] = 1
    blind = tmp_path / "blind.jsonl"
    blind.write_text(
        json.dumps(
            {
                "audit_id": "audit-1",
                "passage": "Pasaż",
                "query_a": "A",
                "query_b": "B",
                "orientation_commitment": "commit",
                "automatic_preference": "A",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="leaks selection fields"):
        build_dual_llm_request_plan(config, blind)
