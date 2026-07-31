from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, cast

import pytest

from doc2query.evaluation import groq_audits
from doc2query.utils.records import read_durable_jsonl_prefix


def _config() -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads(
            Path("configs/evaluation/task05_groq_llm_audit_v1.json").read_text(encoding="utf-8")
        ),
    )


def _request(model_id: str = "qwen/qwen3.6-27b") -> groq_audits.PlannedRequest:
    return groq_audits.PlannedRequest(
        request_id="request-1",
        model_id=model_id,
        reviewer_id="reviewer",
        audit_type="label",
        item_ids=("L-1",),
        api_request={"model": model_id, "messages": [], "max_completion_tokens": 100},
        estimated_tokens=100,
    )


def _success(model_id: str = "qwen/qwen3.6-27b") -> groq_audits.HttpResult:
    content = {
        "ratings": [
            {
                "audit_id": "L-1",
                "gold_form": "full_question",
                "gold_intent": "fact_lookup",
                "intent_adequate": "yes",
                "ambiguous": "no",
                "encoding_error": "no",
                "comment": "ok",
            }
        ]
    }
    return groq_audits.HttpResult(
        200,
        {"x-ratelimit-remaining-requests": "999"},
        {
            "id": "response-1",
            "model": model_id,
            "choices": [{"message": {"content": json.dumps(content)}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
        },
    )


def test_production_plan_uses_every_item_once_and_stays_below_safety_budgets(
    tmp_path: Path,
) -> None:
    config = _config()
    label_path = tmp_path / "labels.csv"
    concept_path = tmp_path / "concepts.csv"
    with label_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["audit_id", "query", "positive_passage"])
        writer.writeheader()
        for index in range(500):
            writer.writerow(
                {
                    "audit_id": f"L-{index:04d}",
                    "query": f"pytanie {index}",
                    "positive_passage": f"pasaż {index}",
                }
            )
    with concept_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["audit_id", "passage", "candidate_concepts"])
        writer.writeheader()
        for index in range(200):
            writer.writerow(
                {
                    "audit_id": f"C-{index:04d}",
                    "passage": f"pasaż {index}",
                    "candidate_concepts": "[]",
                }
            )
    config["label_blind_csv"] = str(label_path)
    config["concept_blind_csv"] = str(concept_path)
    plan = groq_audits.build_request_plan(config)
    summary = groq_audits.plan_summary(plan)
    assert summary["item_count"] == 700
    assert summary["request_count"] == 225
    assert summary["by_type"] == {"concept": 200, "label": 500}
    ids = [audit_id for request in plan for audit_id in request.item_ids]
    assert len(ids) == len(set(ids)) == 700
    for row in summary["by_model"].values():
        assert row["requests"] < config["limits_per_model"]["safety_requests_per_day"]
        assert row["estimated_tokens"] < config["limits_per_model"]["safety_tokens_per_day"]


def test_contract_pins_two_second_interval_and_reasoning_modes() -> None:
    config = groq_audits.load_groq_contract(
        Path("configs/evaluation/task05_groq_llm_audit_v1.json")
    )
    assert config["limits_per_model"]["minimum_seconds_between_requests"] >= 2.0
    modes = {row["model_id"]: row["reasoning_effort"] for row in config["models"]}
    assert modes == {"qwen/qwen3.6-27b": "none", "openai/gpt-oss-120b": "low"}


def test_worker_retries_only_definitive_429_and_persists_every_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    clock = 100.0

    def monotonic() -> float:
        return clock

    def sleep(seconds: float) -> None:
        nonlocal clock
        clock += max(seconds, 0.001)

    monkeypatch.setattr("doc2query.evaluation.groq_audits.time.monotonic", monotonic)
    responses = [
        groq_audits.HttpResult(429, {"retry-after": "2"}, {"error": "rate"}),
        _success(),
    ]

    def transport(*_args: Any) -> groq_audits.HttpResult:
        return responses.pop(0)

    result = groq_audits._run_model_worker(
        [_request()],
        config=config,
        output_dir=tmp_path,
        api_key="secret-not-persisted",
        transport=transport,
        sleep=sleep,
        max_new_requests=None,
        allow_ambiguous_resend=False,
    )
    assert result["completed"] == 1
    journal = tmp_path / "ledgers" / "qwen__qwen3.6-27b.jsonl"
    events = read_durable_jsonl_prefix(journal)
    assert [event["event"] for event in events] == [
        "request_started",
        "rate_limited",
        "request_started",
        "response_received",
    ]
    assert "secret-not-persisted" not in journal.read_text(encoding="utf-8")


def test_worker_retries_over_capacity_with_exponential_backoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    clock = 100.0
    sleeps: list[float] = []

    def monotonic() -> float:
        return clock

    def sleep(seconds: float) -> None:
        nonlocal clock
        sleeps.append(seconds)
        clock += max(seconds, 0.001)

    monkeypatch.setattr("doc2query.evaluation.groq_audits.time.monotonic", monotonic)
    responses = [
        groq_audits.HttpResult(503, {}, {"error": {"message": "over capacity"}}),
        _success(),
    ]
    result = groq_audits._run_model_worker(
        [_request()],
        config=config,
        output_dir=tmp_path,
        api_key="secret",
        transport=lambda *_args: responses.pop(0),
        sleep=sleep,
        max_new_requests=None,
        allow_ambiguous_resend=False,
    )
    assert result["completed"] == 1
    assert groq_audits.TRANSIENT_BACKOFF_INITIAL_SECONDS in sleeps
    events = read_durable_jsonl_prefix(tmp_path / "ledgers" / "qwen__qwen3.6-27b.jsonl")
    assert [event["event"] for event in events] == [
        "request_started",
        "transient_http_error",
        "request_started",
        "response_received",
    ]


def test_worker_expands_completion_budget_after_json_generation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    clock = 100.0

    def sleep(seconds: float) -> None:
        nonlocal clock
        clock += max(seconds, 0.001)

    monkeypatch.setattr("doc2query.evaluation.groq_audits.time.monotonic", lambda: clock)
    responses = [
        groq_audits.HttpResult(
            400,
            {},
            {
                "error": {
                    "code": "json_validate_failed",
                    "failed_generation": (
                        "max completion tokens reached before generating a valid document"
                    ),
                }
            },
        ),
        _success(),
    ]
    sent: list[dict[str, Any]] = []

    def transport(_url: str, _key: str, payload: Any, _timeout: float) -> Any:
        sent.append(dict(payload))
        return responses.pop(0)

    result = groq_audits._run_model_worker(
        [_request()],
        config=config,
        output_dir=tmp_path,
        api_key="secret",
        transport=transport,
        sleep=sleep,
        max_new_requests=None,
        allow_ambiguous_resend=False,
    )
    assert result["completed"] == 1
    assert sent[0]["max_completion_tokens"] == 100
    assert sent[1]["max_completion_tokens"] == (
        100 + groq_audits.JSON_RETRY_EXTRA_COMPLETION_TOKENS
    )


def test_ambiguous_started_request_is_never_resent(tmp_path: Path) -> None:
    config = _config()
    journal = tmp_path / "ledgers" / "qwen__qwen3.6-27b.jsonl"
    groq_audits._append_event(
        journal,
        {
            "schema": groq_audits.LEDGER_SCHEMA,
            "event": "request_started",
            "timestamp": "2026-07-26T00:00:00+00:00",
            "request_id": "request-1",
        },
    )
    called = False

    def transport(*_args: Any) -> groq_audits.HttpResult:
        nonlocal called
        called = True
        return _success()

    with pytest.raises(RuntimeError, match="refusing automatic resend"):
        groq_audits._run_model_worker(
            [_request()],
            config=config,
            output_dir=tmp_path,
            api_key="secret",
            transport=transport,
            sleep=lambda _seconds: None,
            max_new_requests=None,
            allow_ambiguous_resend=False,
        )
    assert not called


def test_ambiguous_request_requires_explicit_operator_authorization(tmp_path: Path) -> None:
    config = _config()
    journal = tmp_path / "ledgers" / "qwen__qwen3.6-27b.jsonl"
    groq_audits._append_event(
        journal,
        {
            "schema": groq_audits.LEDGER_SCHEMA,
            "event": "request_started",
            "timestamp": "2026-07-26T00:00:00+00:00",
            "request_id": "request-1",
        },
    )
    result = groq_audits._run_model_worker(
        [_request()],
        config=config,
        output_dir=tmp_path,
        api_key="secret",
        transport=lambda *_args: _success(),
        sleep=lambda _seconds: None,
        max_new_requests=None,
        allow_ambiguous_resend=True,
    )
    assert result["completed"] == 1
    events = read_durable_jsonl_prefix(journal)
    assert [event["event"] for event in events] == [
        "request_started",
        "operator_authorized_ambiguous_resend",
        "request_started",
        "response_received",
    ]


def test_response_validation_rejects_missing_or_reordered_ids() -> None:
    request = _request()
    response = {
        "body": {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "ratings": [
                                    {
                                        "audit_id": "different",
                                        "gold_form": "full_question",
                                        "gold_intent": "fact_lookup",
                                        "intent_adequate": "yes",
                                        "ambiguous": "no",
                                        "encoding_error": "no",
                                    }
                                ]
                            }
                        )
                    }
                }
            ]
        }
    }
    with pytest.raises(ValueError, match="IDs/order"):
        groq_audits._validated_ratings(request, response)


def test_length_truncated_array_recovers_only_complete_valid_prefix() -> None:
    request = groq_audits.PlannedRequest(
        request_id="request-prefix",
        model_id="openai/gpt-oss-120b",
        reviewer_id="reviewer",
        audit_type="label",
        item_ids=("L-1", "L-2"),
        api_request={},
        estimated_tokens=100,
    )
    first = {
        "audit_id": "L-1",
        "gold_form": "keyword_query",
        "gold_intent": "definition",
        "intent_adequate": "yes",
        "ambiguous": "no",
        "encoding_error": "no",
        "comment": "",
    }
    response = {
        "body": {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": "[" + json.dumps(first) + ',{"audit_id":"L-2"'},
                }
            ]
        }
    }
    assert groq_audits._validated_truncated_prefix(request, response) == [
        groq_audits._validate_rating("label", first)
    ]


def test_complete_bare_array_is_accepted_only_with_exact_ids() -> None:
    request = _request()
    response = _success().body
    ratings = json.loads(response["choices"][0]["message"]["content"])["ratings"]
    event = {"body": response | {"choices": [{"message": {"content": json.dumps(ratings)}}]}}
    assert groq_audits._validated_ratings(request, event)[0]["audit_id"] == "L-1"


def test_singleton_enum_lists_are_normalized_without_changing_values() -> None:
    rating = {
        "audit_id": "L-1",
        "gold_form": ["full_question"],
        "gold_intent": ["definition"],
        "intent_adequate": ["yes"],
        "ambiguous": ["no"],
        "encoding_error": ["no"],
        "comment": "",
    }
    normalized = groq_audits._validate_rating("label", rating)
    assert normalized["gold_form"] == "full_question"
    assert normalized["gold_intent"] == "definition"
    assert normalized["intent_adequate"] == "yes"


def test_complete_bare_object_recovers_exact_first_item_only() -> None:
    request = groq_audits.PlannedRequest(
        request_id="request-prefix-object",
        model_id="openai/gpt-oss-120b",
        reviewer_id="reviewer",
        audit_type="label",
        item_ids=("L-1", "L-2"),
        api_request={},
        estimated_tokens=100,
    )
    content = {
        "audit_id": "L-1",
        "gold_form": "keyword_query",
        "gold_intent": "fact_lookup",
        "intent_adequate": "yes",
        "ambiguous": "no",
        "encoding_error": "no",
        "comment": "",
    }
    response = {
        "body": {
            "choices": [{"finish_reason": "length", "message": {"content": json.dumps(content)}}]
        }
    }
    assert [
        row["audit_id"] for row in groq_audits._validated_truncated_prefix(request, response)
    ] == ["L-1"]
