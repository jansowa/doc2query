"""Prospective blind dual-LLM audit planning for Task 06 preference pairs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from doc2query.utils.records import read_records

CONTRACT = "task06-groq-dual-llm-preference-audit-v1"
APPROVED_MODELS = frozenset({"openai/gpt-oss-120b", "qwen/qwen3.6-27b"})
# Liczebności próby audytowej zamrożone prospektywnie przez ADR-y właściciela:
#   500 — rozwojowa bramka dual-LLM pilota (waiver 2026-08-12, audyty v1 i v2);
#   800 — komórka bramkowa polityki v2.1, wyprowadzona z rachunku mocy
#         (reports/decisions/task06_defect_pair_policy_v2_1.md §4-5), plus
#         amendment task06_groq_audit_sample_size_amendment_2026-08-23.md.
# Bramka pozostaje fail-closed: dowolna inna liczba jest odrzucana, żeby nikt nie
# zmienił liczebności audytu bez prospektywnej decyzji. Prompt, rubryka, modele i
# limity są tą zmianą nietknięte.
APPROVED_PAIR_COUNTS = frozenset({500, 800})
FORBIDDEN_BLIND_FIELDS = frozenset(
    {
        "chosen",
        "rejected",
        "candidate_a_id",
        "candidate_b_id",
        "generator_a",
        "generator_b",
        "primary_score",
        "shadow_score",
        "total_score",
        "automatic_preference",
    }
)


@dataclass(frozen=True)
class LlmAuditRequest:
    request_id: str
    model_id: str
    reviewer_id: str
    item_ids: tuple[str, ...]
    api_request: dict[str, Any]


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def load_llm_audit_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("contract") != CONTRACT:
        raise ValueError("unsupported Task 06 LLM audit contract")
    if value.get("final_tests_used") != [] or value.get("human_evidence_claimed") is not False:
        raise ValueError("LLM audit cannot use final tests or claim human evidence")
    models = value.get("models")
    if not isinstance(models, list) or {row.get("model_id") for row in models} != APPROVED_MODELS:
        raise ValueError("dual audit must pin both owner-approved Groq models")
    if int(value.get("pair_count", 0)) not in APPROVED_PAIR_COUNTS:
        raise ValueError(
            "audit sample size must be one frozen by an owner-approved ADR "
            f"({sorted(APPROVED_PAIR_COUNTS)}), not an arbitrary number"
        )
    if value.get("assignment") != "every_pair_reviewed_by_both_models":
        raise ValueError("each pair must receive two independent ratings")
    if value.get("disagreement_policy") != "exclude_from_automatic_acceptance":
        raise ValueError("LLM disagreements must fail closed")
    api = value.get("api")
    if not isinstance(api, dict) or float(api.get("temperature", -1)) != 0.0:
        raise ValueError("blind LLM audit must use deterministic temperature zero")
    limits = value.get("limits_per_model")
    if not isinstance(limits, dict):
        raise ValueError("missing Groq limits")
    if float(limits.get("minimum_seconds_between_requests", 0)) < 4.0:
        raise ValueError("Task 06 Groq requests require at least four seconds spacing")
    if int(limits.get("safety_requests_per_minute", 0)) > 14:
        raise ValueError("four-second spacing allows at most 14 safety requests per minute")
    for safety, hard in (
        ("safety_tokens_per_minute", "tokens_per_minute"),
        ("safety_requests_per_day", "requests_per_day"),
        ("safety_tokens_per_day", "tokens_per_day"),
    ):
        if int(limits.get(safety, 0)) > int(limits.get(hard, 0)):
            raise ValueError(f"{safety} exceeds {hard}")
    scheduler = value.get("quota_scheduler")
    if not isinstance(scheduler, dict) or scheduler != {
        "global_request_serialization": True,
        "on_model_quota_exhausted": "defer_model_and_switch_to_other",
        "when_both_models_deferred": "stop_cleanly_incomplete_quota_deferred",
        "resume_on_next_operator_run": True,
        "automatic_next_day_wakeup": False,
        "persist_not_before_timestamp": True,
        "respect_retry_after_header": True,
    }:
        raise ValueError("Groq quota scheduler must switch models and stop resumably")
    return value


def _load_blind_rows(path: Path, expected_count: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for raw in read_records(path):
        forbidden = FORBIDDEN_BLIND_FIELDS & set(raw)
        if forbidden:
            raise ValueError(f"blind input leaks selection fields: {sorted(forbidden)}")
        required = {"audit_id", "passage", "query_a", "query_b", "orientation_commitment"}
        if set(raw) != required:
            raise ValueError("blind input has an unexpected schema")
        row = {field: str(raw[field]).strip() for field in required}
        if not all(row.values()) or row["query_a"] == row["query_b"]:
            raise ValueError("blind pair must contain two distinct non-empty queries")
        rows.append(row)
    if len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} blind pairs, got {len(rows)}")
    ids = [row["audit_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("blind audit IDs must be unique")
    return sorted(rows, key=lambda row: row["audit_id"])


def _system_prompt() -> str:
    return (
        "Jesteś niezależnym polskim audytorem par zapytań wyszukiwawczych. "
        "Nie wiesz, który model wygenerował A lub B ani co wybrał automat. "
        "Dla każdego rekordu oceń answerability, naturalność, użyteczność "
        "wyszukiwawczą, nadmierne kopiowanie i zdradzanie odpowiedzi. Wybierz "
        "preference z [A,B,tie,both_bad,uncertain] i reason_code z "
        "[grounding,naturalness,retrieval_usefulness,copying,answer_leakage,mixed,uncertain]. "
        "Zwróć wyłącznie JSON z polem ratings; każdy rating ma audit_id, preference, "
        "reason_code, answerable_a, answerable_b, format_valid_a, format_valid_b "
        "jako boolean oraz confidence w [0,1]. Bez toku rozumowania i komentarzy."
    )


def _api_request(
    rows: Sequence[Mapping[str, str]], model: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    api = cast(Mapping[str, Any], config["api"])
    records = [
        {
            "audit_id": row["audit_id"],
            "passage": row["passage"],
            "query_a": row["query_a"],
            "query_b": row["query_b"],
        }
        for row in rows
    ]
    return {
        "model": model["model_id"],
        "messages": [
            {"role": "system", "content": _system_prompt()},
            {
                "role": "user",
                "content": "JSON do ślepej oceny:\n"
                + json.dumps(records, ensure_ascii=False, separators=(",", ":")),
            },
        ],
        "temperature": 0.0,
        "max_completion_tokens": int(api["max_completion_tokens_per_pair"]) * len(rows)
        + int(model.get("completion_token_overhead", 0)),
        "response_format": {"type": "json_object"},
        "reasoning_effort": model["reasoning_effort"],
        "include_reasoning": False,
    }


def build_dual_llm_request_plan(
    config: Mapping[str, Any], blind_path: Path
) -> list[LlmAuditRequest]:
    """Create two complete, independent and deterministic review assignments."""
    rows = _load_blind_rows(blind_path, int(config["pair_count"]))
    size = int(config["batch_size"])
    if size < 1:
        raise ValueError("batch_size must be positive")
    requests: list[LlmAuditRequest] = []
    for model_raw in config["models"]:
        model = cast(Mapping[str, Any], model_raw)
        for start in range(0, len(rows), size):
            batch = rows[start : start + size]
            payload = _api_request(batch, model, config)
            identity = {
                "contract": CONTRACT,
                "prompt_version": config["prompt_version"],
                "model_id": model["model_id"],
                "item_ids": [row["audit_id"] for row in batch],
                "api_request": payload,
            }
            requests.append(
                LlmAuditRequest(
                    request_id=_canonical_sha256(identity),
                    model_id=str(model["model_id"]),
                    reviewer_id=str(model["reviewer_id"]),
                    item_ids=tuple(row["audit_id"] for row in batch),
                    api_request=payload,
                )
            )
    return requests


def plan_summary(plan: Sequence[LlmAuditRequest]) -> dict[str, Any]:
    models: dict[str, dict[str, int]] = {}
    for request in plan:
        row = models.setdefault(request.model_id, {"requests": 0, "ratings": 0})
        row["requests"] += 1
        row["ratings"] += len(request.item_ids)
    return {
        "request_count": len(plan),
        "rating_count": sum(len(request.item_ids) for request in plan),
        "pair_count": len({item for request in plan for item in request.item_ids}),
        "by_model": dict(sorted(models.items())),
        "human_evidence_claimed": False,
        "final_tests_used": [],
    }
