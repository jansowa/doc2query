"""Quota-safe, resumable Groq labelling for frozen Task 05 audit samples."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict, deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from doc2query.evaluation.natural_audits import FORM_VALUES, INTENT_VALUES
from doc2query.utils.records import read_durable_jsonl_prefix, write_json
from doc2query.utils.tracking import collect_code_provenance

CONTRACT = "task05-groq-llm-audit-v1"
LEDGER_SCHEMA = "task05-groq-request-ledger-v1"
RESULT_SCHEMA = "task05-groq-llm-ratings-v1"
YES_NO_UNCERTAIN = frozenset({"yes", "no", "uncertain"})
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class AuditItem:
    audit_type: str
    audit_id: str
    payload: dict[str, str]


@dataclass(frozen=True)
class PlannedRequest:
    request_id: str
    model_id: str
    reviewer_id: str
    audit_type: str
    item_ids: tuple[str, ...]
    api_request: dict[str, Any]
    estimated_tokens: int


@dataclass(frozen=True)
class HttpResult:
    status: int
    headers: dict[str, str]
    body: dict[str, Any]


Transport = Callable[[str, str, Mapping[str, Any], float], HttpResult]


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_groq_contract(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("contract") != CONTRACT:
        raise ValueError(f"unsupported Groq audit contract: {path}")
    if value.get("final_tests_used") != [] or value.get("d01_results_used") != []:
        raise ValueError("Groq audit must not use final tests or D01 results")
    models = value.get("models")
    if not isinstance(models, list) or {row.get("model_id") for row in models} != {
        "qwen/qwen3.6-27b",
        "openai/gpt-oss-120b",
    }:
        raise ValueError("contract must pin both approved Groq models")
    limits = cast(Mapping[str, Any], value["limits_per_model"])
    if float(limits["minimum_seconds_between_requests"]) < 2.0:
        raise ValueError("minimum request interval cannot be below two seconds")
    if int(limits["safety_requests_per_day"]) > int(limits["requests_per_day"]):
        raise ValueError("request safety budget exceeds the hard daily limit")
    if int(limits["safety_tokens_per_day"]) > int(limits["tokens_per_day"]):
        raise ValueError("token safety budget exceeds the hard daily limit")
    return value


def load_api_key(env_path: Path) -> str:
    """Read only the requested field without logging any secret material."""
    if not env_path.is_file():
        raise FileNotFoundError(env_path)
    values: dict[str, str] = {}
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    secret = values.get("api_key", "")
    if not secret:
        raise ValueError(f"{env_path} has no non-empty api_key field")
    return secret


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _carry_forward_rows(
    config: Mapping[str, Any],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    carry = config.get("carry_forward_ratings")
    if not isinstance(carry, Mapping):
        return [], []
    labels = _read_csv(Path(str(carry["label_csv"])))
    concepts = _read_csv(Path(str(carry["concept_csv"])))
    ids = [row["audit_id"] for row in labels + concepts]
    if len(ids) != len(set(ids)):
        raise ValueError("carry-forward ratings contain duplicate audit IDs")
    return labels, concepts


def _load_items(config: Mapping[str, Any]) -> list[AuditItem]:
    labels = _read_csv(Path(str(config["label_blind_csv"])))
    concepts = _read_csv(Path(str(config["concept_blind_csv"])))
    carried_labels, carried_concepts = _carry_forward_rows(config)
    carried_ids = {row["audit_id"] for row in carried_labels + carried_concepts}
    items = [
        AuditItem(
            "label",
            row["audit_id"],
            {
                "query": row["query"],
                "passage": row["positive_passage"],
            },
        )
        for row in labels
        if row["audit_id"] not in carried_ids
    ]
    items.extend(
        AuditItem(
            "concept",
            row["audit_id"],
            {
                "passage": row["passage"],
                "candidate_concepts": row["candidate_concepts"],
            },
        )
        for row in concepts
        if row["audit_id"] not in carried_ids
    )
    ids = [item.audit_id for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("audit inputs contain duplicate audit IDs")
    return sorted(items, key=lambda item: (item.audit_type, item.audit_id))


def _system_prompt(audit_type: str, config: Mapping[str, Any]) -> str:
    terse = bool(config.get("terse_output", False))
    common = (
        "Jesteś precyzyjnym polskim anotatorem danych wyszukiwawczych. "
        "Oceń każdy rekord niezależnie. Zwróć wyłącznie obiekt JSON z polem ratings. "
        "Nie dodawaj rekordów, nie pomijaj ID i nie wyjaśniaj toku rozumowania."
    )
    if terse:
        common += " Zapisz JSON bez wcięć i ustaw każde pole comment na pusty string."
    if audit_type == "label":
        return common + (
            " Dla każdego rekordu zwróć audit_id, gold_form z "
            "[full_question,keyword_query,unknown], gold_intent z "
            "[fact_lookup,definition,entity_lookup,procedure,comparison,unknown], "
            "intent_adequate, ambiguous i encoding_error z [yes,no,uncertain] oraz "
            + ("comment równy pustemu stringowi. " if terse else "krótki comment. ")
            + "Form dotyczy brzmienia query; intent jego celu; "
            "intent_adequate określa, czy passage wspiera tę intencję."
        )
    return common + (
        " Dla każdego rekordu oceń podane candidate_concepts względem passage. Zwróć "
        "audit_id, correct_concept_ids i spurious_concept_ids jako listy istniejących ID, "
        "missing_important_concepts jako krótki tekst lub pusty string; "
        "numbers_units_correct, over_fragmented, duplicate_concepts, useful_for_coverage, "
        "ambiguous i encoding_error z [yes,no,uncertain] oraz "
        + ("comment równy pustemu stringowi." if terse else "krótki comment.")
    )


def _user_payload(items: Sequence[AuditItem]) -> str:
    records = [{"audit_id": item.audit_id, **item.payload} for item in items]
    return "JSON do oceny:\n" + json.dumps(records, ensure_ascii=False, separators=(",", ":"))


def _estimate_tokens(payload: Mapping[str, Any], config: Mapping[str, Any]) -> int:
    estimator = cast(Mapping[str, Any], config["token_estimator"])
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return math.ceil(len(serialized) / float(estimator["characters_per_token"])) + int(
        estimator["fixed_request_tokens"]
    )


def _api_payload(
    items: Sequence[AuditItem], model: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    audit_type = items[0].audit_type
    per_item = int(config["max_completion_tokens_per_item"][audit_type])
    api = cast(Mapping[str, Any], config["api"])
    return {
        "model": model["model_id"],
        "messages": [
            {"role": "system", "content": _system_prompt(audit_type, config)},
            {"role": "user", "content": _user_payload(items)},
        ],
        "temperature": float(api["temperature"]),
        "max_completion_tokens": per_item * len(items)
        + 40
        + int(model.get("completion_token_overhead", 0)),
        "response_format": {"type": str(api["response_format"])},
        "reasoning_effort": model["reasoning_effort"],
        "include_reasoning": bool(api["include_reasoning"]),
    }


def build_request_plan(config: Mapping[str, Any]) -> list[PlannedRequest]:
    items = _load_items(config)
    batches: list[list[AuditItem]] = []
    for audit_type in ("label", "concept"):
        selected = [item for item in items if item.audit_type == audit_type]
        size = int(config["batch_sizes"][audit_type])
        batches.extend(selected[index : index + size] for index in range(0, len(selected), size))
    model_loads: dict[str, int] = {str(row["model_id"]): 0 for row in config["models"]}
    model_requests: CounterLike = defaultdict(int)
    models = {str(row["model_id"]): cast(Mapping[str, Any], row) for row in config["models"]}
    plan: list[PlannedRequest] = []
    for batch in batches:
        choices: list[tuple[int, int, str, dict[str, Any], int]] = []
        for model_id, model in models.items():
            payload = _api_payload(batch, model, config)
            estimate = _estimate_tokens(payload, config)
            if config.get("include_completion_budget_in_estimate", False):
                estimate += int(payload["max_completion_tokens"])
            choices.append(
                (
                    model_loads[model_id] + estimate,
                    model_requests[model_id],
                    model_id,
                    payload,
                    estimate,
                )
            )
        _, _, model_id, payload, estimate = min(choices, key=lambda row: (row[0], row[1], row[2]))
        model = models[model_id]
        identity = {
            "contract": config["contract"],
            "prompt_version": config["prompt_version"],
            "model_id": model_id,
            "audit_type": batch[0].audit_type,
            "item_ids": [item.audit_id for item in batch],
            "api_request": payload,
        }
        plan.append(
            PlannedRequest(
                request_id=_canonical_sha256(identity),
                model_id=model_id,
                reviewer_id=str(model["reviewer_id"]),
                audit_type=batch[0].audit_type,
                item_ids=tuple(item.audit_id for item in batch),
                api_request=payload,
                estimated_tokens=estimate,
            )
        )
        model_loads[model_id] += estimate
        model_requests[model_id] += 1
    limits = cast(Mapping[str, Any], config["limits_per_model"])
    prior_usage = cast(Mapping[str, Any], config.get("prior_daily_usage", {}))
    for model_id in models:
        if model_requests[model_id] > int(limits["safety_requests_per_day"]):
            raise ValueError(f"planned request count exceeds safety budget for {model_id}")
        prior_tokens = int(cast(Mapping[str, Any], prior_usage.get(model_id, {})).get("tokens", 0))
        if model_loads[model_id] + prior_tokens > int(limits["safety_tokens_per_day"]):
            raise ValueError(
                f"estimated token count {model_loads[model_id]} exceeds daily safety budget "
                f"for {model_id}"
            )
    return plan


CounterLike = defaultdict[str, int]


def plan_summary(plan: Sequence[PlannedRequest]) -> dict[str, Any]:
    by_model: defaultdict[str, dict[str, int]] = defaultdict(
        lambda: {"requests": 0, "items": 0, "estimated_tokens": 0}
    )
    by_type: defaultdict[str, int] = defaultdict(int)
    for request in plan:
        row = by_model[request.model_id]
        row["requests"] += 1
        row["items"] += len(request.item_ids)
        row["estimated_tokens"] += request.estimated_tokens
        by_type[request.audit_type] += len(request.item_ids)
    return {
        "request_count": len(plan),
        "item_count": sum(len(request.item_ids) for request in plan),
        "by_model": dict(sorted(by_model.items())),
        "by_type": dict(sorted(by_type.items())),
    }


def _http_transport(
    url: str, api_key: str, payload: Mapping[str, Any], timeout: float
) -> HttpResult:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "doc2query-task05-audit/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode())
            return HttpResult(
                int(response.status),
                {key.casefold(): value for key, value in response.headers.items()},
                cast(dict[str, Any], body),
            )
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode(errors="replace")
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = {"raw_error": raw[:4000]}
        return HttpResult(
            int(exc.code),
            {key.casefold(): value for key, value in exc.headers.items()},
            cast(dict[str, Any], body),
        )


def _append_event(path: Path, event: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class ModelLimiter:
    def __init__(self, config: Mapping[str, Any], *, sleep: Callable[[float], None]) -> None:
        limits = cast(Mapping[str, Any], config["limits_per_model"])
        self.minimum_interval = float(limits["minimum_seconds_between_requests"])
        self.minute_budget = int(limits["safety_tokens_per_minute"])
        self.sleep = sleep
        self.last_request_at: float | None = None
        self.window: deque[dict[str, float]] = deque()

    def reserve(self, estimated_tokens: int) -> dict[str, float]:
        while True:
            now = time.monotonic()
            while self.window and now - self.window[0]["time"] >= 60.0:
                self.window.popleft()
            interval_wait = (
                max(0.0, self.minimum_interval - (now - self.last_request_at))
                if self.last_request_at is not None
                else 0.0
            )
            token_wait = 0.0
            if sum(row["tokens"] for row in self.window) + estimated_tokens > self.minute_budget:
                token_wait = max(0.0, 60.0 - (now - self.window[0]["time"]) + 0.05)
            wait = max(interval_wait, token_wait)
            if wait <= 0:
                entry = {"time": now, "tokens": float(estimated_tokens)}
                self.window.append(entry)
                self.last_request_at = now
                return entry
            self.sleep(min(wait, 5.0))

    @staticmethod
    def finalize(entry: dict[str, float], actual_tokens: int) -> None:
        entry["tokens"] = float(max(actual_tokens, 1))


def _events_by_request(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in read_durable_jsonl_prefix(path):
        grouped[str(event["request_id"])].append(event)
    return dict(grouped)


def _completed_response(events: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    for event in reversed(events):
        if event.get("event") == "response_received":
            return event
    return None


def _has_ambiguous_tail(events: Sequence[Mapping[str, Any]]) -> bool:
    if not events:
        return False
    return events[-1].get("event") in {"request_started", "transport_error_ambiguous"}


def _response_content(body: Mapping[str, Any]) -> str:
    try:
        return str(body["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Groq response has no assistant content") from exc


def _parse_json_content(content: str) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        match = _JSON_OBJECT.search(content)
        if match is None:
            raise ValueError("model response is not JSON") from None
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("model response JSON must be an object")
    return value


def _validate_rating(audit_type: str, rating: Mapping[str, Any]) -> dict[str, Any]:
    audit_id = str(rating.get("audit_id", ""))
    if not audit_id:
        raise ValueError("rating has no audit_id")
    if audit_type == "label":
        if rating.get("gold_form") not in FORM_VALUES:
            raise ValueError(f"{audit_id}: invalid gold_form")
        if rating.get("gold_intent") not in INTENT_VALUES:
            raise ValueError(f"{audit_id}: invalid gold_intent")
        for field in ("intent_adequate", "ambiguous", "encoding_error"):
            if rating.get(field) not in YES_NO_UNCERTAIN:
                raise ValueError(f"{audit_id}: invalid {field}")
        return {
            "audit_id": audit_id,
            "gold_form": rating["gold_form"],
            "gold_intent": rating["gold_intent"],
            "intent_adequate": rating["intent_adequate"],
            "ambiguous": rating["ambiguous"],
            "encoding_error": rating["encoding_error"],
            "comment": str(rating.get("comment", ""))[:500],
        }
    for field in (
        "numbers_units_correct",
        "over_fragmented",
        "duplicate_concepts",
        "useful_for_coverage",
        "ambiguous",
        "encoding_error",
    ):
        if rating.get(field) not in YES_NO_UNCERTAIN:
            raise ValueError(f"{audit_id}: invalid {field}")
    correct = rating.get("correct_concept_ids")
    spurious = rating.get("spurious_concept_ids")
    if not isinstance(correct, list) or not isinstance(spurious, list):
        raise ValueError(f"{audit_id}: concept ID fields must be lists")
    return {
        "audit_id": audit_id,
        "correct_concept_ids": [str(item) for item in correct],
        "spurious_concept_ids": [str(item) for item in spurious],
        "missing_important_concepts": str(rating.get("missing_important_concepts", ""))[:1000],
        "numbers_units_correct": rating["numbers_units_correct"],
        "over_fragmented": rating["over_fragmented"],
        "duplicate_concepts": rating["duplicate_concepts"],
        "useful_for_coverage": rating["useful_for_coverage"],
        "ambiguous": rating["ambiguous"],
        "encoding_error": rating["encoding_error"],
        "comment": str(rating.get("comment", ""))[:500],
    }


def _validated_ratings(
    request: PlannedRequest, response: Mapping[str, Any]
) -> list[dict[str, Any]]:
    content = _response_content(cast(Mapping[str, Any], response["body"]))
    try:
        parsed: Any = json.loads(content)
    except json.JSONDecodeError:
        parsed = _parse_json_content(content)
    raw = parsed if isinstance(parsed, list) else parsed.get("ratings")
    if not isinstance(raw, list):
        raise ValueError("model JSON has no ratings list")
    ratings = [_validate_rating(request.audit_type, cast(Mapping[str, Any], row)) for row in raw]
    if [row["audit_id"] for row in ratings] != list(request.item_ids):
        raise ValueError("model response audit IDs/order differ from the request")
    return ratings


def _validated_truncated_prefix(
    request: PlannedRequest, response: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Recover only complete, valid leading rows from a length-truncated JSON array."""
    body = cast(Mapping[str, Any], response["body"])
    choices = cast(Sequence[Mapping[str, Any]], body.get("choices", []))
    if not choices or choices[0].get("finish_reason") != "length":
        return []
    content = _response_content(body).strip()
    if not content.startswith("["):
        return []
    decoder = json.JSONDecoder()
    position = 1
    recovered: list[dict[str, Any]] = []
    while position < len(content):
        while position < len(content) and content[position] in " \t\r\n,":
            position += 1
        if position >= len(content) or content[position] == "]":
            break
        try:
            raw, position = decoder.raw_decode(content, position)
            if not isinstance(raw, Mapping):
                break
            rating = _validate_rating(request.audit_type, raw)
        except (json.JSONDecodeError, ValueError):
            break
        expected = request.item_ids[len(recovered)]
        if rating["audit_id"] != expected:
            break
        recovered.append(rating)
    return recovered


def _ledger_usage(events: Sequence[Mapping[str, Any]], today: str) -> tuple[int, int]:
    attempts = sum(
        event.get("event") == "request_started"
        and str(event.get("timestamp", "")).startswith(today)
        for event in events
    )
    tokens = sum(
        int(cast(Mapping[str, Any], event.get("usage", {})).get("total_tokens", 0))
        for event in events
        if event.get("event") == "response_received"
        and str(event.get("timestamp", "")).startswith(today)
    )
    return attempts, tokens


def _resume_cooldown_seconds(events: Sequence[Mapping[str, Any]]) -> float:
    timestamps: list[datetime] = []
    for event in events:
        if event.get("event") != "request_started":
            continue
        try:
            timestamps.append(datetime.fromisoformat(str(event["timestamp"])))
        except (KeyError, ValueError):
            continue
    if not timestamps:
        return 0.0
    age = (datetime.now(UTC) - max(timestamps)).total_seconds()
    return max(0.0, 60.05 - age)


def _run_model_worker(
    requests: Sequence[PlannedRequest],
    *,
    config: Mapping[str, Any],
    output_dir: Path,
    api_key: str,
    transport: Transport,
    sleep: Callable[[float], None],
    max_new_requests: int | None,
    allow_ambiguous_resend: bool,
) -> dict[str, Any]:
    if not requests:
        return {"model_id": None, "completed": 0, "pending": 0}
    model_id = requests[0].model_id
    safe_name = model_id.replace("/", "__")
    journal = output_dir / "ledgers" / f"{safe_name}.jsonl"
    prior_events = read_durable_jsonl_prefix(journal)
    grouped = _events_by_request(journal)
    ambiguous = [
        request.request_id
        for request in requests
        if _has_ambiguous_tail(grouped.get(request.request_id, []))
    ]
    if ambiguous and not allow_ambiguous_resend:
        raise RuntimeError(
            f"{model_id} has {len(ambiguous)} ambiguous request(s); refusing automatic resend: "
            + ",".join(ambiguous[:3])
        )
    if ambiguous:
        for request_id in ambiguous:
            _append_event(
                journal,
                {
                    "schema": LEDGER_SCHEMA,
                    "event": "operator_authorized_ambiguous_resend",
                    "timestamp": _utc_now(),
                    "request_id": request_id,
                },
            )
    limits = cast(Mapping[str, Any], config["limits_per_model"])
    retry = cast(Mapping[str, Any], config["retry"])
    api = cast(Mapping[str, Any], config["api"])
    today = datetime.now(UTC).date().isoformat()
    attempts_today, tokens_today = _ledger_usage(prior_events, today)
    prior_usage = cast(Mapping[str, Any], config.get("prior_daily_usage", {}))
    model_prior = cast(Mapping[str, Any], prior_usage.get(model_id, {}))
    ledger_date = next(
        (
            str(event.get("timestamp", ""))[:10]
            for event in prior_events
            if event.get("event") == "request_started"
        ),
        "",
    )
    if model_prior.get("date", ledger_date) == today:
        attempts_today += int(model_prior.get("requests", 0))
        tokens_today += int(model_prior.get("tokens", 0))
    cooldown = _resume_cooldown_seconds(prior_events)
    if cooldown:
        sleep(cooldown)
    limiter = ModelLimiter(config, sleep=sleep)
    new_requests = 0
    completed = 0
    for position, request in enumerate(requests, 1):
        events = grouped.get(request.request_id, [])
        if _completed_response(events) is not None:
            completed += 1
            continue
        if max_new_requests is not None and new_requests >= max_new_requests:
            continue
        attempt = 0
        while True:
            attempt += 1
            if attempts_today + 1 > int(limits["safety_requests_per_day"]):
                raise RuntimeError(f"daily request safety budget exhausted for {model_id}")
            if tokens_today + request.estimated_tokens > int(limits["safety_tokens_per_day"]):
                raise RuntimeError(f"daily token safety budget exhausted for {model_id}")
            reservation = limiter.reserve(request.estimated_tokens)
            started = {
                "schema": LEDGER_SCHEMA,
                "event": "request_started",
                "timestamp": _utc_now(),
                "request_id": request.request_id,
                "attempt": attempt,
                "model_id": model_id,
                "audit_type": request.audit_type,
                "item_ids": list(request.item_ids),
                "estimated_tokens": request.estimated_tokens,
                "api_request": request.api_request,
            }
            _append_event(journal, started)
            attempts_today += 1
            new_requests += 1
            try:
                result = transport(
                    str(api["url"]),
                    api_key,
                    request.api_request,
                    float(api["timeout_seconds"]),
                )
            except Exception as exc:
                _append_event(
                    journal,
                    {
                        "schema": LEDGER_SCHEMA,
                        "event": "transport_error_ambiguous",
                        "timestamp": _utc_now(),
                        "request_id": request.request_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:1000],
                    },
                )
                raise RuntimeError(
                    f"ambiguous transport failure for {request.request_id}; not retrying"
                ) from exc
            if result.status == 429:
                _append_event(
                    journal,
                    {
                        "schema": LEDGER_SCHEMA,
                        "event": "rate_limited",
                        "timestamp": _utc_now(),
                        "request_id": request.request_id,
                        "attempt": attempt,
                        "status": result.status,
                        "headers": result.headers,
                        "body": result.body,
                    },
                )
                ModelLimiter.finalize(reservation, 1)
                if attempt > int(retry["max_rate_limit_retries"]):
                    raise RuntimeError(f"rate-limit retries exhausted for {request.request_id}")
                retry_after = min(
                    float(result.headers.get("retry-after", retry["default_retry_after_seconds"])),
                    float(retry["maximum_retry_after_seconds"]),
                )
                sleep(max(2.0, retry_after))
                continue
            if result.status != 200:
                ModelLimiter.finalize(reservation, 1)
                _append_event(
                    journal,
                    {
                        "schema": LEDGER_SCHEMA,
                        "event": "http_error",
                        "timestamp": _utc_now(),
                        "request_id": request.request_id,
                        "status": result.status,
                        "headers": result.headers,
                        "body": result.body,
                    },
                )
                raise RuntimeError(f"Groq HTTP {result.status} for {request.request_id}")
            usage = cast(Mapping[str, Any], result.body.get("usage", {}))
            actual_tokens = int(usage.get("total_tokens", request.estimated_tokens))
            ModelLimiter.finalize(reservation, actual_tokens)
            tokens_today += actual_tokens
            event = {
                "schema": LEDGER_SCHEMA,
                "event": "response_received",
                "timestamp": _utc_now(),
                "request_id": request.request_id,
                "attempt": attempt,
                "status": result.status,
                "model_id": model_id,
                "headers": result.headers,
                "usage": dict(usage),
                "body": result.body,
            }
            _append_event(journal, event)
            grouped.setdefault(request.request_id, []).extend([started, event])
            completed += 1
            print(
                f"[{model_id}] request={position}/{len(requests)} items={len(request.item_ids)} "
                f"tokens={actual_tokens} daily_tokens={tokens_today} "
                f"daily_requests={attempts_today}",
                flush=True,
            )
            break
    return {
        "model_id": model_id,
        "completed": completed,
        "planned": len(requests),
        "new_attempts": new_requests,
        "tokens_today": tokens_today,
        "requests_today": attempts_today,
    }


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _materialize_ratings(
    plan: Sequence[PlannedRequest], output_dir: Path, config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    carried_labels, carried_concepts = _carry_forward_rows(config)
    labels: list[dict[str, Any]] = [dict(row) for row in carried_labels]
    concepts: list[dict[str, Any]] = [
        dict(row)
        | {
            "correct_concept_ids": row["correct_concept_ids"].split("|")
            if row["correct_concept_ids"]
            else [],
            "spurious_concept_ids": row["spurious_concept_ids"].split("|")
            if row["spurious_concept_ids"]
            else [],
        }
        for row in carried_concepts
    ]
    invalid: list[str] = []
    for request in plan:
        journal = output_dir / "ledgers" / f"{request.model_id.replace('/', '__')}.jsonl"
        response = _completed_response(_events_by_request(journal).get(request.request_id, []))
        if response is None:
            invalid.extend(request.item_ids)
            continue
        try:
            ratings = _validated_ratings(request, response)
        except ValueError:
            ratings = _validated_truncated_prefix(request, response)
            invalid.extend(request.item_ids[len(ratings) :])
        for rating in ratings:
            row = dict(rating) | {
                "reviewer_id": request.reviewer_id,
                "model_id": request.model_id,
                "request_id": request.request_id,
            }
            (labels if request.audit_type == "label" else concepts).append(row)
    label_fields = [
        "audit_id",
        "reviewer_id",
        "gold_form",
        "gold_intent",
        "intent_adequate",
        "ambiguous",
        "encoding_error",
        "comment",
        "model_id",
        "request_id",
    ]
    concept_fields = [
        "audit_id",
        "reviewer_id",
        "correct_concept_ids",
        "spurious_concept_ids",
        "missing_important_concepts",
        "numbers_units_correct",
        "over_fragmented",
        "duplicate_concepts",
        "useful_for_coverage",
        "ambiguous",
        "encoding_error",
        "comment",
        "model_id",
        "request_id",
    ]
    serialized_concepts = [
        row
        | {
            "correct_concept_ids": "|".join(row["correct_concept_ids"]),
            "spurious_concept_ids": "|".join(row["spurious_concept_ids"]),
        }
        for row in concepts
    ]
    _write_csv(output_dir / "label_llm_ratings.csv", label_fields, labels)
    _write_csv(output_dir / "concept_llm_ratings.csv", concept_fields, serialized_concepts)
    return labels, concepts, invalid


def _identity(
    config_path: Path, config: Mapping[str, Any], plan: Sequence[PlannedRequest]
) -> dict[str, Any]:
    source_manifest = Path(str(config["source_materialization_manifest"]))
    base = {
        "schema": RESULT_SCHEMA,
        "contract_sha256": _file_sha256(config_path),
        "resolved_contract_sha256": _canonical_sha256(config),
        "source_materialization_manifest_sha256": _file_sha256(source_manifest),
        "label_blind_sha256": _file_sha256(Path(str(config["label_blind_csv"]))),
        "concept_blind_sha256": _file_sha256(Path(str(config["concept_blind_csv"]))),
        "request_plan_sha256": _canonical_sha256(
            [
                {
                    "request_id": row.request_id,
                    "model_id": row.model_id,
                    "item_ids": row.item_ids,
                    "estimated_tokens": row.estimated_tokens,
                }
                for row in plan
            ]
        ),
        "final_tests_used": [],
        "d01_results_used": [],
    }
    carry = config.get("carry_forward_ratings")
    if isinstance(carry, Mapping):
        base["carry_forward_label_sha256"] = _file_sha256(Path(str(carry["label_csv"])))
        base["carry_forward_concept_sha256"] = _file_sha256(Path(str(carry["concept_csv"])))
    return base | {"identity_sha256": _canonical_sha256(base)}


def run_groq_audit(
    config_path: Path,
    *,
    output_dir: Path,
    api_key: str,
    transport: Transport = _http_transport,
    sleep: Callable[[float], None] = time.sleep,
    max_new_requests_per_model: int | None = None,
    allow_ambiguous_resend: bool = False,
) -> dict[str, Any]:
    config = load_groq_contract(config_path)
    plan = build_request_plan(config)
    identity = _identity(config_path, config, plan)
    output_dir.mkdir(parents=True, exist_ok=True)
    identity_path = output_dir / "identity.json"
    if identity_path.exists():
        previous = json.loads(identity_path.read_text(encoding="utf-8"))
        if previous != identity:
            raise ValueError("Groq audit resume identity mismatch")
    else:
        if (output_dir / "ledgers").exists():
            raise ValueError("Groq ledgers exist without identity")
        write_json(identity_path, identity)
        write_json(
            output_dir / "request_plan.json",
            {
                "identity": identity,
                "summary": plan_summary(plan),
                "requests": [
                    {
                        "request_id": row.request_id,
                        "model_id": row.model_id,
                        "reviewer_id": row.reviewer_id,
                        "audit_type": row.audit_type,
                        "item_ids": list(row.item_ids),
                        "estimated_tokens": row.estimated_tokens,
                    }
                    for row in plan
                ],
            },
        )
    by_model: defaultdict[str, list[PlannedRequest]] = defaultdict(list)
    for request in plan:
        by_model[request.model_id].append(request)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                _run_model_worker,
                requests,
                config=config,
                output_dir=output_dir,
                api_key=api_key,
                transport=transport,
                sleep=sleep,
                max_new_requests=max_new_requests_per_model,
                allow_ambiguous_resend=allow_ambiguous_resend,
            )
            for _, requests in sorted(by_model.items())
        ]
        workers = [future.result() for future in futures]
    labels, concepts, invalid = _materialize_ratings(plan, output_dir, config)
    complete = len(labels) == 500 and len(concepts) == 200 and not invalid
    result = {
        "schema": RESULT_SCHEMA,
        "status": "complete" if complete else "incomplete",
        "identity": identity,
        "plan": plan_summary(plan),
        "workers": workers,
        "ratings": {"label": len(labels), "concept": len(concepts), "invalid_or_missing": invalid},
        "provenance": collect_code_provenance(),
        "final_tests_used": [],
        "d01_results_used": [],
    }
    temporary = output_dir / "summary.json.tmp"
    write_json(temporary, result)
    os.replace(temporary, output_dir / "summary.json")
    return result
