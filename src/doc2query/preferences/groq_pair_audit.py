"""Globally serialized, resumable dual-LLM audit of frozen Task 06 preference pairs.

The frozen contract (``configs/preferences/task06_groq_preference_audit_v1.json``) demands
properties the Task 05 runner deliberately does not have, so this runner is separate:

* **global request serialization** — one request in flight at a time across *both* models,
  with at least ``minimum_seconds_between_requests`` between any two requests;
* **per-model durable ledger** with resume, and a refusal to resend a request whose
  response was never recorded (an ambiguous tail needs an explicit operator flag);
* **quota deferral** — an exhausted model is deferred and the other one continues; when both
  are deferred the run stops cleanly as ``incomplete_quota_deferred``;
* **disagreement fails closed** — a pair the two models rank differently is excluded from
  automatic acceptance rather than resolved by a tie-break.

The analysis half never sees a judge score or a selection field: it joins blind ratings with
the separately stored unblinding key.  Nothing here is human evidence, and nothing here
authorizes DPO training.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from collections import Counter, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias, cast

from doc2query.evaluation.groq_audits import HttpResult, _http_transport, load_api_key
from doc2query.evaluation.retrieval import percentile
from doc2query.preferences.llm_audit import (
    CONTRACT,
    LlmAuditRequest,
    build_dual_llm_request_plan,
    load_llm_audit_config,
    plan_summary,
)
from doc2query.preferences.pair_audit_export import (
    BlindExportManifest,
    load_blind_export_manifest,
)
from doc2query.utils.records import JsonlWriter, read_durable_jsonl_prefix, read_records, write_json

if TYPE_CHECKING:  # pragma: no cover - tylko dla typów
    # Import wyłącznie typowy: `pair_audit_export_v2` prowadzi przez `pair_policy_v2` do
    # `answerability_judge`, który importuje helpery z tego modułu, więc import
    # modułowy zamknąłby cykl. Runtime'owy import siedzi w `load_export_manifest`.
    from doc2query.preferences.pair_audit_export_v2 import DefectBlindExportManifest
    from doc2query.preferences.pair_audit_export_v2_1 import DefectBlindExportManifestV21

ExportManifest: TypeAlias = (
    "BlindExportManifest | DefectBlindExportManifest | DefectBlindExportManifestV21"
)

LEDGER_SCHEMA = "task06-groq-pair-audit-ledger-v1"
RESULT_SCHEMA = "task06-groq-pair-audit-result-v1"
ANALYSIS_SCHEMA = "task06-groq-pair-audit-analysis-v1"
PREFERENCES = frozenset({"A", "B", "tie", "both_bad", "uncertain"})
DECIDED_PREFERENCES = frozenset({"A", "B"})
REASON_CODES = frozenset(
    {
        "grounding",
        "naturalness",
        "retrieval_usefulness",
        "copying",
        "answer_leakage",
        "mixed",
        "uncertain",
    }
)
BOOLEAN_FIELDS = ("answerable_a", "answerable_b", "format_valid_a", "format_valid_b")
# Zachowawcze oszacowanie promptu w tokenach: mniej znaków na token oznacza
# wyższy szacunek, czyli większy zapas względem limitów minutowych i dziennych.
CHARACTERS_PER_TOKEN = 3.0
MAX_TRANSIENT_RETRIES = 8
# Liczba wiodących znaków `audit_id`, po których wolno rozpoznać zniekształcone
# ID (patrz `_resolve_audit_id`). Osiem znaków heksadecymalnych to 32 bity, a
# dopasowanie jest ograniczone do jednego requestu; jednoznaczność i pełne
# pokrycie requestu są sprawdzane osobno.
AUDIT_ID_MATCH_PREFIX = 8
OUT_OF_SCHEMA_REASON = "out_of_schema"

Transport = Callable[[str, str, Mapping[str, Any], float], HttpResult]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _append_event(path: Path, event: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _estimate_tokens(payload: Mapping[str, Any]) -> int:
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    prompt = math.ceil(len(serialized) / CHARACTERS_PER_TOKEN)
    return prompt + int(payload.get("max_completion_tokens", 0))


class GlobalSerializer:
    """One request at a time across both models, with a global minimum spacing.

    The clock is injected so the spacing contract can be tested without spending the
    wall-clock time the frozen limits demand.
    """

    def __init__(
        self,
        minimum_interval: float,
        *,
        sleep: Callable[[float], None],
        monotonic: Callable[[], float],
    ) -> None:
        self.minimum_interval = minimum_interval
        self.sleep = sleep
        self.monotonic = monotonic
        self.last_request_at: float | None = None

    def wait(self) -> None:
        if self.last_request_at is None:
            self.last_request_at = self.monotonic()
            return
        while True:
            waited = self.monotonic() - self.last_request_at
            if waited >= self.minimum_interval:
                self.last_request_at = self.monotonic()
                return
            self.sleep(self.minimum_interval - waited)


class ModelBudget:
    """Per-model minute and day budgets; exhaustion defers the model, never crashes."""

    def __init__(
        self, model_id: str, limits: Mapping[str, Any], *, monotonic: Callable[[], float]
    ) -> None:
        self.model_id = model_id
        self.monotonic = monotonic
        self.requests_per_minute = int(limits["safety_requests_per_minute"])
        self.tokens_per_minute = int(limits["safety_tokens_per_minute"])
        self.requests_per_day = int(limits["safety_requests_per_day"])
        self.tokens_per_day = int(limits["safety_tokens_per_day"])
        self.requests_today = 0
        self.tokens_today = 0
        self.window: deque[tuple[float, int]] = deque()
        self.deferred_reason: str | None = None

    def replay(self, requests: int, tokens: int) -> None:
        self.requests_today += requests
        self.tokens_today += tokens

    def _trim(self) -> None:
        now = self.monotonic()
        while self.window and now - self.window[0][0] >= 60.0:
            self.window.popleft()

    def daily_room(self, estimated_tokens: int) -> str | None:
        if self.requests_today + 1 > self.requests_per_day:
            return "daily_request_budget_exhausted"
        if self.tokens_today + estimated_tokens > self.tokens_per_day:
            return "daily_token_budget_exhausted"
        return None

    def minute_wait(self, estimated_tokens: int) -> float:
        self._trim()
        if not self.window:
            return 0.0
        requests = len(self.window)
        tokens = sum(value for _, value in self.window)
        if requests + 1 <= self.requests_per_minute and (
            tokens + estimated_tokens <= self.tokens_per_minute
        ):
            return 0.0
        return max(0.0, 60.0 - (self.monotonic() - self.window[0][0]) + 0.05)

    def charge(self, estimated_tokens: int, actual_tokens: int | None) -> None:
        spent = max(actual_tokens or estimated_tokens, 1)
        self.window.append((self.monotonic(), spent))
        self.requests_today += 1
        self.tokens_today += spent


def _events_by_request(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for event in read_durable_jsonl_prefix(path):
        grouped.setdefault(str(event["request_id"]), []).append(event)
    return grouped


def _completed(events: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    for event in reversed(events):
        if event.get("event") == "response_received":
            return event
    return None


def _has_ambiguous_tail(events: Sequence[Mapping[str, Any]]) -> bool:
    if not events:
        return False
    return events[-1].get("event") in {"request_started", "transport_error_ambiguous"}


def _ledger_usage(events: Sequence[Mapping[str, Any]], today: str) -> tuple[int, int]:
    requests = 0
    tokens = 0
    for event in events:
        if event.get("event") != "request_started":
            continue
        if str(event.get("timestamp", ""))[:10] != today:
            continue
        requests += 1
        tokens += int(event.get("estimated_tokens", 0))
    return requests, tokens


def _response_content(body: Mapping[str, Any]) -> str:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Groq response carries no choice")
    message = cast(Mapping[str, Any], cast(Mapping[str, Any], choices[0])["message"])
    return str(message.get("content") or "")


def _resolve_audit_id(raw_id: str, allowed_ids: frozenset[str]) -> tuple[str, bool]:
    """Resolve a mangled ID only when its leading characters match exactly one expected ID.

    Both judges corrupt the 24-hex `audit_id` deterministically at temperature zero, so a
    strict match leaves single pairs permanently unratable and quietly biases coverage:
    `qwen/qwen3.6-27b` drops characters from the **middle** (`241b038fc3b311a023a2cf1e` →
    `241b038fc3b023a2cf1e`), which no whole-string prefix rule can recover.

    Matching therefore uses the first `AUDIT_ID_MATCH_PREFIX` hex characters and demands a
    unique hit inside the request, which holds at most `batch_size` items.  This cannot
    merge two pairs: the caller separately requires the resolved IDs to cover the request
    exactly, so the mapping stays injective.  Every repair is recorded on the rating and
    counted in the analysis, so the blinding remains verifiable after the fact.
    """
    if raw_id in allowed_ids:
        return raw_id, False
    if len(raw_id) >= AUDIT_ID_MATCH_PREFIX:
        head = raw_id[:AUDIT_ID_MATCH_PREFIX]
        matches = sorted(
            value for value in allowed_ids if value[:AUDIT_ID_MATCH_PREFIX] == head
        )
        if len(matches) == 1:
            return matches[0], True
    raise ValueError(f"rating references an audit_id outside its request: {raw_id!r}")


def _validate_rating(raw: Mapping[str, Any], allowed_ids: frozenset[str]) -> dict[str, Any]:
    audit_id, repaired = _resolve_audit_id(str(raw.get("audit_id", "")), allowed_ids)
    preference = str(raw.get("preference", ""))
    if preference not in PREFERENCES:
        raise ValueError(f"invalid preference for {audit_id}: {preference!r}")
    # `reason_code` jest metadaną diagnostyczną, a nie pomiarem: nie wchodzi do
    # żadnej zgodności ani do konsensusu. Odrzucanie całego requestu za kod poza
    # listą kasowałoby właśnie te pary, które sędzia opisuje inaczej, czyli
    # wprowadzałoby obciążenie pokrycia. Kod poza schematem jest więc zachowywany
    # dosłownie i policzony, a surowość pozostaje na `preference`.
    raw_reason = str(raw.get("reason_code", ""))
    if not raw_reason:
        raise ValueError(f"missing reason_code for {audit_id}")
    in_schema = raw_reason in REASON_CODES
    rating: dict[str, Any] = {
        "audit_id": audit_id,
        "audit_id_repaired": repaired,
        "preference": preference,
        "reason_code": raw_reason if in_schema else OUT_OF_SCHEMA_REASON,
        "reason_code_raw": raw_reason,
    }
    for field in BOOLEAN_FIELDS:
        value = raw.get(field)
        if not isinstance(value, bool):
            raise ValueError(f"{field} must be boolean for {audit_id}")
        rating[field] = value
    confidence = raw.get("confidence")
    if not isinstance(confidence, int | float) or not 0.0 <= float(confidence) <= 1.0:
        raise ValueError(f"confidence must lie in [0,1] for {audit_id}")
    rating["confidence"] = float(confidence)
    return rating


def _parse_ratings(content: str, item_ids: Sequence[str]) -> list[dict[str, Any]]:
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise ValueError("Groq response is not a JSON object")
    ratings = payload.get("ratings")
    if not isinstance(ratings, list):
        raise ValueError("Groq response has no ratings list")
    allowed = frozenset(item_ids)
    parsed = [_validate_rating(cast(Mapping[str, Any], row), allowed) for row in ratings]
    seen = {row["audit_id"] for row in parsed}
    if seen != allowed:
        raise ValueError(f"ratings cover {sorted(seen)} instead of {sorted(allowed)}")
    return sorted(parsed, key=lambda row: str(row["audit_id"]))


def _run_requests(
    plan: Sequence[LlmAuditRequest],
    *,
    config: Mapping[str, Any],
    output_dir: Path,
    api_key: str,
    transport: Transport,
    sleep: Callable[[float], None],
    monotonic: Callable[[], float],
    max_new_requests_per_model: int | None,
    allow_ambiguous_resend: bool,
) -> dict[str, Any]:
    api = cast(Mapping[str, Any], config["api"])
    limits = cast(Mapping[str, Any], config["limits_per_model"])
    retry = cast(Mapping[str, Any], config["retry"])
    serializer = GlobalSerializer(
        float(limits["minimum_seconds_between_requests"]), sleep=sleep, monotonic=monotonic
    )
    today = datetime.now(UTC).date().isoformat()
    by_model: dict[str, list[LlmAuditRequest]] = {}
    for request in plan:
        by_model.setdefault(request.model_id, []).append(request)

    budgets: dict[str, ModelBudget] = {}
    ledgers: dict[str, Path] = {}
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    new_requests: Counter[str] = Counter()
    completed: Counter[str] = Counter()
    for model_id, requests in by_model.items():
        ledger = output_dir / "ledgers" / f"{model_id.replace('/', '__')}.jsonl"
        ledgers[model_id] = ledger
        events = read_durable_jsonl_prefix(ledger)
        grouped[model_id] = _events_by_request(ledger)
        budget = ModelBudget(model_id, limits, monotonic=monotonic)
        budget.replay(*_ledger_usage(events, today))
        budgets[model_id] = budget
        ambiguous = [
            request.request_id
            for request in requests
            if _has_ambiguous_tail(grouped[model_id].get(request.request_id, []))
        ]
        if ambiguous and not allow_ambiguous_resend:
            raise RuntimeError(
                f"{model_id} has {len(ambiguous)} request(s) without a recorded response; "
                "refusing automatic resend: " + ",".join(ambiguous[:3])
            )
        for request_id in ambiguous:
            _append_event(
                ledger,
                {
                    "schema": LEDGER_SCHEMA,
                    "event": "operator_authorized_ambiguous_resend",
                    "timestamp": _utc_now(),
                    "request_id": request_id,
                },
            )

    # Global serialization means one interleaved queue, not one worker per model.
    pending = {model_id: list(requests) for model_id, requests in by_model.items()}
    for model_id, requests in pending.items():
        for request in list(requests):
            if _completed(grouped[model_id].get(request.request_id, [])) is not None:
                completed[model_id] += 1
                requests.remove(request)

    order = sorted(pending)
    while any(
        pending[model_id] and budgets[model_id].deferred_reason is None for model_id in order
    ):
        for model_id in order:
            queue = pending[model_id]
            budget = budgets[model_id]
            if not queue or budget.deferred_reason is not None:
                continue
            if (
                max_new_requests_per_model is not None
                and new_requests[model_id] >= max_new_requests_per_model
            ):
                budget.deferred_reason = "operator_request_cap_reached"
                continue
            request = queue[0]
            estimated = _estimate_tokens(request.api_request)
            exhausted = budget.daily_room(estimated)
            if exhausted is not None:
                budget.deferred_reason = exhausted
                continue
            wait = budget.minute_wait(estimated)
            if wait > 0:
                sleep(wait)
                continue
            outcome = _perform_request(
                request,
                ledger=ledgers[model_id],
                events=grouped[model_id].get(request.request_id, []),
                api=api,
                retry=retry,
                api_key=api_key,
                transport=transport,
                sleep=sleep,
                serializer=serializer,
                budget=budget,
                estimated_tokens=estimated,
            )
            new_requests[model_id] += outcome["attempts"]
            if outcome["status"] == "completed":
                completed[model_id] += 1
                queue.pop(0)
            elif outcome["status"] == "rate_limited":
                budget.deferred_reason = "rate_limit_retries_exhausted"
            else:
                budget.deferred_reason = outcome["status"]
    return {
        "by_model": {
            model_id: {
                "completed": completed[model_id],
                "pending": len(pending[model_id]),
                "new_requests": new_requests[model_id],
                "deferred_reason": budgets[model_id].deferred_reason,
                "requests_today": budgets[model_id].requests_today,
                "tokens_today": budgets[model_id].tokens_today,
            }
            for model_id in order
        },
        "deferred_models": sorted(
            model_id for model_id in order if budgets[model_id].deferred_reason is not None
        ),
    }


def _perform_request(
    request: LlmAuditRequest,
    *,
    ledger: Path,
    events: Sequence[Mapping[str, Any]],
    api: Mapping[str, Any],
    retry: Mapping[str, Any],
    api_key: str,
    transport: Transport,
    sleep: Callable[[float], None],
    serializer: GlobalSerializer,
    budget: ModelBudget,
    estimated_tokens: int,
) -> dict[str, Any]:
    del events
    attempts = 0
    rate_limit_retries = 0
    transient_retries = 0
    while True:
        attempts += 1
        serializer.wait()
        _append_event(
            ledger,
            {
                "schema": LEDGER_SCHEMA,
                "event": "request_started",
                "timestamp": _utc_now(),
                "request_id": request.request_id,
                "model_id": request.model_id,
                "reviewer_id": request.reviewer_id,
                "attempt": attempts,
                "item_ids": list(request.item_ids),
                "estimated_tokens": estimated_tokens,
                "api_request": dict(request.api_request),
            },
        )
        try:
            result = transport(
                str(api["url"]), api_key, request.api_request, float(api["timeout_seconds"])
            )
        except Exception as exc:
            _append_event(
                ledger,
                {
                    "schema": LEDGER_SCHEMA,
                    "event": "transport_error_ambiguous",
                    "timestamp": _utc_now(),
                    "request_id": request.request_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                },
            )
            budget.charge(estimated_tokens, None)
            return {"status": "ambiguous_transport_failure", "attempts": attempts}
        usage = result.body.get("usage") if isinstance(result.body, Mapping) else None
        actual = (
            int(cast(Mapping[str, Any], usage).get("total_tokens", 0))
            if isinstance(usage, Mapping)
            else None
        )
        budget.charge(estimated_tokens, actual)
        if result.status == 429:
            rate_limit_retries += 1
            header = result.headers.get("retry-after")
            delay = float(header) if header and header.isdigit() else float(
                retry["default_retry_after_seconds"]
            )
            delay = min(delay, float(retry["maximum_retry_after_seconds"]))
            _append_event(
                ledger,
                {
                    "schema": LEDGER_SCHEMA,
                    "event": "rate_limited",
                    "timestamp": _utc_now(),
                    "request_id": request.request_id,
                    "retry_after_seconds": delay,
                    "attempt": attempts,
                },
            )
            if rate_limit_retries > int(retry["max_rate_limit_retries_per_request"]):
                return {"status": "rate_limited", "attempts": attempts}
            sleep(delay)
            continue
        if result.status != 200:
            transient_retries += 1
            _append_event(
                ledger,
                {
                    "schema": LEDGER_SCHEMA,
                    "event": "http_error",
                    "timestamp": _utc_now(),
                    "request_id": request.request_id,
                    "status": result.status,
                    "body": result.body,
                },
            )
            if transient_retries > MAX_TRANSIENT_RETRIES:
                return {"status": "http_error_retries_exhausted", "attempts": attempts}
            backoff = min(
                float(retry["transient_backoff_initial_seconds"]) * (2 ** (transient_retries - 1)),
                float(retry["transient_backoff_maximum_seconds"]),
            )
            sleep(backoff)
            continue
        try:
            ratings = _parse_ratings(_response_content(result.body), request.item_ids)
        except (ValueError, json.JSONDecodeError) as exc:
            transient_retries += 1
            _append_event(
                ledger,
                {
                    "schema": LEDGER_SCHEMA,
                    "event": "invalid_ratings",
                    "timestamp": _utc_now(),
                    "request_id": request.request_id,
                    "error": str(exc)[:1000],
                },
            )
            if transient_retries > MAX_TRANSIENT_RETRIES:
                return {"status": "invalid_ratings_retries_exhausted", "attempts": attempts}
            sleep(float(retry["transient_backoff_initial_seconds"]))
            continue
        _append_event(
            ledger,
            {
                "schema": LEDGER_SCHEMA,
                "event": "response_received",
                "timestamp": _utc_now(),
                "request_id": request.request_id,
                "model_id": request.model_id,
                "reviewer_id": request.reviewer_id,
                "attempt": attempts,
                "ratings": ratings,
                "usage": usage,
            },
        )
        return {"status": "completed", "attempts": attempts}


def collect_ratings(output_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Read every recorded rating out of the durable per-model ledgers."""
    ratings: dict[str, dict[str, dict[str, Any]]] = {}
    ledger_dir = output_dir / "ledgers"
    if not ledger_dir.is_dir():
        return ratings
    for ledger in sorted(ledger_dir.glob("*.jsonl")):
        for event in read_durable_jsonl_prefix(ledger):
            if event.get("event") != "response_received":
                continue
            model_id = str(event["model_id"])
            target = ratings.setdefault(model_id, {})
            for rating in event["ratings"]:
                target[str(rating["audit_id"])] = dict(rating)
    return ratings


def _proportion_ci(
    values: Sequence[bool], *, samples: int = 2000, seed: int = 42
) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "rate": None, "ci95_low": None, "ci95_high": None}
    rng = random.Random(seed)
    population = [1 if value else 0 for value in values]
    count = len(population)
    estimates = [sum(rng.choices(population, k=count)) / count for _ in range(samples)]
    return {
        "count": len(values),
        "rate": sum(values) / len(values),
        "ci95_low": percentile(estimates, 0.025),
        "ci95_high": percentile(estimates, 0.975),
    }


CONFIDENCE_BUCKETS = ((0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0))


def _confidence_bucket(value: float) -> str:
    """Descriptive reporting buckets; they gate no decision and no threshold."""
    for low, high in CONFIDENCE_BUCKETS:
        if low <= value < high:
            return f"[{low},{high})"
    return f"[{CONFIDENCE_BUCKETS[-1][1]}]"


def _role_fields(rating: Mapping[str, Any], automatic: str) -> dict[str, dict[str, Any]]:
    """Map the judge's per-option answers onto the chosen/rejected roles via the key."""
    chosen_suffix, rejected_suffix = ("a", "b") if automatic == "A" else ("b", "a")
    return {
        "chosen": {
            "answerable": rating[f"answerable_{chosen_suffix}"],
            "format_valid": rating[f"format_valid_{chosen_suffix}"],
        },
        "rejected": {
            "answerable": rating[f"answerable_{rejected_suffix}"],
            "format_valid": rating[f"format_valid_{rejected_suffix}"],
        },
    }


@dataclass(frozen=True)
class ExportReadAdapter:
    """Which key fields a blind export supplies, per its frozen contract.

    Amendment `reports/decisions/task06_groq_audit_reader_axis_amendment_2026-08-21.md`:
    the v1 export stratifies the descriptive slices by primary-margin band, the v2 export
    by defect **axis**, because policy v2 does not order pairs by margin and therefore
    publishes no band.  Only these *reporting* dimensions differ — the prompt, rubric,
    models, budgets and every decision rule are shared and untouched.
    """

    slice_field: str
    label_field: str
    slice_analysis_key: str
    decided_slice_analysis_key: str
    label_analysis_key: str


_V1_ADAPTER = ExportReadAdapter(
    slice_field="primary_margin_gap_band",
    label_field="rejected_failure_types",
    slice_analysis_key="agreement_by_primary_margin_gap_band",
    decided_slice_analysis_key="decided_agreement_by_primary_margin_gap_band",
    label_analysis_key="agreement_by_rejected_failure_type",
)
_V2_ADAPTER = ExportReadAdapter(
    slice_field="axis",
    label_field="rejected_defect_labels",
    slice_analysis_key="agreement_by_axis",
    decided_slice_analysis_key="decided_agreement_by_axis",
    label_analysis_key="agreement_by_rejected_defect_label",
)
# Polityka v2.1 ma jedną oś, więc `axis` przestaje różnicować cokolwiek; wymiarem
# opisowym zostaje jednowartościowa etykieta defektu (ADR v2.1 §3.3). Prompt,
# rubryka, modele, limity i reguły decyzyjne pozostają bez zmian.
_V2_1_ADAPTER = ExportReadAdapter(
    slice_field="rejected_defect_label",
    label_field="rejected_defect_labels",
    slice_analysis_key="agreement_by_primary_defect_label",
    decided_slice_analysis_key="decided_agreement_by_primary_defect_label",
    label_analysis_key="agreement_by_rejected_defect_label",
)


def load_export_manifest(path: Path) -> tuple[ExportManifest, ExportReadAdapter]:
    """Validate a blind export manifest with the model its own contract pins."""
    from doc2query.preferences.pair_audit_export_v2 import (
        EXPORT_CONTRACT as V2_EXPORT_CONTRACT,
    )
    from doc2query.preferences.pair_audit_export_v2 import (
        load_defect_blind_export_manifest,
    )
    from doc2query.preferences.pair_audit_export_v2_1 import (
        EXPORT_CONTRACT as V2_1_EXPORT_CONTRACT,
    )
    from doc2query.preferences.pair_audit_export_v2_1 import (
        load_defect_blind_export_manifest_v2_1,
    )

    raw = json.loads(path.read_text(encoding="utf-8"))
    if str(raw.get("contract")) == V2_1_EXPORT_CONTRACT:
        return load_defect_blind_export_manifest_v2_1(path), _V2_1_ADAPTER
    if str(raw.get("contract")) == V2_EXPORT_CONTRACT:
        return load_defect_blind_export_manifest(path), _V2_ADAPTER
    return load_blind_export_manifest(path), _V1_ADAPTER


def _load_sample(export_dir: Path, manifest: ExportManifest) -> dict[str, dict[str, Any]]:
    """Read the sampled pair records so judge answers can be crossed with the pipeline."""
    path = export_dir / str(manifest.sample["path"])
    if not path.is_file():
        raise ValueError(f"missing sampled pair records declared by the manifest: {path}")
    rows = {str(row["pair_id"]): dict(row) for row in read_records(path)}
    if len(rows) != manifest.sampled_pair_count:
        raise ValueError("sampled pair record count drifted from the export manifest")
    return rows


def analyze_pair_audit(*, export_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Join blind ratings with the unblinding key and report every required agreement."""
    manifest, adapter = load_export_manifest(export_dir / "manifest.json")
    key = {str(row["audit_id"]): row for row in read_records(export_dir / "machine_key.jsonl")}
    sample = _load_sample(export_dir, manifest)
    ratings = collect_ratings(output_dir)
    models = sorted(ratings)
    verdicts: list[dict[str, Any]] = []
    per_model_agreement: dict[str, list[bool]] = {model: [] for model in models}
    inter_model: list[bool] = []
    consensus: Counter[str] = Counter()
    positions: dict[str, Counter[str]] = {model: Counter() for model in models}
    reason_codes: dict[str, Counter[str]] = {model: Counter() for model in models}
    repairs: Counter[str] = Counter()
    out_of_schema: dict[str, Counter[str]] = {model: Counter() for model in models}
    by_failure: dict[str, list[bool]] = {}
    by_band: dict[str, list[bool]] = {}
    by_confidence: dict[str, dict[str, list[bool]]] = {model: {} for model in models}
    by_band_per_model: dict[str, dict[str, list[bool]]] = {model: {} for model in models}
    tie_by_confidence: dict[str, dict[str, list[bool]]] = {model: {} for model in models}
    format_cross: dict[str, dict[str, Counter[str]]] = {
        model: {"chosen": Counter(), "rejected": Counter()} for model in models
    }
    answerability_cross: dict[str, dict[str, Counter[str]]] = {
        model: {"chosen": Counter(), "rejected": Counter()} for model in models
    }

    for audit_id, key_row in sorted(key.items()):
        automatic = str(key_row["automatic_chosen_option"])
        row: dict[str, Any] = {
            "audit_id": audit_id,
            "pair_id": key_row["pair_id"],
            "cohort_id": key_row["cohort_id"],
            "automatic_chosen_option": automatic,
            adapter.slice_field: key_row[adapter.slice_field],
            adapter.label_field: list(key_row[adapter.label_field]),
            "ratings": {},
        }
        decided: dict[str, str] = {}
        for model in models:
            rating = ratings[model].get(audit_id)
            if rating is None:
                continue
            preference = str(rating["preference"])
            positions[model][preference] += 1
            reason_codes[model][str(rating["reason_code"])] += 1
            if rating.get("audit_id_repaired"):
                repairs[model] += 1
            if str(rating["reason_code"]) == OUT_OF_SCHEMA_REASON:
                out_of_schema[model][str(rating.get("reason_code_raw", ""))] += 1
            bucket = _confidence_bucket(float(rating["confidence"]))
            tie_by_confidence[model].setdefault(bucket, []).append(
                preference not in DECIDED_PREFERENCES
            )
            if preference in DECIDED_PREFERENCES:
                by_confidence[model].setdefault(bucket, []).append(preference == automatic)
                by_band_per_model[model].setdefault(
                    str(key_row[adapter.slice_field]), []
                ).append(preference == automatic)
            pair = sample[str(key_row["pair_id"])]
            for role, answers in _role_fields(rating, automatic).items():
                components = pair[f"{role}_components"]
                pipeline_format = bool(components["format_valid"])
                judge_format = bool(answers["format_valid"])
                format_cross[model][role][
                    f"pipeline_{'valid' if pipeline_format else 'invalid'}"
                    f"__judge_{'valid' if judge_format else 'invalid'}"
                ] += 1
                pipeline_round_trip = float(components["corpus_round_trip_at_20"]) >= 1.0
                judge_answerable = bool(answers["answerable"])
                answerability_cross[model][role][
                    f"round_trip20_{'hit' if pipeline_round_trip else 'miss'}"
                    f"__judge_{'answerable' if judge_answerable else 'unanswerable'}"
                ] += 1
            row["ratings"][model] = rating
            if preference in DECIDED_PREFERENCES:
                decided[model] = preference
                per_model_agreement[model].append(preference == automatic)
        if len(row["ratings"]) < len(models) or len(models) < 2:
            row["consensus"] = "incomplete_ratings"
        elif len(decided) < len(models):
            row["consensus"] = "abstained"
        elif len(set(decided.values())) > 1:
            row["consensus"] = "disagreement"
            inter_model.append(False)
        else:
            agrees = next(iter(decided.values())) == automatic
            row["consensus"] = (
                "consensus_supports_automatic" if agrees else "consensus_contradicts_automatic"
            )
            inter_model.append(True)
            for label in row[adapter.label_field] or ["unclassified"]:
                by_failure.setdefault(label, []).append(agrees)
            by_band.setdefault(str(row[adapter.slice_field]), []).append(agrees)
        row["eligible_for_automatic_acceptance"] = (
            row["consensus"] == "consensus_supports_automatic"
        )
        consensus[str(row["consensus"])] += 1
        verdicts.append(row)

    rated = [row for row in verdicts if row["consensus"] != "incomplete_ratings"]
    analysis: dict[str, Any] = {
        "schema": ANALYSIS_SCHEMA,
        "contract": CONTRACT,
        "status": "complete" if len(rated) == manifest.sampled_pair_count else "incomplete",
        "export_policy_id": manifest.policy_id,
        "export_contract": manifest.contract,
        "export_audit_ids_fingerprint": manifest.audit_ids_fingerprint,
        "sampled_pair_count": manifest.sampled_pair_count,
        "development_gate_met": manifest.development_gate_met,
        "rated_pair_count": len(rated),
        "models": models,
        "automatic_selector_agreement": {
            model: _proportion_ci(per_model_agreement[model]) for model in models
        },
        "model_agreement": _proportion_ci(inter_model),
        "consensus_counts": dict(sorted(consensus.items())),
        "excluded_from_automatic_acceptance": sum(
            1 for row in rated if not row["eligible_for_automatic_acceptance"]
        ),
        "position_balance": {
            model: dict(sorted(positions[model].items())) for model in models
        },
        "reason_code_counts": {
            model: dict(sorted(reason_codes[model].items())) for model in models
        },
        "audit_id_prefix_repairs": {model: repairs[model] for model in models},
        # Slice'y konsensusu niżej mają mianownik ograniczony do par, na których
        # oba modele rozstrzygnęły zgodnie. Ten slice patrzy na wszystkie
        # rozstrzygnięte oceny osobno dla każdego modelu, więc jest lepiej
        # obsadzony; ocen dwóch modeli nie wolno łączyć, bo dotyczą tych samych par.
        adapter.decided_slice_analysis_key: {
            model: {
                band: _proportion_ci(values)
                for band, values in sorted(by_band_per_model[model].items())
            }
            for model in models
        },
        "agreement_by_confidence_bucket": {
            model: {
                bucket: _proportion_ci(values)
                for bucket, values in sorted(by_confidence[model].items())
            }
            for model in models
        },
        "tie_rate_by_confidence_bucket": {
            model: {
                bucket: _proportion_ci(values)
                for bucket, values in sorted(tie_by_confidence[model].items())
            }
            for model in models
        },
        "judge_versus_pipeline_format": {
            model: {role: dict(sorted(counts.items())) for role, counts in roles.items()}
            for model, roles in format_cross.items()
        },
        "judge_versus_pipeline_answerability": {
            model: {role: dict(sorted(counts.items())) for role, counts in roles.items()}
            for model, roles in answerability_cross.items()
        },
        "out_of_schema_reason_codes": {
            model: dict(sorted(out_of_schema[model].items())) for model in models
        },
        adapter.label_analysis_key: {
            label: _proportion_ci(values) for label, values in sorted(by_failure.items())
        },
        adapter.slice_analysis_key: {
            label: _proportion_ci(values) for label, values in sorted(by_band.items())
        },
        "human_evidence_claimed": False,
        "safe_anchor_selection_signal": False,
        "task07_training_authorized": False,
        "final_tests_used": [],
    }
    with JsonlWriter(output_dir / "pair_verdicts.jsonl") as writer:
        for row in verdicts:
            writer.write(row)
    write_json(output_dir / "analysis.json", analysis)
    return analysis


def run_pair_audit(
    config_path: Path,
    *,
    export_dir: Path,
    output_dir: Path,
    api_key: str,
    transport: Transport = _http_transport,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    max_new_requests_per_model: int | None = None,
    allow_ambiguous_resend: bool = False,
) -> dict[str, Any]:
    """Run or resume the dual-LLM audit of an already frozen blind export."""
    config = load_llm_audit_config(config_path)
    manifest, _ = load_export_manifest(export_dir / "manifest.json")
    if manifest.sampled_pair_count != int(config["pair_count"]):
        raise ValueError(
            f"the frozen audit contract pins {config['pair_count']} pairs but the export holds "
            f"{manifest.sampled_pair_count}; an owner amendment is required before running"
        )
    blind_path = export_dir / "blind_pairs.jsonl"
    plan = build_dual_llm_request_plan(config, blind_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    identity = {
        "contract": CONTRACT,
        "config_sha256": manifest.policy_sha256,
        "export_audit_ids_fingerprint": manifest.audit_ids_fingerprint,
        "prompt_version": config["prompt_version"],
        "request_ids": sorted(request.request_id for request in plan),
    }
    identity_path = output_dir / "identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("dual-LLM audit resume identity mismatch")
    else:
        if (output_dir / "ledgers").exists():
            raise ValueError("audit ledgers exist without an identity file")
        write_json(identity_path, identity)
        write_json(output_dir / "request_plan.json", plan_summary(plan))

    workers = _run_requests(
        plan,
        config=config,
        output_dir=output_dir,
        api_key=api_key,
        transport=transport,
        sleep=sleep,
        monotonic=monotonic,
        max_new_requests_per_model=max_new_requests_per_model,
        allow_ambiguous_resend=allow_ambiguous_resend,
    )
    analysis = analyze_pair_audit(export_dir=export_dir, output_dir=output_dir)
    deferred = workers["deferred_models"]
    status = (
        "complete"
        if analysis["status"] == "complete"
        else ("incomplete_quota_deferred" if deferred else "incomplete")
    )
    result = {
        "schema": RESULT_SCHEMA,
        "contract": CONTRACT,
        "status": status,
        "identity": identity,
        "plan": plan_summary(plan),
        "workers": workers,
        "analysis_status": analysis["status"],
        "human_evidence_claimed": False,
        "task07_training_authorized": False,
        "final_tests_used": [],
    }
    temporary = output_dir / "summary.json.tmp"
    write_json(temporary, result)
    os.replace(temporary, output_dir / "summary.json")
    return result


__all__ = [
    "analyze_pair_audit",
    "collect_ratings",
    "load_api_key",
    "run_pair_audit",
]
