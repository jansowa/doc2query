from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest

from doc2query.evaluation.groq_audits import HttpResult
from doc2query.preferences.groq_pair_audit import (
    LEDGER_SCHEMA,
    _estimate_tokens,
    _parse_ratings,
    analyze_pair_audit,
    collect_ratings,
    run_pair_audit,
)
from doc2query.training.dpo import canonical_fingerprint

CONFIG_PATH = Path("configs/preferences/task06_groq_preference_audit_v1.json")
MODELS = ("openai/gpt-oss-120b", "qwen/qwen3.6-27b")
# Zamrożony kontrakt pinuje dokładnie 500 par i `load_llm_audit_config` odrzuca
# każdą inną wartość, więc testy używają prawdziwego configu bez modyfikacji.
FROZEN_PAIR_COUNT = 500
REQUESTS_PER_MODEL = FROZEN_PAIR_COUNT // 2
MINIMUM_SPACING_SECONDS = 4.0


def _audit_id(index: int) -> str:
    return f"audit-{index:04d}"


def _automatic_option(index: int) -> str:
    return "A" if index % 2 == 0 else "B"


def _export(tmp_path: Path, *, pair_count: int = FROZEN_PAIR_COUNT) -> Path:
    """Write a blind export by hand so the runner is tested without the builder."""
    export = tmp_path / "export"
    export.mkdir()
    blind = []
    key = []
    for index in range(pair_count):
        blind.append(
            {
                "audit_id": _audit_id(index),
                "passage": f"Pasaż numer {index} o wirusach oddechowych.",
                "query_a": f"zapytanie a {index}",
                "query_b": f"zapytanie b {index}",
                "orientation_commitment": f"{index:064d}",
            }
        )
        key.append(
            {
                "audit_id": _audit_id(index),
                "pair_id": f"pair-{index:04d}",
                "cohort_id": "same_prompt_expansion_v1",
                "group_id": f"group-{index}",
                "automatic_chosen_option": _automatic_option(index),
                "orientation_commitment": f"{index:064d}",
                "primary_margin_gap": 1.5 + index,
                "primary_margin_gap_band": "[1.0,2.0)" if index % 2 else "[2.0,4.0)",
                "requested_form": "full_question",
                "requested_intent": "fact_lookup",
                "rejected_failure_types": ["lower_primary_margin"],
                "split": "train",
            }
        )
    sample = [
        {
            "pair_id": f"pair-{index:04d}",
            "cohort_id": "same_prompt_expansion_v1",
            "chosen": f"zapytanie chosen {index}",
            "rejected": f"zapytanie rejected {index}",
            "primary_margin_gap": 1.5 + index,
            "chosen_components": {
                "format_valid": True,
                "corpus_round_trip_at_20": 1.0,
            },
            "rejected_components": {
                "format_valid": True,
                "corpus_round_trip_at_20": 1.0 if index % 2 else 0.0,
            },
        }
        for index in range(pair_count)
    ]
    for name, rows in (
        ("blind_pairs.jsonl", blind),
        ("machine_key.jsonl", key),
        ("sample.jsonl", sample),
    ):
        with (export / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "contract": "task06-preference-audit-blind-export-v1",
        "status": "blind_export_frozen_not_reviewed",
        "policy_id": "task06-tentative-pair-policy-v1",
        "policy_sha256": "a" * 64,
        "source_cohorts": ["same_prompt_expansion_v1"],
        "source_pair_manifest_sha256": {"same_prompt_expansion_v1": "b" * 64},
        "population_pair_count": pair_count,
        "target_pair_count": pair_count,
        "sampled_pair_count": pair_count,
        "shortfall_pair_count": 0,
        "development_gate_met": True,
        "seed": 20260816,
        "strata": [
            {
                "cohort_id": "same_prompt_expansion_v1",
                "requested_form": "full_question",
                "primary_margin_gap_band": "[1.0,2.0)",
                "population": pair_count,
                "allocated": pair_count,
            }
        ],
        "orientation_commitment_salt": "salt",
        "orientation_balance": {"A": pair_count // 2, "B": pair_count - pair_count // 2},
        "audit_ids_fingerprint": "c" * 64,
        "blind_pairs": {
            "path": "blind_pairs.jsonl",
            "sha256": "d" * 64,
            "record_count": pair_count,
        },
        "machine_key": {
            "path": "machine_key.jsonl",
            "sha256": "e" * 64,
            "record_count": pair_count,
        },
        "sample": {"path": "sample.jsonl", "sha256": "f" * 64, "record_count": pair_count},
        "report": {"path": "report.json", "sha256": "0" * 64, "record_count": 1},
        "blind_fields": [
            "audit_id",
            "passage",
            "query_a",
            "query_b",
            "orientation_commitment",
        ],
        "ratings_collected": False,
        "human_evidence_claimed": False,
        "task07_training_authorized": False,
        "final_tests_used": [],
    }
    manifest["manifest_fingerprint"] = canonical_fingerprint(manifest)
    (export / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return export


class _Clock:
    """Fake monotonic clock so the frozen 4 s spacing costs no wall-clock time."""

    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        if seconds < 0:
            raise AssertionError("the runner must never sleep a negative interval")
        self.now += seconds


class _Recorder:
    """Fake Groq transport with a per-model answer policy and a call log."""

    def __init__(self, clock: _Clock, *, preference: Mapping[str, str] | None = None) -> None:
        self.clock = clock
        self.calls: list[dict[str, Any]] = []
        self.preference = dict(preference or {})

    def __call__(
        self, url: str, api_key: str, payload: Mapping[str, Any], timeout: float
    ) -> HttpResult:
        del url, api_key, timeout
        model = str(payload["model"])
        self.calls.append({"model": model, "payload": dict(payload), "at": self.clock.now})
        content = json.loads(str(payload["messages"][1]["content"]).split("\n", 1)[1])
        ratings = [
            {
                "audit_id": row["audit_id"],
                "preference": self.preference.get(model, "A"),
                "reason_code": "grounding",
                "answerable_a": True,
                "answerable_b": True,
                "format_valid_a": True,
                "format_valid_b": True,
                "confidence": 0.8,
            }
            for row in content
        ]
        body = {
            "choices": [{"message": {"content": json.dumps({"ratings": ratings})}}],
            "usage": {"total_tokens": 120},
        }
        return HttpResult(200, {}, body)


def _run(
    tmp_path: Path,
    transport: Any,
    clock: _Clock,
    *,
    export: Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    return run_pair_audit(
        CONFIG_PATH,
        export_dir=export if export is not None else _export(tmp_path),
        output_dir=tmp_path / "audit",
        api_key="secret",
        transport=transport,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
        **kwargs,
    )


def _write_ledger(output_dir: Path, model_id: str, ratings: Sequence[Mapping[str, Any]]) -> None:
    """Record ratings straight into a durable ledger, bypassing the transport."""
    ledger = output_dir / "ledgers" / f"{model_id.replace('/', '__')}.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as handle:
        for index in range(0, len(ratings), 2):
            batch = ratings[index : index + 2]
            handle.write(
                json.dumps(
                    {
                        "schema": LEDGER_SCHEMA,
                        "event": "response_received",
                        "timestamp": "2026-08-17T10:00:00+00:00",
                        "request_id": f"{model_id}-{index}",
                        "model_id": model_id,
                        "reviewer_id": f"reviewer-{model_id}",
                        "attempt": 1,
                        "ratings": list(batch),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


def _rating(index: int, preference: str, **overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "audit_id": _audit_id(index),
        "preference": preference,
        "reason_code": "grounding",
        "answerable_a": True,
        "answerable_b": True,
        "format_valid_a": True,
        "format_valid_b": True,
        "confidence": 0.8,
    }
    row.update(overrides)
    return row


def _analysis_for(
    tmp_path: Path, preferences: Mapping[str, Any], *, pair_count: int = 8
) -> dict[str, Any]:
    export = _export(tmp_path, pair_count=pair_count)
    output = tmp_path / "audit"
    for model, policy in preferences.items():
        _write_ledger(
            output,
            model,
            [
                _rating(index, policy(index) if callable(policy) else str(policy))
                for index in range(pair_count)
            ],
        )
    return analyze_pair_audit(export_dir=export, output_dir=output)


def _export_v2(tmp_path: Path, *, pair_count: int = 8) -> Path:
    """Write a v2 (defect-anchored) blind export by hand, with axis instead of margin band.

    Amendment `task06_groq_audit_reader_axis_amendment_2026-08-21.md`: the reader must
    accept this shape without any change to the prompt, rubric, models or decision rules.
    """
    export = tmp_path / "export_v2"
    export.mkdir()
    blind, key, sample = [], [], []
    for index in range(pair_count):
        axis = "A" if index % 2 == 0 else "B"
        blind.append(
            {
                "audit_id": _audit_id(index),
                "passage": f"Pasaż numer {index} o wirusach oddechowych.",
                "query_a": f"zapytanie a {index}",
                "query_b": f"zapytanie b {index}",
                "orientation_commitment": f"{index:064d}",
            }
        )
        key.append(
            {
                "audit_id": _audit_id(index),
                "pair_id": f"pair-{index:04d}",
                "cohort_id": "same_prompt_expansion_v1",
                "group_id": f"group-{index}",
                "axis": axis,
                "automatic_chosen_option": _automatic_option(index),
                "orientation_commitment": f"{index:064d}",
                "chosen_verdict": "yes",
                "rejected_verdict": "no" if axis == "A" else "yes",
                "rejected_defect_labels": (
                    ["judge_unanswerable"] if axis == "A" else ["high_lexical_overlap"]
                ),
                "requested_form": "full_question",
                "requested_intent": "fact_lookup",
                "split": "train",
            }
        )
        sample.append(
            {
                "pair_id": f"pair-{index:04d}",
                "cohort_id": "same_prompt_expansion_v1",
                "axis": axis,
                "chosen": f"zapytanie chosen {index}",
                "rejected": f"zapytanie rejected {index}",
                "chosen_components": {"format_valid": True, "corpus_round_trip_at_20": 1.0},
                "rejected_components": {"format_valid": True, "corpus_round_trip_at_20": 0.0},
            }
        )
    for name, rows in (
        ("blind_pairs.jsonl", blind),
        ("machine_key.jsonl", key),
        ("sample.jsonl", sample),
    ):
        with (export / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    half = pair_count // 2
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "contract": "task06-defect-pair-audit-blind-export-v2",
        "status": "blind_export_frozen_not_reviewed",
        "policy_id": "task06-defect-pair-policy-v2.0",
        "policy_sha256": "a" * 64,
        "source_cohorts": ["same_prompt_expansion_v1"],
        "source_pair_manifest_sha256": {"same_prompt_expansion_v1": "b" * 64},
        "population_pair_count": pair_count,
        "population_axis_counts": {"A": pair_count - half, "B": half},
        "target_pair_count": pair_count,
        "sampled_pair_count": pair_count,
        "shortfall_pair_count": 0,
        "development_gate_met": True,
        "axis_quotas": [
            {
                "axis": "A",
                "quota": pair_count - half,
                "effective_quota": pair_count - half,
                "population": pair_count - half,
                "allocated": pair_count - half,
                "shortfall": 0,
            },
            {
                "axis": "B",
                "quota": half,
                "effective_quota": half,
                "population": half,
                "allocated": half,
                "shortfall": 0,
            },
        ],
        "axis_quota_shortfall": {"A": 0, "B": 0},
        "seed": 20260820,
        "strata": [
            {
                "cohort_id": "same_prompt_expansion_v1",
                "axis": "A",
                "requested_form": "full_question",
                "population": pair_count - half,
                "allocated": pair_count - half,
            },
            {
                "cohort_id": "same_prompt_expansion_v1",
                "axis": "B",
                "requested_form": "full_question",
                "population": half,
                "allocated": half,
            },
        ],
        "orientation_commitment_salt": "salt",
        "orientation_balance": {"A": half, "B": pair_count - half},
        "audit_ids_fingerprint": "c" * 64,
        "blind_pairs": {
            "path": "blind_pairs.jsonl",
            "sha256": "d" * 64,
            "record_count": pair_count,
        },
        "machine_key": {
            "path": "machine_key.jsonl",
            "sha256": "e" * 64,
            "record_count": pair_count,
        },
        "sample": {"path": "sample.jsonl", "sha256": "f" * 64, "record_count": pair_count},
        "report": {"path": "report.json", "sha256": "0" * 64, "record_count": 1},
        "blind_fields": [
            "audit_id",
            "passage",
            "query_a",
            "query_b",
            "orientation_commitment",
        ],
        "margin_used_for_stratification": False,
        "ratings_collected": False,
        "human_evidence_claimed": False,
        "task07_training_authorized": False,
        "final_tests_used": [],
    }
    manifest["manifest_fingerprint"] = canonical_fingerprint(manifest)
    (export / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return export


def test_v2_export_is_read_and_sliced_by_axis_not_by_margin_band(tmp_path: Path) -> None:
    export = _export_v2(tmp_path)
    output = tmp_path / "audit"
    for model in MODELS:
        _write_ledger(
            output, model, [_rating(index, _automatic_option(index)) for index in range(8)]
        )

    analysis = analyze_pair_audit(export_dir=export, output_dir=output)

    assert analysis["export_contract"] == "task06-defect-pair-audit-blind-export-v2"
    assert analysis["consensus_counts"] == {"consensus_supports_automatic": 8}
    assert sorted(analysis["agreement_by_axis"]) == ["A", "B"]
    assert analysis["agreement_by_axis"]["A"]["count"] == 4
    assert sorted(analysis["agreement_by_rejected_defect_label"]) == [
        "high_lexical_overlap",
        "judge_unanswerable",
    ]
    assert "agreement_by_primary_margin_gap_band" not in analysis
    assert "agreement_by_rejected_failure_type" not in analysis
    for model in MODELS:
        assert sorted(analysis["decided_agreement_by_axis"][model]) == ["A", "B"]


def test_v1_export_keeps_its_frozen_slice_keys_after_the_amendment(tmp_path: Path) -> None:
    """Ścieżka v1 pozostaje odtwarzalna: te same nazwy kluczy, te same wartości."""
    analysis = _analysis_for(tmp_path, {model: _automatic_option for model in MODELS})

    assert analysis["export_contract"] == "task06-preference-audit-blind-export-v1"
    assert sorted(analysis["agreement_by_primary_margin_gap_band"]) == ["[1.0,2.0)", "[2.0,4.0)"]
    assert analysis["agreement_by_rejected_failure_type"]["lower_primary_margin"]["rate"] == 1.0
    assert "agreement_by_axis" not in analysis
    assert "agreement_by_rejected_defect_label" not in analysis


def test_v2_export_still_faces_the_frozen_pair_count_contract(tmp_path: Path) -> None:
    """Amendment dotyczy tylko czytania pól; kontrakt 500 par obowiązuje bez zmian."""
    export = _export_v2(tmp_path, pair_count=8)
    clock = _Clock()

    with pytest.raises(ValueError, match="owner amendment is required"):
        _run(tmp_path, _Recorder(clock), clock, export=export)


def test_export_smaller_than_the_frozen_contract_is_refused(tmp_path: Path) -> None:
    """447 par z kohort v1+v2 nie mogą wejść do audytu pinującego 500 par."""
    export = _export(tmp_path, pair_count=447)
    clock = _Clock()

    with pytest.raises(ValueError, match="owner amendment is required"):
        _run(tmp_path, _Recorder(clock), clock, export=export)


def test_requests_are_globally_serialized_with_the_frozen_spacing(tmp_path: Path) -> None:
    clock = _Clock()
    recorder = _Recorder(clock)

    _run(tmp_path, recorder, clock, max_new_requests_per_model=6)

    assert len(recorder.calls) == 12
    moments = [call["at"] for call in recorder.calls]
    assert moments == sorted(moments)
    gaps = [later - earlier for earlier, later in pairwise(moments)]
    assert all(gap >= MINIMUM_SPACING_SECONDS - 1e-9 for gap in gaps)
    # Serializacja jest globalna, więc modele przeplatają się, a nie jadą równolegle.
    assert {call["model"] for call in recorder.calls} == set(MODELS)


def test_blind_payload_never_carries_the_automatic_choice(tmp_path: Path) -> None:
    clock = _Clock()
    recorder = _Recorder(clock)

    _run(tmp_path, recorder, clock, max_new_requests_per_model=4)

    for call in recorder.calls:
        rendered = json.dumps(call["payload"], ensure_ascii=False)
        assert "automatic_chosen_option" not in rendered
        assert "primary_margin" not in rendered
        assert "pair_id" not in rendered
        assert "chosen" not in rendered


def test_full_run_rates_every_pair_twice_and_respects_the_spacing(tmp_path: Path) -> None:
    clock = _Clock()
    recorder = _Recorder(clock)

    result = _run(tmp_path, recorder, clock)

    assert result["status"] == "complete"
    assert len(recorder.calls) == 2 * REQUESTS_PER_MODEL
    assert result["plan"]["rating_count"] == 2 * FROZEN_PAIR_COUNT
    gaps = [later - earlier for earlier, later in pairwise(call["at"] for call in recorder.calls)]
    assert all(gap >= MINIMUM_SPACING_SECONDS - 1e-9 for gap in gaps)
    analysis = json.loads((tmp_path / "audit" / "analysis.json").read_text(encoding="utf-8"))
    assert analysis["status"] == "complete"
    assert analysis["rated_pair_count"] == FROZEN_PAIR_COUNT
    limits = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["limits_per_model"]
    for model in MODELS:
        worker = result["workers"]["by_model"][model]
        assert worker["completed"] == REQUESTS_PER_MODEL
        assert worker["pending"] == 0
        assert worker["requests_today"] <= limits["safety_requests_per_day"]
        assert worker["tokens_today"] <= limits["safety_tokens_per_day"]


def test_daily_token_budget_defers_both_models_and_stops_cleanly(tmp_path: Path) -> None:
    """Wyczerpanie dziennego budżetu tokenów odracza model, a nie wywala runu.

    Faktyczne zużycie jest znane dopiero po wywołaniu, więc miękki budżet bezpieczeństwa
    wolno przekroczyć o jeden request — to jest właśnie jego zapas względem limitu
    twardego, którego przekroczyć nie wolno.
    """

    class _Expensive(_Recorder):
        def __call__(
            self, url: str, api_key: str, payload: Mapping[str, Any], timeout: float
        ) -> HttpResult:
            result = super().__call__(url, api_key, payload, timeout)
            body = dict(result.body)
            body["usage"] = {"total_tokens": 12000}
            return HttpResult(result.status, result.headers, body)

    clock = _Clock()
    recorder = _Expensive(clock)

    result = _run(tmp_path, recorder, clock)

    assert result["status"] == "incomplete_quota_deferred"
    limits = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["limits_per_model"]
    for model in MODELS:
        worker = result["workers"]["by_model"][model]
        assert worker["deferred_reason"] == "daily_token_budget_exhausted"
        assert worker["tokens_today"] > limits["safety_tokens_per_day"] - 12000
        assert worker["tokens_today"] <= limits["tokens_per_day"]
        assert worker["pending"] > 0
        assert worker["completed"] > 0


def test_resume_skips_completed_requests(tmp_path: Path) -> None:
    clock = _Clock()
    first = _Recorder(clock)
    export = _export(tmp_path)
    _run(tmp_path, first, clock, export=export, max_new_requests_per_model=4)
    completed = len(first.calls)

    second = _Recorder(clock)
    _run(tmp_path, second, clock, export=export, max_new_requests_per_model=2)

    assert completed == 8
    # Wznowienie nie powtarza żadnego ukończonego requestu.
    started = {call["payload"]["messages"][1]["content"] for call in first.calls}
    assert all(call["payload"]["messages"][1]["content"] not in started for call in second.calls)


def test_rate_limit_is_retried_then_deferred(tmp_path: Path) -> None:
    class _AlwaysRateLimited:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(
            self, url: str, api_key: str, payload: Mapping[str, Any], timeout: float
        ) -> HttpResult:
            del url, api_key, payload, timeout
            self.calls += 1
            return HttpResult(429, {"retry-after": "1"}, {"error": {"code": "rate_limit"}})

    clock = _Clock()
    transport = _AlwaysRateLimited()

    result = _run(tmp_path, transport, clock)

    assert result["status"] == "incomplete_quota_deferred"
    assert sorted(result["workers"]["deferred_models"]) == sorted(MODELS)
    for model in MODELS:
        assert result["workers"]["by_model"][model]["deferred_reason"] == (
            "rate_limit_retries_exhausted"
        )


def test_ambiguous_transport_failure_is_not_resent_automatically(tmp_path: Path) -> None:
    class _Exploding:
        def __call__(
            self, url: str, api_key: str, payload: Mapping[str, Any], timeout: float
        ) -> HttpResult:
            del url, api_key, payload, timeout
            raise TimeoutError("połączenie zerwane")

    clock = _Clock()
    export = _export(tmp_path)
    _run(tmp_path, _Exploding(), clock, export=export)

    with pytest.raises(RuntimeError, match="refusing automatic resend"):
        _run(tmp_path, _Recorder(clock), clock, export=export)


def test_operator_can_authorize_a_resend_and_the_ledger_records_it(tmp_path: Path) -> None:
    class _Exploding:
        def __call__(
            self, url: str, api_key: str, payload: Mapping[str, Any], timeout: float
        ) -> HttpResult:
            del url, api_key, payload, timeout
            raise TimeoutError("połączenie zerwane")

    clock = _Clock()
    export = _export(tmp_path)
    _run(tmp_path, _Exploding(), clock, export=export)

    _run(
        tmp_path,
        _Recorder(clock),
        clock,
        export=export,
        allow_ambiguous_resend=True,
        max_new_requests_per_model=1,
    )

    events = [
        json.loads(line)
        for path in (tmp_path / "audit" / "ledgers").glob("*.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert any(event["event"] == "operator_authorized_ambiguous_resend" for event in events)


def test_operator_request_cap_stops_the_run_resumably(tmp_path: Path) -> None:
    clock = _Clock()
    recorder = _Recorder(clock)

    result = _run(tmp_path, recorder, clock, max_new_requests_per_model=1)

    assert result["status"] == "incomplete_quota_deferred"
    assert len(recorder.calls) == 2
    for model in MODELS:
        worker = result["workers"]["by_model"][model]
        assert worker["deferred_reason"] == "operator_request_cap_reached"
        assert worker["pending"] == REQUESTS_PER_MODEL - 1


def test_resume_identity_mismatch_is_refused(tmp_path: Path) -> None:
    clock = _Clock()
    export = _export(tmp_path)
    _run(tmp_path, _Recorder(clock), clock, export=export, max_new_requests_per_model=1)
    identity_path = tmp_path / "audit" / "identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    identity["prompt_version"] = "forged"
    identity_path.write_text(json.dumps(identity), encoding="utf-8")

    with pytest.raises(ValueError, match="resume identity mismatch"):
        _run(tmp_path, _Recorder(clock), clock, export=export)


def test_ratings_are_collected_from_the_durable_ledger(tmp_path: Path) -> None:
    clock = _Clock()
    _run(tmp_path, _Recorder(clock), clock, max_new_requests_per_model=3)

    ratings = collect_ratings(tmp_path / "audit")

    assert sorted(ratings) == sorted(MODELS)
    for model in MODELS:
        assert len(ratings[model]) == 6


def test_agreement_is_reported_per_model_and_between_models(tmp_path: Path) -> None:
    # Model 1 zawsze wskazuje A, model 2 zawsze B; automat wybiera A dla parzystych.
    analysis = _analysis_for(tmp_path, {MODELS[0]: "A", MODELS[1]: "B"})

    assert analysis["automatic_selector_agreement"][MODELS[0]]["rate"] == pytest.approx(0.5)
    assert analysis["automatic_selector_agreement"][MODELS[1]]["rate"] == pytest.approx(0.5)
    assert analysis["model_agreement"]["rate"] == pytest.approx(0.0)
    assert analysis["consensus_counts"] == {"disagreement": 8}
    assert analysis["excluded_from_automatic_acceptance"] == 8
    for model in MODELS:
        interval = analysis["automatic_selector_agreement"][model]
        assert interval["ci95_low"] <= interval["rate"] <= interval["ci95_high"]


def test_full_agreement_with_the_selector_is_reported_as_consensus(tmp_path: Path) -> None:
    analysis = _analysis_for(tmp_path, {model: _automatic_option for model in MODELS})

    assert analysis["consensus_counts"] == {"consensus_supports_automatic": 8}
    assert analysis["excluded_from_automatic_acceptance"] == 0
    assert analysis["model_agreement"]["rate"] == pytest.approx(1.0)
    for model in MODELS:
        assert analysis["automatic_selector_agreement"][model]["rate"] == pytest.approx(1.0)
    slice_rates = analysis["agreement_by_rejected_failure_type"]["lower_primary_margin"]
    assert slice_rates["rate"] == pytest.approx(1.0)


def test_consensus_against_the_selector_is_reported_not_hidden(tmp_path: Path) -> None:
    flipped = {model: (lambda index: "B" if index % 2 == 0 else "A") for model in MODELS}

    analysis = _analysis_for(tmp_path, flipped)

    assert analysis["consensus_counts"] == {"consensus_contradicts_automatic": 8}
    assert analysis["excluded_from_automatic_acceptance"] == 8
    assert analysis["model_agreement"]["rate"] == pytest.approx(1.0)
    for model in MODELS:
        assert analysis["automatic_selector_agreement"][model]["rate"] == pytest.approx(0.0)


def test_abstention_is_kept_separate_from_disagreement(tmp_path: Path) -> None:
    analysis = _analysis_for(tmp_path, {MODELS[0]: "A", MODELS[1]: "uncertain"})

    assert analysis["consensus_counts"] == {"abstained": 8}
    assert analysis["automatic_selector_agreement"][MODELS[1]]["count"] == 0
    assert analysis["position_balance"][MODELS[1]] == {"uncertain": 8}


def test_missing_second_model_leaves_the_analysis_incomplete(tmp_path: Path) -> None:
    analysis = _analysis_for(tmp_path, {MODELS[0]: "A"})

    assert analysis["status"] == "incomplete"
    assert analysis["consensus_counts"] == {"incomplete_ratings": 8}
    assert analysis["rated_pair_count"] == 0


def test_analysis_never_claims_human_evidence_or_authorizes_training(tmp_path: Path) -> None:
    analysis = _analysis_for(tmp_path, {model: _automatic_option for model in MODELS})

    assert analysis["human_evidence_claimed"] is False
    assert analysis["safe_anchor_selection_signal"] is False
    assert analysis["task07_training_authorized"] is False
    assert analysis["final_tests_used"] == []


def test_ratings_outside_the_request_are_rejected() -> None:
    payload = json.dumps({"ratings": [_rating(9999, "A")]})

    with pytest.raises(ValueError, match="outside its request"):
        _parse_ratings(payload, [_audit_id(1)])


def test_partial_coverage_of_a_request_is_rejected() -> None:
    payload = json.dumps({"ratings": [_rating(1, "A")]})

    with pytest.raises(ValueError, match="instead of"):
        _parse_ratings(payload, [_audit_id(1), _audit_id(2)])


def test_invalid_preference_and_confidence_are_rejected() -> None:
    def rating(**overrides: Any) -> str:
        base = _rating(1, "A")
        base.update(overrides)
        return json.dumps({"ratings": [base]})

    with pytest.raises(ValueError, match="invalid preference"):
        _parse_ratings(rating(preference="maybe"), [_audit_id(1)])
    with pytest.raises(ValueError, match=r"confidence must lie"):
        _parse_ratings(rating(confidence=1.5), [_audit_id(1)])
    with pytest.raises(ValueError, match="must be boolean"):
        _parse_ratings(rating(answerable_a="yes"), [_audit_id(1)])


def test_token_estimate_is_conservative() -> None:
    payload = {"messages": [{"role": "user", "content": "x" * 300}], "max_completion_tokens": 360}

    estimate = _estimate_tokens(payload)

    assert estimate > 360
    assert estimate >= 360 + 300 / 4


# Prawdziwe audit_id mają 24 znaki heksadecymalne, jak w ślepym eksporcie.
REAL_IDS = ("241b038fc3b023a2cf1e77aa", "9f0c5512ab77d301ee4b1c60")


def _rating_with_id(audit_id: str, preference: str) -> dict[str, Any]:
    row = _rating(0, preference)
    row["audit_id"] = audit_id
    return row


def test_truncated_audit_id_is_repaired_only_when_unambiguous() -> None:
    """qwen deterministycznie psuje audit_id; unikalny przedrostek wolno rozwinąć."""
    truncated = _rating_with_id(REAL_IDS[0][:20], "A")
    second = _rating_with_id(REAL_IDS[1], "B")

    parsed = _parse_ratings(json.dumps({"ratings": [truncated, second]}), REAL_IDS)

    assert [row["audit_id"] for row in parsed] == sorted(REAL_IDS)
    by_id = {row["audit_id"]: row for row in parsed}
    assert by_id[REAL_IDS[0]]["audit_id_repaired"] is True
    assert by_id[REAL_IDS[1]]["audit_id_repaired"] is False


def test_id_mangled_in_the_middle_is_resolved_by_its_leading_characters() -> None:
    """Prawdziwy przypadek z runu: qwen zgubił cztery znaki w środku ID."""
    mangled = _rating_with_id("241b038fc3b023a2cf1e", "A")
    expected = ("241b038fc3b311a023a2cf1e", "248804e63b0f4f53bb18723c")

    parsed = _parse_ratings(
        json.dumps({"ratings": [mangled, _rating_with_id(expected[1], "B")]}), expected
    )

    by_id = {row["audit_id"]: row for row in parsed}
    assert sorted(by_id) == sorted(expected)
    assert by_id[expected[0]]["audit_id_repaired"] is True


def test_too_short_or_ambiguous_prefix_is_still_rejected() -> None:
    shared = ("aabbccddeeff00112233aaaa", "aabbccddeeff00112233bbbb")
    too_short = _rating_with_id(REAL_IDS[0][:6], "A")

    with pytest.raises(ValueError, match="outside its request"):
        _parse_ratings(
            json.dumps({"ratings": [too_short, _rating_with_id(REAL_IDS[1], "B")]}), REAL_IDS
        )

    # Oba ID mają identyczny przedrostek, więc dopasowanie jest niejednoznaczne.
    ambiguous = _rating_with_id(shared[0][:20], "A")
    with pytest.raises(ValueError, match="outside its request"):
        _parse_ratings(
            json.dumps({"ratings": [ambiguous, _rating_with_id(shared[1], "B")]}), shared
        )


def test_two_mangled_ids_collapsing_onto_one_pair_are_rejected() -> None:
    """Naprawa nie może scalić dwóch ocen w jedną parę — pokrycie requestu jest sprawdzane."""
    expected = ("241b038fc3b311a023a2cf1e", "248804e63b0f4f53bb18723c")
    first = _rating_with_id("241b038fc3b023a2cf1e", "A")
    second = _rating_with_id("241b038fc3xxxx", "B")

    with pytest.raises(ValueError, match="instead of"):
        _parse_ratings(json.dumps({"ratings": [first, second]}), expected)


def test_repairs_are_counted_in_the_analysis(tmp_path: Path) -> None:
    export = _export(tmp_path, pair_count=4)
    output = tmp_path / "audit"
    for model in MODELS:
        rows = []
        for index in range(4):
            row = _rating(index, _automatic_option(index))
            row["audit_id_repaired"] = index == 0
            rows.append(row)
        _write_ledger(output, model, rows)

    analysis = analyze_pair_audit(export_dir=export, output_dir=output)

    assert analysis["audit_id_prefix_repairs"] == {model: 1 for model in MODELS}


def test_off_schema_reason_code_is_kept_verbatim_not_rejected() -> None:
    """Odrzucanie requestu za kod diagnostyczny obciążałoby pokrycie par."""
    off_schema = _rating_with_id(REAL_IDS[0], "A")
    off_schema["reason_code"] = "answerability"

    parsed = _parse_ratings(
        json.dumps({"ratings": [off_schema, _rating_with_id(REAL_IDS[1], "B")]}), REAL_IDS
    )

    by_id = {row["audit_id"]: row for row in parsed}
    assert by_id[REAL_IDS[0]]["reason_code"] == "out_of_schema"
    assert by_id[REAL_IDS[0]]["reason_code_raw"] == "answerability"
    assert by_id[REAL_IDS[1]]["reason_code"] == "grounding"


def test_missing_reason_code_is_still_rejected() -> None:
    broken = _rating_with_id(REAL_IDS[0], "A")
    broken["reason_code"] = ""

    with pytest.raises(ValueError, match="missing reason_code"):
        _parse_ratings(
            json.dumps({"ratings": [broken, _rating_with_id(REAL_IDS[1], "B")]}), REAL_IDS
        )


def test_out_of_schema_reason_codes_are_counted_in_the_analysis(tmp_path: Path) -> None:
    export = _export(tmp_path, pair_count=4)
    output = tmp_path / "audit"
    for model in MODELS:
        rows = []
        for index in range(4):
            row = _rating(index, _automatic_option(index))
            if index == 0:
                row["reason_code"] = "out_of_schema"
                row["reason_code_raw"] = "answerability"
            rows.append(row)
        _write_ledger(output, model, rows)

    analysis = analyze_pair_audit(export_dir=export, output_dir=output)

    assert analysis["out_of_schema_reason_codes"] == {
        model: {"answerability": 1} for model in MODELS
    }


def test_judge_format_answers_are_crossed_with_the_pipeline(tmp_path: Path) -> None:
    """Polityka wymusza format_valid=True po obu stronach, więc każde `invalid`
    od sędziego jest kandydatem na ślepą plamkę `format.py`."""
    export = _export(tmp_path, pair_count=4)
    output = tmp_path / "audit"
    for model in MODELS:
        rows = []
        for index in range(4):
            # Automat wybiera A dla parzystych, więc strona `chosen` to raz a, raz b.
            chosen_suffix = "a" if _automatic_option(index) == "A" else "b"
            row = _rating(index, _automatic_option(index))
            if index == 0:
                row[f"format_valid_{chosen_suffix}"] = False
            rows.append(row)
        _write_ledger(output, model, rows)

    analysis = analyze_pair_audit(export_dir=export, output_dir=output)

    for model in MODELS:
        chosen = analysis["judge_versus_pipeline_format"][model]["chosen"]
        assert chosen["pipeline_valid__judge_invalid"] == 1
        assert chosen["pipeline_valid__judge_valid"] == 3


def test_judge_answerability_is_crossed_with_corpus_round_trip(tmp_path: Path) -> None:
    export = _export(tmp_path, pair_count=4)
    output = tmp_path / "audit"
    for model in MODELS:
        rows = []
        for index in range(4):
            rejected_suffix = "b" if _automatic_option(index) == "A" else "a"
            row = _rating(index, _automatic_option(index))
            row[f"answerable_{rejected_suffix}"] = False
            rows.append(row)
        _write_ledger(output, model, rows)

    analysis = analyze_pair_audit(export_dir=export, output_dir=output)

    for model in MODELS:
        rejected = analysis["judge_versus_pipeline_answerability"][model]["rejected"]
        # Fixture: rejected trafia w round_trip@20 tylko dla nieparzystych indeksów.
        assert rejected["round_trip20_hit__judge_unanswerable"] == 2
        assert rejected["round_trip20_miss__judge_unanswerable"] == 2
        chosen = analysis["judge_versus_pipeline_answerability"][model]["chosen"]
        assert chosen["round_trip20_hit__judge_answerable"] == 4


def test_agreement_and_tie_rate_are_reported_per_confidence_bucket(tmp_path: Path) -> None:
    export = _export(tmp_path, pair_count=4)
    output = tmp_path / "audit"
    for model in MODELS:
        rows = [
            _rating(0, _automatic_option(0), confidence=0.95),
            _rating(1, _automatic_option(1), confidence=0.95),
            # Niska pewność: jedna zgodna, jedna remis.
            _rating(2, _automatic_option(2), confidence=0.4),
            _rating(3, "tie", confidence=0.4),
        ]
        _write_ledger(output, model, rows)

    analysis = analyze_pair_audit(export_dir=export, output_dir=output)

    for model in MODELS:
        agreement = analysis["agreement_by_confidence_bucket"][model]
        assert agreement["[0.9,1.0)"]["rate"] == pytest.approx(1.0)
        assert agreement["[0.9,1.0)"]["count"] == 2
        assert agreement["[0.0,0.5)"]["count"] == 1
        ties = analysis["tie_rate_by_confidence_bucket"][model]
        assert ties["[0.0,0.5)"]["rate"] == pytest.approx(0.5)
        assert ties["[0.9,1.0)"]["rate"] == pytest.approx(0.0)


def test_missing_sampled_records_are_refused(tmp_path: Path) -> None:
    export = _export(tmp_path, pair_count=4)
    (export / "sample.jsonl").unlink()
    output = tmp_path / "audit"
    _write_ledger(output, MODELS[0], [_rating(index, "A") for index in range(4)])

    with pytest.raises(ValueError, match="missing sampled pair records"):
        analyze_pair_audit(export_dir=export, output_dir=output)


def test_margin_band_agreement_is_reported_per_model_on_decided_ratings(tmp_path: Path) -> None:
    """Slice konsensusu ma mały mianownik, więc pasma raportujemy też per model."""
    export = _export(tmp_path, pair_count=4)
    output = tmp_path / "audit"
    # Fixture: pasmo [1.0,2.0) dla nieparzystych indeksów, [2.0,4.0) dla parzystych.
    _write_ledger(output, MODELS[0], [_rating(index, "A") for index in range(4)])

    analysis = analyze_pair_audit(export_dir=export, output_dir=output)

    bands = analysis["decided_agreement_by_primary_margin_gap_band"][MODELS[0]]
    assert bands["[2.0,4.0)"]["rate"] == pytest.approx(1.0)
    assert bands["[2.0,4.0)"]["count"] == 2
    assert bands["[1.0,2.0)"]["rate"] == pytest.approx(0.0)
    assert bands["[1.0,2.0)"]["count"] == 2
