from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from doc2query.preferences.pair_selector_v3 import (
    RUBRICS,
    JudgeApiError,
    JudgeEndpoint,
    PairwiseItem,
    analyze_calibration,
    chat_payload,
    endpoint_from_args,
    journal_key,
    load_journal,
    parse_verdict,
    plan_summary,
    probe_endpoint,
    run_pairwise,
    strip_reasoning,
)


def _endpoint(**kwargs: Any) -> JudgeEndpoint:
    return JudgeEndpoint(base_url="http://x/v1", api_key="k", model="m", **kwargs)


def _item(index: int = 0) -> PairwiseItem:
    return PairwiseItem(
        item_id=f"item-{index}",
        passage="Pasaż o orangutanach na Borneo i Sumatrze.",
        query_first="jaki jest zasięg występowania orangutanów",
        query_second="które kraje mają orangutany",
        metadata={"expected_canonical": "A"},
    )


def _transport(answers: Mapping[str, str]) -> Any:
    def call(payload: Mapping[str, Any]) -> dict[str, Any]:
        user = str(payload["messages"][1]["content"])
        first = user.split("Zapytanie A:\n")[1].split("\n")[0]
        return {
            "choices": [
                {"message": {"content": json.dumps({"better": answers[first], "confidence": 0.9})}}
            ]
        }

    return call


def test_rubrics_are_frozen_and_state_the_conflict_hierarchy() -> None:
    assert set(RUBRICS) == {"R1_grounding", "R2_retrieval_usefulness", "R3_holistic"}
    assert "ugruntowanie przed" in RUBRICS["R3_holistic"]
    for text in RUBRICS.values():
        assert "better" in text and "confidence" in text


def test_payload_swaps_queries_and_hides_every_selection_signal() -> None:
    item = _item()
    endpoint = _endpoint()
    forward = chat_payload(item, "R3_holistic", "ab", endpoint)
    reverse = chat_payload(item, "R3_holistic", "ba", endpoint)
    forward_user = str(forward["messages"][1]["content"])
    reverse_user = str(reverse["messages"][1]["content"])
    assert item.query_first in forward_user.split("Zapytanie B:")[0]
    assert item.query_second in reverse_user.split("Zapytanie B:")[0]
    for text in (forward_user, reverse_user):
        for leak in ("chosen", "rejected", "margin", "round_trip", "score"):
            assert leak not in text
    assert forward["temperature"] == 0.0
    assert forward["chat_template_kwargs"] == {"enable_thinking": False}
    assert "chat_template_kwargs" not in chat_payload(
        item, "R3_holistic", "ab", _endpoint(allow_reasoning=True)
    )
    with pytest.raises(ValueError, match="nieznana rubryka"):
        chat_payload(item, "R9", "ab", endpoint)


def test_parse_verdict_accepts_only_the_frozen_labels() -> None:
    assert parse_verdict('{"better": "A", "confidence": 0.8}') == ("A", 0.8)
    assert parse_verdict('bla {"better":"tie","confidence":0.1} bla')[0] == "tie"
    with pytest.raises(ValueError, match="niedozwolony werdykt"):
        parse_verdict('{"better": "oba", "confidence": 1}')
    with pytest.raises(ValueError, match="nie zawiera obiektu JSON"):
        parse_verdict("A jest lepsze")


def test_position_swap_is_normalized_to_canonical_order(tmp_path: Path) -> None:
    """Sędzia zawsze mówi 'A'; po normalizacji to raz A, raz B — czyli position_flip."""
    item = _item()
    journal = tmp_path / "j.jsonl"
    summary = run_pairwise(
        items=[item],
        endpoint=_endpoint(),
        journal_path=journal,
        transport=_transport({item.query_first: "A", item.query_second: "A"}),
        rubrics=["R3_holistic"],
        progress_every=0,
    )
    assert summary["judgments"] == 2
    rows = load_journal(journal)
    assert rows[journal_key("item-0", "R3_holistic", "ab")]["canonical_verdict"] == "A"
    assert rows[journal_key("item-0", "R3_holistic", "ba")]["canonical_verdict"] == "B"


def test_consistent_judge_yields_full_votes_and_no_flip(tmp_path: Path) -> None:
    item = _item()
    journal = tmp_path / "j.jsonl"
    run_pairwise(
        items=[item],
        endpoint=_endpoint(),
        journal_path=journal,
        transport=_transport({item.query_first: "A", item.query_second: "B"}),
        progress_every=0,
    )
    report = analyze_calibration(journal)
    assert report["complete_items"] == 1
    assert report["position_flip_counts"] == {}
    assert report["aggregation_curve"]["min_votes_6"]["kept_items"] == 1
    assert report["aggregation_curve"]["min_votes_6"]["precision"] == 1.0
    assert report["per_rubric"]["R3_holistic"]["correct_share"] == 1.0
    assert report["threshold_frozen_here"] is False


def test_concurrent_run_records_every_judgment_exactly_once(tmp_path: Path) -> None:
    """Równoległość nie może zdublować ani zgubić werdyktu, a resume musi ją widzieć."""
    items = [_item(index) for index in range(25)]
    journal = tmp_path / "j.jsonl"
    transport = _transport({items[0].query_first: "A", items[0].query_second: "B"})
    summary = run_pairwise(
        items=items,
        endpoint=_endpoint(),
        journal_path=journal,
        transport=transport,
        concurrency=8,
        progress_every=0,
    )
    expected = 25 * 3 * 2
    assert summary["collected"] == expected
    assert summary["failures"] == 0
    rows = load_journal(journal)
    assert len(rows) == expected
    keys = [json.loads(line)["key"] for line in journal.read_text().splitlines()]
    assert len(keys) == len(set(keys)), "żaden klucz nie może się powtórzyć"
    again = run_pairwise(
        items=items,
        endpoint=_endpoint(),
        journal_path=journal,
        transport=transport,
        concurrency=8,
        progress_every=0,
    )
    assert again["planned_calls"] == 0
    assert again["collected"] == 0
    assert again["resumed"] == expected


def test_partial_run_resumes_only_the_missing_judgments(tmp_path: Path) -> None:
    items = [_item(index) for index in range(4)]
    journal = tmp_path / "j.jsonl"
    transport = _transport({items[0].query_first: "A", items[0].query_second: "B"})
    run_pairwise(
        items=items[:2],
        endpoint=_endpoint(),
        journal_path=journal,
        transport=transport,
        concurrency=4,
        progress_every=0,
    )
    summary = run_pairwise(
        items=items,
        endpoint=_endpoint(),
        journal_path=journal,
        transport=transport,
        concurrency=4,
        progress_every=0,
    )
    assert summary["resumed"] == 2 * 3 * 2
    assert summary["collected"] == 2 * 3 * 2
    assert summary["planned_calls"] == 2 * 3 * 2


def test_run_aborts_when_nothing_succeeds(tmp_path: Path) -> None:
    """Zły model albo nieznane pole API nie może zapisać tysięcy cichych porażek."""

    def broken(payload: Mapping[str, Any]) -> dict[str, Any]:
        raise ValueError("model not found")

    with pytest.raises(RuntimeError, match="ani jedno nie przeszło"):
        run_pairwise(
            items=[_item(index) for index in range(20)],
            endpoint=_endpoint(),
            journal_path=tmp_path / "j.jsonl",
            transport=broken,
            rubrics=["R3_holistic"],
            progress_every=0,
        )


def test_concurrency_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="concurrency"):
        run_pairwise(
            items=[_item()],
            endpoint=_endpoint(),
            journal_path=tmp_path / "j.jsonl",
            transport=_transport({}),
            concurrency=0,
        )


def test_run_is_resumable_and_skips_finished_judgments(tmp_path: Path) -> None:
    item = _item()
    journal = tmp_path / "j.jsonl"
    kwargs: dict[str, Any] = {
        "items": [item],
        "endpoint": _endpoint(),
        "journal_path": journal,
        "transport": _transport({item.query_first: "A", item.query_second: "B"}),
        "progress_every": 0,
    }
    first = run_pairwise(**kwargs)
    second = run_pairwise(**kwargs)
    assert first["collected"] == 6
    assert second["collected"] == 0
    assert second["resumed"] == 6


def test_endpoint_requires_a_key_and_a_full_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    with pytest.raises(ValueError, match="brak klucza API"):
        endpoint_from_args(
            base_url="http://x/v1", api_key=None, api_key_env="QWEN_API_KEY",
            model="m", allow_reasoning=False, max_completion_tokens=64,
        )
    monkeypatch.setenv("QWEN_API_KEY", "z-env")
    assert endpoint_from_args(
        base_url="http://x/v1", api_key=None, api_key_env="QWEN_API_KEY",
        model="m", allow_reasoning=False, max_completion_tokens=64,
    ).api_key == "z-env"
    with pytest.raises(ValueError, match="pełnym adresem"):
        endpoint_from_args(
            base_url="x/v1", api_key="k", api_key_env="QWEN_API_KEY",
            model="m", allow_reasoning=False, max_completion_tokens=64,
        )


def test_server_without_auth_is_allowed_only_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    kwargs: dict[str, Any] = {
        "base_url": "http://127.0.0.1:8000/v1",
        "api_key": None,
        "api_key_env": "QWEN_API_KEY",
        "model": "m",
        "allow_reasoning": False,
        "max_completion_tokens": 64,
    }
    with pytest.raises(ValueError, match="allow-no-auth"):
        endpoint_from_args(**kwargs)
    endpoint = endpoint_from_args(**kwargs, allow_no_auth=True)
    assert endpoint.api_key == ""


def test_plan_summary_reports_the_call_budget() -> None:
    plan = plan_summary([_item(0), _item(1)], list(RUBRICS))
    assert plan["calls"] == 2 * 3 * 2
    # Czasu nie szacujemy z góry: 19,1 it/s zmierzono na sędzim generującym 24 tokeny.
    assert "estimated_minutes_at_19_1_per_second" not in plan
    assert plan["throughput_note"]


# --- obsługa błędów -------------------------------------------------------------


def _failing(status: int | None, body: str, *, retryable: bool) -> Any:
    def call(payload: Mapping[str, Any]) -> dict[str, Any]:
        raise JudgeApiError("odmowa", status=status, body=body, retryable=retryable)

    return call


def test_api_error_keeps_status_and_body_in_its_summary() -> None:
    error = JudgeApiError("x", status=400, body='{"error": "model  not\n found"}', retryable=False)
    assert "HTTP 400" in error.summary()
    assert "model not found" in error.summary()


def test_reasoning_block_with_braces_does_not_break_parsing() -> None:
    """Model rozumujący wstawia nawiasy w toku myślenia; werdykt jest na końcu."""
    content = (
        "<think>Rozważam {A} kontra {B}, chyba {\"better\": \"B\"} nie...</think>\n"
        '{"better": "A", "confidence": 0.7}'
    )
    assert strip_reasoning(content).startswith("{")
    assert parse_verdict(content) == ("A", 0.7)


def test_unterminated_reasoning_block_is_reported_not_guessed() -> None:
    with pytest.raises(ValueError, match="pusta po usunięciu"):
        parse_verdict("<think>myślę i nie kończę")


def test_permanent_api_error_is_not_retried(tmp_path: Path) -> None:
    """Przy 400 cztery próby z backoffem tylko wydłużały ciszę."""
    calls = {"n": 0}

    def call(payload: Mapping[str, Any]) -> dict[str, Any]:
        calls["n"] += 1
        raise JudgeApiError("zły model", status=400, body="model not found", retryable=False)

    row = run_pairwise(
        items=[_item()],
        endpoint=_endpoint(),
        journal_path=tmp_path / "j.jsonl",
        transport=call,
        rubrics=["R3_holistic"],
        orders=("ab",),
        progress_every=0,
    )
    assert calls["n"] == 1, "błąd trwały nie może być ponawiany"
    assert row["failures"] == 1
    assert "model not found" in json.loads((tmp_path / "j.jsonl").read_text())["reason"]


def test_rejected_optional_field_is_dropped_and_the_call_retried(tmp_path: Path) -> None:
    """Serwer bez `chat_template_kwargs` nie może wywalić całego runu."""
    seen: list[bool] = []

    def call(payload: Mapping[str, Any]) -> dict[str, Any]:
        has_field = "chat_template_kwargs" in payload
        seen.append(has_field)
        if has_field:
            raise JudgeApiError(
                "nieznane pole",
                status=400,
                body="unrecognized request argument: chat_template_kwargs",
                retryable=False,
            )
        return {"choices": [{"message": {"content": '{"better": "A", "confidence": 1.0}'}}]}

    summary = run_pairwise(
        items=[_item()],
        endpoint=_endpoint(),
        journal_path=tmp_path / "j.jsonl",
        transport=call,
        rubrics=["R3_holistic"],
        orders=("ab",),
        progress_every=0,
    )
    assert seen == [True, False]
    assert summary["collected"] == 1
    row = json.loads((tmp_path / "j.jsonl").read_text().splitlines()[0])
    assert row["dropped_payload_fields"] == ["chat_template_kwargs"]


def test_response_without_choices_or_content_is_a_named_failure(tmp_path: Path) -> None:
    cases: list[tuple[dict[str, Any], str]] = [
        ({"choices": []}, "bez pola choices"),
        ({"choices": [{"message": {}}]}, "bez treści"),
    ]
    for response, expected in cases:

        def call(payload: Mapping[str, Any], reply: dict[str, Any] = response) -> dict[str, Any]:
            return dict(reply)

        journal = tmp_path / f"j{expected[:5]}.jsonl"
        run_pairwise(
            items=[_item()],
            endpoint=_endpoint(),
            journal_path=journal,
            transport=call,
            rubrics=["R3_holistic"],
            orders=("ab",),
            progress_every=0,
        )
        assert expected in json.loads(journal.read_text())["reason"]


def test_probe_surfaces_the_server_error_body() -> None:
    with pytest.raises(JudgeApiError) as caught:
        probe_endpoint(_endpoint(), _failing(404, "model 'zly' does not exist", retryable=False))
    assert "does not exist" in caught.value.summary()


def test_probe_returns_a_verdict_when_the_server_answers() -> None:
    def call(payload: Mapping[str, Any]) -> dict[str, Any]:
        return {"choices": [{"message": {"content": '{"better": "A", "confidence": 0.9}'}}]}

    probe = probe_endpoint(_endpoint(), call)
    assert probe["ok"] is True
    assert probe["verdict"] == "A"
    assert probe["dropped_payload_fields"] == []


def test_curve_separates_precision_from_yield(tmp_path: Path) -> None:
    """Filtr trzymający pary, w których sędzia myli kierunek, musi to pokazać."""
    journal = tmp_path / "j.jsonl"
    lines = []
    # Dwie pary spójnie poprawne, jedna spójnie ODWROTNA, jedna z position_flip.
    plan = {
        "ok-1": ("A", "A"),
        "ok-2": ("A", "A"),
        "zla": ("B", "B"),
        "flip": ("A", "B"),
    }
    for item_id, (forward, reverse) in plan.items():
        for rubric in ("R1_grounding", "R2_retrieval_usefulness", "R3_holistic"):
            for order, verdict in (("ab", forward), ("ba", reverse)):
                lines.append(
                    json.dumps(
                        {
                            "event": "judgment",
                            "key": f"{item_id}|{rubric}|{order}",
                            "item_id": item_id,
                            "rubric": rubric,
                            "order": order,
                            "canonical_verdict": verdict,
                        }
                    )
                )
    journal.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = analyze_calibration(journal)
    assert report["complete_items"] == 4
    six = report["aggregation_curve"]["min_votes_6"]
    # Zatrzymane: dwie poprawne i jedna odwrotna; para z flipem nie ma żadnych głosów.
    assert six["kept_items"] == 3
    assert six["precision"] == pytest.approx(2 / 3)
    assert six["kept_with_wrong_direction"] == 1
    assert six["yield"] == pytest.approx(3 / 4)
    assert report["position_flip_counts"]["R3_holistic"] == 1
    assert report["achievable_thresholds"] == [2, 4, 6]
