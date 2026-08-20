"""Testy samodzielnego skryptu operatorskiego (ładowany po ścieżce, nie z paczki).

Pokrywają dokładnie te dwie własności, na których operatorowi zależy: wznawianie
(journal odporny na ucięcie ostatniej linii) i uczciwy postęp/ETA.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "task06_judge_remote.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("task06_judge_remote", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script() -> ModuleType:
    return _load()


def test_journal_tolerates_a_truncated_final_line(script: ModuleType, tmp_path: Path) -> None:
    """Zabicie procesu w trakcie zapisu nie może unieważnić całego wznowienia."""
    journal = tmp_path / "verdicts.jsonl"
    good = json.dumps({"event": "verdict", "item_id": "a", "verdict": "yes"})
    journal.write_text(good + "\n" + '{"event": "verdict", "item_i', encoding="utf-8")

    done = script.load_done(str(journal))

    assert set(done) == {"a"}


def test_lanes_keep_a_passage_inside_one_lane(script: ModuleType) -> None:
    items = [
        {"item_id": f"i{index}", "query": f"q{index}", "passage": f"p{index % 3}"}
        for index in range(12)
    ]

    buckets = script.lanes_by_passage(items, 3)

    assert sum(len(bucket) for bucket in buckets) == 12
    for bucket in buckets:
        passages = {item["passage"] for item in bucket}
        assert len(passages) <= 1  # pasaż nigdy nie jest dzielony między pasy
    assert script.lanes_by_passage(items, 3) == buckets  # deterministyczne


def test_progress_counts_resumed_work_but_rates_only_this_run(script: ModuleType) -> None:
    progress = script.Progress(total=100, already_done=40, stream=None)
    for _ in range(10):
        progress.counts["yes"] += 1

    assert progress.judged_now == 10
    # Pasek pokazuje 50/100 wobec całego pakietu, ale tempo liczy z 10 zrobionych teraz.
    assert progress.already_done + progress.judged_now == 50


def test_duration_formatting_switches_to_hours(script: ModuleType) -> None:
    assert script.format_duration(59) == "0:59"
    assert script.format_duration(605) == "10:05"
    assert script.format_duration(3725) == "1:02:05"


def test_verdict_parsing_refuses_anything_outside_the_schema(script: ModuleType) -> None:
    assert script.parse_verdict('{"verdict": "YES"}') == "yes"
    for bad in ('{"verdict": "maybe"}', "[]", '{"other": "yes"}'):
        with pytest.raises(ValueError):
            script.parse_verdict(bad)


def test_batches_never_mix_passages_and_respect_the_cap(script: ModuleType) -> None:
    items = [{"item_id": f"i{n}", "query": f"q{n}", "passage": "P1"} for n in range(9)]
    items += [{"item_id": f"j{n}", "query": f"r{n}", "passage": "P2"} for n in range(2)]

    batches = script.batches_in_lane(items, 4)

    assert [len(batch) for batch in batches] == [4, 4, 1, 2]
    for batch in batches:
        assert len({item["passage"] for item in batch}) == 1


def test_batch_payload_sends_the_passage_once_with_local_ids(script: ModuleType) -> None:
    from types import SimpleNamespace

    batch = [
        {"item_id": "a", "query": "pierwsze", "passage": "PASAZ"},
        {"item_id": "b", "query": "drugie", "passage": "PASAZ"},
    ]
    args = SimpleNamespace(
        model="m",
        seed=1,
        max_tokens=24,
        decoding="json_schema_enum",
        tokens_per_verdict=40,
        batch_token_overhead=48,
    )

    payload = script.build_batch_payload(batch, args)
    user = json.loads(payload["messages"][1]["content"])

    assert user["passage"] == "PASAZ"  # pasaż raz, nie N razy
    assert user["queries"] == [{"id": 1, "query": "pierwsze"}, {"id": 2, "query": "drugie"}]
    assert payload["messages"][1]["content"].index('"passage"') < payload["messages"][1][
        "content"
    ].index('"queries"')  # pasaż przed zapytaniami => wspólny prefiks
    schema = payload["response_format"]["json_schema"]["schema"]["properties"]["verdicts"]
    assert schema["minItems"] == schema["maxItems"] == 2
    # Budzet liczony z argumentow, nie ze stalej: 40*2 + 48
    assert payload["max_tokens"] == 128
    # Po truncation ponowienie ma dostac podwojony budzet.
    assert script.batch_token_budget(2, args, multiplier=2) == 256


def test_batch_parsing_ignores_order_and_refuses_broken_contracts(script: ModuleType) -> None:
    shuffled = json.dumps({"verdicts": [{"id": 2, "verdict": "no"}, {"id": 1, "verdict": "yes"}]})

    assert script.parse_batch_verdicts(shuffled, [1, 2]) == {1: "yes", 2: "no"}

    for broken, reason in (
        ('{"verdicts": [{"id": 1, "verdict": "yes"}]}', "id nie zgadza"),
        ('{"verdicts": [{"id": 1, "verdict": "yes"}, {"id": 1, "verdict": "no"}]}', "zduplikowane"),
        (
            '{"verdicts": [{"id": 1, "verdict": "verdict"}, {"id": 2, "verdict": "no"}]}',
            "schematem",
        ),
        ('{"inne": []}', "brak tablicy"),
    ):
        with pytest.raises(script.BatchError, match=reason):
            script.parse_batch_verdicts(broken, [1, 2])


def test_single_and_batched_prompts_have_distinct_pinned_hashes(script: ModuleType) -> None:
    assert (
        script.PROMPT_SHA256[script.PROMPT_VERSION_SINGLE]
        != (script.PROMPT_SHA256[script.PROMPT_VERSION_BATCHED])
    )
    assert script.EXPECTED_SYSTEM_PROMPT_SHA256 == (
        "74d3ee07757decbdf5655e1878c070f66bf05c90a60bf4dc5f56b1c520cfee84"
    )
