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
