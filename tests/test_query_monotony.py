from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from doc2query.evaluation.query_monotony import (
    MONOTONY_CONTRACT,
    accumulate_cohort,
    distinct_n,
    first_word,
    run_monotony_baseline,
    tokenize,
)


def _row(query: str, group: str, **control: Any) -> dict[str, Any]:
    base = {"form": "full_question", "length": "medium", "intent": "fact_lookup"}
    base.update(control)
    return {
        "generated": query,
        "evaluation_id": f"{group}::{query}",
        "evaluation_group_id": group,
        "word_length": float(len(query.split())),
        "character_length": float(len(query)),
        "control": base,
    }


def test_tokenizer_is_lowercased_and_punctuation_free() -> None:
    assert tokenize("Jak wygodne są gogle UVEX?") == ["jak", "wygodne", "są", "gogle", "uvex"]
    assert first_word("  Kiedy powstał Rzym?") == "kiedy"
    assert first_word("???") is None


def test_distinct_n_measures_set_level_repetition() -> None:
    identical = ["jak to działa", "jak to działa"]
    varied = ["jak to działa", "kiedy powstał rzym"]

    assert distinct_n(identical, 1) == pytest.approx(0.5)
    assert distinct_n(varied, 1) == pytest.approx(1.0)
    assert distinct_n(identical, 2) == pytest.approx(0.5)
    assert distinct_n(["jedno"], 2) is None


def test_first_word_concentration_detects_a_single_opening_word() -> None:
    rows = [_row(f"jak numer {index} działa", f"g{index}") for index in range(4)]

    report = accumulate_cohort(rows)
    concentration = report["pooled"]["first_word"]

    assert concentration["distinct"] == 1
    assert concentration["top1_share"] == pytest.approx(1.0)
    # Jedno słowo początkowe to zerowa entropia, czyli maksymalna monotonia.
    assert concentration["normalized_entropy"] == pytest.approx(0.0)


def test_uniform_openings_reach_full_normalized_entropy() -> None:
    rows = [
        _row("jak to działa", "g1"),
        _row("kiedy to powstało", "g2"),
        _row("gdzie to leży", "g3"),
        _row("czym to jest", "g4"),
    ]

    concentration = accumulate_cohort(rows)["pooled"]["first_word"]

    assert concentration["distinct"] == 4
    assert concentration["normalized_entropy"] == pytest.approx(1.0)


def test_slices_follow_the_frozen_controls() -> None:
    rows = [
        _row("jak to działa", "g1", form="full_question", length="short"),
        _row("co to jest rzym i kiedy powstał", "g2", form="keyword_query", length="long"),
    ]

    report = accumulate_cohort(rows)

    assert set(report["by_control"]["form"]) == {"full_question", "keyword_query"}
    assert report["by_control"]["length"]["short"]["word_length"]["p50"] == pytest.approx(3.0)
    assert report["by_control"]["length"]["long"]["word_length"]["p50"] == pytest.approx(7.0)


def test_group_level_diversity_is_reported_per_group() -> None:
    rows = [
        _row("jak to działa", "g1"),
        _row("jak to działa", "g1"),
        _row("kiedy powstał rzym", "g2"),
    ]

    set_level = accumulate_cohort(rows)["set_level_per_group"]

    assert set_level["groups"] == 2
    assert set_level["distinct_1"]["count"] == 2
    assert set_level["distinct_1"]["min"] == pytest.approx(0.5)


def test_missing_control_block_is_refused() -> None:
    row = _row("jak to działa", "g1")
    del row["control"]

    with pytest.raises(ValueError, match="has no control block"):
        accumulate_cohort([row])


def _write_cohort(directory: Path, queries: list[str]) -> None:
    scoring = directory / "d01_controlled" / "scoring"
    scoring.mkdir(parents=True)
    rows = [_row(query, f"g{index}") for index, query in enumerate(queries)]
    (scoring / "per_generation.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def test_baseline_publishes_a_pinned_artifact_that_freezes_nothing(tmp_path: Path) -> None:
    cohort = tmp_path / "same_prompt_expansion_v1"
    _write_cohort(cohort, ["jak to działa", "kiedy powstał rzym"])
    output = tmp_path / "out" / "summary.json"

    report = run_monotony_baseline(cohort_dirs=[cohort], output_path=output)

    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert report["contract"] == MONOTONY_CONTRACT
    assert report["thresholds_frozen_here"] is False
    assert report["pairs_built"] is False
    assert report["final_tests_used"] == []
    assert len(report["inputs_sha256"]["same_prompt_expansion_v1"]) == 64


def test_missing_cohort_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing scored cohort"):
        run_monotony_baseline(cohort_dirs=[tmp_path / "nope"], output_path=tmp_path / "out.json")
