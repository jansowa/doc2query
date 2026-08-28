"""Testy ślepego spot-checku właściciela: arkusz nie może zdradzać strony.

Kontrola jest operacyjna, nie panelowa, ale jej jedyna wartość leży w ślepości —
gdyby arkusz ujawniał, którą stronę wybrał selektor, pomiar zgodności byłby
pomiarem czytania klucza. Dlatego testy pilnują ślepości, deterministycznego
losowania stron i tego, że scoring nie przemyca żadnego progu.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from doc2query.utils.records import JsonlWriter

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "task07_owner_spot_check.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("task07_owner_spot_check", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script() -> ModuleType:
    return _load()


def _pairs(path: Path, count: int) -> None:
    with JsonlWriter(path) as writer:
        for index in range(count):
            writer.write(
                {
                    "preference_id": f"pair-{index:03d}",
                    "prompt": (
                        "Wygeneruj jedno polskie zapytanie wyszukiwawcze.\n\n"
                        f"Pasaż:\nPasaż numer {index} o kotach.\n\nZapytanie:\n"
                    ),
                    "chosen": f"ile kotów opisuje pasaż {index}",
                    "rejected": f"czy pasaż {index} to legenda o psach",
                }
            )


def _args(script: ModuleType, mode: str, tmp_path: Path, count: int = 6) -> argparse.Namespace:
    pairs = tmp_path / "pairs.jsonl"
    if mode == "export":
        _pairs(pairs, count * 2)
    return argparse.Namespace(
        mode=mode,
        pairs=[pairs],
        count=count,
        seed=7,
        output_dir=tmp_path / "spot_check",
    )


def _key(directory: Path) -> dict[str, Any]:
    payload = json.loads((directory / "spot_check_key.json").read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_sheet_is_blind(script: ModuleType, tmp_path: Path) -> None:
    args = _args(script, "export", tmp_path)
    script._export(args)
    sheet = (args.output_dir / "spot_check_sheet.md").read_text(encoding="utf-8")
    # Ani etykiet stron, ani identyfikatorów par: arkusz musi być nieodgadywalny.
    for leak in ("chosen", "rejected", "pair-0", "preference_id", "margin", "vote"):
        assert leak not in sheet, f"arkusz zdradza {leak}"
    assert sheet.count("- **A:**") == args.count
    assert sheet.count("- **B:**") == args.count


def test_side_assignment_is_deterministic_and_order_free(script: ModuleType) -> None:
    first = [script._side_for_a(f"pair-{index}", 7) for index in range(50)]
    second = [script._side_for_a(f"pair-{index}", 7) for index in reversed(range(50))]
    assert first == list(reversed(second))
    assert set(first) == {"chosen", "rejected"}, "obie strony muszą trafiać na A"


def test_score_counts_agreement_against_hidden_key(script: ModuleType, tmp_path: Path) -> None:
    args = _args(script, "export", tmp_path)
    script._export(args)
    key = _key(args.output_dir)
    lines = []
    for position, row in enumerate(key["items"]):
        correct = str(row["chosen_letter"])
        if position == 0:
            answer = "B" if correct == "A" else "A"
        elif position == 1:
            answer = "="
        else:
            answer = correct
        lines.append(f"{row['item']} {answer}")
    (args.output_dir / "spot_check_answers.txt").write_text(
        "# odpowiedzi\n" + "\n".join(lines) + "\n", encoding="utf-8"
    )
    script._score(argparse.Namespace(**{**vars(args), "mode": "score"}))
    result = json.loads((args.output_dir / "spot_check_result.json").read_text(encoding="utf-8"))
    assert result["agreements"] == args.count - 2
    assert result["disagreements"] == 1
    assert result["ties"] == 1
    assert result["decided"] == args.count - 1
    low, high = result["agreement_ci95"]
    assert 0.0 <= low <= result["agreement_rate"] <= high <= 1.0
    assert result["no_frozen_threshold"] is True
    assert result["role"] == "operational_sanity_check_not_panel_9_3"


def test_score_refuses_unreadable_and_duplicated_answers(
    script: ModuleType, tmp_path: Path
) -> None:
    args = _args(script, "export", tmp_path)
    script._export(args)
    answers = args.output_dir / "spot_check_answers.txt"
    score_args = argparse.Namespace(**{**vars(args), "mode": "score"})
    answers.write_text("1 C\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        script._score(score_args)
    answers.write_text("1 A\n1 B\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        script._score(score_args)
    answers.write_text(f"{args.count + 99} A\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        script._score(score_args)
