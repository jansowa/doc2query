from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from doc2query.evaluation.translation_audit import STRATA, freeze_translation_audit
from doc2query.utils.records import JsonlWriter


def _record(index: int, *, flagged: bool = False) -> dict[str, Any]:
    return {
        "example_id": f"q-{index:03d}",
        "query": "ile kosztuje produkt" + (" Ã" if flagged else ""),
        "positives": [
            {
                "doc_id": f"d-{index:03d}",
                "text": "Produkt kosztuje sto złotych." + (" Ë" if flagged else ""),
                "metadata": {
                    "source_en_score": 23.5 + index / 10,
                    "source_score_language": "en",
                    "text_quality_flags": ["possible_mojibake"] if flagged else [],
                },
            }
        ],
        "hard_negatives": [],
        "metadata": {
            "source": "fixture",
            "source_revision": "a" * 40,
            "split": "train",
            "query_text_quality_flags": ["possible_mojibake"] if flagged else [],
            "source_en_difference_between_max_scores": 6 + ((index + 40) % 80) / 10,
        },
    }


def _write_input(path: Path) -> None:
    with JsonlWriter(path) as writer:
        for index in range(80):
            writer.write(_record(index, flagged=20 <= index < 28))


def test_freeze_translation_audit_is_disjoint_blind_and_deterministic(tmp_path: Path) -> None:
    input_path = tmp_path / "train.jsonl"
    _write_input(input_path)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = freeze_translation_audit(input_path, output_dir=first_dir, seed=42, stratum_size=5)
    second = freeze_translation_audit(input_path, output_dir=second_dir, seed=42, stratum_size=5)

    assert first["sample_count"] == 20
    assert first["stratum_counts"] == {stratum: 5 for stratum in STRATA}
    assert first["selected_ids_sha256"] == second["selected_ids_sha256"]
    selected = first["selected_records"]
    assert len({row["example_id"] for row in selected}) == 20
    assert first["final_tests_used"] == []
    assert first["diagnostic_policy"]["drop_threshold_defined"] is False

    with (first_dir / "blind_review_form.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 20
    assert (first_dir / "review_instructions.md").is_file()
    forbidden = {
        "example_id",
        "doc_id",
        "stratum",
        "source_en_positive_score",
        "source_en_margin",
        "surface_risk",
        "primary_margin",
        "shadow_margin",
    }
    assert forbidden.isdisjoint(rows[0])

    diagnostics = [
        json.loads(line)
        for line in (first_dir / "triage_diagnostics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(diagnostics) == 20
    assert all(row["primary_margin"] is None for row in diagnostics)
    assert all(row["shadow_margin"] is None for row in diagnostics)


def test_quality_stratum_fills_from_surface_risk_when_flags_are_scarce(tmp_path: Path) -> None:
    input_path = tmp_path / "train.jsonl"
    with JsonlWriter(input_path) as writer:
        for index in range(80):
            writer.write(_record(index, flagged=index == 30))
    manifest = freeze_translation_audit(
        input_path, output_dir=tmp_path / "audit", seed=42, stratum_size=5
    )
    assert manifest["quality_stratum"] == {
        "flagged_selected": 1,
        "surface_risk_fill_selected": 4,
    }
