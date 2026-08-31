#!/usr/bin/env python3
"""Paczka v3 dla serwera: potwierdzenia par bez S4 i powtórka osi językowej.

Trzy źródła par do potwierdzenia (wszystkie przeszły filtry deterministyczne
lokalnie, tutaj; ADR §5 wymaga jeszcze jednomyślnego potwierdzenia preferencji
w obu kolejnościach, którego nocne zadania generacyjne nie wykonywały):

1. `class_backfill` — mutacje dogenerowane nocą do grup bez danej klasy;
   filtry §4 + wymogi klasowe §2 liczone tutaj z werdyktów answerability.
2. mutacje `lexical_contrast` — 222 pary zmontowane w night_results.
3. `wrong_form` — przepisania formy zweryfikowane regexem: wynik musi łamać
   formę oryginału (to jest wada) i zachować słownictwo (Jaccard ≥0,3).

Do tego `polish_flagged`: 1 388 par oflagowanych na osi językowej, do powtórnej
oceny ostrzejszym promptem (przegląd wykazał zawyżanie przez anglicyzmy).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from doc2query.preferences.defect_pairs_v1 import (
    class_reject,
    deterministic_reject,
    form_violation,
    jaccard,
    load_journal,
)
from doc2query.utils.records import JsonlWriter, read_records, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--night-journal",
        type=Path,
        default=Path("artifacts/task06/night_jobs_v1/verdicts/night_jobs.journal.jsonl"),
    )
    parser.add_argument(
        "--night-input",
        type=Path,
        default=Path("artifacts/task06/night_jobs_v1/input"),
    )
    parser.add_argument(
        "--groups",
        type=Path,
        default=Path("artifacts/task06/defect_pipeline_v1/input/groups.jsonl"),
    )
    parser.add_argument(
        "--lexical-pairs",
        type=Path,
        default=Path("artifacts/task06/night_results_v1/lexical_contrast_pairs.jsonl"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/task06/night_jobs_v3/input")
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"wyjście już istnieje: {args.output_dir}")

    journal = load_journal(args.night_journal)
    groups = {str(row["group_id"]): dict(row) for row in read_records(args.groups)}
    args.output_dir.mkdir(parents=True)
    counts: dict[str, int] = {}

    def bump(name: str) -> None:
        counts[name] = counts.get(name, 0) + 1

    with JsonlWriter(args.output_dir / "pairs_to_confirm.jsonl") as confirm:
        # 1. class_backfill: filtry deterministyczne + wymogi klasowe z werdyktów.
        for row in read_records(args.night_input / "class_backfill.jsonl"):
            entry = journal.get(f"class_backfill::{row['id']}")
            group = groups.get(str(row["group_id"]))
            if entry is None or group is None:
                bump("backfill_no_verdict")
                continue
            query = str(entry.get("verdict", {}).get("query", "")).strip()
            chosen = str(row["chosen"])
            if not query:
                bump("backfill_empty")
                continue
            if deterministic_reject(query, chosen, str(row["form"])) is not None:
                bump("backfill_filtered")
                continue
            answerable = entry.get("answerable_check", {}).get("answerable")
            if not isinstance(answerable, bool):
                bump("backfill_no_answerability")
                continue
            defect = str(row["defect_class"])
            if class_reject(
                defect, query=query, passage=str(row["passage"]), answerable=answerable
            ):
                bump("backfill_class_reject")
                continue
            confirm.write(
                {
                    "id": f"backfill::{row['id']}",
                    "passage": str(row["passage"]),
                    "chosen": chosen,
                    "rejected": query,
                    "defect_class": defect,
                    "source": "class_backfill",
                    "group_id": str(row["group_id"]),
                }
            )
            bump("confirm_backfill")

        # 2. mutacje lexical_contrast bez S4.
        if args.lexical_pairs.is_file():
            for row in read_records(args.lexical_pairs):
                if str(row.get("negative_population")) != "mutated_synthetic":
                    continue
                confirm.write(
                    {
                        "id": f"lexical::{row['group_id']}",
                        "passage": str(row["passage"]),
                        "chosen": str(row["chosen"]),
                        "rejected": str(row["rejected"]),
                        "defect_class": "not_answerable",
                        "source": "lexical_mutated",
                        "group_id": str(row["group_id"]),
                    }
                )
                bump("confirm_lexical")

        # 3. wrong_form: regex — wynik musi łamać formę oryginału i dzielić słownictwo.
        for row in read_records(args.night_input / "answer_leak_groups.jsonl"):
            entry = journal.get(f"wrong_form::{row['id']}")
            group = groups.get(str(row["id"]))
            if entry is None or group is None:
                continue
            query = str(entry.get("verdict", {}).get("query", "")).strip()
            chosen = str(row["chosen"])
            form = str(row["form"])
            if not query:
                bump("wrongform_empty")
                continue
            # Wada obecna = wynik ma formę PRZECIWNĄ do kontraktu grupy.
            if not form_violation(query, form):
                bump("wrongform_no_violation")
                continue
            if jaccard(query, chosen) < 0.3:
                bump("wrongform_content_lost")
                continue
            confirm.write(
                {
                    "id": f"wrongform::{row['id']}",
                    "passage": str(row["passage"]),
                    "chosen": chosen,
                    "rejected": query,
                    "defect_class": "wrong_form",
                    "source": "wrong_form",
                    "group_id": str(row["id"]),
                }
            )
            bump("confirm_wrongform")

    # 4. powtórka osi językowej.
    sample = {
        str(row["id"]): dict(row) for row in read_records(args.night_input / "sft_sample.jsonl")
    }
    with JsonlWriter(args.output_dir / "polish_flagged.jsonl") as flagged:
        for pid, item in sorted(sample.items()):
            entry = journal.get(f"sft_data_audit::{pid}")
            if entry is None:
                continue
            if not bool(entry.get("verdict", {}).get("polszczyzna", True)):
                flagged.write({"id": pid, "query": str(item["query"])})
                bump("polish_recheck")

    summary = {
        "schema_version": 1,
        "contract": "task06-night-jobs-v3-input-v1",
        "items_per_job": dict(sorted(counts.items())),
        "estimated_calls": (
            2
            * (
                counts.get("confirm_backfill", 0)
                + counts.get("confirm_lexical", 0)
                + counts.get("confirm_wrongform", 0)
            )
            + counts.get("polish_recheck", 0)
        ),
        "pairs_built": 0,
        "final_tests_used": [],
    }
    write_json(args.output_dir / "manifest.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
