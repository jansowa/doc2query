#!/usr/bin/env python3
"""Zbuduj paczkę wejściową czterech zadań nocnych dla serwera inferencji.

Każde zadanie dostaje własny plik z jednolitym polem `id`, żeby journal runnera
miał stabilny klucz. Nic tu nie jest generowane ani oceniane — to przepakowanie
istniejących artefaktów.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from doc2query.preferences.defect_pairs_v1 import load_journal
from doc2query.utils.records import JsonlWriter, read_records, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--groups",
        type=Path,
        default=Path("artifacts/task06/defect_pipeline_v1/input/groups.jsonl"),
    )
    parser.add_argument(
        "--journal",
        type=Path,
        default=Path("artifacts/task06/defect_pipeline_v1/verdicts/verdicts.journal.jsonl"),
    )
    parser.add_argument(
        "--lexical-worklist",
        type=Path,
        default=Path("artifacts/task06/lexical_contrast_v1/worklist.jsonl"),
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        default=Path("artifacts/task06/defect_pairs_v1/pairs_trainable.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/task06/night_jobs_v1/input"),
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"wyjście już istnieje: {args.output_dir}")

    args.output_dir.mkdir(parents=True)
    journal = load_journal(args.journal)
    groups = {str(row["group_id"]): dict(row) for row in read_records(args.groups)}
    counts: dict[str, int] = {}

    # 1. lexical_contrast — grupy bez organicznego negatywu o wysokim pokryciu.
    with JsonlWriter(args.output_dir / "lexical_worklist.jsonl") as writer:
        for row in read_records(args.lexical_worklist):
            writer.write(
                {
                    "id": str(row["group_id"]),
                    "passage": str(row["passage"]),
                    "chosen": str(row["chosen"]["query"]),
                    "form": str(row["form"]),
                    "preference_id": str(row["preference_id"]),
                }
            )
            counts["lexical_mutation"] = counts.get("lexical_mutation", 0) + 1

    # 2. label_purity — wszystkie pary kohorty trenowalnej.
    with JsonlWriter(args.output_dir / "pairs_to_verify.jsonl") as writer:
        for row in read_records(args.pairs):
            writer.write(
                {
                    "id": str(row["pair_id"]),
                    "passage": str(row["passage"]),
                    "chosen": str(row["chosen"]),
                    "rejected": str(row["rejected"]),
                    "defect_class": str(row["defect_class"]),
                }
            )
            counts["label_purity"] = counts.get("label_purity", 0) + 1

    # 3. answer_leak_v2 — grupy, które przeszły answerability chosen.
    # 4. chosen_recheck — grupy odrzucone na answerability chosen.
    with (
        JsonlWriter(args.output_dir / "answer_leak_groups.jsonl") as leak,
        JsonlWriter(args.output_dir / "dropped_groups.jsonl") as dropped,
    ):
        for gid, group in sorted(groups.items()):
            row = journal.get(f"{gid}::answerable::chosen")
            if row is None:
                continue
            payload: dict[str, Any] = {
                "id": gid,
                "passage": str(group["passage"]),
                "chosen": str(group["chosen"]["query"]),
                "form": str(group["form"]),
                "preference_id": str(group["preference_id"]),
            }
            if bool(row.get("verdict", {}).get("answerable")):
                leak.write(payload)
                counts["answer_leak_v2"] = counts.get("answer_leak_v2", 0) + 1
            else:
                dropped.write(payload)
                counts["chosen_recheck"] = counts.get("chosen_recheck", 0) + 1

    summary = {
        "schema_version": 1,
        "contract": "task06-night-jobs-input-v1",
        "adr": "reports/decisions/task06_defect_pair_pipeline_v1.md",
        "items_per_job": dict(sorted(counts.items())),
        "estimated_calls": (
            counts.get("lexical_mutation", 0) * 2
            + counts.get("label_purity", 0)
            + counts.get("answer_leak_v2", 0) * 2
            + counts.get("chosen_recheck", 0)
        ),
        "pairs_built": 0,
        "final_tests_used": [],
    }
    write_json(args.output_dir / "manifest.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
