#!/usr/bin/env python3
"""Złóż pary z wadami z verdictów serwera i uruchom bramkę anty-skrótową.

Lokalny, deterministyczny etap po przeniesieniu `verdicts/` z maszyny z serwerem
inferencji. Nie woła żadnego modelu i nie autoryzuje treningu.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from doc2query.preferences.defect_pairs_v1 import assemble_defect_pairs, shortcut_audit
from doc2query.utils.records import read_records, write_json


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
        "--v3-pairs",
        type=Path,
        default=Path("artifacts/task06/v3_pairs_v1/bottom/pairs.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/task06/defect_pairs_v1"),
    )
    args = parser.parse_args()

    summary = assemble_defect_pairs(
        groups_path=args.groups,
        journal_path=args.journal,
        v3_pairs_path=args.v3_pairs,
        output_dir=args.output_dir,
    )
    pairs = [dict(row) for row in read_records(args.output_dir / "pairs.jsonl")]
    audit = shortcut_audit(pairs)
    per_class = {}
    for defect in sorted({str(row["defect_class"]) for row in pairs}):
        subset = [row for row in pairs if row["defect_class"] == defect]
        per_class[defect] = {"pairs": len(subset), **shortcut_audit(subset)}
    audit_payload = {
        "schema_version": 1,
        "contract": "task06-defect-pairs-shortcut-audit-v1",
        "overall": audit,
        "per_defect_class": per_class,
        "note": (
            "AUC powyżej progu blokuje klasę do czasu amendmentu (ADR §7.2); audyt nie zmienia par."
        ),
        "final_tests_used": [],
    }
    write_json(args.output_dir / "shortcut_audit.json", audit_payload)
    print(json.dumps({"summary": summary, "audit": audit_payload}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
