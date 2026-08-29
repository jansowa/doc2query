#!/usr/bin/env python3
"""Odłóż klasy par, które oblały audyt anty-skrótowy (ADR §7.2).

Audyt jest bramką zamrożoną **przed** pomiarem: klasa z AUC powyżej progu nie
wchodzi do treningu, dopóki nie powstanie amendment. Ten skrypt nie zmienia
progu ani nie ocenia wyniku — czyta `shortcut_audit.json` i wypisuje kohortę
trenowalną obok pełnego zbioru par, który zostaje nietknięty jako pomiar.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from doc2query.utils.records import JsonlWriter, read_records, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-dir", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads((args.pairs_dir / "shortcut_audit.json").read_text(encoding="utf-8"))
    threshold = float(audit["overall"].get("threshold_blocking", 0.80))
    blocked = sorted(
        name
        for name, stats in audit["per_defect_class"].items()
        if stats.get("auc") is not None and float(stats["auc"]) > threshold
    )
    kept = [
        dict(row)
        for row in read_records(args.pairs_dir / "pairs.jsonl")
        if str(row["defect_class"]) not in blocked
    ]
    output = args.pairs_dir / "pairs_trainable.jsonl"
    with JsonlWriter(output) as writer:
        for row in kept:
            writer.write(row)
    summary = {
        "schema_version": 1,
        "contract": "task06-defect-pairs-trainable-v1",
        "adr": "reports/decisions/task06_defect_pair_pipeline_v1.md",
        "threshold_blocking": threshold,
        "blocked_classes": blocked,
        "blocked_class_auc": {name: audit["per_defect_class"][name]["auc"] for name in blocked},
        "pairs_total": sum(1 for _ in read_records(args.pairs_dir / "pairs.jsonl")),
        "pairs_trainable": len(kept),
        "task07_training_authorized": False,
        "final_tests_used": [],
    }
    write_json(args.pairs_dir / "trainable_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
