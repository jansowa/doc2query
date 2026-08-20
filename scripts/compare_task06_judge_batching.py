#!/usr/bin/env python3
"""Bramka A/B paczkowania sędziego: porównaj journal pojedynczy z paczkowym.

Baseline to przyrząd, na którym zamrożono kryteria K1-K3 (jedno zapytanie na request),
kandydat to wariant paczkowy. Progi (zgodność ≥ 0,98, brak istotnego dryfu) są zamrożone
w amendmencie o paczkowaniu **przed** tym uruchomieniem. Raport idzie do pliku, nie
tylko na stdout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from doc2query.preferences.answerability_judge import (
    calibration_items_from_audit,
    calibration_items_from_reward_corpus,
)
from doc2query.preferences.answerability_remote import (
    candidate_pool_items,
    compare_journal_verdicts,
    journal_provenance,
    load_remote_journal,
    render_ab_report,
)
from doc2query.utils.records import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-journal", type=Path, required=True)
    parser.add_argument("--candidate-journal", type=Path, required=True)
    parser.add_argument(
        "--packet-dir", type=Path, default=Path("artifacts/task06/answerability_packet_v1")
    )
    parser.add_argument(
        "--items-from",
        choices=("calibration", "pool"),
        default="calibration",
        help="Skad wziac liste itemow do walidacji journali.",
    )
    parser.add_argument(
        "--export-dir", type=Path, default=Path("artifacts/task06/preference_audit_v2")
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("artifacts/task06/reward_validation_corpus_v1/corpus.jsonl"),
    )
    parser.add_argument(
        "--cohort-records",
        type=Path,
        default=Path("artifacts/task06/candidate_pilot_v1/cohort.records.jsonl"),
    )
    parser.add_argument("--cohort-dir", type=Path, action="append")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/measurements/task06/judge_batching_ab_v1"),
    )
    args = parser.parse_args()

    manifest = json.loads((args.packet_dir / "manifest.json").read_text(encoding="utf-8"))
    if args.items_from == "calibration":
        items = calibration_items_from_audit(args.export_dir)
        items += calibration_items_from_reward_corpus(args.corpus, args.cohort_records)
        unique = {item.item_id: item for item in items}
        ordered = [unique[item_id] for item_id in sorted(unique)]
    else:
        cohorts = args.cohort_dir or [
            Path(f"artifacts/task06/same_prompt_expansion_v{index}") for index in (1, 2, 3)
        ]
        ordered = candidate_pool_items(cohorts)

    baseline = load_remote_journal(args.baseline_journal, manifest, ordered)
    candidate = load_remote_journal(args.candidate_journal, manifest, ordered)
    result = compare_journal_verdicts(baseline, candidate)
    provenance = {
        "baseline": journal_provenance(baseline),
        "candidate": journal_provenance(candidate),
    }
    payload = result | {
        "packet": {
            "items_sha256": manifest["items_sha256"],
            "item_count": manifest["item_count"],
        },
        "provenance": provenance,
        "baseline_journal": str(args.baseline_journal),
        "candidate_journal": str(args.candidate_journal),
    }
    args.report.mkdir(parents=True, exist_ok=True)
    write_json(args.report / "summary.json", payload)
    (args.report / "report.md").write_text(
        render_ab_report(result, provenance["candidate"]), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"\nraport: {args.report / 'report.md'}")


if __name__ == "__main__":
    main()
