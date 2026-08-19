#!/usr/bin/env python3
"""Zaimportuj werdykty sędziego z drugiej maszyny i policz kalibrację wg ADR V2-01.

Import jest fail-closed: odrzuca journal z innym promptem, z itemami spoza zamrożonego
pakietu, mieszający modele albo sprzeczny sam ze sobą. Progi akceptacji są zamrożone w
`reports/decisions/task06_answerability_judge_v1.md` **przed** tym uruchomieniem.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from doc2query.preferences.answerability_judge import (
    analyze_calibration,
    calibration_items_from_audit,
    calibration_items_from_reward_corpus,
    write_calibration_report,
)
from doc2query.preferences.answerability_remote import (
    apply_acceptance_criteria,
    load_remote_journal,
    remote_identity,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument(
        "--packet-dir", type=Path, default=Path("artifacts/task06/answerability_packet_v1")
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
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/measurements/task06/answerability_judge_v1/summary.json"),
    )
    args = parser.parse_args()

    manifest = json.loads((args.packet_dir / "manifest.json").read_text(encoding="utf-8"))
    items = calibration_items_from_audit(args.export_dir)
    items += calibration_items_from_reward_corpus(args.corpus, args.cohort_records)
    unique = {item.item_id: item for item in items}
    ordered = [unique[item_id] for item_id in sorted(unique)]

    verdicts = load_remote_journal(args.journal, manifest, ordered)
    analysis = analyze_calibration(ordered, verdicts)
    report = analysis | {
        "adr": "reports/decisions/task06_answerability_judge_v1.md",
        "packet": {
            "items_sha256": manifest["items_sha256"],
            "item_ids_fingerprint": manifest["item_ids_fingerprint"],
            "item_count": manifest["item_count"],
        },
        "judge_identity": remote_identity(verdicts),
        "acceptance": apply_acceptance_criteria(ordered, verdicts, analysis),
    }
    write_calibration_report(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
