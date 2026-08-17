#!/usr/bin/env python3
"""V2-01: uruchom lub skalibruj lokalnego sędziego odpowiadalności.

Runner jest fail-closed: odmawia pracy, dopóki digest lokalnych wag nie jest
przypięty w configu i zgodny z tym, co raportuje backend. `--print-model-info`
wypisuje digesty serwowane lokalnie (pomoc przy zamrażaniu ADR V2-01), niczego
nie oceniając.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from doc2query.preferences.answerability_judge import (
    _http_transport,
    analyze_calibration,
    calibration_items_from_audit,
    calibration_items_from_reward_corpus,
    load_judge_config,
    load_judgments,
    run_judgments,
    write_calibration_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/preferences/task06_answerability_judge_v1.json"),
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=Path("artifacts/task06/preference_audit_v2"),
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
        "--output-dir",
        type=Path,
        default=Path("artifacts/task06/answerability_judge_v1/calibration"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/measurements/task06/answerability_calibration_v1/summary.json"),
    )
    parser.add_argument("--max-new-judgments", type=int)
    parser.add_argument("--print-model-info", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()

    config = load_judge_config(args.config)
    if args.print_model_info:
        listing = _http_transport(
            str(config["judge"]["tags_url"]), {}, float(config["judge"]["timeout_seconds"])
        )
        print(json.dumps(listing, ensure_ascii=False, indent=2, sort_keys=True))
        return

    items = calibration_items_from_audit(args.export_dir)
    items += calibration_items_from_reward_corpus(args.corpus, args.cohort_records)
    # Ten sam (query, passage) może wystąpić w obu źródłach — werdykt jest wspólny.
    unique = {item.item_id: item for item in items}
    ordered = [unique[item_id] for item_id in sorted(unique)]

    if not args.analyze_only:
        summary = run_judgments(
            ordered,
            config=config,
            output_dir=args.output_dir,
            max_new_judgments=args.max_new_judgments,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))

    judgments = load_judgments(args.output_dir / "judgments.journal.jsonl")
    report = analyze_calibration(ordered, judgments)
    write_calibration_report(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
