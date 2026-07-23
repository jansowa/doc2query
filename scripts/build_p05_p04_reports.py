#!/usr/bin/env python3
"""Build P-04 dev-screen reports and run the fail-closed decision engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from doc2query.evaluation.p04_decision import evaluate_p04_comparison
from doc2query.evaluation.p05_guardrails import build_dev_screen_report
from doc2query.evaluation.statistical_contract import StatisticalContract
from doc2query.utils.records import write_json


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root", type=Path, default=Path("runs/task04_p05_dev_screen/dev_screen")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/measurements/task04_p05_dev_screen/p04_gate"),
    )
    parser.add_argument(
        "--comparison-contract",
        type=Path,
        default=Path("configs/evaluation/comparison_contract_v1.yaml"),
    )
    args = parser.parse_args()
    contract = StatisticalContract.load(args.comparison_contract)
    control_id = "P05-GOLD-NATURAL-S42"
    control_dir = args.run_root / control_id
    control_result = _load(control_dir / "result.json")
    statuses: dict[str, Any] = {}
    for arm_id in ("P05-MIXED50-S42", "P05-W05-SYNTHETIC-S42"):
        arm_dir = args.run_root / arm_id
        report = build_dev_screen_report(
            arm_id=arm_id,
            control_id=control_id,
            arm_result=_load(arm_dir / "result.json"),
            control_result=control_result,
            arm_per_query_path=arm_dir / "corpus_retrieval_per_query.jsonl",
            control_per_query_path=control_dir / "corpus_retrieval_per_query.jsonl",
            arm_guardrails_path=arm_dir / "p04_guardrails_per_query.jsonl",
            control_guardrails_path=control_dir / "p04_guardrails_per_query.jsonl",
            contract=contract,
        )
        decision = evaluate_p04_comparison(
            report, control_manifest=control_result, contract=contract
        )
        write_json(args.output_dir / f"{arm_id}.report.json", report)
        write_json(args.output_dir / f"{arm_id}.decision.json", decision)
        statuses[arm_id] = decision
    summary = {
        "schema_version": 1,
        "stage": "dev_screen",
        "final_tests_used": [],
        "decisions": statuses,
        "dev_confirm_authorized_arms": [
            arm_id for arm_id, decision in statuses.items() if decision["status"] == "eligible"
        ],
    }
    write_json(args.output_dir / "decision_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
