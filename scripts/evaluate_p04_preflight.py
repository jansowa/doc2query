#!/usr/bin/env python3
"""Evaluate a dev-only P-04 comparison report without opening final tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from doc2query.evaluation.p04_decision import evaluate_p04_comparison
from doc2query.evaluation.statistical_contract import StatisticalContract


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--control-manifest", type=Path, required=True)
    parser.add_argument(
        "--comparison-contract",
        type=Path,
        default=Path("configs/evaluation/comparison_contract_v1.yaml"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate_p04_comparison(
        _load(args.report),
        control_manifest=_load(args.control_manifest),
        contract=StatisticalContract.load(args.comparison_contract),
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 2 if result["status"] == "incomplete" else 0


if __name__ == "__main__":
    raise SystemExit(main())
