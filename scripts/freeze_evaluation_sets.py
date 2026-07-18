#!/usr/bin/env python3
"""Freeze Task 04 evaluation IDs and fingerprints without changing v1 splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from doc2query.evaluation.datasets import freeze_evaluation_sets, verify_frozen_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", type=Path, default=Path("data/processed/v1/dev.parquet"))
    parser.add_argument("--test", type=Path, default=Path("data/processed/v1/test.parquet"))
    parser.add_argument(
        "--adversarial",
        type=Path,
        default=Path("tests/fixtures/task02_manual_holdout.jsonl"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed/v1/evaluation/task04-v1")
    )
    parser.add_argument("--human-panel-size", type=int, default=300)
    parser.add_argument("--generation-panel-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    manifest_path = args.output_dir / "manifest.json"
    result = (
        verify_frozen_manifest(manifest_path)
        if args.verify
        else freeze_evaluation_sets(
            dev_path=args.dev,
            test_path=args.test,
            adversarial_path=args.adversarial,
            output_dir=args.output_dir,
            human_panel_size=args.human_panel_size,
            generation_panel_size=args.generation_panel_size,
            seed=args.seed,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
