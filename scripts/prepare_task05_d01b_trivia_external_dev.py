#!/usr/bin/env python3
"""Materialize the preregistered external TriviaQA development cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from doc2query.evaluation.d01b_trivia import prepare_trivia_external_dev


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/raw/trivia-mined-negatives/train_pl.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/task05/d01b_trivia_external_dev_v1"),
    )
    args = parser.parse_args()
    result = prepare_trivia_external_dev(
        source_path=args.source,
        readme_path=args.source.parent / "README.md",
        policy_path=Path("reports/decisions/task05_d01b_trivia_external_dev_cohort_policy_v1.md"),
        pilot_inputs=(
            Path(
                "artifacts/task05/d01b_scale_interaction_4_5b_pilot_v1/probe_inputs/baseline.jsonl"
            ),
            Path("artifacts/task05/d01b_scale_interaction_4_5b_pilot_v1/probe_inputs/hybrid.jsonl"),
        ),
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
