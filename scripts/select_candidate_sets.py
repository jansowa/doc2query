#!/usr/bin/env python3
"""Select deterministic chosen/rejected candidate sets from frozen scores."""

import argparse
import json
from pathlib import Path

from doc2query.preferences.commands import run_select
from doc2query.preferences.schemas import SelectionPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--strategy",
        choices=("top_vs_bottom", "top_vs_near_miss"),
        default="top_vs_near_miss",
    )
    parser.add_argument("--min-score-margin", type=float, default=0.25)
    parser.add_argument("--max-pairs-per-passage", type=int, default=1)
    parser.add_argument("--min-rejected-ground-score", type=float)
    args = parser.parse_args()
    policy = SelectionPolicy(
        strategy=args.strategy,
        min_score_margin=args.min_score_margin,
        max_pairs_per_passage=args.max_pairs_per_passage,
        min_rejected_ground_score=args.min_rejected_ground_score,
    )
    print(json.dumps(run_select(args.input, args.output, args.report, policy), indent=2))


if __name__ == "__main__":
    main()
