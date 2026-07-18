#!/usr/bin/env python3
"""Compare generator evaluations with query-level paired bootstrap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from doc2query.evaluation.comparison import compare_generator_runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-dir", type=Path, required=True)
    parser.add_argument("--right-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = compare_generator_runs(
        args.left_dir / "summary.json",
        args.right_dir / "summary.json",
        left_per_generation_path=args.left_dir / "per_generation.jsonl",
        right_per_generation_path=args.right_dir / "per_generation.jsonl",
        output_path=args.output,
        samples=args.samples,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
