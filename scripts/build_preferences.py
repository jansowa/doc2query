#!/usr/bin/env python3
"""Build TRL preference splits and mandatory continued-SFT controls."""

import argparse
import json
from pathlib import Path

from doc2query.preferences.commands import run_build


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--candidate-sets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--format", choices=("jsonl", "parquet"), default="parquet")
    args = parser.parse_args()
    print(
        json.dumps(
            run_build(args.candidates, args.candidate_sets, args.output_dir, args.format),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
