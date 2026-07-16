#!/usr/bin/env python3
"""Compare SFT summaries in a compact table with explicit missing metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

from doc2query.training.sft import compare_run_summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(compare_run_summaries(args.summaries, args.output))


if __name__ == "__main__":
    main()
