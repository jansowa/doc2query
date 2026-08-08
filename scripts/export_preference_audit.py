#!/usr/bin/env python3
"""Export a deterministic blind A/B audit sample."""

import argparse
import json
from pathlib import Path

from doc2query.preferences.commands import run_export_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preferences", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(
        json.dumps(
            run_export_audit(args.preferences, args.output_dir, args.sample_size, args.seed),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
