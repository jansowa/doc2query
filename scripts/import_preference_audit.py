#!/usr/bin/env python3
"""Validate and unblind a completed preference audit."""

import argparse
import json
from pathlib import Path

from doc2query.preferences.commands import run_import_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completed", type=Path, required=True)
    parser.add_argument("--machine-key", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run_import_audit(
                args.completed,
                args.machine_key,
                args.output_dir,
                not args.allow_incomplete,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
