#!/usr/bin/env python3
"""Validate generated candidates against a frozen Task 06 request plan."""

import argparse
import json
from pathlib import Path

from doc2query.preferences.evidence import validate_generated_candidate_files
from doc2query.utils.records import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = validate_generated_candidate_files(args.requests, args.candidates)
    if args.report is not None:
        if args.report.exists():
            raise FileExistsError(f"validation report already exists: {args.report}")
        write_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
