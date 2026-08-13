#!/usr/bin/env python3
"""Validate Task 06 execution design without loading models or starting work."""

import argparse
import json
from pathlib import Path

from doc2query.preferences.execution_design import preflight_execution_design
from doc2query.utils.records import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = preflight_execution_design(args.config, args.audit)
    if args.output is not None:
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
