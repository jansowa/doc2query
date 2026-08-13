#!/usr/bin/env python3
"""Run the Task 06 train-only, ID-only cohort audit."""

import argparse
import json
from pathlib import Path

from doc2query.preferences.execution_design import run_id_only_cohort_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_id_only_cohort_audit(args.config, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
