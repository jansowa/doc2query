#!/usr/bin/env python3
"""Prepare, preflight, or compare the prospective D01b 4.5B scale pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from doc2query.evaluation.d01b_scale_pilot import (
    compare_scale_pilot,
    preflight_scale_pilot,
    prepare_scale_pilot_cohorts,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare-cohorts", "preflight", "compare"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/evaluation/d01b_scale_interaction_4_5b_pilot_v1.yaml"),
    )
    args = parser.parse_args()
    if args.command == "prepare-cohorts":
        result = prepare_scale_pilot_cohorts(args.config)
    elif args.command == "preflight":
        result = preflight_scale_pilot(args.config)
    else:
        result = compare_scale_pilot(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
