#!/usr/bin/env python3
"""Validate or compare the preregistered D01b probe dev confirm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from doc2query.evaluation.d01_probe import (
    build_d01b_probe_dev_confirm_decision,
    preflight_d01b_probe_dev_confirm,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "compare"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/evaluation/d01b_probe_dev_confirm_v1.yaml"),
    )
    args = parser.parse_args()
    result = (
        preflight_d01b_probe_dev_confirm(args.config)
        if args.command == "preflight"
        else build_d01b_probe_dev_confirm_decision(args.config)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
