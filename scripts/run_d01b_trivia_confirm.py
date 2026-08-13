#!/usr/bin/env python3
"""Preflight, stage seed 42, or compare the D01b 4.5B TriviaQA confirm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from doc2query.evaluation.d01b_trivia_confirm import (
    compare_trivia_confirm,
    preflight_trivia_confirm,
    stage_reused_seed42,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "stage-seed42", "compare"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/evaluation/d01b_scale_interaction_4_5b_trivia_dev_confirm_v1.yaml"),
    )
    parser.add_argument("--require-staged-seed42", action="store_true")
    args = parser.parse_args()
    if args.command == "preflight":
        result = preflight_trivia_confirm(
            args.config, require_staged_seed42=args.require_staged_seed42
        )
    elif args.command == "stage-seed42":
        result = stage_reused_seed42(args.config)
    else:
        result = compare_trivia_confirm(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
