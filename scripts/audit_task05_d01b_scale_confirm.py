#!/usr/bin/env python3
"""Run the CPU-only, ID-only D01b 4.5B dev-confirm feasibility audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from doc2query.evaluation.d01b_scale_confirm import assess_confirm_feasibility


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot-config",
        type=Path,
        default=Path("configs/evaluation/d01b_scale_interaction_4_5b_pilot_v1.yaml"),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            assess_confirm_feasibility(args.pilot_config),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
