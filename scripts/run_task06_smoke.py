#!/usr/bin/env python3
"""Run individual resumable stages of the authorized Task 06 smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from doc2query.preferences.task06_smoke import (
    generate_role,
    generate_same_prompt_expansion,
    prepare_smoke_cohort,
    score_natural_queries,
    score_role,
    select_safe_queries,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("prepare", "generate", "score", "natural", "select", "expand")
    )
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--role", choices=("w06_anchor", "d01_controlled"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stage-name", choices=("smoke", "pilot"), default="smoke")
    parser.add_argument("--passages", type=int, default=32)
    parser.add_argument("--exclude-ids", type=Path)
    parser.add_argument("--experiment-id")
    args = parser.parse_args()
    root = args.output_root
    cohort = root / "cohort.records.jsonl"
    result: dict[str, Any]
    if args.stage == "expand":
        result = generate_same_prompt_expansion(args.design, root)
    elif args.stage == "prepare":
        result = prepare_smoke_cohort(
            args.design,
            root,
            passage_count=args.passages,
            stage=args.stage_name,
            excluded_ids_path=args.exclude_ids,
        )
    elif args.stage == "generate":
        if args.role is None:
            parser.error("generate requires --role")
        result = generate_role(
            args.design,
            cohort,
            root / args.role / "generations.jsonl",
            role=args.role,
            passage_count=args.passages,
            stage=args.stage_name,
        )
    elif args.stage == "score":
        if args.role is None:
            parser.error("score requires --role")
        arm = root / args.role
        result = score_role(
            args.design,
            arm / "generations.jsonl",
            arm / "scoring",
            role=args.role,
            device=args.device,
            stage=args.stage_name,
            experiment_id=args.experiment_id,
        )
    elif args.stage == "natural":
        result = score_natural_queries(
            args.design,
            cohort,
            root / "natural_primary.jsonl",
            device=args.device,
            stage=args.stage_name,
        )
    else:
        result = select_safe_queries(
            args.design,
            root / "natural_primary.jsonl",
            root / "w06_anchor/scoring/per_generation.jsonl",
            root / "d01_controlled/scoring/per_generation.jsonl",
            root / "selection",
            device=args.device,
            stage=args.stage_name,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
