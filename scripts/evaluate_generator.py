#!/usr/bin/env python3
"""Evaluate one generator checkpoint in deterministic and diverse modes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from doc2query.evaluation.generator import run_checkpoint_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--subset", default="test_generator_panel_rank10")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--primary-judge", type=Path)
    parser.add_argument("--shadow-judge", type=Path)
    parser.add_argument("--judge-device", choices=("cpu", "cuda"))
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--generations", type=Path)
    parser.add_argument("--generation-only", action="store_true")
    args = parser.parse_args()
    result = run_checkpoint_evaluation(
        args.config,
        frozen_manifest=args.frozen_manifest,
        subset=args.subset,
        output_dir=args.output_dir,
        adapter_path=args.adapter,
        primary_config=args.primary_judge,
        shadow_config=args.shadow_judge,
        judge_device=args.judge_device,
        max_examples=args.max_examples,
        generations_path=args.generations,
        generation_only=args.generation_only,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
