#!/usr/bin/env python3
"""Resume offline primary margins for every natural train query-positive pair."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from doc2query.reranker.base import FrozenRerankerConfig
from doc2query.reranker.train_margins import score_natural_train_margins


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--judge-config", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--calibration-scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--group-batch-size", type=int, default=128)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--progress-interval-seconds", type=float, default=10.0)
    args = parser.parse_args()
    raw: Any = yaml.safe_load(args.judge_config.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("judge config must be a YAML mapping")
    try:
        result = score_natural_train_margins(
            input_path=args.input,
            output_dir=args.output_dir,
            judge=FrozenRerankerConfig(**raw),
            calibration_path=args.calibration,
            calibration_scores_path=args.calibration_scores,
            group_batch_size=args.group_batch_size,
            progress_every=args.progress_every,
            progress_interval_seconds=args.progress_interval_seconds,
        )
    except KeyboardInterrupt:
        print("[train-margins] interrupted safely; rerun the same command", file=sys.stderr)
        raise SystemExit(130) from None
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
