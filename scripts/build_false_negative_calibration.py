#!/usr/bin/env python3
"""Build a reproducible possible-false-negative threshold artifact from frozen dev scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from doc2query.reranker.base import FrozenRerankerConfig
from doc2query.reranker.false_negative_calibration import build_calibration_artifact
from doc2query.utils.records import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--fit-dataset", type=Path, required=True)
    parser.add_argument("--fit-split", required=True)
    parser.add_argument("--judge-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()
    raw: Any = yaml.safe_load(args.judge_config.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("judge config must be a YAML mapping")
    artifact = build_calibration_artifact(
        scores_path=args.scores,
        fit_dataset_path=args.fit_dataset,
        fit_split=args.fit_split,
        judge=FrozenRerankerConfig(**raw),
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    write_json(args.output, artifact)
    print(json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
