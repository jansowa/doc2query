#!/usr/bin/env python3
"""Score frozen-dev positive and inherited-negative pairs for P-03 calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from doc2query.reranker.base import FrozenRerankerConfig
from doc2query.reranker.false_negative_calibration import (
    score_frozen_dev_for_false_negative_calibration,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--judge-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw: Any = yaml.safe_load(args.judge_config.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("judge config must be a YAML mapping")
    result = score_frozen_dev_for_false_negative_calibration(
        input_path=args.input,
        output_path=args.output,
        judge=FrozenRerankerConfig(**raw),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
