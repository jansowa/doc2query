#!/usr/bin/env python3
"""Freeze Task 04 P-02 native/translated holdouts without downloading data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from doc2query.evaluation.native_holdout import (
    freeze_native_pl_holdout,
    verify_native_holdout_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--translated-manifest",
        type=Path,
        default=Path("data/processed/v1/evaluation/task04-v1/manifest.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/v1/evaluation/task04-native-pl-v1"),
    )
    parser.add_argument(
        "--polqa-test",
        type=Path,
        help="Pinned official PolQA test CSV/Parquet; omitted means an explicit blocker.",
    )
    parser.add_argument(
        "--polqa-passages",
        type=Path,
        help="Optional pinned passages.jsonl; streamed only for the full profile.",
    )
    parser.add_argument(
        "--polqa-revision",
        default="d78d036ef08ab3b9f4d85a2893f4d3a0c95a6f37",
    )
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = (
        verify_native_holdout_manifest(args.output_dir / "manifest.json")
        if args.verify
        else freeze_native_pl_holdout(
            translated_manifest=args.translated_manifest,
            output_dir=args.output_dir,
            polqa_test_path=args.polqa_test,
            polqa_passages_path=args.polqa_passages,
            polqa_revision=args.polqa_revision,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
