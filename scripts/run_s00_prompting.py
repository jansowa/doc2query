#!/usr/bin/env python3
"""Prepare or run the prospective, development-only S00 prompting baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from doc2query.evaluation.s00_prompting import generate_s00, prepare_s00


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/evaluation/s00_prompting_v1.yaml"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Prompt batch size; sampling expands it by num_return_sequences.",
    )
    parser.add_argument(
        "--min-batch-size",
        type=int,
        default=1,
        help="Smallest automatic retry batch after CUDA OOM.",
    )
    parser.add_argument("--mock", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--interrupt-after", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--mock-oom-above", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.prepare_only:
        result = prepare_s00(args.contract, output_dir=args.output_dir)
    else:
        result = generate_s00(
            args.contract,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            min_batch_size=args.min_batch_size,
            mock=args.mock,
            interrupt_after=args.interrupt_after,
            mock_oom_above=args.mock_oom_above,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
