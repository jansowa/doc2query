#!/usr/bin/env python3
"""Train one SFT/QLoRA run with optional smoke/probe overrides."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from doc2query.config import load_config
from doc2query.training.sft import run_sft


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-panel", action="store_true")
    parser.add_argument(
        "--resume-if-available",
        action="store_true",
        help="Start from scratch or automatically resume the newest compatible checkpoint.",
    )
    args = parser.parse_args()
    summary = run_sft(
        load_config(args.config),
        max_steps=args.max_steps,
        max_length=args.max_length,
        output_dir=args.output_dir,
        generate_fixed_panel=not args.no_panel,
        resume_if_available=args.resume_if_available,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
