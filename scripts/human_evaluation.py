#!/usr/bin/env python3
"""Export blind A/B panels or import completed human ratings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from doc2query.evaluation.human import export_blind_ab, import_ratings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export")
    export.add_argument("--left", type=Path, required=True)
    export.add_argument("--right", type=Path, required=True)
    export.add_argument("--output-jsonl", type=Path, required=True)
    export.add_argument("--output-csv", type=Path, required=True)
    export.add_argument("--seed", type=int, default=42)
    export.add_argument("--max-examples", type=int, default=300)
    export.add_argument("--mode", default="deterministic")
    ingest = subparsers.add_parser("import")
    ingest.add_argument("--ratings", type=Path, required=True)
    ingest.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = (
        export_blind_ab(
            args.left,
            args.right,
            output_jsonl=args.output_jsonl,
            output_csv=args.output_csv,
            seed=args.seed,
            max_examples=args.max_examples,
            mode=args.mode,
        )
        if args.command == "export"
        else import_ratings(args.ratings, output_path=args.output)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
