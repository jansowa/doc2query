#!/usr/bin/env python3
"""Build Markdown and HTML reports from measured generator artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from doc2query.evaluation.report import build_generator_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--per-generation", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=100)
    args = parser.parse_args()
    result = build_generator_report(
        args.summary,
        args.per_generation,
        markdown_path=args.markdown,
        html_path=args.html,
        max_examples=args.max_examples,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
