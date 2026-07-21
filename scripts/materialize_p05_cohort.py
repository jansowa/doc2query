#!/usr/bin/env python3
"""Materialize the development-only P-05 common cohort; never execute its plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from doc2query.evaluation.p05_materializer import materialize_p05_cohort


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--natural-pairs", type=Path, required=True)
    parser.add_argument("--w05-generations", type=Path, required=True)
    parser.add_argument("--natural-fingerprint", type=Path, required=True)
    parser.add_argument("--w05-fingerprint", type=Path, required=True)
    parser.add_argument("--budget", type=Path, required=True)
    parser.add_argument("--gold-output", type=Path, required=True)
    parser.add_argument("--synthetic-output", type=Path, required=True)
    parser.add_argument("--mixed-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--queries-per-passage",
        type=int,
        help="Explicit uniform K; omitted means exactly one query per passage.",
    )
    args = parser.parse_args()
    manifest = materialize_p05_cohort(
        natural_pairs_path=args.natural_pairs,
        w05_generations_path=args.w05_generations,
        natural_fingerprint_path=args.natural_fingerprint,
        w05_fingerprint_path=args.w05_fingerprint,
        budget_path=args.budget,
        gold_output_path=args.gold_output,
        synthetic_output_path=args.synthetic_output,
        mixed_output_path=args.mixed_output,
        manifest_output_path=args.manifest_output,
        seed=args.seed,
        queries_per_passage=args.queries_per_passage,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
