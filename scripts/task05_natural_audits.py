#!/usr/bin/env python3
"""Materialize and aggregate prospective Task 05 natural-query audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from doc2query.evaluation.natural_audits import (
    aggregate_concept_audit,
    aggregate_label_audit,
    materialize_natural_audits,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    materialize = sub.add_parser("materialize")
    materialize.add_argument(
        "--contract", type=Path, default=Path("configs/evaluation/task05_natural_audits_v1.json")
    )
    materialize.add_argument("--output-dir", type=Path, required=True)
    materialize.add_argument("--archive-incompatible", action="store_true")
    materialize.add_argument("--max-records", type=int)
    labels = sub.add_parser("aggregate-labels")
    labels.add_argument("--machine-key", type=Path, required=True)
    labels.add_argument("--ratings", type=Path, nargs="+", required=True)
    labels.add_argument("--adjudication", type=Path)
    labels.add_argument("--output-dir", type=Path, required=True)
    labels.add_argument("--required-reviewers", type=int, default=2)
    concepts = sub.add_parser("aggregate-concepts")
    concepts.add_argument("--machine-proposals", type=Path, required=True)
    concepts.add_argument("--ratings", type=Path, nargs="+", required=True)
    concepts.add_argument("--adjudication", type=Path)
    concepts.add_argument("--output-dir", type=Path, required=True)
    concepts.add_argument("--required-reviewers", type=int, default=2)
    args = parser.parse_args()
    if args.command == "materialize":
        result = materialize_natural_audits(
            args.contract,
            output_dir=args.output_dir,
            archive_incompatible=args.archive_incompatible,
            max_records=args.max_records,
        )
    elif args.command == "aggregate-labels":
        result = aggregate_label_audit(
            args.machine_key,
            args.ratings,
            adjudication=args.adjudication,
            output_dir=args.output_dir,
            required_reviewers=args.required_reviewers,
        )
    else:
        result = aggregate_concept_audit(
            args.machine_proposals,
            args.ratings,
            adjudication=args.adjudication,
            output_dir=args.output_dir,
            required_reviewers=args.required_reviewers,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
