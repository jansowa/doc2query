#!/usr/bin/env python3
"""Train and evaluate the frozen Task 04 probe-embedder recipe."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import yaml

from doc2query.evaluation.embedder_probe import ProbeRecipe, run_probe_experiment
from doc2query.utils.records import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--test-subset", default="test_embedder_rank10")
    parser.add_argument(
        "--query-source", choices=("natural", "copy_control", "synthetic"), required=True
    )
    parser.add_argument("--synthetic-generations", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument(
        "--smoke-steps",
        type=int,
        help="Explicit smoke override; outputs are not comparable to frozen full-budget runs.",
    )
    args = parser.parse_args()
    raw = yaml.safe_load(args.recipe.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("probe recipe must be a YAML mapping")
    recipe = ProbeRecipe(**raw)
    if args.smoke_steps is not None:
        recipe = ProbeRecipe(**(asdict(recipe) | {"max_steps": args.smoke_steps}))
    result = run_probe_experiment(
        train_path=args.train_input,
        frozen_manifest=args.frozen_manifest,
        test_subset=args.test_subset,
        output_dir=args.output_dir,
        recipe=recipe,
        query_source=args.query_source,
        synthetic_generations=args.synthetic_generations,
        train_limit=args.train_limit,
    )
    if args.smoke_steps is not None:
        result["comparability"] = "smoke_only_not_comparable"
        result["training"]["comparability"] = "smoke_only_not_comparable"
        result["retrieval"]["comparability"] = "smoke_only_not_comparable"
        write_json(args.output_dir / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
