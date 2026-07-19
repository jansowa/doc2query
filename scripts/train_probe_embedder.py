#!/usr/bin/env python3
"""Train and evaluate the frozen Task 04 probe-embedder recipe."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from doc2query.evaluation.corpus import load_corpus_index
from doc2query.evaluation.embedder_probe import ProbeRecipe, run_probe_experiment
from doc2query.evaluation.probe_negatives import ProbeNegativeBlocker
from doc2query.reranker.base import FrozenRerankerConfig
from doc2query.reranker.load import load_frozen_reranker
from doc2query.utils.records import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--test-subset", default="test_embedder")
    parser.add_argument(
        "--holdout-manifest",
        type=Path,
        help="P-02 manifest containing test_native_pl and translated provenance.",
    )
    parser.add_argument(
        "--native-corpus",
        type=Path,
        help="Adapted native corpus; quick/medium use the manifest's judged corpus by default.",
    )
    parser.add_argument(
        "--holdout-profile",
        choices=("quick", "medium", "full"),
        default="quick",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="Full frozen documents.parquet used by corpus_retrieval.",
    )
    parser.add_argument(
        "--query-source", choices=("natural", "copy_control", "synthetic"), required=True
    )
    parser.add_argument("--synthetic-generations", type=Path)
    parser.add_argument(
        "--generator-id",
        help="Required for synthetic query-source reporting and manifest provenance.",
    )
    parser.add_argument(
        "--primary-judge-config",
        type=Path,
        help="Pinned primary reranker config; required by HN0+filter and HN1.",
    )
    parser.add_argument(
        "--bm25-index",
        type=Path,
        help="Frozen P-01 BM25 index directory; required by HN1.",
    )
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
    recipe = ProbeRecipe.from_dict(raw)
    if args.smoke_steps is not None:
        recipe = ProbeRecipe.from_dict(asdict(recipe) | {"max_steps": args.smoke_steps})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models_loaded = False
    corpus_index = None
    try:
        calibration = recipe.negative_recipe.load_calibration()
        if args.query_source == "synthetic" and not args.generator_id:
            raise ProbeNegativeBlocker("P-03 BLOCKED: synthetic probe runs require --generator-id")
        if recipe.negative_recipe.strategy == "hn1_bm25":
            if args.bm25_index is None:
                raise ProbeNegativeBlocker("P-03 BLOCKED: HN1 requires --bm25-index")
            corpus_index = load_corpus_index(args.bm25_index)
        primary = None
        if recipe.negative_recipe.requires_filter:
            if args.primary_judge_config is None:
                raise ProbeNegativeBlocker(
                    "P-03 BLOCKED: filtered probe recipe requires --primary-judge-config"
                )
            judge_raw: Any = yaml.safe_load(args.primary_judge_config.read_text(encoding="utf-8"))
            if not isinstance(judge_raw, dict):
                raise ValueError("primary judge config must be a YAML mapping")
            judge_config = FrozenRerankerConfig(**judge_raw)
            if calibration is None or (
                judge_config.name_or_path != calibration.primary_judge_name
                or judge_config.revision != calibration.primary_judge_revision
            ):
                raise ProbeNegativeBlocker(
                    "P-03 BLOCKED: primary judge config does not match calibration provenance"
                )
            primary = load_frozen_reranker(judge_config)
            models_loaded = True
        result = run_probe_experiment(
            train_path=args.train_input,
            frozen_manifest=args.frozen_manifest,
            test_subset=args.test_subset,
            output_dir=args.output_dir,
            recipe=recipe,
            query_source=args.query_source,
            synthetic_generations=args.synthetic_generations,
            train_limit=args.train_limit,
            documents_path=args.corpus,
            holdout_manifest=args.holdout_manifest,
            native_documents_path=args.native_corpus,
            holdout_profile=args.holdout_profile,
            primary_scorer=primary,
            bm25_index=corpus_index,
            generator_id=args.generator_id,
        )
    except ProbeNegativeBlocker as exc:
        blocker = {
            "schema_version": 1,
            "status": "blocked",
            "scope": "P-03",
            "reason": str(exc),
            "recipe_version": recipe.recipe_version,
            "negative_recipe_version": recipe.negative_recipe.version,
            "hard_negative_strategy": recipe.negative_recipe.strategy,
            "possible_false_negative_policy": recipe.negative_recipe.false_negative_policy,
            "models_loaded": models_loaded,
            "tests_used_for_threshold_tuning": [],
        }
        write_json(args.output_dir / "p03_preflight.json", blocker)
        print(json.dumps(blocker, ensure_ascii=False, indent=2, sort_keys=True))
        raise SystemExit(2) from exc
    finally:
        if corpus_index is not None:
            corpus_index.close()
    if args.smoke_steps is not None:
        result["comparability"] = "smoke_only_not_comparable"
        result["training"]["comparability"] = "smoke_only_not_comparable"
        result["corpus_retrieval"]["comparability"] = "smoke_only_not_comparable"
        write_json(args.output_dir / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
