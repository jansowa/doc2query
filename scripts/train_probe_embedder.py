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
from doc2query.evaluation.statistical_contract import StatisticalContract
from doc2query.reranker.base import FrozenRerankerConfig
from doc2query.reranker.load import load_frozen_reranker
from doc2query.utils.records import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument(
        "--comparison-contract",
        type=Path,
        default=Path("configs/evaluation/comparison_contract_v1.yaml"),
        help="Versioned P-04 statistical and four-dimensional budget contract.",
    )
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
        "--train-prefix-limit",
        type=int,
        help="Use the exact ordered prefix of a pre-materialized common cohort.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Explicit P-04 training-seed override recorded in the resolved recipe.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Explicit preregistered P-04 halving-stage step budget.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Explicit microbatch override, recorded in the resolved probe recipe.",
    )
    parser.add_argument(
        "--checkpoint-interval-steps",
        type=int,
        default=0,
        help="Atomically save a resumable rolling training checkpoint every N steps (0 disables).",
    )
    parser.add_argument(
        "--evaluation-encode-batch-size",
        type=int,
        default=64,
        help=(
            "Execution-only batch size for corpus/query encoding; "
            "does not alter the frozen recipe."
        ),
    )
    parser.add_argument(
        "--retrieval-query-batch-size",
        type=int,
        default=512,
        help="Number of cached query embeddings per exact blockwise corpus scan.",
    )
    parser.add_argument(
        "--retrieval-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device for exact sharded inner-product retrieval.",
    )
    parser.add_argument(
        "--smoke-steps",
        type=int,
        help="Explicit smoke override; outputs are not comparable to frozen full-budget runs.",
    )
    args = parser.parse_args()
    if args.train_limit is not None and args.train_prefix_limit is not None:
        raise ValueError("--train-limit and --train-prefix-limit are mutually exclusive")
    raw = yaml.safe_load(args.recipe.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("probe recipe must be a YAML mapping")
    recipe = ProbeRecipe.from_dict(raw)
    statistical_contract = StatisticalContract.load(args.comparison_contract)
    recipe_updates: dict[str, Any] = {}
    if args.smoke_steps is not None:
        recipe_updates["max_steps"] = args.smoke_steps
    if args.seed is not None:
        recipe_updates["seed"] = args.seed
    if args.max_steps is not None:
        if args.max_steps < 1:
            raise ValueError("--max-steps must be positive")
        recipe_updates["max_steps"] = args.max_steps
    if args.batch_size is not None:
        if args.batch_size < 1:
            raise ValueError("--batch-size must be positive")
        recipe_updates["batch_size"] = args.batch_size
    if args.checkpoint_interval_steps < 0:
        raise ValueError("--checkpoint-interval-steps cannot be negative")
    if min(args.evaluation_encode_batch_size, args.retrieval_query_batch_size) < 1:
        raise ValueError("evaluation and retrieval batch sizes must be positive")
    if recipe_updates:
        recipe = ProbeRecipe.from_dict(asdict(recipe) | recipe_updates)
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
        completed_training_artifacts = (args.output_dir / "train_summary.json").is_file() and (
            args.output_dir / "model"
        ).is_dir()
        if recipe.negative_recipe.requires_filter and not completed_training_artifacts:
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
            print(
                f"[probe] loading primary judge for negative filtering on "
                f"{judge_config.device}",
                flush=True,
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
            statistical_contract=statistical_contract,
            synthetic_generations=args.synthetic_generations,
            train_limit=args.train_limit,
            train_prefix_limit=args.train_prefix_limit,
            documents_path=args.corpus,
            holdout_manifest=args.holdout_manifest,
            native_documents_path=args.native_corpus,
            holdout_profile=args.holdout_profile,
            primary_scorer=primary,
            bm25_index=corpus_index,
            generator_id=args.generator_id,
            checkpoint_interval_steps=args.checkpoint_interval_steps,
            evaluation_encode_batch_size=args.evaluation_encode_batch_size,
            retrieval_query_batch_size=args.retrieval_query_batch_size,
            retrieval_device=args.retrieval_device,
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
