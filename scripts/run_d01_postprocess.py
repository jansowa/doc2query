#!/usr/bin/env python3
"""Run frozen-dev D01 generation, scoring, matched reporting, or probe materialization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from doc2query.evaluation.d01_pipeline import (
    assemble_matched_report,
    generate_frozen_dev,
    materialize_probe_inputs,
    score_d01_artifact,
)


def _generation(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--subset", default="dev_intrinsic_rank10")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--model-checkpoint", type=Path)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--archive-incompatible", action="store_true")


def _scoring(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--generation-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--primary-judge", type=Path, required=True)
    parser.add_argument("--shadow-judge", type=Path, required=True)
    parser.add_argument("--judge-device", choices=("cpu", "cuda"))
    parser.add_argument("--primary-judge-device", choices=("cpu", "cuda"))
    parser.add_argument("--shadow-judge-device", choices=("cpu", "cuda"))
    parser.add_argument("--corpus-index", type=Path)
    parser.add_argument("--scoring-batch-size", type=int, default=16)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--archive-incompatible", action="store_true")


def _comparison(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--baseline-rows", type=Path, required=True)
    parser.add_argument("--variant-summary", type=Path, required=True)
    parser.add_argument("--variant-rows", type=Path, required=True)
    parser.add_argument("--comparison-contract", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_721)


def _materialization(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--generation-summary", type=Path, required=True)
    parser.add_argument("--scoring-summary", type=Path, required=True)
    parser.add_argument("--scoring-rows", type=Path, required=True)
    parser.add_argument("--comparison-report", type=Path, required=True)
    parser.add_argument(
        "--probe-recipe",
        type=Path,
        default=Path("configs/evaluation/probe_v1.yaml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-policy", default="all_matched_k4")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _generation(subparsers.add_parser("generation-only"))
    _scoring(subparsers.add_parser("score"))
    _comparison(subparsers.add_parser("compare"))
    _materialization(subparsers.add_parser("materialize-probe-inputs"))
    args = parser.parse_args()
    if args.command == "generation-only":
        result = generate_frozen_dev(
            args.config,
            frozen_manifest=args.frozen_manifest,
            subset=args.subset,
            output_path=args.output,
            adapter_path=args.adapter,
            model_path=args.model_checkpoint,
            max_examples=args.max_examples,
            archive_incompatible=args.archive_incompatible,
            progress_every=args.progress_every,
        )
    elif args.command == "score":
        result = score_d01_artifact(
            args.generations,
            generation_summary_path=args.generation_summary,
            output_dir=args.output_dir,
            primary_config=args.primary_judge,
            shadow_config=args.shadow_judge,
            judge_device=args.judge_device,
            primary_judge_device=args.primary_judge_device,
            shadow_judge_device=args.shadow_judge_device,
            corpus_index_path=args.corpus_index,
            scoring_batch_size=args.scoring_batch_size,
            progress_every=args.progress_every,
            archive_incompatible=args.archive_incompatible,
        )
    elif args.command == "compare":
        result = assemble_matched_report(
            baseline_summary_path=args.baseline_summary,
            baseline_rows_path=args.baseline_rows,
            variant_summary_path=args.variant_summary,
            variant_rows_path=args.variant_rows,
            comparison_contract_path=args.comparison_contract,
            output_json=args.output_json,
            output_markdown=args.output_markdown,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
    else:
        result = materialize_probe_inputs(
            generations_path=args.generations,
            generation_summary_path=args.generation_summary,
            scoring_summary_path=args.scoring_summary,
            scoring_rows_path=args.scoring_rows,
            comparison_report_path=args.comparison_report,
            probe_recipe_path=args.probe_recipe,
            output_path=args.output,
            selection_policy=args.selection_policy,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
