#!/usr/bin/env python3
"""Run frozen-dev D01 generation, scoring, matched reporting, or probe materialization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from doc2query.evaluation.d01_campaign import (
    audit_d01_artifacts,
    prepare_common_exact_k_cohort,
    validate_baseline_provenance,
    validate_corpus_index,
)
from doc2query.evaluation.d01_pipeline import (
    assemble_matched_report,
    generate_frozen_dev,
    generate_frozen_dev_batched,
    materialize_probe_inputs,
    score_d01_artifact,
)
from doc2query.evaluation.d01_quality import D01QualityContract
from doc2query.evaluation.d01_usefulness import analyze_usefulness_and_select


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
    parser.add_argument("--cohort-manifest", type=Path)


def _campaign(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--campaign-config",
        type=Path,
        default=Path("configs/evaluation/d01_campaign_v2.yaml"),
    )


def _load_campaign(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("final_tests_used") != []:
        raise ValueError("post-D01 campaign config must be a dev-only mapping")
    return value


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
    parser.add_argument("--quality-contract", type=Path, required=True)
    parser.add_argument("--semantic-device", choices=("cpu", "cuda"), default="cuda")
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


def _usefulness(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/evaluation/d01b_usefulness_hybrid_v1.yaml"),
    )
    parser.add_argument("--baseline-rows", type=Path, required=True)
    parser.add_argument("--controlled-rows", type=Path, required=True)
    parser.add_argument("--semantic-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--semantic-cache-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--output-selected", type=Path, required=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _campaign(subparsers.add_parser("audit"))
    _campaign(subparsers.add_parser("prepare-common-cohort"))
    _campaign(subparsers.add_parser("preflight"))
    _generation(subparsers.add_parser("generation-only"))
    batched = subparsers.add_parser("generation-batched")
    _generation(batched)
    batched.add_argument("--generation-batch-size", type=int, required=True)
    _scoring(subparsers.add_parser("score"))
    _comparison(subparsers.add_parser("compare"))
    _usefulness(subparsers.add_parser("analyze-usefulness"))
    _materialization(subparsers.add_parser("materialize-probe-inputs"))
    args = parser.parse_args()
    if args.command in {"generation-only", "generation-batched"}:
        generator = (
            generate_frozen_dev_batched
            if args.command == "generation-batched"
            else generate_frozen_dev
        )
        kwargs = {}
        if args.command == "generation-batched":
            kwargs["generation_batch_size"] = args.generation_batch_size
        result = generator(
            args.config,
            frozen_manifest=args.frozen_manifest,
            subset=args.subset,
            output_path=args.output,
            adapter_path=args.adapter,
            model_path=args.model_checkpoint,
            max_examples=args.max_examples,
            archive_incompatible=args.archive_incompatible,
            progress_every=args.progress_every,
            cohort_manifest=args.cohort_manifest,
            **kwargs,
        )
    elif args.command in {"audit", "prepare-common-cohort", "preflight"}:
        campaign = _load_campaign(args.campaign_config)
        frozen_manifest = Path(str(campaign["frozen_manifest"]))
        subset = str(campaign["frozen_subset"])
        audit_config = campaign["audit"]
        if not isinstance(audit_config, dict):
            raise ValueError("campaign audit config must be a mapping")
        arms = audit_config["arms"]
        if not isinstance(arms, list):
            raise ValueError("campaign audit arms must be a list")
        if args.command == "audit":
            result = audit_d01_artifacts(
                frozen_manifest=frozen_manifest,
                subset=subset,
                arms=arms,
                output_json=Path(str(audit_config["output_json"])),
                output_markdown=Path(str(audit_config["output_markdown"])),
            )
        elif args.command == "prepare-common-cohort":
            recovery = campaign["recovery"]
            if not isinstance(recovery, dict):
                raise ValueError("campaign recovery config must be a mapping")
            recovery_arms = recovery.get("arms")
            if not isinstance(recovery_arms, list):
                raise ValueError("campaign recovery arms must be a list")
            result = prepare_common_exact_k_cohort(
                frozen_manifest=frozen_manifest,
                subset=subset,
                arms=recovery_arms,
                output_dir=Path(str(recovery["output_dir"])),
                target_k=int(recovery["target_k"]),
            )
        else:
            baselines = campaign["baselines"]
            scoring = campaign["scoring"]
            comparison = campaign["comparison"]
            if (
                not isinstance(baselines, list)
                or not isinstance(scoring, dict)
                or not isinstance(comparison, dict)
            ):
                raise ValueError("campaign baseline/scoring config is malformed")
            result = {
                "status": "verified",
                "baselines": [
                    validate_baseline_provenance(
                        config_path=Path(str(item["generation_config"])),
                        adapter_path=Path(str(item["adapter"])),
                        training_manifest_path=Path(str(item["training_manifest"])),
                    )
                    for item in baselines
                ],
                "corpus_index": validate_corpus_index(
                    Path(str(scoring["corpus_index"])),
                    expected_fingerprint=str(scoring["expected_corpus_fingerprint"]),
                ),
                "copy_semantic_quality": D01QualityContract.load(
                    Path(str(comparison["quality_contract"]))
                ).reference(),
                "final_tests_used": [],
            }
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
            quality_contract_path=args.quality_contract,
            output_json=args.output_json,
            output_markdown=args.output_markdown,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
            semantic_device=args.semantic_device,
        )
    elif args.command == "analyze-usefulness":
        result = analyze_usefulness_and_select(
            contract_path=args.contract,
            baseline_rows_path=args.baseline_rows,
            controlled_rows_path=args.controlled_rows,
            output_json=args.output_json,
            output_markdown=args.output_markdown,
            output_selected=args.output_selected,
            semantic_cache_dir=args.semantic_cache_dir,
            semantic_device=args.semantic_device,
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
