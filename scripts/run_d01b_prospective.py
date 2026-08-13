#!/usr/bin/env python3
"""Run preregistered D01b cohort, preflight, and prospective selection phases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from doc2query.evaluation.d01_prospective import (
    assert_exact_k_summary,
    assert_scoring_summary,
    materialize_prospective_probe_inputs,
    preflight_prospective,
    prepare_prospective_cohort,
    select_compare_prospective,
    validate_materialized_cohort,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "preflight",
            "prepare-cohort",
            "validate-cohort",
            "validate-exact-k",
            "validate-scoring",
            "select-compare",
            "materialize-probe-inputs",
        ),
    )
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--cohort-manifest", type=Path)
    parser.add_argument("--baseline-rows", type=Path)
    parser.add_argument("--controlled-rows", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--output-selected", type=Path)
    parser.add_argument("--semantic-cache-dir", type=Path)
    parser.add_argument("--semantic-device", default="cuda")
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--selected-rows", type=Path)
    parser.add_argument("--probe-recipe", type=Path)
    parser.add_argument("--baseline-output", type=Path)
    parser.add_argument("--hybrid-output", type=Path)
    parser.add_argument("--manifest-output", type=Path)
    args = parser.parse_args()
    if args.command == "preflight":
        result = preflight_prospective(args.contract)
    elif args.command == "prepare-cohort":
        if args.cohort_manifest is None:
            parser.error("prepare-cohort requires --cohort-manifest")
        result = prepare_prospective_cohort(args.contract, args.cohort_manifest)
    elif args.command == "validate-cohort":
        if args.cohort_manifest is None:
            parser.error("validate-cohort requires --cohort-manifest")
        result = validate_materialized_cohort(args.contract, args.cohort_manifest)
    elif args.command == "validate-exact-k":
        if args.summary is None:
            parser.error("validate-exact-k requires --summary")
        contract = yaml.safe_load(args.contract.read_text(encoding="utf-8"))
        if not isinstance(contract, dict):
            raise ValueError("prospective contract must be a mapping")
        cohort = contract.get("cohort")
        selector = contract.get("selector")
        if not isinstance(cohort, dict) or not isinstance(selector, dict):
            raise ValueError("prospective contract lacks cohort/selector")
        result = assert_exact_k_summary(
            args.summary,
            groups=int(cohort["selected_count"]),
            k=int(selector["selected_count"]),
        )
    elif args.command == "validate-scoring":
        if args.summary is None:
            parser.error("validate-scoring requires --summary")
        contract = yaml.safe_load(args.contract.read_text(encoding="utf-8"))
        if not isinstance(contract, dict):
            raise ValueError("prospective contract must be a mapping")
        cohort = contract.get("cohort")
        selector = contract.get("selector")
        if not isinstance(cohort, dict) or not isinstance(selector, dict):
            raise ValueError("prospective contract lacks cohort/selector")
        result = assert_scoring_summary(
            args.summary,
            rows=int(cohort["selected_count"]) * int(selector["selected_count"]),
        )
    elif args.command == "select-compare":
        required = {
            "cohort_manifest": args.cohort_manifest,
            "baseline_rows": args.baseline_rows,
            "controlled_rows": args.controlled_rows,
            "output_json": args.output_json,
            "output_markdown": args.output_markdown,
            "output_selected": args.output_selected,
            "semantic_cache_dir": args.semantic_cache_dir,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error(f"select-compare missing arguments: {', '.join(missing)}")
        result = select_compare_prospective(
            args.contract,
            cohort_manifest_path=args.cohort_manifest,
            baseline_rows_path=args.baseline_rows,
            controlled_rows_path=args.controlled_rows,
            output_json=args.output_json,
            output_markdown=args.output_markdown,
            output_selected=args.output_selected,
            semantic_cache_dir=args.semantic_cache_dir,
            semantic_device=args.semantic_device,
        )
    else:
        required = {
            "report": args.report,
            "selected_rows": args.selected_rows,
            "baseline_rows": args.baseline_rows,
            "controlled_rows": args.controlled_rows,
            "probe_recipe": args.probe_recipe,
            "baseline_output": args.baseline_output,
            "hybrid_output": args.hybrid_output,
            "manifest_output": args.manifest_output,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error(f"materialize-probe-inputs missing arguments: {', '.join(missing)}")
        result = materialize_prospective_probe_inputs(
            args.contract,
            report_path=args.report,
            selected_rows_path=args.selected_rows,
            baseline_rows_path=args.baseline_rows,
            controlled_rows_path=args.controlled_rows,
            probe_recipe_path=args.probe_recipe,
            baseline_output=args.baseline_output,
            hybrid_output=args.hybrid_output,
            manifest_output=args.manifest_output,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
