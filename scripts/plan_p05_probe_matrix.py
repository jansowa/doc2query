#!/usr/bin/env python3
"""Generate and validate the first P-05 probe matrix without executing it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from doc2query.evaluation.p05_planner import (
    build_p05_plan,
    load_p05_planner_inputs,
    write_p05_plan,
)
from doc2query.evaluation.statistical_contract import StatisticalContract


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-audit", type=Path, required=True)
    parser.add_argument("--budget", type=Path, required=True)
    parser.add_argument(
        "--comparison-contract",
        type=Path,
        default=Path("configs/evaluation/comparison_contract_v1.yaml"),
    )
    parser.add_argument("--probe-recipe", type=Path, required=True)
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--frozen-dev-manifest", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--primary-judge-config", type=Path, required=True)
    parser.add_argument("--w05-adapter", type=Path, required=True)
    parser.add_argument("--w05-synthetic-generations", type=Path, required=True)
    parser.add_argument("--mixed-50-50-generations", type=Path, required=True)
    parser.add_argument("--p05-materialization-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("runs/p05_probe_matrix"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    campaign, budget = load_p05_planner_inputs(args.campaign_audit, args.budget)
    artifacts = {
        "probe_recipe": args.probe_recipe,
        "train_input": args.train_input,
        "frozen_dev_manifest": args.frozen_dev_manifest,
        "corpus": args.corpus,
        "primary_judge_config": args.primary_judge_config,
        "w05_adapter": args.w05_adapter,
        "w05_synthetic_generations": args.w05_synthetic_generations,
        "mixed_50_50_generations": args.mixed_50_50_generations,
        "p05_materialization_manifest": args.p05_materialization_manifest,
    }
    plan = build_p05_plan(
        campaign_audit=campaign,
        contract=StatisticalContract.load(args.comparison_contract),
        comparison_budget=budget,
        artifacts=artifacts,
        output_root=args.output_root,
    )
    write_p05_plan(plan, args.output)
    print(json.dumps(plan, ensure_ascii=False, sort_keys=True))
    return 0 if plan["execution_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
