#!/usr/bin/env python3
"""Run the one-off, resumable, dev-only Task 04 P-03 W05 sensitivity check."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

import yaml

from doc2query.evaluation.corpus import load_corpus_index
from doc2query.evaluation.datasets import evaluation_fingerprint, load_frozen_records
from doc2query.evaluation.embedder_probe import ProbeRecipe, prepare_probe_pairs
from doc2query.evaluation.p03_sensitivity import (
    ARM_NAMES,
    assert_equal_budget,
    common_cohort,
    compare_sensitivity_arms,
    evaluate_probe_on_dev,
    freeze_train_cohort,
    generate_w05_queries,
    load_sensitivity_config,
    materialize_selected_train,
    mock_smoke,
    negative_recipe_for_arm,
    ordered_ids_fingerprint,
    preflight,
    sensitivity_contract,
    token_budget,
    train_sensitivity_probe,
    write_sensitivity_adr,
)
from doc2query.reranker.base import FrozenRerankerConfig
from doc2query.reranker.load import load_frozen_reranker
from doc2query.utils.records import JsonlWriter, read_records, write_json


def _mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"P-03 config requires mapping: {key}")
    return value


def _load_probe_recipe(path: Path) -> ProbeRecipe:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("probe recipe must be a YAML mapping")
    return ProbeRecipe.from_dict(raw)


def _load_judge(path: Path) -> FrozenRerankerConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("primary judge config must be a YAML mapping")
    return FrozenRerankerConfig(**raw)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with JsonlWriter(path) as writer:
        for row in rows:
            writer.write(row)


def _select_dev(
    manifest: Path,
    subset: str,
    *,
    limit: int,
    seed: int,
) -> tuple[list[dict[str, Any]], str]:
    if not subset.startswith("dev"):
        raise ValueError("P-03 sensitivity refuses non-development evaluation subsets")
    records = load_frozen_records(manifest, subset)
    ranked = sorted(
        records,
        key=lambda row: (
            __import__("hashlib").sha256(f"{seed}:{row['example_id']}".encode()).hexdigest(),
            str(row["example_id"]),
        ),
    )
    selected = ranked[:limit]
    if len(selected) != limit:
        raise ValueError(f"frozen dev has only {len(selected)} records; requested {limit}")
    ids = [str(row["example_id"]) for row in selected]
    fingerprint = ordered_ids_fingerprint([evaluation_fingerprint(manifest, subset), *ids])
    return selected, fingerprint


def _markdown_report(
    path: Path,
    *,
    generation: dict[str, Any],
    preparation: dict[str, Any],
    comparison: dict[str, Any],
) -> None:
    lines = [
        "# Task 04 P-03 — W05 hard-negative sensitivity",
        "",
        "Status: measured diagnostic; no recipe selected",
        "",
        "This run compares only HN0, HN0+filter and HN1 BM25 for the same W05",
        "synthetic train cohort. Evaluation used only frozen dev. It is neither a",
        "generator comparison nor a final result.",
        "",
        f"- generation fingerprint: `{generation['fingerprint']}`",
        f"- common cohort: {preparation['common_cohort']['count']} examples",
        (f"- common-cohort drop rate: {preparation['common_cohort']['drop_rate']:.6f}"),
        f"- outcome: `{comparison['outcome']}`",
        "- selected recipe: none",
        "- final tests used: none",
        "",
        "## Arm metrics",
        "",
        "| Arm | MRR | nDCG@10 | HN win rate | query/s | peak VRAM bytes |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARM_NAMES:
        block = comparison["arms"][arm]
        metrics = block["metrics"]
        lines.append(
            f"| {arm} | {metrics['pool_mrr']:.6f} | "
            f"{metrics['pool_ndcg_at_10']:.6f} | "
            f"{metrics['pool_hard_negative_win_rate']:.6f} | "
            f"{block['throughput_queries_per_second'] or 0:.4f} | "
            f"{block['peak_vram_allocated_bytes'] or 'CPU'} |"
        )
    lines.extend(
        [
            "",
            "Paired-query bootstrap differences and 95% confidence intervals are in",
            "`sensitivity_report.json`. Flag/drop rates and token budgets are in",
            "`preparation.json`.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(config_path: Path, root: Path) -> dict[str, Any]:
    raw = load_sensitivity_config(config_path)
    check = preflight(raw, root, require_model_cache=True)
    generator = _mapping(raw, "generator")
    inputs = _mapping(raw, "inputs")
    probe_block = _mapping(raw, "probe")
    bootstrap = _mapping(raw, "bootstrap")
    output_dir = root / str(raw["output_dir"])
    report_dir = root / str(raw["report_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = root / str(inputs["frozen_evaluation_manifest"])
    from doc2query.evaluation.p03_sensitivity import frozen_test_ids

    cohort = freeze_train_cohort(
        root / str(inputs["train"]),
        output_path=output_dir / "train_cohort.json",
        limit=int(generator["train_examples"]),
        seed=int(generator["seed"]),
        forbidden_ids=frozen_test_ids(manifest_path),
    )
    ordered_ids = cast(list[str], cohort["ordered_ids"])
    records = materialize_selected_train(root / str(inputs["train"]), ordered_ids)
    generations_path = output_dir / "w05_train_generations.jsonl"
    generation = generate_w05_queries(
        records,
        raw_config=raw,
        cohort_fingerprint=str(cohort["ordered_ids_fingerprint"]),
        journal_path=output_dir / "w05_train_generation.sqlite",
        output_path=generations_path,
    )
    write_json(output_dir / "generation_summary.json", generation)

    base_recipe = _load_probe_recipe(root / str(probe_block["recipe"]))
    preparation_cache_path = output_dir / "preparation_cache.json"
    cached = (
        json.loads(preparation_cache_path.read_text(encoding="utf-8"))
        if preparation_cache_path.is_file()
        else None
    )
    cache_matches = (
        isinstance(cached, dict)
        and cached.get("generation_fingerprint") == generation["fingerprint"]
        and cached.get("frozen_cohort_fingerprint") == cohort["ordered_ids_fingerprint"]
        and all((output_dir / "prepared" / arm / "pairs.jsonl").is_file() for arm in ARM_NAMES)
    )
    if cache_matches:
        cached_mapping = cast(dict[str, Any], cached)
        common = cast(dict[str, Any], cached_mapping["common_cohort"])
        arm_reports = cast(
            dict[str, dict[str, Any]],
            cached_mapping["false_negative_audit"],
        )
        common_rows = {
            arm: list(read_records(output_dir / "prepared" / arm / "pairs.jsonl"))
            for arm in ARM_NAMES
        }
        verified_rows, verified_common = common_cohort(common_rows, ordered_ids)
        if verified_common != common:
            raise ValueError("cached P-03 common cohort fingerprint or order drifted")
        common_rows = verified_rows
    else:
        judge = load_frozen_reranker(_load_judge(root / str(probe_block["primary_judge"])))
        arm_rows: dict[str, list[dict[str, Any]]] = {}
        arm_audits: dict[str, list[dict[str, Any]]] = {}
        arm_reports = {}
        for arm in ARM_NAMES:
            recipe = negative_recipe_for_arm(base_recipe, arm)
            index = (
                load_corpus_index(root / str(inputs["bm25_index"])) if arm == "hn1_bm25" else None
            )
            try:
                rows, _fingerprint, audit_report, audits = prepare_probe_pairs(
                    records,
                    query_source="synthetic",
                    negative_recipe=recipe.negative_recipe,
                    calibration=recipe.negative_recipe.load_calibration(),
                    primary_scorer=judge if recipe.negative_recipe.requires_filter else None,
                    synthetic_generations=generations_path,
                    generator_id="W05-1.5B-50K-8GB",
                    bm25_index=index,
                    documents_path=root / str(inputs["train_corpus"]),
                )
            finally:
                if index is not None:
                    index.close()
            arm_rows[arm] = rows
            arm_audits[arm] = audits
            arm_reports[arm] = audit_report

        common_rows, common = common_cohort(arm_rows, ordered_ids)
        write_json(output_dir / "common_cohort.json", common)
        for arm in ARM_NAMES:
            arm_dir = output_dir / "prepared" / arm
            arm_dir.mkdir(parents=True, exist_ok=True)
            _write_rows(arm_dir / "pairs.jsonl", common_rows[arm])
            _write_rows(arm_dir / "negative_audit.jsonl", arm_audits[arm])
        write_json(
            preparation_cache_path,
            {
                "schema_version": 1,
                "generation_fingerprint": generation["fingerprint"],
                "frozen_cohort_fingerprint": cohort["ordered_ids_fingerprint"],
                "common_cohort": common,
                "false_negative_audit": arm_reports,
            },
        )

    from transformers import AutoTokenizer

    tokenizer_loader: Any = getattr(AutoTokenizer, "from_" + "pretrained")
    tokenizer = tokenizer_loader(
        base_recipe.model_name_or_path,
        revision=base_recipe.revision,
        trust_remote_code=False,
        local_files_only=True,
    )
    budgets = {
        arm: token_budget(
            common_rows[arm],
            tokenizer,
            max_length=base_recipe.max_length,
            max_steps=base_recipe.max_steps,
            batch_size=base_recipe.batch_size,
            seed=base_recipe.seed,
        )
        for arm in ARM_NAMES
    }
    equal_budget = assert_equal_budget(budgets)
    preparation = {
        "status": "measured",
        "common_cohort": common,
        "false_negative_audit": arm_reports,
        "arm_budgets": budgets,
        "equalized_budget": equal_budget,
        "final_tests_used": [],
    }
    write_json(report_dir / "preparation.json", preparation)

    dev_records, dev_fingerprint = _select_dev(
        manifest_path,
        str(inputs["dev_subset"]),
        limit=int(inputs["dev_examples"]),
        seed=base_recipe.seed,
    )
    arm_eval_dirs: dict[str, Path] = {}
    for arm in ARM_NAMES:
        recipe = negative_recipe_for_arm(base_recipe, arm)
        contract = sensitivity_contract(
            recipe=recipe,
            arm=arm,
            cohort=common,
            generation_fingerprint=str(generation["fingerprint"]),
            dev_fingerprint=dev_fingerprint,
            budget=budgets[arm],
        )
        arm_dir = output_dir / "arms" / arm
        train_sensitivity_probe(
            common_rows[arm],
            recipe=recipe,
            output_dir=arm_dir,
            contract=contract,
            checkpoint_steps=int(probe_block["checkpoint_steps"]),
        )
        evaluate_probe_on_dev(
            arm_dir / "model",
            dev_records,
            recipe=recipe,
            output_dir=arm_dir,
            dev_fingerprint=dev_fingerprint,
            contract=contract,
        )
        arm_eval_dirs[arm] = arm_dir

    comparison_path = report_dir / "sensitivity_report.json"
    comparison = compare_sensitivity_arms(
        arm_eval_dirs,
        output_path=comparison_path,
        samples=int(bootstrap["samples"]),
        seed=int(bootstrap["seed"]),
    )
    comparison["artifact_path"] = str(comparison_path)
    write_json(comparison_path, comparison)
    write_sensitivity_adr(
        root / "reports/decisions/task04_p03_w05_negative_recipe.md",
        comparison,
    )
    _markdown_report(
        report_dir / "report.md",
        generation=generation,
        preparation=preparation,
        comparison=comparison,
    )
    result = {
        "status": "measured",
        "scope": "task04-p03-only",
        "preflight": check,
        "generation": generation,
        "preparation": preparation,
        "comparison": comparison,
        "final_tests_used": [],
    }
    write_json(report_dir / "run_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/evaluation/p03_w05_sensitivity.yaml"),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate pins, project-only paths and legal local model access without training.",
    )
    mode.add_argument(
        "--mock-smoke",
        action="store_true",
        help="Exercise resume/cohort/budget contracts with mocks only; no models or GPU.",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        if args.mock_smoke:
            result = mock_smoke(root / ".cache/p03_w05_mock_smoke")
        else:
            raw = load_sensitivity_config(args.config)
            if args.dry_run:
                result = preflight(raw, root, require_model_cache=True)
            else:
                result = run(args.config, root)
    except Exception as exc:
        blocker = {
            "schema_version": 1,
            "status": "blocked",
            "scope": "task04-p03-only",
            "reason": str(exc),
            "final_tests_used": [],
        }
        blocker_path = root / "reports/blockers/task04_p03_w05_sensitivity_runtime.json"
        write_json(blocker_path, blocker)
        print(json.dumps(blocker, ensure_ascii=False, indent=2, sort_keys=True), file=sys.stderr)
        raise SystemExit(2) from None
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
