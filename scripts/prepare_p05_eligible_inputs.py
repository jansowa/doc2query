#!/usr/bin/env python3
"""Build the dual-source HN0+filter eligible inputs required by P-05."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from doc2query.evaluation.corpus import sha256_file
from doc2query.evaluation.embedder_probe import ProbeRecipe, prepare_probe_pairs
from doc2query.evaluation.p05_materializer import P05_NEGATIVE_RECIPE, W05_GENERATOR_ID
from doc2query.evaluation.statistical_contract import build_budget_manifest
from doc2query.reranker.base import FrozenRerankerConfig
from doc2query.reranker.load import load_frozen_reranker
from doc2query.utils.records import JsonlWriter, read_records, write_json


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with JsonlWriter(path) as writer:
        for row in rows:
            writer.write(row)


def _progress(stage: str) -> Callable[[int, int], None]:
    started = time.monotonic()
    last_completed = -1

    def report(completed: int, total: int) -> None:
        nonlocal last_completed
        if completed == last_completed:
            return
        last_completed = completed
        percent = 100.0 * completed / max(1, total)
        elapsed = time.monotonic() - started
        print(
            f"[{stage}] {completed:,}/{total:,} examples ({percent:5.1f}%) elapsed={elapsed:.1f}s",
            file=sys.stderr,
            flush=True,
        )

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--w05-generations", type=Path, required=True)
    parser.add_argument("--p03-common-cohort", type=Path, required=True)
    parser.add_argument("--probe-recipe", type=Path, required=True)
    parser.add_argument("--primary-judge-config", type=Path, required=True)
    parser.add_argument("--natural-output", type=Path, required=True)
    parser.add_argument("--w05-output", type=Path, required=True)
    parser.add_argument("--natural-fingerprint-output", type=Path, required=True)
    parser.add_argument("--w05-fingerprint-output", type=Path, required=True)
    parser.add_argument("--budget-output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()

    print("[preflight] loading pinned recipe and calibration", file=sys.stderr, flush=True)

    recipe_raw = yaml.safe_load(args.probe_recipe.read_text(encoding="utf-8"))
    judge_raw = yaml.safe_load(args.primary_judge_config.read_text(encoding="utf-8"))
    if not isinstance(recipe_raw, dict) or not isinstance(judge_raw, dict):
        raise ValueError("probe recipe and primary judge config must be mappings")
    recipe = ProbeRecipe.from_dict(recipe_raw)
    if recipe.negative_recipe.strategy != "hn0_filter" or (
        recipe.negative_recipe.false_negative_policy != "drop"
    ):
        raise ValueError("P-05 input preparation requires HN0+filter/drop")
    calibration = recipe.negative_recipe.load_calibration()
    if calibration is None:
        raise ValueError("P-05 input preparation requires pinned calibration")
    judge = FrozenRerankerConfig(**judge_raw)
    if judge.name_or_path != calibration.primary_judge_name or (
        judge.revision != calibration.primary_judge_revision
    ):
        raise ValueError("primary judge does not match calibration provenance")

    common = _load_object(args.p03_common_cohort)
    ordered_raw = common.get("ordered_example_ids")
    if not isinstance(ordered_raw, list) or not all(isinstance(item, str) for item in ordered_raw):
        raise ValueError("P-03 common cohort requires ordered_example_ids")
    ordered_ids = list(ordered_raw)
    if common.get("count") != len(ordered_ids) or len(set(ordered_ids)) != len(ordered_ids):
        raise ValueError("P-03 common cohort count/IDs are inconsistent")
    if common.get("ordered_example_ids_fingerprint") != _canonical_hash(ordered_ids):
        raise ValueError("P-03 common cohort fingerprint drift")
    wanted = set(ordered_ids)

    natural_by_id: dict[str, dict[str, Any]] = {}
    for record in read_records(args.train_input):
        example_id = str(record.get("example_id", ""))
        if example_id not in wanted:
            continue
        positives = record.get("positives")
        if not isinstance(positives, list) or not positives:
            raise ValueError(f"missing positive for P-05 example {example_id}")
        first = sorted(positives, key=lambda item: str(item["doc_id"]))[0]
        natural_by_id[example_id] = dict(record) | {
            "example_id": example_id,
            "pair_id": example_id,
            "doc_id": str(first["doc_id"]),
            "passage": str(first["text"]),
            "split": "train",
        }
    if set(natural_by_id) != wanted:
        raise ValueError("P-03 common cohort is not fully present in train input")

    source_w05: dict[str, dict[str, Any]] = {}
    for row in read_records(args.w05_generations):
        example_id = str(row.get("example_id", ""))
        if example_id in wanted:
            if example_id in source_w05:
                raise ValueError(f"duplicate W05 generation: {example_id}")
            source_w05[example_id] = dict(row)
    if set(source_w05) != wanted:
        raise ValueError("P-03 common cohort is not fully present in W05 generations")

    ordered_natural = [natural_by_id[example_id] for example_id in ordered_ids]
    print(
        f"[preflight] loaded {len(ordered_natural):,} common examples; "
        f"loading frozen judge on {judge.device}",
        file=sys.stderr,
        flush=True,
    )
    scorer = load_frozen_reranker(judge)
    print("[natural] scoring inherited hard negatives", file=sys.stderr, flush=True)
    natural_pairs, _, natural_report, _ = prepare_probe_pairs(
        ordered_natural,
        query_source="natural",
        negative_recipe=recipe.negative_recipe,
        calibration=calibration,
        primary_scorer=scorer,
        progress=_progress("natural"),
    )
    print(
        f"[natural] eligible={len(natural_pairs):,}; phase complete",
        file=sys.stderr,
        flush=True,
    )
    print("[synthetic_w05] scoring inherited hard negatives", file=sys.stderr, flush=True)
    synthetic_pairs, _, synthetic_report, _ = prepare_probe_pairs(
        ordered_natural,
        query_source="synthetic",
        negative_recipe=recipe.negative_recipe,
        calibration=calibration,
        primary_scorer=scorer,
        synthetic_generations=args.w05_generations,
        generator_id=W05_GENERATOR_ID,
        progress=_progress("synthetic_w05"),
    )
    print(
        f"[synthetic_w05] eligible={len(synthetic_pairs):,}; phase complete",
        file=sys.stderr,
        flush=True,
    )
    eligible = {str(row["example_id"]) for row in natural_pairs} & {
        str(row["example_id"]) for row in synthetic_pairs
    }
    eligible_ids = []
    selected_docs: set[str] = set()
    for example_id in ordered_ids:
        doc_id = str(natural_by_id[example_id]["doc_id"])
        if example_id in eligible and doc_id not in selected_docs:
            eligible_ids.append(example_id)
            selected_docs.add(doc_id)
    budget_count = len(eligible_ids) - len(eligible_ids) % 8
    if budget_count < 8:
        raise ValueError("dual-source HN0+filter eligible cohort is too small")
    eligible_ids = eligible_ids[:budget_count]
    eligible_hash = _canonical_hash(sorted(eligible_ids))

    natural_rows = [natural_by_id[example_id] for example_id in eligible_ids]
    w05_rows = [
        {
            "example_id": example_id,
            "pair_id": example_id,
            "doc_id": natural_by_id[example_id]["doc_id"],
            "passage": natural_by_id[example_id]["passage"],
            "generated": str(source_w05[example_id]["generated"]),
            "generator_id": W05_GENERATOR_ID,
            "mode": "deterministic",
            "candidate_index": 0,
            "split": "train",
        }
        for example_id in eligible_ids
    ]
    _write_jsonl(args.natural_output, natural_rows)
    _write_jsonl(args.w05_output, w05_rows)
    common_manifest = {
        "splits": ["train"],
        "final_tests_used": [],
        "negative_recipe": P05_NEGATIVE_RECIPE,
        "eligible_pair_ids_sha256": eligible_hash,
        "calibration": calibration.to_manifest(),
    }
    natural_sha = sha256_file(args.natural_output)
    write_json(
        args.natural_fingerprint_output,
        common_manifest
        | {"artifact_path": str(args.natural_output.resolve()), "sha256": natural_sha},
    )
    write_json(
        args.w05_fingerprint_output,
        common_manifest
        | {
            "artifact_path": str(args.w05_output.resolve()),
            "sha256": sha256_file(args.w05_output),
            "source_data_sha256": natural_sha,
            "generator_id": W05_GENERATOR_ID,
        },
    )
    budget = build_budget_manifest(
        token_count=(
            recipe.max_steps
            * recipe.batch_size
            * recipe.max_length
            * (2 + recipe.negatives_per_example)
        ),
        pair_count=budget_count,
        unique_passage_count=budget_count,
        queries_per_passage=1,
    )
    write_json(args.budget_output, budget)
    audit = {
        "schema_version": 1,
        "status": "prepared",
        "input_sha256": {
            "train": sha256_file(args.train_input),
            "w05_generations": sha256_file(args.w05_generations),
            "p03_common_cohort": sha256_file(args.p03_common_cohort),
            "probe_recipe": sha256_file(args.probe_recipe),
            "primary_judge_config": sha256_file(args.primary_judge_config),
        },
        "p03_common_count": len(ordered_ids),
        "natural_eligible_count": len(natural_pairs),
        "synthetic_eligible_count": len(synthetic_pairs),
        "intersection_budget_count": budget_count,
        "eligible_pair_ids_sha256": eligible_hash,
        "natural_false_negative_report": natural_report,
        "synthetic_false_negative_report": synthetic_report,
        "probe_recipe": asdict(recipe),
        "comparison_budget": budget,
        "final_tests_used": [],
    }
    write_json(args.audit_output, audit)
    print(
        f"[done] common eligible budget={budget_count:,}; audit={args.audit_output}",
        file=sys.stderr,
        flush=True,
    )
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
