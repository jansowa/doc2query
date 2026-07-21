#!/usr/bin/env python3
"""Finalize already-scored P-05 inputs to one deterministic query per passage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from doc2query.evaluation.corpus import sha256_file
from doc2query.evaluation.p05_materializer import P05_NEGATIVE_RECIPE, W05_GENERATOR_ID
from doc2query.evaluation.statistical_contract import build_budget_manifest
from doc2query.utils.records import JsonlWriter, read_records, write_json


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with JsonlWriter(path) as writer:
        for row in rows:
            writer.write(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--natural-input", type=Path, required=True)
    parser.add_argument("--w05-input", type=Path, required=True)
    parser.add_argument("--natural-fingerprint", type=Path, required=True)
    parser.add_argument("--w05-fingerprint", type=Path, required=True)
    parser.add_argument("--natural-output", type=Path, required=True)
    parser.add_argument("--w05-output", type=Path, required=True)
    parser.add_argument("--natural-fingerprint-output", type=Path, required=True)
    parser.add_argument("--w05-fingerprint-output", type=Path, required=True)
    parser.add_argument("--budget-output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--token-count", type=int, required=True)
    args = parser.parse_args()

    natural_manifest = _load(args.natural_fingerprint)
    w05_manifest = _load(args.w05_fingerprint)
    for label, manifest, path in (
        ("natural", natural_manifest, args.natural_input),
        ("W05", w05_manifest, args.w05_input),
    ):
        if manifest.get("sha256") != sha256_file(path):
            raise ValueError(f"{label} input SHA-256 drift")
        if manifest.get("negative_recipe") != P05_NEGATIVE_RECIPE:
            raise ValueError(f"{label} input does not prove HN0+filter/drop")
        if manifest.get("final_tests_used") != []:
            raise ValueError(f"{label} input must declare final_tests_used=[]")
    if natural_manifest.get("eligible_pair_ids_sha256") != w05_manifest.get(
        "eligible_pair_ids_sha256"
    ):
        raise ValueError("natural/W05 eligible cohort drift")

    natural_rows = list(read_records(args.natural_input))
    w05_by_id = {str(row["pair_id"]): row for row in read_records(args.w05_input)}
    if len(w05_by_id) != len(natural_rows):
        raise ValueError("natural/W05 inputs differ or contain duplicate pair IDs")
    selected: list[dict[str, Any]] = []
    seen_docs: set[str] = set()
    for row in natural_rows:
        pair_id = str(row["pair_id"])
        doc_id = str(row["doc_id"])
        if pair_id not in w05_by_id:
            raise ValueError(f"missing W05 pair: {pair_id}")
        if doc_id in seen_docs:
            continue
        seen_docs.add(doc_id)
        selected.append(row)
    budget_count = len(selected) - len(selected) % 8
    selected = selected[:budget_count]
    selected_ids = [str(row["pair_id"]) for row in selected]
    selected_w05 = [dict(w05_by_id[pair_id]) for pair_id in selected_ids]
    _write_rows(args.natural_output, selected)
    _write_rows(args.w05_output, selected_w05)

    eligible_hash = _canonical_hash(sorted(selected_ids))
    common = {
        "splits": ["train"],
        "final_tests_used": [],
        "negative_recipe": P05_NEGATIVE_RECIPE,
        "eligible_pair_ids_sha256": eligible_hash,
        "calibration": natural_manifest.get("calibration"),
    }
    natural_sha = sha256_file(args.natural_output)
    write_json(
        args.natural_fingerprint_output,
        common | {"artifact_path": str(args.natural_output.resolve()), "sha256": natural_sha},
    )
    write_json(
        args.w05_fingerprint_output,
        common
        | {
            "artifact_path": str(args.w05_output.resolve()),
            "sha256": sha256_file(args.w05_output),
            "source_data_sha256": natural_sha,
            "generator_id": W05_GENERATOR_ID,
        },
    )
    budget = build_budget_manifest(
        token_count=args.token_count,
        pair_count=budget_count,
        unique_passage_count=budget_count,
        queries_per_passage=1,
    )
    write_json(args.budget_output, budget)
    audit = {
        "schema_version": 1,
        "status": "prepared",
        "selection": "first post-filter pair in pinned P-03 order per unique doc_id",
        "input_pair_count": len(natural_rows),
        "input_unique_passage_count": len(seen_docs),
        "materialization_budget_count": budget_count,
        "dropped_duplicate_passage_pairs": len(natural_rows) - len(seen_docs),
        "dropped_divisibility_tail": len(seen_docs) - budget_count,
        "eligible_pair_ids_sha256": eligible_hash,
        "comparison_budget": budget,
        "final_tests_used": [],
    }
    write_json(args.audit_output, audit)
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
