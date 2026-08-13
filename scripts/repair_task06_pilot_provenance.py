#!/usr/bin/env python3
"""Mechanically repair the legacy smoke label in completed Task 06 pilot artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from doc2query.utils.records import write_json

OLD_GROUP = "task06-smoke::"
NEW_GROUP = "task06-pilot::"
OLD_EXPERIMENT = "TASK06-SMOKE-"
NEW_EXPERIMENT = "TASK06-PILOT-"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _replace_jsonl(path: Path, expected_rows: int) -> dict[str, Any]:
    before = _sha256(path)
    temporary = path.with_suffix(path.suffix + ".provenance-repair.tmp")
    rows = 0
    group_replacements = 0
    experiment_replacements = 0
    with path.open(encoding="utf-8") as source, temporary.open("w", encoding="utf-8") as target:
        for line in source:
            value = json.loads(line)
            group = value.get("evaluation_group_id")
            if isinstance(group, str) and group.startswith(OLD_GROUP):
                value["evaluation_group_id"] = NEW_GROUP + group.removeprefix(OLD_GROUP)
                group_replacements += 1
            experiment = value.get("experiment_id")
            if isinstance(experiment, str) and experiment.startswith(OLD_EXPERIMENT):
                value["experiment_id"] = NEW_EXPERIMENT + experiment.removeprefix(OLD_EXPERIMENT)
                experiment_replacements += 1
            target.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
            rows += 1
        target.flush()
        os.fsync(target.fileno())
    if rows != expected_rows or group_replacements != expected_rows:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"unexpected pilot repair coverage for {path}")
    os.replace(temporary, path)
    return {
        "path": str(path),
        "rows": rows,
        "group_replacements": group_replacements,
        "experiment_replacements": experiment_replacements,
        "before_sha256": before,
        "after_sha256": _sha256(path),
    }


def _update_json(path: Path, replacements: dict[str, str]) -> dict[str, Any]:
    before = _sha256(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON mapping: {path}")
    value.update(replacements)
    temporary = path.with_suffix(path.suffix + ".provenance-repair.tmp")
    write_json(temporary, value)
    os.replace(temporary, path)
    return {"path": str(path), "before_sha256": before, "after_sha256": _sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    if args.report.exists():
        raise FileExistsError(args.report)
    changes: list[dict[str, Any]] = []
    generation_hashes: dict[str, str] = {}
    for role in ("w06_anchor", "d01_controlled"):
        arm = root / role
        for relative in (
            "generations.jsonl",
            "generations.jsonl.journal.jsonl",
            "scoring/per_generation.jsonl",
            "scoring/scoring.journal.jsonl",
        ):
            changes.append(_replace_jsonl(arm / relative, 2048))
        generation_hash = _sha256(arm / "generations.jsonl")
        generation_hashes[role] = generation_hash
        generation_summary = arm / "generations.jsonl.summary.json"
        changes.append(_update_json(generation_summary, {"output_sha256": generation_hash}))
        scoring_resume = arm / "scoring/scoring.resume.json"
        experiment_id = f"TASK06-PILOT-{role.upper()}"
        changes.append(
            _update_json(
                scoring_resume,
                {
                    "experiment_id": experiment_id,
                    "records_sha256": generation_hash,
                    "test_fingerprint": generation_hash,
                },
            )
        )
        changes.append(
            _update_json(
                arm / "scoring/summary.json",
                {"experiment_id": experiment_id, "test_fingerprint": generation_hash},
            )
        )
    selection = root / "selection"
    archived_selection = root / "invalid_provenance_selection_before_repair"
    if archived_selection.exists():
        raise FileExistsError(archived_selection)
    os.replace(selection, archived_selection)
    report = {
        "schema_version": 1,
        "contract": "task06-pilot-provenance-repair-v1",
        "status": "mechanical_label_repair_selection_requires_rebuild",
        "reason": "pilot rows incorrectly inherited the smoke group and experiment labels",
        "semantic_scores_changed": False,
        "generated_text_changed": False,
        "row_order_changed": False,
        "generation_sha256": generation_hashes,
        "changes": changes,
        "archived_selection": str(archived_selection),
        "final_tests_used": [],
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
