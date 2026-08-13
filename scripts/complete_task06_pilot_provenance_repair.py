#!/usr/bin/env python3
"""Complete the nested Task 06 pilot label repair and propagate all fingerprints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from doc2query.utils.records import write_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _repair(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("task06-smoke::", "task06-pilot::").replace(
            "TASK06-SMOKE-", "TASK06-PILOT-"
        )
    if isinstance(value, list):
        return [_repair(item) for item in value]
    if isinstance(value, dict):
        return {
            (
                "task06_pilot"
                if key == "task06_smoke"
                else key.replace("task06-smoke::", "task06-pilot::").replace(
                    "TASK06-SMOKE-", "TASK06-PILOT-"
                )
            ): _repair(item)
            for key, item in value.items()
        }
    return value


def _rewrite_jsonl(path: Path, transform: Any) -> dict[str, Any]:
    before = _sha256(path)
    temporary = path.with_suffix(path.suffix + ".nested-repair.tmp")
    count = 0
    with path.open(encoding="utf-8") as source, temporary.open("w", encoding="utf-8") as target:
        for line in source:
            value = transform(json.loads(line))
            target.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, path)
    return {
        "path": str(path),
        "rows": count,
        "before_sha256": before,
        "after_sha256": _sha256(path),
    }


def _rewrite_json(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    before = _sha256(path)
    temporary = path.with_suffix(path.suffix + ".nested-repair.tmp")
    write_json(temporary, value)
    os.replace(temporary, path)
    return {"path": str(path), "before_sha256": before, "after_sha256": _sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    root = args.root
    changes: list[dict[str, Any]] = []

    cohort_path = root / "cohort.records.jsonl"
    changes.append(_rewrite_jsonl(cohort_path, _repair))
    cohort_sha = _sha256(cohort_path)
    cohort_manifest_path = root / "cohort.manifest.json"
    cohort_manifest = _repair(json.loads(cohort_manifest_path.read_text(encoding="utf-8")))
    cohort_manifest["records_sha256"] = cohort_sha
    changes.append(_rewrite_json(cohort_manifest_path, cohort_manifest))

    generation_hashes: dict[str, str] = {}
    for role in ("w06_anchor", "d01_controlled"):
        arm = root / role
        identity_path = arm / "generations.jsonl.identity.json"
        identity = _repair(json.loads(identity_path.read_text(encoding="utf-8")))
        identity["cohort_sha256"] = cohort_sha
        identity.pop("identity_sha256", None)
        identity_sha = _canonical_sha256(identity)
        identity["identity_sha256"] = identity_sha
        changes.append(_rewrite_json(identity_path, identity))

        def generation_transform(value: Any, identity_sha_value: str = identity_sha) -> Any:
            result = _repair(value)
            result["frozen_cohort_fingerprint"] = cohort_sha
            result["generation_identity_sha256"] = identity_sha_value
            return result

        for name in ("generations.jsonl", "generations.jsonl.journal.jsonl"):
            changes.append(_rewrite_jsonl(arm / name, generation_transform))
        generation_sha = _sha256(arm / "generations.jsonl")
        generation_hashes[role] = generation_sha
        summary_path = arm / "generations.jsonl.summary.json"
        summary = _repair(json.loads(summary_path.read_text(encoding="utf-8")))
        summary["output_sha256"] = generation_sha
        changes.append(_rewrite_json(summary_path, summary))

        def scoring_transform(value: Any) -> Any:
            result = generation_transform(value)
            return result

        for name in ("scoring/per_generation.jsonl", "scoring/scoring.journal.jsonl"):
            changes.append(_rewrite_jsonl(arm / name, scoring_transform))
        experiment_id = f"TASK06-PILOT-{role.upper()}"
        resume_path = arm / "scoring/scoring.resume.json"
        resume = _repair(json.loads(resume_path.read_text(encoding="utf-8")))
        resume.update(
            {
                "experiment_id": experiment_id,
                "records_sha256": generation_sha,
                "test_fingerprint": generation_sha,
            }
        )
        changes.append(_rewrite_json(resume_path, resume))
        scoring_summary_path = arm / "scoring/summary.json"
        scoring_summary = _repair(
            json.loads(scoring_summary_path.read_text(encoding="utf-8"))
        )
        scoring_summary.update(
            {"experiment_id": experiment_id, "test_fingerprint": generation_sha}
        )
        changes.append(_rewrite_json(scoring_summary_path, scoring_summary))

    report = {
        "schema_version": 1,
        "contract": "task06-pilot-nested-provenance-repair-v1",
        "status": "complete_nested_repair_selection_requires_rebuild",
        "cohort_sha256": cohort_sha,
        "generation_sha256": generation_hashes,
        "generated_text_changed": False,
        "semantic_scores_changed": False,
        "row_order_changed": False,
        "changes": changes,
        "final_tests_used": [],
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
