"""Blind human-audit export and import for selected preference pairs."""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from doc2query.utils.records import JsonlWriter, write_json


def export_blind_audit(
    preferences: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    sample_size: int,
    seed: int,
) -> dict[str, Any]:
    """Create a deterministic blind A/B form and a separate machine key."""
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    ids = [str(row.get("preference_id", "")) for row in preferences]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("preference_id must be present and unique")
    ordered = sorted(preferences, key=lambda row: str(row["preference_id"]))
    rng = random.Random(seed)
    sample = rng.sample(ordered, k=min(sample_size, len(ordered)))
    sample.sort(key=lambda row: str(row["preference_id"]))
    form_path = output_dir / "blind_review_form.jsonl"
    key_path = output_dir / "machine_key.jsonl"
    with JsonlWriter(form_path) as form_writer, JsonlWriter(key_path) as key_writer:
        for row in sample:
            preference_id = str(row["preference_id"])
            chosen_is_a = bool(rng.getrandbits(1))
            audit_id = hashlib.sha256(f"{seed}\0{preference_id}".encode()).hexdigest()[:24]
            option_a = str(row["chosen"] if chosen_is_a else row["rejected"])
            option_b = str(row["rejected"] if chosen_is_a else row["chosen"])
            form_writer.write(
                {
                    "audit_id": audit_id,
                    "prompt": row["prompt"],
                    "option_a": option_a,
                    "option_b": option_b,
                    "human_preference": None,
                    "reason": None,
                    "option_a_answerable": None,
                    "option_b_answerable": None,
                    "notes": None,
                }
            )
            key_writer.write(
                {
                    "audit_id": audit_id,
                    "preference_id": preference_id,
                    "automatic_chosen_option": "A" if chosen_is_a else "B",
                    "split": row.get("split"),
                    "rejected_failure_types": row.get("rejected_failure_types", []),
                }
            )
    manifest = {
        "schema_version": 1,
        "contract": "task06-preference-blind-audit-v1",
        "population_count": len(preferences),
        "sample_count": len(sample),
        "sample_size_requested": sample_size,
        "seed": seed,
        "blind_form": str(form_path),
        "machine_key": str(key_path),
        "final_tests_used": [],
    }
    write_json(output_dir / "audit_manifest.json", manifest)
    return manifest


def import_blind_audit(
    completed_rows: Sequence[Mapping[str, Any]],
    machine_key_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Validate, unblind and summarize human agreement with automatic ranking."""
    key = {str(row.get("audit_id", "")): row for row in machine_key_rows}
    if "" in key or len(key) != len(machine_key_rows):
        raise ValueError("machine key audit_id values must be present and unique")
    completed: dict[str, Mapping[str, Any]] = {}
    allowed = {"A", "B", "tie", "invalid"}
    for row in completed_rows:
        audit_id = str(row.get("audit_id", ""))
        if not audit_id or audit_id in completed:
            raise ValueError("completed audit_id values must be present and unique")
        if audit_id not in key:
            raise ValueError(f"unknown audit_id: {audit_id}")
        preference = row.get("human_preference")
        if preference not in allowed:
            raise ValueError(f"invalid human_preference for {audit_id}: {preference}")
        completed[audit_id] = row
    missing = sorted(set(key) - set(completed))
    if missing and require_complete:
        raise ValueError(f"audit is incomplete: {len(missing)} rows missing")

    counts: Counter[str] = Counter()
    by_failure: dict[str, Counter[str]] = {}
    unblinded_path = output_dir / "unblinded_audit.jsonl"
    with JsonlWriter(unblinded_path) as writer:
        for audit_id, row in sorted(completed.items()):
            key_row = key[audit_id]
            human = str(row["human_preference"])
            automatic = str(key_row["automatic_chosen_option"])
            outcome = (
                "tie_or_invalid"
                if human in {"tie", "invalid"}
                else ("agree" if human == automatic else "disagree")
            )
            counts[outcome] += 1
            failure_types = [str(value) for value in key_row.get("rejected_failure_types", [])]
            option_a_answerable = row.get("option_a_answerable")
            option_b_answerable = row.get("option_b_answerable")
            chosen_answerable = option_a_answerable if automatic == "A" else option_b_answerable
            rejected_answerable = option_b_answerable if automatic == "A" else option_a_answerable
            for failure_type in failure_types or ["unclassified"]:
                by_failure.setdefault(failure_type, Counter())[outcome] += 1
            writer.write(
                {
                    "audit_id": audit_id,
                    "preference_id": key_row["preference_id"],
                    "human_preference": human,
                    "automatic_chosen_option": automatic,
                    "outcome": outcome,
                    "reason": row.get("reason"),
                    "chosen_answerable": chosen_answerable,
                    "rejected_answerable": rejected_answerable,
                    "rejected_failure_types": failure_types,
                    "notes": row.get("notes"),
                }
            )
    decided = counts["agree"] + counts["disagree"]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "contract": "task06-preference-audit-result-v1",
        "key_count": len(key),
        "completed_count": len(completed),
        "missing_count": len(missing),
        "counts": dict(sorted(counts.items())),
        "automatic_human_agreement": counts["agree"] / decided if decided else None,
        "by_rejected_failure_type": {
            name: dict(sorted(value.items())) for name, value in sorted(by_failure.items())
        },
        "unblinded_rows": str(unblinded_path),
        "final_tests_used": [],
    }
    write_json(output_dir / "audit_summary.json", summary)
    return summary
