"""Thin command helpers for model-free Task 06 stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from doc2query.preferences.audit import export_blind_audit, import_blind_audit
from doc2query.preferences.build import (
    build_preference_dataset,
    candidate_fingerprint,
    select_candidate_sets,
    serialize_candidate_sets,
)
from doc2query.preferences.schemas import CandidateSet, ScoredCandidate, SelectionPolicy
from doc2query.utils.records import read_records, write_json


def load_candidates(path: Path) -> list[ScoredCandidate]:
    return [ScoredCandidate.model_validate(row) for row in read_records(path)]


def load_candidate_sets(path: Path) -> list[CandidateSet]:
    return [CandidateSet.model_validate(row) for row in read_records(path)]


def run_select(
    input_path: Path,
    output_path: Path,
    report_path: Path,
    policy: SelectionPolicy,
) -> dict[str, Any]:
    candidates = load_candidates(input_path)
    selected, report = select_candidate_sets(candidates, policy)
    report["candidate_fingerprint"] = candidate_fingerprint(candidates)
    serialize_candidate_sets(output_path, selected)
    write_json(report_path, report)
    return report


def run_build(
    candidates_path: Path,
    candidate_sets_path: Path,
    output_dir: Path,
    output_format: str,
) -> dict[str, Any]:
    return build_preference_dataset(
        load_candidates(candidates_path),
        load_candidate_sets(candidate_sets_path),
        output_dir,
        output_format=output_format,
    )


def run_export_audit(
    preferences_path: Path, output_dir: Path, sample_size: int, seed: int
) -> dict[str, Any]:
    return export_blind_audit(
        list(read_records(preferences_path)), output_dir, sample_size=sample_size, seed=seed
    )


def run_import_audit(
    completed_path: Path,
    machine_key_path: Path,
    output_dir: Path,
    require_complete: bool,
) -> dict[str, Any]:
    return import_blind_audit(
        list(read_records(completed_path)),
        list(read_records(machine_key_path)),
        output_dir,
        require_complete=require_complete,
    )
