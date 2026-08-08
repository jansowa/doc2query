"""Fail-closed, model-free assembly of Task 06 scoring evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TypeVar

from doc2query.generation.deduplicate import query_key
from doc2query.preferences.schemas import (
    CandidateEvidenceBundle,
    CandidateGenerationRequest,
    CorpusRetrievalEvidence,
    EvidenceIdentity,
    FocusEvidence,
    FormatEvidence,
    GeneratedCandidate,
    LexicalCopyEvidence,
    PrimaryJudgeEvidence,
    ShadowJudgeEvidence,
    StyleEvidence,
)
from doc2query.utils.records import JsonParquetWriter, read_records, write_json

E = TypeVar("E", bound=EvidenceIdentity)

EVIDENCE_CONTRACT_VERSIONS = {
    "bundle": "task06-candidate-evidence-v1",
    "candidate": "task06-generated-candidate-v1",
    "request": "task06-candidate-generation-request-v1",
    "component": "task06-scoring-evidence-v1",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_unique(path: Path, model: type[E], id_field: str) -> dict[str, E]:
    records: dict[str, E] = {}
    duplicates: list[str] = []
    for raw in read_records(path):
        record = model.model_validate(raw)
        record_id = str(getattr(record, id_field))
        if record_id in records:
            duplicates.append(record_id)
        records[record_id] = record
    if duplicates:
        raise ValueError(f"duplicate {id_field} in {path}: {sorted(set(duplicates))}")
    return records


def _coverage(label: str, expected: set[str], actual: set[str]) -> None:
    missing = sorted(expected - actual)
    orphan = sorted(actual - expected)
    if missing or orphan:
        raise ValueError(
            f"{label} coverage mismatch: missing={len(missing)} {missing[:5]}, "
            f"orphan={len(orphan)} {orphan[:5]}"
        )


def _request_candidate_drift(
    request: CandidateGenerationRequest, candidate: GeneratedCandidate
) -> list[str]:
    drift: list[str] = []
    pairs = {
        "request_id": (request.request_id, candidate.request_id),
        "plan_id": (request.plan_id, candidate.plan_id),
        "plan_fingerprint": (request.plan_fingerprint, candidate.plan_fingerprint),
        "passage_id": (request.passage_id, candidate.passage_id),
        "passage_cluster_id": (request.passage_cluster_id, candidate.passage_cluster_id),
        "passage": (request.passage, candidate.passage),
        "split": (request.split, candidate.split),
        "prompt": (request.prompt, candidate.prompt),
        "control": (
            request.control.model_dump(mode="json"),
            candidate.control.model_dump(mode="json"),
        ),
        "temperature": (request.temperature, candidate.provenance.decoding.temperature),
        "top_p": (request.top_p, candidate.provenance.decoding.top_p),
        "max_new_tokens": (
            request.max_new_tokens,
            candidate.provenance.decoding.max_new_tokens,
        ),
        "seed": (request.seed, candidate.provenance.decoding.seed),
    }
    for field, (expected, actual) in pairs.items():
        if expected != actual:
            drift.append(field)
    return drift


def validate_generated_candidates(
    requests: Iterable[dict[str, Any]], candidates: Iterable[dict[str, Any]]
) -> tuple[list[CandidateGenerationRequest], list[GeneratedCandidate], dict[str, Any]]:
    """Validate exact request coverage without invoking a generator."""
    parsed_requests = [CandidateGenerationRequest.model_validate(row) for row in requests]
    parsed_candidates = [GeneratedCandidate.model_validate(row) for row in candidates]
    request_ids = [row.request_id for row in parsed_requests]
    candidate_ids = [row.candidate_id for row in parsed_candidates]
    candidate_request_ids = [row.request_id for row in parsed_candidates]
    for label, values in (
        ("request_id", request_ids),
        ("candidate_id", candidate_ids),
        ("candidate request_id", candidate_request_ids),
    ):
        duplicates = sorted(key for key, count in Counter(values).items() if count > 1)
        if duplicates:
            raise ValueError(f"duplicate {label}: {duplicates[:5]}")
    _coverage("generated candidates", set(request_ids), set(candidate_request_ids))
    request_by_id = {row.request_id: row for row in parsed_requests}
    generator_identities: set[tuple[str, ...]] = set()
    cluster_splits: dict[str, str] = {}
    passage_identity: dict[str, tuple[str, str, str]] = {}
    normalized_queries: dict[str, str] = {}
    for candidate in parsed_candidates:
        request = request_by_id[candidate.request_id]
        drift = _request_candidate_drift(request, candidate)
        if drift:
            raise ValueError(f"request/candidate drift for {candidate.candidate_id}: {drift}")
        provenance = candidate.provenance
        generator_identities.add(
            (
                provenance.model_id,
                provenance.model_revision,
                provenance.checkpoint_id,
                provenance.checkpoint_fingerprint,
                provenance.adapter_id or "",
                provenance.adapter_fingerprint or "",
                provenance.plan_id,
                provenance.plan_fingerprint,
            )
        )
        prior_split = cluster_splits.setdefault(candidate.passage_cluster_id, candidate.split)
        if prior_split != candidate.split:
            raise ValueError(
                f"passage cluster {candidate.passage_cluster_id} crosses generation splits"
            )
        identity = (candidate.passage, candidate.split, candidate.passage_cluster_id)
        prior_identity = passage_identity.setdefault(candidate.passage_id, identity)
        if prior_identity != identity:
            raise ValueError(f"passage identity drift for {candidate.passage_id}")
        normalized = query_key(candidate.query)
        if not normalized:
            raise ValueError(f"empty normalized query for {candidate.candidate_id}")
        if normalized in normalized_queries:
            raise ValueError(
                "duplicate query after normalization: "
                f"{normalized_queries[normalized]} and {candidate.candidate_id}"
            )
        normalized_queries[normalized] = candidate.candidate_id
        if candidate.duplicate_within_request or candidate.duplicate_candidate_ids:
            raise ValueError(f"candidate {candidate.candidate_id} is flagged as duplicate")
    if len(generator_identities) != 1:
        raise ValueError("generator model/revision/checkpoint/adapter or plan drift")
    ordered_requests = sorted(parsed_requests, key=lambda row: row.request_id)
    ordered_candidates = sorted(parsed_candidates, key=lambda row: row.request_id)
    report = {
        "contract_versions": EVIDENCE_CONTRACT_VERSIONS,
        "request_count": len(ordered_requests),
        "candidate_count": len(ordered_candidates),
        "complete_count": len(ordered_candidates),
        "missing_count": 0,
        "orphan_count": 0,
        "duplicate_count": 0,
        "status": "generated_candidates_validated_not_scored",
        "model_scoring_performed": False,
        "final_tests_used": [],
    }
    return ordered_requests, ordered_candidates, report


def validate_generated_candidate_files(
    requests_path: Path, candidates_path: Path
) -> dict[str, Any]:
    _, _, report = validate_generated_candidates(
        read_records(requests_path), read_records(candidates_path)
    )
    report["input_sha256"] = {
        "requests": sha256_file(requests_path),
        "candidates": sha256_file(candidates_path),
    }
    return report


def _identity(candidate: GeneratedCandidate) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "request_id": candidate.request_id,
        "plan_id": candidate.plan_id,
        "plan_fingerprint": candidate.plan_fingerprint,
        "passage_id": candidate.passage_id,
        "passage_cluster_id": candidate.passage_cluster_id,
        "passage": candidate.passage,
        "split": candidate.split,
    }


def _validate_component_identity(label: str, candidate: GeneratedCandidate, evidence: E) -> None:
    expected = _identity(candidate)
    actual = evidence.model_dump(mode="json", include=set(expected))
    drift = sorted(field for field in expected if expected[field] != actual[field])
    if drift:
        raise ValueError(f"{label} identity drift for {candidate.candidate_id}: {drift}")


def _load_evidence(label: str, path: Path, model: type[E], candidate_ids: set[str]) -> dict[str, E]:
    records = _load_unique(path, model, "candidate_id")
    _coverage(label, candidate_ids, set(records))
    return records


def _write_bundles_atomic(path: Path, bundles: list[CandidateEvidenceBundle]) -> None:
    if path.exists():
        raise FileExistsError(f"evidence output already exists: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if path.suffix == ".jsonl":
            with temporary.open("w", encoding="utf-8") as handle:
                for bundle in bundles:
                    payload = json.dumps(
                        bundle.model_dump(mode="json"), ensure_ascii=False, sort_keys=True
                    )
                    handle.write(payload + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        elif path.suffix == ".parquet":
            with JsonParquetWriter(temporary) as writer:
                for bundle in bundles:
                    writer.write(bundle.model_dump(mode="json"))
        else:
            raise ValueError("evidence output must have .jsonl or .parquet suffix")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def assemble_candidate_evidence(
    *,
    requests_path: Path,
    candidates_path: Path,
    primary_path: Path,
    shadow_path: Path,
    corpus_path: Path,
    lexical_path: Path,
    focus_path: Path,
    style_path: Path,
    format_path: Path,
    output_path: Path,
    manifest_path: Path,
    primary_judge_id: str,
    primary_judge_revision: str,
    shadow_judge_id: str,
    shadow_judge_revision: str,
    margin_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Join precomputed evidence exactly once; never score, calibrate or rank."""
    if margin_tolerance < 0 or not math.isfinite(margin_tolerance):
        raise ValueError("margin_tolerance must be finite and non-negative")
    if (primary_judge_id, primary_judge_revision) == (
        shadow_judge_id,
        shadow_judge_revision,
    ):
        raise ValueError("primary and shadow judges must be independently pinned")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError("evidence output or manifest already exists")
    _, candidates, _ = validate_generated_candidates(
        read_records(requests_path), read_records(candidates_path)
    )
    candidate_ids = {row.candidate_id for row in candidates}
    sources: dict[str, Path] = {
        "requests": requests_path,
        "candidates": candidates_path,
        "primary_judge": primary_path,
        "shadow_judge": shadow_path,
        "corpus_retrieval": corpus_path,
        "lexical_copy": lexical_path,
        "focus": focus_path,
        "style": style_path,
        "format": format_path,
    }
    primary = _load_evidence("primary judge", primary_path, PrimaryJudgeEvidence, candidate_ids)
    shadow = _load_evidence("shadow judge", shadow_path, ShadowJudgeEvidence, candidate_ids)
    corpus = _load_evidence("corpus retrieval", corpus_path, CorpusRetrievalEvidence, candidate_ids)
    lexical = _load_evidence("lexical/copy", lexical_path, LexicalCopyEvidence, candidate_ids)
    focus = _load_evidence("focus", focus_path, FocusEvidence, candidate_ids)
    style = _load_evidence("style", style_path, StyleEvidence, candidate_ids)
    formats = _load_evidence("format", format_path, FormatEvidence, candidate_ids)
    bundles: list[CandidateEvidenceBundle] = []
    for candidate in sorted(candidates, key=lambda row: row.candidate_id):
        candidate_id = candidate.candidate_id
        components: list[tuple[str, EvidenceIdentity]] = [
            ("primary judge", primary[candidate_id]),
            ("shadow judge", shadow[candidate_id]),
            ("corpus retrieval", corpus[candidate_id]),
            ("lexical/copy", lexical[candidate_id]),
            ("focus", focus[candidate_id]),
            ("style", style[candidate_id]),
            ("format", formats[candidate_id]),
        ]
        for label, component in components:
            _validate_component_identity(label, candidate, component)
        primary_row = primary[candidate_id]
        shadow_row = shadow[candidate_id]
        if (primary_row.judge_id, primary_row.judge_revision) != (
            primary_judge_id,
            primary_judge_revision,
        ):
            raise ValueError(f"primary judge ID/revision drift for {candidate_id}")
        if (shadow_row.judge_id, shadow_row.judge_revision) != (
            shadow_judge_id,
            shadow_judge_revision,
        ):
            raise ValueError(f"shadow judge ID/revision drift for {candidate_id}")
        if primary_row.raw_score_scale_id == shadow_row.raw_score_scale_id:
            raise ValueError(f"primary and shadow raw score scales are mixed for {candidate_id}")
        for label, judge in (("primary", primary_row), ("shadow", shadow_row)):
            recomputed = judge.positive_score - judge.max_negative_score
            if not math.isclose(judge.margin, recomputed, rel_tol=0.0, abs_tol=margin_tolerance):
                raise ValueError(f"incorrect {label} margin for {candidate_id}")
        if (
            candidate.format_valid is not None
            and candidate.format_valid != formats[candidate_id].valid
        ):
            raise ValueError(f"format validity drift for {candidate_id}")
        bundles.append(
            CandidateEvidenceBundle(
                candidate=candidate,
                primary_judge=primary_row,
                shadow_judge=shadow_row,
                corpus_retrieval=corpus[candidate_id],
                lexical_copy=lexical[candidate_id],
                focus=focus[candidate_id],
                style=style[candidate_id],
                format=formats[candidate_id],
            )
        )
    _write_bundles_atomic(output_path, bundles)
    flag_distributions = {
        "candidate_format_valid": dict(
            sorted(Counter(str(row.format_valid) for row in candidates).items())
        ),
        "format_valid": dict(sorted(Counter(str(row.valid) for row in formats.values()).items())),
        "copy_risk": dict(sorted(Counter(str(row.copy_risk) for row in lexical.values()).items())),
        "primary_all_scores_close": dict(
            sorted(Counter(str(row.all_scores_close) for row in primary.values()).items())
        ),
        "shadow_all_scores_close": dict(
            sorted(Counter(str(row.all_scores_close) for row in shadow.values()).items())
        ),
    }
    artifact_fingerprint = sha256_file(output_path)
    manifest: dict[str, Any] = {
        "contract_versions": EVIDENCE_CONTRACT_VERSIONS,
        "status": "evidence_assembled_not_ranked",
        "input_sha256": {label: sha256_file(path) for label, path in sorted(sources.items())},
        "output_sha256": artifact_fingerprint,
        "artifact_fingerprint": artifact_fingerprint,
        "counts": {
            "complete": len(bundles),
            "missing": 0,
            "orphan": 0,
            "duplicate": 0,
        },
        "flag_distributions": flag_distributions,
        "primary_judge": {"id": primary_judge_id, "revision": primary_judge_revision},
        "shadow_judge": {"id": shadow_judge_id, "revision": shadow_judge_revision},
        "margin_tolerance": margin_tolerance,
        "model_scoring_performed_by_assembler": False,
        "final_tests_used": [],
    }
    write_json(manifest_path, manifest)
    return manifest
