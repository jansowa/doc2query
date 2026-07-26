"""Prospective, blind translation-integrity audit for natural train records."""

from __future__ import annotations

import csv
import hashlib
import math
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, cast

from doc2query.data.validate import polish_confidence
from doc2query.evaluation.corpus import sha256_file
from doc2query.utils.records import JsonlWriter, read_records, write_json
from doc2query.utils.tracking import collect_code_provenance

AUDIT_SCHEMA = "p06-translation-integrity-audit-v1"
STRATUM_SIZE = 75
STRATA = (
    "low_source_positive_score_decile",
    "low_source_margin_decile",
    "text_quality_or_surface_risk",
    "random_control",
)
MOJIBAKE_MARKERS = ("Ã", "Â", "Ë", "�", "â", "É")
RATING_FIELDS = (
    "query_intent_preserved",
    "answerable_from_positive_passage",
    "translation_semantically_damaged",
    "encoding_or_text_error",
    "repeated_error_class_optional",
    "reviewer_note_optional",
)

REVIEW_INSTRUCTIONS = """# P06-T — instrukcja ślepej oceny

Oceniaj wyłącznie polskie `query` i `positive_passage`. Nie otwieraj manifestu
ani pliku diagnostycznego przed zakończeniem oceny. Formularz celowo nie
zawiera stratum, source score, marginów sędziów ani identyfikatorów źródłowych.

Dozwolone wartości:

- `query_intent_preserved`: `yes`, `no`, `uncertain`;
- `answerable_from_positive_passage`: `yes`, `no`, `uncertain`;
- `translation_semantically_damaged`: `no`, `query`, `passage`, `both`, `uncertain`;
- `encoding_or_text_error`: `no`, `query`, `passage`, `both`, `uncertain`;
- `repeated_error_class_optional`: krótka, spójna etykieta tylko wtedy, gdy ten
  sam typ błędu powtarza się w wielu rekordach;
- `reviewer_note_optional`: krótka uwaga uzasadniająca niejednoznaczną ocenę.

Nie ustalaj automatycznego progu odrzucenia na podstawie tego formularza.
Ewentualna powtarzalna klasa błędu wymaga osobnego prospektywnego ADR.
"""


def _metadata(value: Any) -> dict[str, Any]:
    return cast(dict[str, Any], value) if isinstance(value, dict) else {}


def _stable_key(seed: int, namespace: str, example_id: str) -> str:
    return hashlib.sha256(f"{seed}:{namespace}:{example_id}".encode()).hexdigest()


def _lower_decile(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot calculate a decile from an empty population")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.1 * len(ordered)) - 1)]


def _flag_list(metadata: dict[str, Any], field: str) -> list[str]:
    value = metadata.get(field, [])
    return sorted(str(item) for item in value) if isinstance(value, list) else []


def _surface_diagnostics(query: str, passage: str) -> dict[str, Any]:
    joined = f"{query}\n{passage}"
    marker_count = sum(joined.count(marker) for marker in MOJIBAKE_MARKERS)
    control_count = sum(
        unicodedata.category(character) in {"Cc", "Cf"} and character not in "\n\t"
        for character in joined
    )
    replacement_count = joined.count("�")
    non_ascii_rate = sum(ord(character) > 127 for character in joined) / max(len(joined), 1)
    query_pl = polish_confidence(query)
    passage_pl = polish_confidence(passage)
    # This is only an ordering heuristic. It is deliberately not a classifier or drop score.
    risk = (
        4.0 * marker_count
        + 4.0 * replacement_count
        + 2.0 * control_count
        + (1.0 - query_pl)
        + (1.0 - passage_pl)
        + (0.5 if len(query.split()) <= 1 else 0.0)
        + (0.5 if len(passage.split()) <= 3 else 0.0)
    )
    return {
        "surface_risk": risk,
        "mojibake_marker_count": marker_count,
        "replacement_character_count": replacement_count,
        "control_character_count": control_count,
        "non_ascii_character_rate": non_ascii_rate,
        "query_polish_confidence": query_pl,
        "passage_polish_confidence": passage_pl,
        "query_word_count": len(query.split()),
        "passage_word_count": len(passage.split()),
    }


def _audit_candidate(record: dict[str, Any]) -> dict[str, Any]:
    example_id = str(record["example_id"])
    query = str(record["query"])
    positives = record.get("positives")
    if not isinstance(positives, list) or not positives:
        raise ValueError(f"record {example_id} has no positive passages")

    def positive_key(value: Any) -> tuple[float, str]:
        positive = cast(dict[str, Any], value)
        score = _metadata(positive.get("metadata")).get("source_en_score")
        if not isinstance(score, (int, float)):
            raise ValueError(f"record {example_id} has a positive without source_en_score")
        return float(score), str(positive["doc_id"])

    positive = cast(dict[str, Any], min(positives, key=positive_key))
    positive_metadata = _metadata(positive.get("metadata"))
    record_metadata = _metadata(record.get("metadata"))
    score = positive_metadata.get("source_en_score")
    margin = record_metadata.get("source_en_difference_between_max_scores")
    if not isinstance(score, (int, float)) or not isinstance(margin, (int, float)):
        raise ValueError(f"record {example_id} is missing source provenance")
    query_flags = _flag_list(record_metadata, "query_text_quality_flags")
    passage_flags = _flag_list(positive_metadata, "text_quality_flags")
    query_text = str(query)
    passage_text = str(positive["text"])
    return {
        "example_id": example_id,
        "doc_id": str(positive["doc_id"]),
        "query": query_text,
        "passage": passage_text,
        "positive_count": len(positives),
        "source_en_positive_score": float(score),
        "source_en_margin": float(margin),
        "source_score_language": str(positive_metadata.get("source_score_language", "en")),
        "source": str(record_metadata.get("source", "unknown")),
        "source_revision": str(record_metadata.get("source_revision", "unknown")),
        "split": str(record_metadata.get("split", "unknown")),
        "query_text_quality_flags": query_flags,
        "passage_text_quality_flags": passage_flags,
        "has_text_quality_flags": bool(query_flags or passage_flags),
        **_surface_diagnostics(query_text, passage_text),
    }


def _take(
    candidates: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
    namespace: str,
) -> list[dict[str, Any]]:
    if len(candidates) < count:
        raise ValueError(f"stratum {namespace} has only {len(candidates)} candidates for {count}")
    return sorted(
        candidates,
        key=lambda row: _stable_key(seed, namespace, str(row["example_id"])),
    )[:count]


def _write_blind_form(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["review_order", "audit_id", "query", "positive_passage", *RATING_FIELDS]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def freeze_translation_audit(
    input_path: Path,
    *,
    output_dir: Path,
    seed: int = 42,
    stratum_size: int = STRATUM_SIZE,
    registry_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Freeze four disjoint strata and export a score-blind review form."""
    if stratum_size < 1:
        raise ValueError("stratum_size must be positive")
    candidates = [_audit_candidate(record) for record in read_records(input_path)]
    if len({str(row["example_id"]) for row in candidates}) != len(candidates):
        raise ValueError("P06-T requires unique train example IDs")
    if any(row["split"] != "train" for row in candidates):
        raise ValueError("P06-T accepts train records only")

    score_p10 = _lower_decile([float(row["source_en_positive_score"]) for row in candidates])
    margin_p10 = _lower_decile([float(row["source_en_margin"]) for row in candidates])
    selected: list[dict[str, Any]] = []
    used: set[str] = set()

    def add(rows: list[dict[str, Any]], stratum: str) -> None:
        for row in rows:
            copied = dict(row)
            copied["stratum"] = stratum
            selected.append(copied)
            used.add(str(row["example_id"]))

    low_score = [row for row in candidates if row["source_en_positive_score"] <= score_p10]
    add(
        _take(low_score, count=stratum_size, seed=seed, namespace=STRATA[0]),
        STRATA[0],
    )
    low_margin = [
        row
        for row in candidates
        if row["example_id"] not in used and row["source_en_margin"] <= margin_p10
    ]
    add(
        _take(low_margin, count=stratum_size, seed=seed, namespace=STRATA[1]),
        STRATA[1],
    )
    remaining = [row for row in candidates if row["example_id"] not in used]
    flagged = [row for row in remaining if row["has_text_quality_flags"]]
    flagged_selected = _take(
        flagged,
        count=min(stratum_size, len(flagged)),
        seed=seed,
        namespace=f"{STRATA[2]}:flagged",
    )
    flag_ids = {str(row["example_id"]) for row in flagged_selected}
    fill_count = stratum_size - len(flagged_selected)
    surface_fill = sorted(
        (row for row in remaining if row["example_id"] not in flag_ids),
        key=lambda row: (
            -float(row["surface_risk"]),
            _stable_key(seed, f"{STRATA[2]}:fill", str(row["example_id"])),
        ),
    )[:fill_count]
    add(flagged_selected + surface_fill, STRATA[2])
    controls = [row for row in candidates if row["example_id"] not in used]
    add(
        _take(controls, count=stratum_size, seed=seed, namespace=STRATA[3]),
        STRATA[3],
    )

    for row in selected:
        row["audit_id"] = (
            "p06t-"
            + hashlib.sha256(f"{seed}:{row['example_id']}:{row['doc_id']}".encode()).hexdigest()[
                :12
            ]
        )
    review_rows = sorted(
        selected,
        key=lambda row: (
            -float(row["surface_risk"]),
            _stable_key(seed, "review-order", str(row["example_id"])),
        ),
    )
    blind_rows = [
        {
            "review_order": index,
            "audit_id": row["audit_id"],
            "query": row["query"],
            "positive_passage": row["passage"],
            **{field: "" for field in RATING_FIELDS},
        }
        for index, row in enumerate(review_rows, start=1)
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    form_csv = output_dir / "blind_review_form.csv"
    form_jsonl = output_dir / "blind_review_form.jsonl"
    diagnostics_path = output_dir / "triage_diagnostics.jsonl"
    instructions_path = output_dir / "review_instructions.md"
    manifest_path = output_dir / "manifest.json"
    _write_blind_form(form_csv, blind_rows)
    instructions_path.write_text(REVIEW_INSTRUCTIONS, encoding="utf-8")
    with JsonlWriter(form_jsonl) as writer:
        for row in blind_rows:
            writer.write(row)
    with JsonlWriter(diagnostics_path) as writer:
        for index, row in enumerate(review_rows, start=1):
            writer.write(
                {
                    "review_order": index,
                    "audit_id": row["audit_id"],
                    "example_id": row["example_id"],
                    "doc_id": row["doc_id"],
                    "stratum": row["stratum"],
                    "has_text_quality_flags": row["has_text_quality_flags"],
                    "query_text_quality_flags": row["query_text_quality_flags"],
                    "passage_text_quality_flags": row["passage_text_quality_flags"],
                    **{
                        key: row[key]
                        for key in (
                            "surface_risk",
                            "mojibake_marker_count",
                            "replacement_character_count",
                            "control_character_count",
                            "non_ascii_character_rate",
                            "query_polish_confidence",
                            "passage_polish_confidence",
                            "query_word_count",
                            "passage_word_count",
                        )
                    },
                    "primary_margin": None,
                    "shadow_margin": None,
                    "judge_rank_disagreement": None,
                }
            )

    selected_by_id = sorted(selected, key=lambda row: str(row["audit_id"]))
    selected_ids_payload = "\n".join(
        f"{row['audit_id']}\t{row['example_id']}\t{row['doc_id']}\t{row['stratum']}"
        for row in selected_by_id
    )
    stratum_counts = Counter(str(row["stratum"]) for row in selected)
    manifest = {
        "schema": AUDIT_SCHEMA,
        "schema_version": 1,
        "status": "sample_frozen_manual_review_pending",
        "adr": "docs/decisions/task03_p06_source_provenance_2026-07-26.md",
        "seed": seed,
        "unit": "unique_train_query_record_with_lowest_source_score_positive",
        "population_count": len(candidates),
        "sample_count": len(selected),
        "stratum_size": stratum_size,
        "stratum_order": list(STRATA),
        "stratum_counts": dict(sorted(stratum_counts.items())),
        "thresholds": {
            "source_en_positive_score_lower_decile_inclusive": score_p10,
            "source_en_margin_lower_decile_inclusive": margin_p10,
        },
        "quality_stratum": {
            "flagged_selected": len(flagged_selected),
            "surface_risk_fill_selected": len(surface_fill),
        },
        "input": {"path": str(input_path), "sha256": sha256_file(input_path)},
        "artifacts": {
            "blind_review_form_csv": {"path": str(form_csv), "sha256": sha256_file(form_csv)},
            "blind_review_form_jsonl": {
                "path": str(form_jsonl),
                "sha256": sha256_file(form_jsonl),
            },
            "triage_diagnostics": {
                "path": str(diagnostics_path),
                "sha256": sha256_file(diagnostics_path),
            },
            "review_instructions": {
                "path": str(instructions_path),
                "sha256": sha256_file(instructions_path),
            },
        },
        "selected_ids_sha256": hashlib.sha256(selected_ids_payload.encode()).hexdigest(),
        "selected_records": [
            {
                "audit_id": row["audit_id"],
                "example_id": row["example_id"],
                "doc_id": row["doc_id"],
                "stratum": row["stratum"],
                "source": row["source"],
                "source_revision": row["source_revision"],
                "source_score_language": row["source_score_language"],
                "source_en_positive_score": row["source_en_positive_score"],
                "source_en_margin": row["source_en_margin"],
                "positive_count": row["positive_count"],
            }
            for row in selected_by_id
        ],
        "blindness": {
            "form_excludes": [
                "example_id",
                "doc_id",
                "stratum",
                "source_en_positive_score",
                "source_en_margin",
                "primary_margin",
                "shadow_margin",
                "judge_rank_disagreement",
                "surface diagnostics",
            ],
            "review_order": "surface-risk-descending with deterministic hash tie-break",
        },
        "diagnostic_policy": {
            "purpose": "manual-review triage only",
            "drop_threshold_defined": False,
            "training_weights_defined": False,
            "may_override_source_labels": False,
        },
        "final_tests_used": [],
        "code": collect_code_provenance(),
    }
    write_json(manifest_path, manifest)
    if registry_manifest_path is not None:
        write_json(registry_manifest_path, manifest)
    return manifest
