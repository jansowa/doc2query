"""Prospective CPU-only calibration and blind audits for Task 05."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any, cast

from doc2query.data.style_labels import QueryLabels, intent_applicable, label_query
from doc2query.evaluation.datasets import evaluation_fingerprint, load_frozen_records
from doc2query.generation.concepts import extract_concepts
from doc2query.text.normalization import SimplePolishNormalizer
from doc2query.utils.records import read_durable_jsonl_prefix, read_records, write_json
from doc2query.utils.tracking import collect_code_provenance

CONTRACT = "task05-natural-calibration-audits-v1"
ALLOWED_SUBSET = "dev_intrinsic_rank10"
MATERIALIZATION_SCHEMA = "task05-natural-audits-materialization-v1"
LABEL_AGGREGATION_SCHEMA = "task05-label-audit-aggregation-v1"
CONCEPT_AGGREGATION_SCHEMA = "task05-concept-audit-aggregation-v1"
FORM_VALUES = ("full_question", "keyword_query", "unknown")
INTENT_VALUES = ("fact_lookup", "definition", "entity_lookup", "procedure", "comparison", "unknown")
_PROPER_NAME = re.compile(r"(?<![.!?]\s)\b[A-ZĄĆĘŁŃÓŚŹŻ][\wąćęłńóśźż-]{2,}\b")
REVIEW_INSTRUCTIONS = """# Task 05 — blind natural-query audits

Use a separate copy of each blind CSV per reviewer. Never open the machine key,
automatic proposals, calibration report, or confidence values before submitting
the blind ratings. Do not fill ratings automatically.

Label audit values:

- `gold_form`: `full_question`, `keyword_query`, or `unknown`;
- `gold_intent`: `fact_lookup`, `definition`, `entity_lookup`, `procedure`,
  `comparison`, or `unknown`;
- `intent_adequate`, `ambiguous`, `encoding_error`: `yes`, `no`, or `uncertain`.

Concept audit values:

- concept ID lists use `|` separators; list all correct and spurious IDs;
- `missing_important_concepts` is free text, empty only when nothing important is missing;
- `numbers_units_correct`, `over_fragmented`, `duplicate_concepts`,
  `useful_for_coverage`, `ambiguous`, `encoding_error`: `yes`, `no`, or `uncertain`.

Every row requires `reviewer_id`. Two independent ratings are required by the
contract. Resolve every disagreement in the separate adjudication CSV. Aggregators
remain `incomplete` while ratings or required adjudications are missing.
"""


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("contract") != CONTRACT:
        raise ValueError(f"unsupported Task 05 audit contract: {path}")
    if value.get("frozen_subset") != ALLOWED_SUBSET:
        raise ValueError("Task 05 natural audits may use only frozen dev_intrinsic_rank10")
    if value.get("final_tests_used") != []:
        raise ValueError("Task 05 natural audit contract must have final_tests_used=[]")
    forbidden = set(cast(list[str], value.get("forbidden_inputs", [])))
    if not {"D01_results", "final_tests"}.issubset(forbidden):
        raise ValueError("contract must explicitly forbid D01 results and final tests")
    if int(value.get("label_audit_size", 0)) < 500:
        raise ValueError("label audit must contain at least 500 records")
    return value


def _positive(record: Mapping[str, Any]) -> Mapping[str, Any]:
    positives = record.get("positives")
    if not isinstance(positives, list) or not positives:
        raise ValueError(f"record {record.get('example_id')} has no positive passage")
    return cast(Mapping[str, Any], sorted(positives, key=lambda row: str(row["doc_id"]))[0])


def _metadata(value: Any) -> Mapping[str, Any]:
    return cast(Mapping[str, Any], value) if isinstance(value, dict) else {}


def _domain_source(record: Mapping[str, Any]) -> tuple[str, str]:
    metadata = _metadata(record.get("metadata"))
    source = str(metadata.get("source") or "unknown")
    domain = str(metadata.get("domain") or source or "unknown")
    return domain, source


def _bin(value: float, boundaries: Sequence[float]) -> str:
    lower = 0.0
    for upper in boundaries:
        if value <= upper:
            return f"{lower:g}-{upper:g}"
        lower = upper
    return f">{boundaries[-1]:g}"


def _record_features(
    record: Mapping[str, Any], labels: QueryLabels, config: Mapping[str, Any]
) -> dict[str, Any]:
    positive = _positive(record)
    query = str(record.get("query", ""))
    passage = str(positive.get("text", ""))
    domain, source = _domain_source(record)
    normalizer = SimplePolishNormalizer()
    query_analysis = normalizer.analyze(query)
    passage_analysis = normalizer.analyze(passage)
    confidence = min(labels.form_confidence, labels.intent_confidence)
    return {
        "example_id": str(record["example_id"]),
        "doc_id": str(positive["doc_id"]),
        "query": query,
        "passage": passage,
        "domain": domain,
        "source": source,
        "predicted_form": labels.form.value,
        "predicted_intent": labels.intent.value,
        "form_confidence": labels.form_confidence,
        "intent_confidence": labels.intent_confidence,
        "confidence": confidence,
        "confidence_bin": _bin(confidence, cast(list[float], config["confidence_bins"])),
        "abstention": labels.form.value == "unknown" or labels.intent.value == "unknown",
        "intent_applicable": intent_applicable(labels.intent, passage),
        "query_word_count": len(query_analysis.tokens),
        "passage_word_count": len(passage_analysis.tokens),
        "query_length_bin": _bin(
            len(query_analysis.tokens), cast(list[int], config["query_length_bins_words"])
        ),
        "passage_length_bin": _bin(
            len(passage_analysis.tokens), cast(list[int], config["passage_length_bins_words"])
        ),
        "has_numbers": bool(query_analysis.numbers or passage_analysis.numbers),
        "has_units": bool(query_analysis.units or passage_analysis.units),
        "has_entity_signal": bool(_PROPER_NAME.search(query) or _PROPER_NAME.search(passage)),
    }


def _concept_features(record: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    positive = _positive(record)
    passage = str(positive["text"])
    concepts = extract_concepts(
        passage, max_concepts=int(config["concept_extractor"]["max_concepts"])
    )
    domain, source = _domain_source(record)
    analysis = SimplePolishNormalizer().analyze(passage)
    return {
        "example_id": str(record["example_id"]),
        "doc_id": str(positive["doc_id"]),
        "passage": passage,
        "domain": domain,
        "source": source,
        "passage_word_count": len(analysis.tokens),
        "passage_length_bin": _bin(
            len(analysis.tokens), cast(list[int], config["passage_length_bins_words"])
        ),
        "has_numbers": bool(analysis.numbers),
        "has_units": bool(analysis.units),
        "has_entity_signal": bool(_PROPER_NAME.search(passage)),
        "concept_count": len(concepts),
        "concept_count_bin": _bin(len(concepts), [4, 8, 16]),
        "concepts": [
            {"concept_id": f"c{index:02d}", "text": item.text, "kind": item.kind}
            for index, item in enumerate(concepts, 1)
        ],
    }


def _append_durable(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_json(temporary, value)
    os.replace(temporary, path)


def _archive(output_dir: Path, previous: Any, requested: Any) -> Path:
    archive = (
        output_dir
        / "interrupted"
        / f"{_canonical_sha256(previous)[:12]}-to-{_canonical_sha256(requested)[:12]}"
    )
    suffix = 1
    while archive.exists():
        archive = archive.with_name(f"{archive.name}-{suffix}")
        suffix += 1
    archive.mkdir(parents=True)
    for source in list(output_dir.iterdir()):
        if source.is_file():
            shutil.move(str(source), archive / source.name)
    write_json(
        archive / "archive_manifest.json",
        {"reason": "incompatible_resume_identity", "previous": previous, "requested": requested},
    )
    return archive


def _journal_stage(
    records: Sequence[Mapping[str, Any]],
    journal: Path,
    builder: Callable[[Mapping[str, Any]], dict[str, Any]],
    *,
    stage: str,
) -> list[dict[str, Any]]:
    rows = read_durable_jsonl_prefix(journal)
    expected = [str(row["example_id"]) for row in records]
    actual = [str(row["example_id"]) for row in rows]
    if actual != expected[: len(actual)]:
        raise ValueError(f"{stage} journal is not the deterministic frozen cohort prefix")
    if len(rows) > len(records):
        raise ValueError(f"{stage} journal exceeds frozen cohort")
    started = time.monotonic()
    resumed = len(rows)
    for index, record in enumerate(records[len(rows) :], start=len(rows) + 1):
        built = builder(record)
        _append_durable(journal, built)
        rows.append(built)
        if index == len(records) or index % 250 == 0:
            elapsed = max(time.monotonic() - started, 1e-9)
            rate = (index - resumed) / elapsed
            remaining = len(records) - index
            eta_seconds = remaining / rate if rate else None
            print(
                f"[{stage}] {index}/{len(records)} remaining={remaining} "
                f"rate={rate:.1f}/s eta_seconds={eta_seconds:.1f}",
                flush=True,
            )
    return rows


def _allocation_domain(
    row: Mapping[str, Any], domain_counts: Mapping[str, int], minimum: int
) -> str:
    domain = str(row["domain"])
    return domain if domain_counts[domain] >= minimum else "__small_domains__"


def stratified_sample(
    rows: Sequence[Mapping[str, Any]],
    *,
    size: int,
    axes: Sequence[str],
    seed: int,
    small_domain_minimum: int,
) -> list[dict[str, Any]]:
    """Greedily maximize marginal categorical coverage with a stable hash tie-break."""
    if size > len(rows):
        raise ValueError(f"requested sample {size} exceeds population {len(rows)}")
    domain_counts = Counter(str(row["domain"]) for row in rows)
    normalized: list[dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        copied["allocation_domain"] = _allocation_domain(row, domain_counts, small_domain_minimum)
        normalized.append(copied)
    selected: list[dict[str, Any]] = []
    counts: Counter[tuple[str, str]] = Counter()
    remaining = list(normalized)
    for _ in range(size):

        def key(row: Mapping[str, Any]) -> tuple[float, str, str]:
            score = sum(
                1.0
                / (
                    1
                    + counts[
                        (axis, str(row["allocation_domain"] if axis == "domain" else row[axis]))
                    ]
                )
                for axis in axes
            )
            stable = hashlib.sha256(
                f"{seed}:{row['example_id']}:{row['doc_id']}".encode()
            ).hexdigest()
            return (-score, stable, str(row["example_id"]))

        winner = min(remaining, key=key)
        remaining.remove(winner)
        selected.append(winner)
        for axis in axes:
            value = winner["allocation_domain"] if axis == "domain" else winner[axis]
            counts[(axis, str(value))] += 1
    return selected


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _distribution(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    return dict(
        sorted(
            Counter(str(row[field]) if row[field] is not None else "null" for row in rows).items()
        )
    )


def _calibration_summary(
    rows: Sequence[Mapping[str, Any]], minimum_reportable: int
) -> dict[str, Any]:
    def block(group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "count": len(group),
            "form": _distribution(group, "predicted_form"),
            "intent": _distribution(group, "predicted_intent"),
            "abstention_count": sum(bool(row["abstention"]) for row in group),
            "intent_applicable": _distribution(group, "intent_applicable"),
            "confidence": {
                "mean": fmean(float(row["confidence"]) for row in group),
                "bins": _distribution(group, "confidence_bin"),
            },
            "query_length_bins": _distribution(group, "query_length_bin"),
            "passage_length_bins": _distribution(group, "passage_length_bin"),
            "has_numbers": _distribution(group, "has_numbers"),
            "has_entity_signal": _distribution(group, "has_entity_signal"),
        }

    def slices(field: str) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, group in sorted(_group(rows, field).items()):
            result[name] = block(group) | {"calibration_reliable": len(group) >= minimum_reportable}
        return result

    return {
        "global": block(rows),
        "domains": slices("domain"),
        "sources": slices("source"),
        "minimum_reportable_count": minimum_reportable,
    }


def _write_materialized_artifacts(
    output_dir: Path,
    calibration: list[dict[str, Any]],
    concepts: list[dict[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    stratification = cast(Mapping[str, Any], config["stratification"])
    small = cast(Mapping[str, Any], stratification["small_domains"])
    label_rows = stratified_sample(
        calibration,
        size=int(config["label_audit_size"]),
        axes=cast(list[str], stratification["label_audit"]),
        seed=int(config["seed"]),
        small_domain_minimum=int(small["minimum_population"]),
    )
    unique_concepts = list({str(row["doc_id"]): row for row in concepts}.values())
    concept_rows = stratified_sample(
        unique_concepts,
        size=int(config["concept_audit_size"]),
        axes=cast(list[str], stratification["concept_audit"]),
        seed=int(config["seed"]) + 1,
        small_domain_minimum=int(small["minimum_population"]),
    )
    label_blind: list[dict[str, Any]] = []
    label_key: list[dict[str, Any]] = []
    for order, row in enumerate(label_rows, 1):
        audit_key = f"{config['seed']}:{row['example_id']}"
        audit_id = f"L-{order:04d}-{hashlib.sha256(audit_key.encode()).hexdigest()[:8]}"
        label_blind.append(
            {
                "review_order": order,
                "audit_id": audit_id,
                "query": row["query"],
                "positive_passage": row["passage"],
                "domain": row["domain"],
                "source": row["source"],
            }
        )
        label_key.append({"audit_id": audit_id, **row})
    concept_blind: list[dict[str, Any]] = []
    concept_key: list[dict[str, Any]] = []
    for order, row in enumerate(concept_rows, 1):
        audit_key = f"{config['seed']}:{row['doc_id']}"
        audit_id = f"C-{order:04d}-{hashlib.sha256(audit_key.encode()).hexdigest()[:8]}"
        concept_blind.append(
            {
                "review_order": order,
                "audit_id": audit_id,
                "passage": row["passage"],
                "domain": row["domain"],
                "source": row["source"],
                "candidate_concepts": " | ".join(
                    f"{item['concept_id']}={item['text']}" for item in row["concepts"]
                ),
            }
        )
        concept_key.append({"audit_id": audit_id, **row})
    label_fields = [
        "review_order",
        "audit_id",
        "query",
        "positive_passage",
        "domain",
        "source",
        "reviewer_id",
        "gold_form",
        "gold_intent",
        "intent_adequate",
        "ambiguous",
        "encoding_error",
        "comment",
    ]
    concept_fields = [
        "review_order",
        "audit_id",
        "passage",
        "domain",
        "source",
        "candidate_concepts",
        "reviewer_id",
        "correct_concept_ids",
        "spurious_concept_ids",
        "missing_important_concepts",
        "numbers_units_correct",
        "over_fragmented",
        "duplicate_concepts",
        "useful_for_coverage",
        "ambiguous",
        "encoding_error",
        "comment",
    ]
    _write_csv(output_dir / "label_audit_blind.csv", label_fields, label_blind)
    _atomic_jsonl(output_dir / "label_audit_machine_key.jsonl", label_key)
    _write_csv(
        output_dir / "label_adjudication.csv",
        [
            "audit_id",
            "adjudicator_id",
            "final_gold_form",
            "final_gold_intent",
            "final_intent_adequate",
            "resolution_note",
        ],
        [{"audit_id": row["audit_id"]} for row in label_blind],
    )
    _write_csv(output_dir / "concept_audit_blind.csv", concept_fields, concept_blind)
    _atomic_jsonl(output_dir / "concept_audit_machine_proposals.jsonl", concept_key)
    _write_csv(
        output_dir / "concept_adjudication.csv",
        ["audit_id", "adjudicator_id", "resolved", "resolution_note"],
        [{"audit_id": row["audit_id"]} for row in concept_blind],
    )
    summary = _calibration_summary(calibration, int(small["minimum_reportable_count"]))
    write_json(output_dir / "calibration_summary.json", summary)
    markdown = [
        "# Task 05 natural-query calibration",
        "",
        f"Population: {len(calibration)} natural frozen-dev queries.",
        "",
        "This is descriptive calibration, not a style-accuracy gate.",
        "",
        "## Global distributions",
        "",
        f"- Form: `{summary['global']['form']}`",
        f"- Intent: `{summary['global']['intent']}`",
        f"- Abstention: `{summary['global']['abstention_count']}`",
        "",
        "Small domain slices are marked unreliable in the JSON report.",
        "",
    ]
    (output_dir / "calibration_report.md").write_text("\n".join(markdown), encoding="utf-8")
    (output_dir / "REVIEW_INSTRUCTIONS.md").write_text(REVIEW_INSTRUCTIONS, encoding="utf-8")
    return {
        "label_audit_count": len(label_blind),
        "concept_audit_count": len(concept_blind),
        "calibration": summary,
    }


def materialize_natural_audits(
    contract_path: Path,
    *,
    output_dir: Path,
    archive_incompatible: bool = False,
    max_records: int | None = None,
    labeler: Callable[[str], QueryLabels] = label_query,
) -> dict[str, Any]:
    config = load_contract(contract_path)
    manifest = Path(str(config["frozen_manifest"]))
    records = load_frozen_records(manifest, str(config["frozen_subset"]))
    if max_records is not None:
        if max_records < max(int(config["label_audit_size"]), int(config["concept_audit_size"])):
            config = dict(config) | {
                "label_audit_size": min(max_records, 12),
                "concept_audit_size": min(max_records, 8),
                "smoke_max_records": max_records,
            }
        records = records[:max_records]
    identity_base = {
        "schema": MATERIALIZATION_SCHEMA,
        "contract_sha256": _file_sha256(contract_path),
        "resolved_contract_sha256": _canonical_sha256(config),
        "frozen_manifest_sha256": _file_sha256(manifest),
        "frozen_subset": config["frozen_subset"],
        "frozen_cohort_fingerprint": evaluation_fingerprint(manifest, str(config["frozen_subset"])),
        "record_ids_sha256": _canonical_sha256([row["example_id"] for row in records]),
        "seed": config["seed"],
        "final_tests_used": [],
    }
    identity = identity_base | {"identity_sha256": _canonical_sha256(identity_base)}
    output_dir.mkdir(parents=True, exist_ok=True)
    identity_path = output_dir / "identity.json"
    previous = (
        json.loads(identity_path.read_text(encoding="utf-8")) if identity_path.exists() else None
    )
    if previous is not None and previous != identity:
        if not archive_incompatible:
            raise ValueError(
                "resume identity mismatch; pass --archive-incompatible to archive recoverably"
            )
        _archive(output_dir, previous, identity)
        previous = None
    if previous is None and any(
        (output_dir / name).exists()
        for name in ("calibration.journal.jsonl", "concepts.journal.jsonl")
    ):
        raise ValueError("journal exists without identity")
    if previous is None:
        write_json(identity_path, identity)
    calibration = _journal_stage(
        records,
        output_dir / "calibration.journal.jsonl",
        lambda record: _record_features(record, labeler(str(record.get("query", ""))), config),
        stage="calibration",
    )
    concepts = _journal_stage(
        records,
        output_dir / "concepts.journal.jsonl",
        lambda record: _concept_features(record, config),
        stage="concepts",
    )
    _atomic_jsonl(output_dir / "calibration_rows.jsonl", calibration)
    artifacts = _write_materialized_artifacts(output_dir, calibration, concepts, config)
    manifest_payload = {
        "schema": MATERIALIZATION_SCHEMA,
        "status": "materialized_unreviewed",
        "identity": identity,
        "counts": {
            "population": len(records),
            **{key: value for key, value in artifacts.items() if key.endswith("count")},
        },
        "files": {
            path.name: _file_sha256(path)
            for path in sorted(output_dir.iterdir())
            if path.is_file() and path.name != "materialization_manifest.json"
        },
        "review_status": {"label_audit": "NOT MEASURED", "concept_audit": "NOT MEASURED"},
        "provenance": collect_code_provenance(),
        "final_tests_used": [],
    }
    _atomic_json(output_dir / "materialization_manifest.json", manifest_payload)
    return manifest_payload


def _read_csvs(paths: Sequence[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(encoding="utf-8", newline="") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
    return rows


def _kappa(pairs: Sequence[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    observed = sum(a == b for a, b in pairs) / len(pairs)
    left, right = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    labels = set(left) | set(right)
    expected = sum(left[label] / len(pairs) * right[label] / len(pairs) for label in labels)
    return None if expected == 1.0 else (observed - expected) / (1.0 - expected)


def _fleiss(matrix: Sequence[Sequence[str]]) -> float | None:
    if not matrix or any(len(row) != len(matrix[0]) for row in matrix):
        return None
    n = len(matrix[0])
    if n < 2:
        return None
    labels = sorted({value for row in matrix for value in row})
    p_bar = fmean(
        (sum(count * count for count in Counter(row).values()) - n) / (n * (n - 1))
        for row in matrix
    )
    totals = Counter(value for row in matrix for value in row)
    p_e = sum((totals[label] / (len(matrix) * n)) ** 2 for label in labels)
    return None if p_e == 1.0 else (p_bar - p_e) / (1.0 - p_e)


def _agreement(grouped: Mapping[str, Sequence[Mapping[str, str]]], field: str) -> dict[str, Any]:
    ordered = [
        sorted(rows, key=lambda row: row["reviewer_id"]) for _, rows in sorted(grouped.items())
    ]
    reviewer_count = min((len(rows) for rows in ordered), default=0)
    if reviewer_count == 2:
        return {
            "method": "cohen_kappa",
            "value": _kappa([(rows[0][field], rows[1][field]) for rows in ordered]),
        }
    return {
        "method": "fleiss_kappa",
        "value": _fleiss([[row[field] for row in rows] for rows in ordered]),
    }


def _classification_metrics(
    items: Sequence[Mapping[str, Any]], field: str, gold_field: str, labels: Sequence[str]
) -> dict[str, Any]:
    matrix = {gold: {pred: 0 for pred in labels} for gold in labels}
    for item in items:
        matrix[str(item[gold_field])][str(item[field])] += 1
    per_class: dict[str, Any] = {}
    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[gold][label] for gold in labels if gold != label)
        fn = sum(matrix[label][pred] for pred in labels if pred != label)
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall
            else None
        )
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(matrix[label].values()),
        }
    covered = [item for item in items if item[field] != "unknown"]
    return {
        "confusion_matrix": matrix,
        "per_class": per_class,
        "coverage": len(covered) / len(items) if items else 0.0,
        "accuracy_covered": sum(item[field] == item[gold_field] for item in covered) / len(covered)
        if covered
        else None,
        "abstention_count": len(items) - len(covered),
    }


def _adjudications(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    return {
        row["audit_id"]: row
        for row in _read_csvs([path])
        if any(value.strip() for key, value in row.items() if key != "audit_id")
    }


def _write_aggregation_report(path: Path, result: Mapping[str, Any], title: str) -> None:
    missing = len(cast(Sequence[Any], result["missing_required_ratings"]))
    unresolved = len(cast(Sequence[Any], result["unresolved_adjudications"]))
    lines = [
        f"# {title}",
        "",
        f"Status: `{result['status']}`.",
        "",
        f"Expected audited items: {result['expected_items']}.",
        f"Missing required ratings: {missing}.",
        f"Unresolved adjudications: {unresolved}.",
        "",
        "Agreement estimates and detailed metrics are in the adjacent JSON report.",
        "A status of `incomplete` is intentional and must not be reported as a completed audit.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def aggregate_label_audit(
    machine_key: Path,
    ratings: Sequence[Path],
    *,
    adjudication: Path | None,
    output_dir: Path,
    required_reviewers: int = 2,
) -> dict[str, Any]:
    key = {str(row["audit_id"]): row for row in read_records(machine_key)}
    raw = _read_csvs(ratings)
    grouped: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    invalid: list[str] = []
    seen_ratings: set[tuple[str, str]] = set()
    for index, row in enumerate(raw, 1):
        pair = (row.get("audit_id", ""), row.get("reviewer_id", ""))
        valid = (
            pair[0] in key
            and bool(pair[1])
            and row.get("gold_form") in FORM_VALUES
            and row.get("gold_intent") in INTENT_VALUES
            and row.get("intent_adequate") in {"yes", "no", "uncertain"}
            and pair not in seen_ratings
        )
        if not valid:
            invalid.append(f"row:{index}")
            continue
        seen_ratings.add(pair)
        grouped[pair[0]].append(row)
    raw_adjudicated = _adjudications(adjudication)
    adjudicated = {
        audit_id: row
        for audit_id, row in raw_adjudicated.items()
        if row.get("adjudicator_id")
        and row.get("final_gold_form") in FORM_VALUES
        and row.get("final_gold_intent") in INTENT_VALUES
        and row.get("final_intent_adequate") in {"yes", "no", "uncertain"}
    }
    missing = sorted(
        audit_id
        for audit_id in key
        if len({row["reviewer_id"] for row in grouped[audit_id]}) < required_reviewers
    )
    disagreements = sorted(
        audit_id
        for audit_id, rows in grouped.items()
        if len({(row["gold_form"], row["gold_intent"], row["intent_adequate"]) for row in rows}) > 1
    )
    unresolved = sorted(set(disagreements) - set(adjudicated))
    complete = not missing and not unresolved and not invalid
    items: list[dict[str, Any]] = []
    if complete:
        for audit_id, predicted in key.items():
            rows = grouped[audit_id]
            if audit_id in adjudicated:
                adj = adjudicated[audit_id]
                gold_form, gold_intent = adj["final_gold_form"], adj["final_gold_intent"]
            else:
                gold_form, gold_intent = rows[0]["gold_form"], rows[0]["gold_intent"]
            items.append(dict(predicted) | {"gold_form": gold_form, "gold_intent": gold_intent})
    result: dict[str, Any] = {
        "schema": LABEL_AGGREGATION_SCHEMA,
        "status": "complete" if complete else "incomplete",
        "required_reviewers": required_reviewers,
        "expected_items": len(key),
        "missing_required_ratings": missing,
        "invalid_rating_rows": invalid,
        "unresolved_adjudications": unresolved,
        "agreement": {
            "form": _agreement(grouped, "gold_form"),
            "intent": _agreement(grouped, "gold_intent"),
        },
        "input_fingerprints": {
            "machine_key_sha256": _file_sha256(machine_key),
            "ratings_sha256": {str(path): _file_sha256(path) for path in ratings},
            "adjudication_sha256": (
                _file_sha256(adjudication)
                if adjudication is not None and adjudication.exists()
                else None
            ),
        },
        "provenance": collect_code_provenance(),
        "final_tests_used": [],
    }
    if complete:
        result["form"] = _classification_metrics(items, "predicted_form", "gold_form", FORM_VALUES)
        result["intent"] = _classification_metrics(
            items, "predicted_intent", "gold_intent", INTENT_VALUES
        )
        result["per_domain"] = {
            domain: {
                "count": len(group),
                "form_accuracy": sum(row["predicted_form"] == row["gold_form"] for row in group)
                / len(group),
                "intent_accuracy": sum(
                    row["predicted_intent"] == row["gold_intent"] for row in group
                )
                / len(group),
            }
            for domain, group in _group(items, "domain").items()
        }
        result["reliability_bins"] = _reliability(items)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "label_audit_report.json", result)
    _write_aggregation_report(output_dir / "label_audit_report.md", result, "Task 05 label audit")
    return result


def _group(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, list[Mapping[str, Any]]]:
    grouped: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(row)
    return dict(grouped)


def _reliability(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for axis in ("form", "intent"):
        bins: dict[str, Any] = {}
        for name, group in _group(items, "confidence_bin").items():
            bins[name] = {
                "count": len(group),
                "mean_confidence": fmean(float(row[f"{axis}_confidence"]) for row in group),
                "accuracy": sum(row[f"predicted_{axis}"] == row[f"gold_{axis}"] for row in group)
                / len(group),
            }
        result[axis] = bins
    return result


def aggregate_concept_audit(
    machine_proposals: Path,
    ratings: Sequence[Path],
    *,
    adjudication: Path | None,
    output_dir: Path,
    required_reviewers: int = 2,
) -> dict[str, Any]:
    key = {str(row["audit_id"]): row for row in read_records(machine_proposals)}
    grouped: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    categorical = (
        "numbers_units_correct",
        "over_fragmented",
        "duplicate_concepts",
        "useful_for_coverage",
    )
    invalid: list[str] = []
    seen_ratings: set[tuple[str, str]] = set()
    for index, row in enumerate(_read_csvs(ratings), 1):
        audit_id = row.get("audit_id", "")
        reviewer_id = row.get("reviewer_id", "")
        pair = (audit_id, reviewer_id)
        known_ids = {
            str(item["concept_id"])
            for item in cast(list[dict[str, Any]], key.get(audit_id, {}).get("concepts", []))
        }
        supplied_ids = _split_ids(row.get("correct_concept_ids", "")) | _split_ids(
            row.get("spurious_concept_ids", "")
        )
        valid = (
            audit_id in key
            and bool(reviewer_id)
            and pair not in seen_ratings
            and supplied_ids.issubset(known_ids)
            and all(row.get(field) in {"yes", "no", "uncertain"} for field in categorical)
        )
        if not valid:
            invalid.append(f"row:{index}")
            continue
        seen_ratings.add(pair)
        grouped[audit_id].append(row)
    raw_adjudicated = _adjudications(adjudication)
    adjudicated = {
        audit_id: row
        for audit_id, row in raw_adjudicated.items()
        if row.get("adjudicator_id") and row.get("resolved") == "yes"
    }
    missing = sorted(
        audit_id
        for audit_id in key
        if len({row["reviewer_id"] for row in grouped[audit_id]}) < required_reviewers
    )
    disagreements = sorted(
        audit_id
        for audit_id, rows in grouped.items()
        if any(len({row[field] for row in rows}) > 1 for field in categorical)
    )
    unresolved = sorted(set(disagreements) - set(adjudicated))
    complete = not missing and not unresolved and not invalid
    result: dict[str, Any] = {
        "schema": CONCEPT_AGGREGATION_SCHEMA,
        "status": "complete" if complete else "incomplete",
        "required_reviewers": required_reviewers,
        "expected_items": len(key),
        "missing_required_ratings": missing,
        "invalid_rating_rows": invalid,
        "unresolved_adjudications": unresolved,
        "agreement": {field: _agreement(grouped, field) for field in categorical},
        "input_fingerprints": {
            "machine_proposals_sha256": _file_sha256(machine_proposals),
            "ratings_sha256": {str(path): _file_sha256(path) for path in ratings},
            "adjudication_sha256": (
                _file_sha256(adjudication)
                if adjudication is not None and adjudication.exists()
                else None
            ),
        },
        "provenance": collect_code_provenance(),
        "final_tests_used": [],
    }
    if complete:
        rows = [row for values in grouped.values() for row in values]
        result["ratings"] = {
            field: dict(Counter(row[field] for row in rows)) for field in categorical
        }
        result["concept_error_totals"] = {
            "correct": sum(len(_split_ids(row["correct_concept_ids"])) for row in rows),
            "spurious": sum(len(_split_ids(row["spurious_concept_ids"])) for row in rows),
            "missing": sum(bool(row["missing_important_concepts"].strip()) for row in rows),
            "duplicates": sum(row["duplicate_concepts"] == "yes" for row in rows),
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "concept_audit_report.json", result)
    _write_aggregation_report(
        output_dir / "concept_audit_report.md", result, "Task 05 concept audit"
    )
    return result


def _split_ids(value: str) -> set[str]:
    return {item.strip() for item in re.split(r"[|,;]", value) if item.strip()}
