"""Streaming adapter for speakleash/msmarco_pl's aligned list fields."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, cast

from doc2query.schemas import DatasetColumnMapping
from doc2query.utils.records import JsonlWriter, read_records, write_json
from doc2query.utils.tracking import collect_code_provenance

DATASET_REPO_ID = "speakleash/msmarco_pl"
DATASET_REVISION = "ffcfc5fbc254bea348a7871133a6a0fa9ca21cb5"
DEFAULT_MIN_POSITIVE_SCORE = 23.5
_MOJIBAKE_MARKERS = ("Ã", "Â", "Ë", "�")


class NoPositiveAfterSourceScoreFilterError(ValueError):
    """Raised when a source row has no positive at or above the configured score."""

    def __init__(self, *, removed_count: int) -> None:
        self.removed_count = removed_count
        super().__init__("no positives remain after source-score filtering")


def _list(record: dict[str, Any], field: str) -> list[Any]:
    value = record.get(field)
    if not isinstance(value, list):
        raise ValueError(f"field {field!r} must be a list")
    return value


def _parallel(record: dict[str, Any], fields: tuple[str, ...]) -> list[list[Any]]:
    values = [_list(record, field) for field in fields]
    lengths = {len(value) for value in values}
    if len(lengths) != 1:
        details = ", ".join(
            f"{field}={len(value)}" for field, value in zip(fields, values, strict=True)
        )
        raise ValueError(f"unaligned parallel fields: {details}")
    return values


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"field {field!r} contains an empty or non-string text")
    return value


def _doc_id(value: Any, field: str) -> str:
    if not isinstance(value, (str, int)) or str(value).strip() == "":
        raise ValueError(f"field {field!r} contains an empty or invalid document ID")
    return str(value)


def _quality_flags(text: str) -> list[str]:
    return ["possible_mojibake"] if any(marker in text for marker in _MOJIBAKE_MARKERS) else []


def adapt_msmarco_pl_record(
    record: dict[str, Any],
    *,
    columns: DatasetColumnMapping | None = None,
    source_revision: str = DATASET_REVISION,
    min_positive_score: float | None = DEFAULT_MIN_POSITIVE_SCORE,
) -> dict[str, Any]:
    """Convert one raw row without trusting English-derived source scores as Polish labels."""
    if len(source_revision) != 40 or any(
        char not in "0123456789abcdef" for char in source_revision
    ):
        raise ValueError("source_revision must be a full 40-character commit SHA")
    mapping = columns or DatasetColumnMapping()
    query = _text(record.get(mapping.query), mapping.query)
    example_id = record.get(mapping.example_id)
    if not isinstance(example_id, (str, int)) or str(example_id).strip() == "":
        raise ValueError(f"field {mapping.example_id!r} must contain a stable ID")

    pos, pos_ids, pos_scores, pos_synthetic = _parallel(
        record,
        (
            mapping.positive_texts,
            mapping.positive_ids,
            mapping.positive_scores,
            mapping.positive_is_synthetic,
        ),
    )
    neg, neg_ids, neg_scores = _parallel(
        record,
        (mapping.negative_texts, mapping.negative_ids, mapping.negative_scores),
    )
    if not pos:
        raise ValueError("at least one positive is required")
    if len(neg) < 10:
        raise ValueError("at least ten hard negatives are required")

    positives: list[dict[str, Any]] = []
    removed_positive_doc_ids: list[str] = []
    for index, (text, doc_id, score, is_synthetic) in enumerate(
        zip(pos, pos_ids, pos_scores, pos_synthetic, strict=True)
    ):
        passage = _text(text, f"{mapping.positive_texts}[{index}]")
        source_score = float(score)
        normalized_doc_id = _doc_id(doc_id, f"{mapping.positive_ids}[{index}]")
        if not isinstance(is_synthetic, bool):
            raise ValueError(f"{mapping.positive_is_synthetic}[{index}] must be boolean")
        if min_positive_score is not None and source_score < min_positive_score:
            removed_positive_doc_ids.append(normalized_doc_id)
            continue
        positives.append(
            {
                "doc_id": normalized_doc_id,
                "text": passage,
                "metadata": {
                    "source_en_score": source_score,
                    "source_score_language": "en",
                    "is_synthetic_positive": is_synthetic,
                    "text_quality_flags": _quality_flags(passage),
                },
            }
        )
    if not positives:
        raise NoPositiveAfterSourceScoreFilterError(removed_count=len(removed_positive_doc_ids))

    hard_negatives: list[dict[str, Any]] = []
    for index, (text, doc_id, score) in enumerate(zip(neg, neg_ids, neg_scores, strict=True)):
        passage = _text(text, f"{mapping.negative_texts}[{index}]")
        hard_negatives.append(
            {
                "doc_id": _doc_id(doc_id, f"{mapping.negative_ids}[{index}]"),
                "text": passage,
                "metadata": {
                    "source_en_score": float(score),
                    "source_score_language": "en",
                    "text_quality_flags": _quality_flags(passage),
                },
            }
        )

    positive_id_counts: Counter[str] = Counter(str(item["doc_id"]) for item in positives)
    positive_ids = set(positive_id_counts)
    negative_id_counts: Counter[str] = Counter(str(item["doc_id"]) for item in hard_negatives)
    overlap = positive_ids & set(negative_id_counts)

    source_difficulty = record.get("difference_between_max_scores")
    return {
        "example_id": str(example_id),
        "query": query,
        "positives": positives,
        "hard_negatives": hard_negatives,
        "metadata": {
            "source": DATASET_REPO_ID,
            "source_revision": source_revision,
            "language": "pl",
            "source_score_language": "en",
            "source_en_difference_between_max_scores": (
                float(source_difficulty) if source_difficulty is not None else None
            ),
            "source_en_positive_score_filter": {
                "minimum_inclusive": min_positive_score,
                "removed_count": len(removed_positive_doc_ids),
                "removed_doc_ids": removed_positive_doc_ids,
            },
            "query_text_quality_flags": _quality_flags(query),
            "duplicate_positive_doc_ids": sorted(
                doc_id for doc_id, count in positive_id_counts.items() if count > 1
            ),
            "duplicate_negative_doc_ids": sorted(
                doc_id for doc_id, count in negative_id_counts.items() if count > 1
            ),
            "positive_negative_overlap_doc_ids": sorted(overlap),
            "synthetic_positive_count": sum(
                bool(cast(dict[str, Any], item["metadata"])["is_synthetic_positive"])
                for item in positives
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Adapt local speakleash/msmarco_pl JSONL.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--source-revision", default=DATASET_REVISION)
    parser.add_argument("--min-positive-score", type=float, default=DEFAULT_MIN_POSITIVE_SCORE)
    args = parser.parse_args(argv)
    input_count = 0
    output_count = 0
    removed_positive_count = 0
    skipped_without_positive_count = 0
    with JsonlWriter(args.output) as writer:
        for record in read_records(args.input):
            input_count += 1
            try:
                adapted = adapt_msmarco_pl_record(
                    record,
                    source_revision=args.source_revision,
                    min_positive_score=args.min_positive_score,
                )
            except NoPositiveAfterSourceScoreFilterError as exc:
                removed_positive_count += exc.removed_count
                skipped_without_positive_count += 1
                continue
            filter_metadata = cast(
                dict[str, Any], adapted["metadata"]["source_en_positive_score_filter"]
            )
            removed_positive_count += int(filter_metadata["removed_count"])
            writer.write(adapted)
            output_count += 1
    if args.report is not None:
        write_json(
            args.report,
            {
                "code": collect_code_provenance(),
                "input_records": input_count,
                "min_source_en_positive_score_inclusive": args.min_positive_score,
                "output_records": output_count,
                "removed_positives": removed_positive_count,
                "skipped_records_without_positive": skipped_without_positive_count,
                "source": DATASET_REPO_ID,
                "source_revision": args.source_revision,
            },
        )
    return 0
