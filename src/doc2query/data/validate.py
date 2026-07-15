"""Streaming validation with explicit warn/drop/error policies and audit output."""

from __future__ import annotations

import hashlib
import html
import json
import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from doc2query.utils.records import JsonlWriter, read_records, write_json
from doc2query.utils.tracking import collect_code_provenance

Mode = Literal["warn", "drop", "error"]
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HTML = re.compile(r"<\s*(?:html|body|div|p|script|style|br|table|a)\b", re.IGNORECASE)
_BOILERPLATE = re.compile(r"(?:cookie|polityka prywatności|wszelkie prawa zastrzeżone)", re.I)
_TOKEN = re.compile(r"\w+", re.UNICODE)
_POLISH_COMMON = frozenset("i w na z do nie jest się że to o dla jak czy oraz przez od po".split())


@dataclass(frozen=True)
class ValidationIssue:
    rule: str
    mode: Mode
    message: str
    example_id: str


DEFAULT_MODES: dict[str, Mode] = {
    "malformed_json": "error",
    "missing_id": "error",
    "duplicate_example_id": "drop",
    "empty_query": "error",
    "invalid_documents": "error",
    "empty_document": "error",
    "too_few_positives": "error",
    "too_few_negatives": "error",
    "duplicate_doc_id": "drop",
    "positive_negative_overlap": "drop",
    "same_text_different_ids": "warn",
    "short_text": "warn",
    "long_text": "warn",
    "control_characters": "drop",
    "html": "warn",
    "boilerplate": "warn",
    "possible_mojibake": "warn",
    "high_query_passage_overlap": "warn",
    "likely_non_polish": "warn",
}


@dataclass(frozen=True)
class ValidationPolicy:
    modes: dict[str, Mode]
    min_negatives: int = 10
    min_text_chars: int = 10
    max_text_chars: int = 20_000
    high_overlap_threshold: float = 0.85

    def __post_init__(self) -> None:
        if any(mode not in {"warn", "drop", "error"} for mode in self.modes.values()):
            raise ValueError("validation modes must be warn, drop, or error")
        if self.min_negatives < 1 or self.min_text_chars < 0:
            raise ValueError("validation thresholds must be non-negative")
        if self.max_text_chars <= self.min_text_chars:
            raise ValueError("max_text_chars must exceed min_text_chars")
        if not 0 < self.high_overlap_threshold <= 1:
            raise ValueError("high_overlap_threshold must be in (0, 1]")

    @classmethod
    def defaults(cls, overrides: dict[str, Mode] | None = None) -> ValidationPolicy:
        modes = dict(DEFAULT_MODES)
        modes.update(overrides or {})
        return cls(modes=modes)

    def mode(self, rule: str) -> Mode:
        return self.modes.get(rule, "warn")


def _documents(record: dict[str, Any], field: str) -> list[dict[str, Any]] | None:
    value = record.get(field)
    if not isinstance(value, list) or not all(
        isinstance(item, dict)
        and isinstance(item.get("doc_id"), str)
        and bool(item.get("doc_id", "").strip())
        and isinstance(item.get("text"), str)
        for item in value
    ):
        return None
    return value


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN.findall(text) if len(token) > 1}


def _query_overlap(query: str, passage: str) -> float:
    query_tokens = _tokens(query)
    return len(query_tokens & _tokens(passage)) / len(query_tokens) if query_tokens else 0.0


def polish_confidence(text: str) -> float:
    tokens = [token.lower() for token in _TOKEN.findall(text)]
    if not tokens:
        return 0.0
    markers = sum(
        token in _POLISH_COMMON or any(char in "ąćęłńóśźż" for char in token) for token in tokens
    )
    return markers / len(tokens)


def validate_record(
    record: dict[str, Any], policy: ValidationPolicy | None = None
) -> list[ValidationIssue]:
    policy = policy or ValidationPolicy.defaults()
    example_id = str(record.get("example_id", ""))
    issues: list[ValidationIssue] = []

    def add(rule: str, message: str) -> None:
        issues.append(ValidationIssue(rule, policy.mode(rule), message, example_id))

    if not example_id.strip():
        add("missing_id", "example_id is absent or empty")
    query = record.get("query")
    if not isinstance(query, str) or not query.strip():
        add("empty_query", "query is absent, empty, or not a string")
        query = ""
    positives = _documents(record, "positives")
    negatives = _documents(record, "hard_negatives")
    if positives is None or negatives is None:
        add("invalid_documents", "positives/hard_negatives must be lists of doc_id/text objects")
        return issues
    if not positives:
        add("too_few_positives", "at least one positive is required")
    if len(negatives) < policy.min_negatives:
        add("too_few_negatives", f"requires at least {policy.min_negatives} hard negatives")

    documents = positives + negatives
    ids = [str(document["doc_id"]) for document in documents]
    duplicate_ids = sorted(doc_id for doc_id, count in Counter(ids).items() if count > 1)
    if duplicate_ids:
        add("duplicate_doc_id", f"duplicate document IDs: {duplicate_ids[:10]}")
    overlap_ids = {str(document["doc_id"]) for document in positives} & {
        str(document["doc_id"]) for document in negatives
    }
    if overlap_ids:
        add("positive_negative_overlap", f"positive/negative ID overlap: {sorted(overlap_ids)}")
    text_to_ids: dict[str, set[str]] = {}
    for document in documents:
        normalized = " ".join(str(document["text"]).lower().split())
        text_to_ids.setdefault(normalized, set()).add(str(document["doc_id"]))
    conflicts = [sorted(values) for values in text_to_ids.values() if len(values) > 1]
    if conflicts:
        add("same_text_different_ids", f"identical text under different IDs: {conflicts[:5]}")

    all_texts = [("query", query), *[(str(doc["doc_id"]), str(doc["text"])) for doc in documents]]
    for label, text in all_texts:
        length = len(text.strip())
        if label != "query" and not text.strip():
            add("empty_document", f"{label} is empty")
        if label != "query" and length < policy.min_text_chars:
            add("short_text", f"{label} has only {length} characters")
        if length > policy.max_text_chars:
            add("long_text", f"{label} has {length} characters")
        if _CONTROL.search(text):
            add("control_characters", f"{label} contains control characters")
        if _HTML.search(html.unescape(text)):
            add("html", f"{label} contains HTML")
        if _BOILERPLATE.search(text):
            add("boilerplate", f"{label} contains boilerplate")
        if any(marker in text for marker in ("Ã", "Â", "Ë", "�")):
            add("possible_mojibake", f"{label} contains possible mojibake")
    if query and any(
        _query_overlap(query, str(document["text"])) >= policy.high_overlap_threshold
        for document in positives
    ):
        add("high_query_passage_overlap", "query nearly copies a positive passage")
    language = (
        record.get("metadata", {}).get("language")
        if isinstance(record.get("metadata"), dict)
        else None
    )
    if language == "pl" and len(_TOKEN.findall(query)) >= 5 and polish_confidence(query) < 0.05:
        add("likely_non_polish", "query has low Polish-language heuristic confidence")
    return issues


def _validation_records(
    input_path: Path,
) -> Iterator[tuple[dict[str, Any] | None, int, str | None]]:
    if input_path.suffix == ".jsonl":
        with input_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    yield None, line_number, line[:2000]
                    continue
                yield value if isinstance(value, dict) else None, line_number, line[:2000]
        return
    for index, record in enumerate(read_records(input_path), start=1):
        yield record, index, None


def validate_dataset(
    input_path: Path,
    *,
    accepted_path: Path,
    rejected_path: Path,
    report_path: Path,
    policy: ValidationPolicy | None = None,
) -> dict[str, Any]:
    policy = policy or ValidationPolicy.defaults()
    counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    fingerprint = hashlib.sha256()
    examples: list[dict[str, Any]] = []
    seen_example_ids: set[str] = set()
    error_seen = False
    with JsonlWriter(accepted_path) as accepted, JsonlWriter(rejected_path) as rejected:
        for record, line_number, raw_line in _validation_records(input_path):
            counts["input"] += 1
            if record is None:
                issue = ValidationIssue(
                    "malformed_json",
                    policy.mode("malformed_json"),
                    f"line {line_number} is not a JSON object",
                    "",
                )
                counts["rule:malformed_json"] += 1
                counts["rejected"] += 1
                mode_counts[issue.mode] += 1
                rejected.write({"raw_line": raw_line, "issues": [asdict(issue)]})
                if len(examples) < 100:
                    examples.append(asdict(issue))
                error_seen = error_seen or issue.mode == "error"
                continue
            issues = validate_record(record, policy)
            example_id = str(record.get("example_id", ""))
            if example_id in seen_example_ids:
                issues.append(
                    ValidationIssue(
                        "duplicate_example_id",
                        policy.mode("duplicate_example_id"),
                        "example_id occurs more than once in the dataset",
                        example_id,
                    )
                )
            elif example_id:
                seen_example_ids.add(example_id)
            for issue in issues:
                counts[f"rule:{issue.rule}"] += 1
                mode_counts[issue.mode] += 1
                if len(examples) < 100:
                    examples.append(asdict(issue))
            should_reject = any(issue.mode in {"drop", "error"} for issue in issues)
            if should_reject:
                counts["rejected"] += 1
                rejected.write({"record": record, "issues": [asdict(issue) for issue in issues]})
                error_seen = error_seen or any(issue.mode == "error" for issue in issues)
            else:
                counts["accepted"] += 1
                accepted.write(record)
                fingerprint.update(
                    json.dumps(
                        record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    ).encode()
                )
    report = {
        "counts": dict(counts),
        "mode_counts": dict(mode_counts),
        "accepted_fingerprint": fingerprint.hexdigest(),
        "policy": asdict(policy),
        "examples": examples,
        "contains_error_policy_violations": error_seen,
        "code": collect_code_provenance(),
    }
    write_json(report_path, report)
    return report
