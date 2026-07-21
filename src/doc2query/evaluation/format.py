"""Single-query and JSON multi-query format/language checks."""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

from doc2query.generation.multiquery import parse_multiquery_json
from doc2query.text.normalization import POLISH_STOPWORDS, SimplePolishNormalizer

_PREFIX = re.compile(
    r"^\s*(?:zapytanie|pytanie|odpowiedź|query|oto|wygenerowane)\s*[:\-]", re.IGNORECASE
)
_LIST = re.compile(r"(?:^|\n)\s*(?:[-*]|\d+[.)])\s+")
_META = re.compile(r"\b(?:na podstawie pasażu|jako model|nie mogę|wygenerowałem)\b", re.IGNORECASE)


def _language_confidence(text: str) -> float | None:
    tokens = SimplePolishNormalizer().analyze(text).tokens
    if not tokens:
        return None
    clues = sum(
        token in POLISH_STOPWORDS or any(char in "ąćęłńóśźż" for char in token) for token in tokens
    )
    return clues / len(tokens)


def _invalid_character_count(text: str) -> int:
    return sum(unicodedata.category(char).startswith("C") and char not in "\n\t" for char in text)


def format_metrics(text: str, *, multi_query_json: bool = False) -> dict[str, Any]:
    stripped = text.strip()
    parsed: Any = None
    json_valid: bool | None = None
    if multi_query_json:
        result = parse_multiquery_json(stripped, allow_minor_repair=False)
        parsed = result.completion
        json_valid = result.valid
        if not json_valid:
            try:
                legacy = json.loads(stripped)
                json_valid = isinstance(legacy, list) and all(
                    isinstance(value, str) and value.strip() for value in legacy
                )
                parsed = legacy if json_valid else None
            except (json.JSONDecodeError, TypeError):
                pass
    multiple = bool(_LIST.search(stripped) or stripped.count("\n") > 0)
    if parsed is not None:
        multiple = False
    return {
        "empty": not stripped,
        "multiple_query": multiple,
        "has_prefix": bool(_PREFIX.search(stripped)),
        "has_metacomment": bool(_META.search(stripped)),
        "word_length": len(SimplePolishNormalizer().analyze(stripped).tokens),
        "character_length": len(stripped),
        "language_confidence_pl": _language_confidence(stripped),
        "invalid_character_count": _invalid_character_count(stripped),
        "json_valid": json_valid,
        "format_valid": bool(
            stripped
            and (multi_query_json or not multiple)
            and not _PREFIX.search(stripped)
            and not _META.search(stripped)
            and not _invalid_character_count(stripped)
            and (json_valid is not False)
        ),
    }
