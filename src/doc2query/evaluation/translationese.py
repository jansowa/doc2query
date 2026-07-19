"""Cheap, transparent translationese indicators for Polish queries.

This is a diagnostic heuristic, not a language-quality classifier.  The
individual flags are deliberately exposed so a report never turns the score
into an unsupported claim that a query was translated.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

_TOKEN = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?", re.UNICODE)
_POLISH_DIACRITICS = frozenset("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")
_ENGLISH_RESIDUE = frozenset(
    {
        "did",
        "does",
        "how",
        "the",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
    }
)
_CALQUE_PATTERNS = (
    re.compile(r"\bco jest (?:nazwą|imię|powodem|celem)\b", re.IGNORECASE),
    re.compile(r"\bjakie jest znaczenie dla\b", re.IGNORECASE),
    re.compile(r"\bw odniesieniu do czego\b", re.IGNORECASE),
)
_BAD_SPACING = re.compile(r"\s+[?!,.:;]|\?\s*\?|!\s*!")


def translationese_indicators(text: str) -> dict[str, Any]:
    """Return auditable surface flags and a bounded heuristic risk score."""
    normalized = " ".join(text.split())
    tokens = [token.casefold() for token in _TOKEN.findall(normalized)]
    english_hits = sorted({token for token in tokens if token in _ENGLISH_RESIDUE})
    calque_hits = [pattern.pattern for pattern in _CALQUE_PATTERNS if pattern.search(normalized)]
    flags = {
        "english_residue": bool(english_hits),
        "calque_pattern": bool(calque_hits),
        "suspicious_punctuation_spacing": bool(_BAD_SPACING.search(text)),
        "ascii_only_long_query": (
            len(tokens) >= 8 and not any(character in _POLISH_DIACRITICS for character in text)
        ),
    }
    # Weak evidence receives weak weight.  ASCII-only Polish is common and must
    # never dominate the diagnostic.
    score = min(
        1.0,
        0.55 * float(flags["english_residue"])
        + 0.35 * float(flags["calque_pattern"])
        + 0.20 * float(flags["suspicious_punctuation_spacing"])
        + 0.10 * float(flags["ascii_only_long_query"]),
    )
    return {
        "heuristic_version": "translationese-surface-v1",
        "risk_score": score,
        "flags": flags,
        "english_tokens": english_hits,
        "calque_patterns": calque_hits,
        "interpretation": "diagnostic_only_not_proof_of_translation",
    }


def aggregate_translationese(texts: Iterable[str]) -> dict[str, Any]:
    """Aggregate the same explicit flags over a collection of query strings."""
    rows = [translationese_indicators(text) for text in texts]
    flag_counts: Counter[str] = Counter()
    for row in rows:
        for flag, active in row["flags"].items():
            flag_counts[flag] += int(bool(active))
    count = len(rows)
    return {
        "heuristic_version": "translationese-surface-v1",
        "interpretation": "diagnostic_only_not_proof_of_translation_or_naturalness",
        "query_count": count,
        "mean_risk_score": (
            sum(float(row["risk_score"]) for row in rows) / count if count else None
        ),
        "flag_rates": {
            flag: flag_counts[flag] / count if count else None
            for flag in (
                "english_residue",
                "calque_pattern",
                "suspicious_punctuation_spacing",
                "ascii_only_long_query",
            )
        },
    }
