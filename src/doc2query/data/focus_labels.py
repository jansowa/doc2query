"""Deterministic sentence splitting and abstaining lexical focus assignment."""

from __future__ import annotations

import re
from dataclasses import dataclass

from doc2query.text.normalization import SimplePolishNormalizer

_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def split_sentences(passage: str) -> list[str]:
    return [value.strip() for value in _BOUNDARY.split(passage.strip()) if value.strip()]


def focus_bucket(sentence_id: int, sentence_count: int) -> str:
    if sentence_count < 1 or not 0 <= sentence_id < sentence_count:
        raise ValueError("sentence index is outside passage")
    if sentence_count == 1:
        return "middle"
    return ("beginning", "middle", "end")[round(sentence_id * 2 / (sentence_count - 1))]


@dataclass(frozen=True)
class FocusAssignment:
    sentence_id: int | None
    bucket: str | None
    confidence: float
    scores: tuple[float, ...]


def assign_focus(query: str, passage: str, *, minimum_confidence: float = 0.2) -> FocusAssignment:
    sentences = split_sentences(passage)
    query_terms = set(SimplePolishNormalizer().analyze(query).content_lemmas)
    if not sentences or not query_terms:
        return FocusAssignment(None, None, 0.0, ())
    scores = tuple(
        len(query_terms & set(SimplePolishNormalizer().analyze(sentence).content_lemmas))
        / len(query_terms)
        for sentence in sentences
    )
    best = max(scores)
    winners = [index for index, score in enumerate(scores) if score == best]
    if best < minimum_confidence or len(winners) != 1:
        return FocusAssignment(None, None, best, scores)
    sentence_id = winners[0]
    return FocusAssignment(sentence_id, focus_bucket(sentence_id, len(sentences)), best, scores)
