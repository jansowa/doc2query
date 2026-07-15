"""Lexical overlap and copying signals for Polish queries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from doc2query.text.normalization import AnalyzedText


@dataclass(frozen=True)
class LexicalMetrics:
    content_jaccard: float
    overlap_coefficient: float
    query_precision: float
    passage_recall: float
    longest_copied_ngram: int
    normalized_lcs: float
    copy_density: float
    number_preservation: float
    unit_preservation: float
    entity_preservation: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ratio(numerator: int, denominator: int, *, empty: float = 0.0) -> float:
    return numerator / denominator if denominator else empty


def _lcs(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    row = [0] * (len(right) + 1)
    for ltoken in left:
        previous = 0
        for index, rtoken in enumerate(right, start=1):
            old = row[index]
            row[index] = previous + 1 if ltoken == rtoken else max(row[index], row[index - 1])
            previous = old
    return row[-1]


def _longest_contiguous(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    row = [0] * (len(right) + 1)
    longest = 0
    for ltoken in left:
        previous_row = row
        row = [0] * (len(right) + 1)
        for index, rtoken in enumerate(right, start=1):
            if ltoken == rtoken:
                row[index] = previous_row[index - 1] + 1
                longest = max(longest, row[index])
    return longest


def _preservation(query: tuple[str, ...], passage: tuple[str, ...]) -> float:
    required = set(query)
    return _ratio(len(required & set(passage)), len(required), empty=1.0)


def lexical_metrics(query: AnalyzedText, passage: AnalyzedText) -> LexicalMetrics:
    qset, pset = set(query.content_lemmas), set(passage.content_lemmas)
    common = qset & pset
    union = qset | pset
    minimum = min(len(qset), len(pset))
    longest = _longest_contiguous(query.lemmas, passage.lemmas)
    lcs = _lcs(query.lemmas, passage.lemmas)
    return LexicalMetrics(
        content_jaccard=_ratio(len(common), len(union)),
        overlap_coefficient=_ratio(len(common), minimum),
        query_precision=_ratio(len(common), len(qset)),
        passage_recall=_ratio(len(common), len(pset)),
        longest_copied_ngram=longest,
        normalized_lcs=_ratio(lcs, len(query.lemmas)),
        copy_density=_ratio(longest, len(query.lemmas)),
        number_preservation=_preservation(query.numbers, passage.numbers),
        unit_preservation=_preservation(query.units, passage.units),
        entity_preservation=_preservation(query.entities, passage.entities),
    )
