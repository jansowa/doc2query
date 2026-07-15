"""Sentence-level focus assignment with ambiguity handling."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Literal

from doc2query.reranker.base import PairScorer

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-ZĄĆĘŁŃÓŚŹŻ0-9])")


def split_sentences(text: str) -> list[str]:
    return [
        sentence.strip() for sentence in _SENTENCE_BOUNDARY.split(text.strip()) if sentence.strip()
    ]


@dataclass(frozen=True)
class FocusLabel:
    focus_sentence_id: int
    focus_score: float
    focus_margin: float
    focus_bucket: Literal["beginning", "middle", "end"]
    focus_is_ambiguous: bool
    sentence_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _bucket(index: int, count: int) -> Literal["beginning", "middle", "end"]:
    relative = (index + 0.5) / count
    return "beginning" if relative <= 1 / 3 else "middle" if relative <= 2 / 3 else "end"


def assign_focus(
    scorer: PairScorer, query: str, passage: str, *, ambiguity_margin: float = 0.1
) -> FocusLabel:
    sentences = split_sentences(passage)
    if not sentences:
        raise ValueError("passage contains no sentences")
    scores = scorer.score_pairs([(query, sentence) for sentence in sentences])
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    best = order[0]
    margin = scores[best] - scores[order[1]] if len(order) > 1 else float("inf")
    return FocusLabel(
        focus_sentence_id=best,
        focus_score=scores[best],
        focus_margin=margin,
        focus_bucket=_bucket(best, len(sentences)),
        focus_is_ambiguous=margin < ambiguity_margin,
        sentence_count=len(sentences),
    )
