"""Deterministic top-N, MMR and coverage-aware candidate selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from doc2query.generation.deduplicate import query_key
from doc2query.text.normalization import SimplePolishNormalizer


class SelectionStrategy(StrEnum):
    TOP_N = "top_n"
    MMR = "mmr"
    COVERAGE_AWARE = "coverage_aware"


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    text: str
    quality: float
    form: str = "unknown"
    intent: str = "unknown"
    focus_bucket: str | None = None
    focus_sentence_ids: frozenset[int] = field(default_factory=frozenset)
    concepts: frozenset[str] = field(default_factory=frozenset)
    embedding: tuple[float, ...] | None = None


@dataclass(frozen=True)
class SelectionWeights:
    diversity: float = 0.25
    focus_coverage: float = 0.2
    style_coverage: float = 0.1
    concept_coverage: float = 0.2
    duplicate_penalty: float = 1.0


def _lexical_similarity(left: str, right: str) -> float:
    normalizer = SimplePolishNormalizer()
    a = set(normalizer.analyze(left).content_lemmas)
    b = set(normalizer.analyze(right).content_lemmas)
    return len(a & b) / len(a | b) if a or b else 1.0


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    import math

    if len(left) != len(right):
        raise ValueError("candidate embedding dimensions differ")
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    denominator = math.sqrt(sum(a * a for a in left) * sum(b * b for b in right))
    return numerator / denominator if denominator else 0.0


def _similarity(left: Candidate, right: Candidate) -> float:
    if left.embedding is not None and right.embedding is not None:
        return _cosine(left.embedding, right.embedding)
    return _lexical_similarity(left.text, right.text)


def select_candidates(
    candidates: list[Candidate],
    *,
    count: int,
    strategy: SelectionStrategy = SelectionStrategy.COVERAGE_AWARE,
    weights: SelectionWeights | None = None,
) -> list[Candidate]:
    if count < 1:
        raise ValueError("selection count must be positive")
    active_weights = weights or SelectionWeights()
    remaining = sorted(candidates, key=lambda item: item.candidate_id)
    if len({candidate.candidate_id for candidate in remaining}) != len(remaining):
        raise ValueError("candidate IDs must be unique")
    if strategy == SelectionStrategy.TOP_N:
        return sorted(remaining, key=lambda item: (-item.quality, item.candidate_id))[:count]
    selected: list[Candidate] = []
    while remaining and len(selected) < count:
        forms = {item.form for item in selected}
        buckets = {item.focus_bucket for item in selected if item.focus_bucket is not None}
        sentences = (
            set().union(*(item.focus_sentence_ids for item in selected)) if selected else set()
        )
        concepts = set().union(*(item.concepts for item in selected)) if selected else set()

        def gain(
            item: Candidate,
            selected_forms: set[str] = forms,
            selected_buckets: set[str] = buckets,
            selected_sentences: set[int] = sentences,
            selected_concepts: set[str] = concepts,
        ) -> tuple[float, float, str]:
            maximum_similarity = max((_similarity(item, old) for old in selected), default=0.0)
            duplicate = any(query_key(item.text) == query_key(old.text) for old in selected)
            if strategy == SelectionStrategy.MMR:
                score = item.quality - active_weights.diversity * maximum_similarity
            else:
                new_focus = len(item.focus_sentence_ids - selected_sentences)
                if item.focus_bucket is not None and item.focus_bucket not in selected_buckets:
                    new_focus += 1
                score = (
                    item.quality
                    + active_weights.diversity * (1.0 - maximum_similarity)
                    + active_weights.focus_coverage * new_focus
                    + active_weights.style_coverage * (item.form not in selected_forms)
                    + active_weights.concept_coverage * len(item.concepts - selected_concepts)
                    - active_weights.duplicate_penalty * duplicate
                )
            return score, item.quality, item.candidate_id

        winner = max(remaining, key=gain)
        selected.append(winner)
        remaining.remove(winner)
    return selected
