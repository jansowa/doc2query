"""Model-free concept extraction and coverage for stateful generation experiments."""

from __future__ import annotations

from dataclasses import dataclass

from doc2query.text.normalization import SimplePolishNormalizer, TextNormalizer


@dataclass(frozen=True)
class Concept:
    text: str
    kind: str


def extract_concepts(
    passage: str, normalizer: TextNormalizer | None = None, *, max_concepts: int = 24
) -> tuple[Concept, ...]:
    analyzed = (normalizer or SimplePolishNormalizer()).analyze(passage)
    candidates: list[Concept] = []
    candidates.extend(Concept(value, "entity") for value in analyzed.entities)
    candidates.extend(Concept(value, "number") for value in analyzed.numbers)
    candidates.extend(Concept(value, "unit") for value in analyzed.units)
    ranked_lemmas = sorted(
        analyzed.content_counts, key=lambda value: (-analyzed.content_counts[value], value)
    )
    candidates.extend(Concept(value, "content_lemma") for value in ranked_lemmas)
    result: list[Concept] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.text not in seen:
            seen.add(candidate.text)
            result.append(candidate)
        if len(result) >= max_concepts:
            break
    return tuple(result)


def covered_concepts(
    concepts: tuple[Concept, ...], queries: list[str], normalizer: TextNormalizer | None = None
) -> set[str]:
    analyzer = normalizer or SimplePolishNormalizer()
    query_terms = {term for query in queries for term in analyzer.analyze(query).lemmas}
    return {
        concept.text
        for concept in concepts
        if set(analyzer.analyze(concept.text).lemmas).issubset(query_terms)
    }


def render_stateful_context(concepts: tuple[Concept, ...], previous_queries: list[str]) -> str:
    covered = covered_concepts(concepts, previous_queries)
    uncovered = [concept.text for concept in concepts if concept.text not in covered]
    previous = "; ".join(previous_queries[-4:]) or "brak"
    return (
        f"Niepokryte koncepcje: {', '.join(uncovered) or 'brak'}.\n"
        f"Poprzednie zapytania (nie powtarzaj): {previous}."
    )
