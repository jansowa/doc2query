"""Group-level diversity metrics with explicit unavailable embedding fields."""

from __future__ import annotations

import math
from itertools import combinations
from statistics import fmean
from typing import Any

from doc2query.text.normalization import SimplePolishNormalizer


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _jaccard(left: set[str], right: set[str]) -> float:
    return _ratio(len(left & right), len(left | right))


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    denominator = math.sqrt(sum(a * a for a in left) * sum(b * b for b in right))
    return numerator / denominator if denominator else 0.0


def _ngram_set(tokens: tuple[str, ...], n: int) -> set[tuple[str, ...]]:
    return {tokens[index : index + n] for index in range(max(0, len(tokens) - n + 1))}


def _self_bleu(tokens: list[tuple[str, ...]]) -> float | None:
    if len(tokens) < 2:
        return None
    scores = []
    for index, query in enumerate(tokens):
        references = [value for other, value in enumerate(tokens) if other != index]
        precisions = []
        for n in (1, 2, 3, 4):
            query_ngrams = _ngram_set(query, n)
            reference_ngrams = set().union(*(_ngram_set(ref, n) for ref in references))
            precisions.append((len(query_ngrams & reference_ngrams) + 1) / (len(query_ngrams) + 1))
        brevity = min(1.0, math.exp(1 - max(map(len, references)) / max(1, len(query))))
        scores.append(brevity * math.exp(fmean(math.log(value) for value in precisions)))
    return fmean(scores)


def _entropy(values: list[str]) -> float | None:
    if not values:
        return None
    counts = {value: values.count(value) for value in set(values)}
    return -sum((count / len(values)) * math.log2(count / len(values)) for count in counts.values())


def _semantic_clusters(embeddings: list[list[float]], threshold: float = 0.85) -> int:
    parents = list(range(len(embeddings)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for left, right in combinations(range(len(embeddings)), 2):
        if _cosine(embeddings[left], embeddings[right]) >= threshold:
            parents[root(right)] = root(left)
    return len({root(index) for index in range(len(embeddings))})


def diversity_metrics(
    queries: list[str],
    *,
    embeddings: list[list[float]] | None = None,
    styles: list[str] | None = None,
    focus_buckets: list[str] | None = None,
) -> dict[str, Any]:
    if not queries:
        raise ValueError("diversity group cannot be empty")
    normalizer = SimplePolishNormalizer()
    analyzed = [normalizer.analyze(query) for query in queries]
    tokens = [value.tokens for value in analyzed]
    lemmas = [set(value.content_lemmas) for value in analyzed]
    pair_jaccards = [_jaccard(lemmas[a], lemmas[b]) for a, b in combinations(range(len(lemmas)), 2)]
    all_unigrams = [token for query in tokens for token in query]
    all_bigrams = [item for query in tokens for item in _ngram_set(query, 2)]
    duplicate_count = len(queries) - len({query.strip().casefold() for query in queries})
    cosines: list[float] | None = None
    if embeddings is not None:
        if len(embeddings) != len(queries):
            raise ValueError("embedding count must match query count")
        cosines = [
            _cosine(embeddings[a], embeddings[b])
            for a, b in combinations(range(len(embeddings)), 2)
        ]
    return {
        "query_count": len(queries),
        "distinct_1": _ratio(len(set(all_unigrams)), len(all_unigrams)),
        "distinct_2": _ratio(len(set(all_bigrams)), len(all_bigrams)),
        "self_bleu": _self_bleu(tokens),
        "mean_pairwise_lemma_jaccard": fmean(pair_jaccards) if pair_jaccards else None,
        "max_pairwise_lemma_jaccard": max(pair_jaccards) if pair_jaccards else None,
        "mean_pairwise_embedding_cosine": fmean(cosines) if cosines else None,
        "max_pairwise_embedding_cosine": max(cosines) if cosines else None,
        "duplicate_rate": duplicate_count / len(queries),
        "semantic_cluster_count": (
            _semantic_clusters(embeddings) if embeddings is not None else None
        ),
        "style_entropy": _entropy(styles or []),
        "focus_entropy": _entropy(focus_buckets or []),
    }
