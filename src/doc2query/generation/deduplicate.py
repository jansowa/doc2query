"""Polish query deduplication robust to case, diacritics and cheap lemma variants."""

from __future__ import annotations

import unicodedata

from doc2query.text.normalization import SimplePolishNormalizer, TextNormalizer

_POLISH_FOLD = str.maketrans({"ł": "l", "Ł": "L"})


def query_key(query: str, normalizer: TextNormalizer | None = None) -> str:
    analyzer = normalizer or SimplePolishNormalizer()
    lemmas = analyzer.analyze(query).lemmas
    folded = " ".join(lemmas).casefold().translate(_POLISH_FOLD)
    return "".join(
        character
        for character in unicodedata.normalize("NFKD", folded)
        if not unicodedata.combining(character)
    )


def deduplicate_queries(queries: list[str], normalizer: TextNormalizer | None = None) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for query in queries:
        key = query_key(query, normalizer)
        if key and key not in seen:
            seen.add(key)
            unique.append(" ".join(query.strip().split()))
    return unique
