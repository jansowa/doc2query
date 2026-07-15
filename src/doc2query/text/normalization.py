"""Polish text analysis backends with a stable serializable contract."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Protocol

_TOKEN = re.compile(r"\d+(?:[.,]\d+)?|[a-ząćęłńóśźż]+(?:-[a-ząćęłńóśźż]+)*", re.IGNORECASE)
_NUMBER = re.compile(r"^\d+(?:[.,]\d+)?$")
_UNITS = frozenset({"kg", "g", "mg", "km", "m", "cm", "mm", "l", "ml", "°c", "kw", "kwh", "%"})
POLISH_STOPWORDS = frozenset(
    "a aby ale albo ani aż bo bowiem by być był była było były co czy dla do gdy gdzie i ich "
    "jak jako jego jej jest jeśli już kiedy która które który lecz lub ma mają mieć na nad nie "
    "o od oraz po pod przez przy się są ta tak ten to tu w we z za ze że".split()
)


@dataclass(frozen=True)
class AnalyzedText:
    tokens: tuple[str, ...]
    lemmas: tuple[str, ...]
    content_lemmas: tuple[str, ...]
    content_counts: dict[str, int]
    entities: tuple[str, ...]
    numbers: tuple[str, ...]
    units: tuple[str, ...]
    backend: str
    version: str

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key in ("tokens", "lemmas", "content_lemmas", "entities", "numbers", "units"):
            result[key] = list(result[key])
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnalyzedText:
        return cls(
            tokens=tuple(data["tokens"]),
            lemmas=tuple(data["lemmas"]),
            content_lemmas=tuple(data["content_lemmas"]),
            content_counts={str(k): int(v) for k, v in data["content_counts"].items()},
            entities=tuple(data["entities"]),
            numbers=tuple(data["numbers"]),
            units=tuple(data["units"]),
            backend=str(data["backend"]),
            version=str(data["version"]),
        )


class TextNormalizer(Protocol):
    @property
    def cache_namespace(self) -> str: ...

    def analyze(self, text: str) -> AnalyzedText: ...


def _normalized(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower()


class SimplePolishNormalizer:
    """Dependency-free tokenizer; tokens are used as cheap pseudo-lemmas."""

    cache_namespace = "simple_pl:v1:nfkc:stopwords-v1"

    def analyze(self, text: str) -> AnalyzedText:
        tokens = tuple(_TOKEN.findall(_normalized(text)))
        content = tuple(
            token for token in tokens if token not in POLISH_STOPWORDS and len(token) > 1
        )
        numbers = tuple(token for token in tokens if _NUMBER.match(token))
        units = tuple(token for token in tokens if token in _UNITS)
        return AnalyzedText(
            tokens=tokens,
            lemmas=tokens,
            content_lemmas=content,
            content_counts=dict(Counter(content)),
            entities=(),
            numbers=numbers,
            units=units,
            backend="simple",
            version="1",
        )


class SpacyPolishNormalizer:
    """CPU-only spaCy backend. Loading fails clearly when the model is absent."""

    def __init__(self, model_name: str = "pl_core_news_lg") -> None:
        try:
            import spacy
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("install the nlp dependency group") from exc
        spacy.require_cpu()  # type: ignore[attr-defined]
        self._nlp = spacy.load(model_name, disable=["parser"])
        self.model_name = model_name
        version = self._nlp.meta.get("version", "unknown")
        self.cache_namespace = f"spacy_pl:{model_name}:{version}:v1"

    def analyze(self, text: str) -> AnalyzedText:
        doc = self._nlp(unicodedata.normalize("NFKC", text))
        tokens = tuple(
            token.text.lower() for token in doc if not token.is_space and not token.is_punct
        )
        lemmas = tuple(
            token.lemma_.lower() for token in doc if not token.is_space and not token.is_punct
        )
        content = tuple(
            token.lemma_.lower()
            for token in doc
            if not token.is_space
            and not token.is_punct
            and not token.is_stop
            and len(token.text) > 1
        )
        entities = tuple(entity.text.lower() for entity in doc.ents)
        numbers = tuple(token.text.lower() for token in doc if token.like_num)
        units = tuple(token.text.lower() for token in doc if token.text.lower() in _UNITS)
        return AnalyzedText(
            tokens=tokens,
            lemmas=lemmas,
            content_lemmas=content,
            content_counts=dict(Counter(content)),
            entities=entities,
            numbers=numbers,
            units=units,
            backend="spacy_pl",
            version=self.cache_namespace,
        )
