"""SQLite cache keyed by text digest and normalizer namespace."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path

from doc2query.text.normalization import AnalyzedText, TextNormalizer


class AnalysisCache:
    def __init__(self, path: Path, normalizer: TextNormalizer) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path)
        self._normalizer = normalizer
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS analyses "
            "(namespace TEXT NOT NULL, digest TEXT NOT NULL, payload TEXT NOT NULL, "
            "PRIMARY KEY(namespace, digest))"
        )

    def analyze(self, text: str) -> AnalyzedText:
        return self._analyze(text, commit=True)

    def analyze_uncommitted(self, text: str) -> AnalyzedText:
        """Analyze while letting a bulk caller control the SQLite commit boundary."""
        return self._analyze(text, commit=False)

    def analyze_many_uncommitted(self, texts: Sequence[str]) -> list[AnalyzedText]:
        """Read cached values and batch only cache misses through the normalizer."""
        results: list[AnalyzedText | None] = [None] * len(texts)
        missing: list[tuple[int, str, str]] = []
        for index, text in enumerate(texts):
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            row = self._connection.execute(
                "SELECT payload FROM analyses WHERE namespace = ? AND digest = ?",
                (self._normalizer.cache_namespace, digest),
            ).fetchone()
            if row is None:
                missing.append((index, text, digest))
            else:
                results[index] = AnalyzedText.from_dict(json.loads(row[0]))
        if not missing:
            return [result for result in results if result is not None]
        batch_method = getattr(self._normalizer, "analyze_many", None)
        analyses = (
            batch_method([text for _index, text, _digest in missing])
            if callable(batch_method)
            else [self._normalizer.analyze(text) for _index, text, _digest in missing]
        )
        self._connection.executemany(
            "INSERT OR REPLACE INTO analyses(namespace, digest, payload) VALUES (?, ?, ?)",
            (
                (
                    self._normalizer.cache_namespace,
                    digest,
                    json.dumps(analysis.to_dict(), ensure_ascii=False, sort_keys=True),
                )
                for (_index, _text, digest), analysis in zip(missing, analyses, strict=True)
            ),
        )
        for (index, _text, _digest), analysis in zip(missing, analyses, strict=True):
            results[index] = analysis
        if any(result is None for result in results):
            raise RuntimeError("batched analysis cache lost result alignment")
        return [result for result in results if result is not None]

    def _analyze(self, text: str, *, commit: bool) -> AnalyzedText:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        key = (self._normalizer.cache_namespace, digest)
        row = self._connection.execute(
            "SELECT payload FROM analyses WHERE namespace = ? AND digest = ?", key
        ).fetchone()
        if row is not None:
            return AnalyzedText.from_dict(json.loads(row[0]))
        analysis = self._normalizer.analyze(text)
        self._connection.execute(
            "INSERT OR REPLACE INTO analyses(namespace, digest, payload) VALUES (?, ?, ?)",
            (*key, json.dumps(analysis.to_dict(), ensure_ascii=False, sort_keys=True)),
        )
        if commit:
            self._connection.commit()
        return analysis

    def commit(self) -> None:
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> AnalysisCache:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
