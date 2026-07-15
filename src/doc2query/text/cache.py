"""SQLite cache keyed by text digest and normalizer namespace."""

from __future__ import annotations

import hashlib
import json
import sqlite3
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
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        key = (self._normalizer.cache_namespace, digest)
        row = self._connection.execute(
            "SELECT payload FROM analyses WHERE namespace = ? AND digest = ?", key
        ).fetchone()
        if row is not None:
            return AnalyzedText.from_dict(json.loads(row[0]))
        analysis = self._normalizer.analyze(text)
        with self._connection:
            self._connection.execute(
                "INSERT OR REPLACE INTO analyses(namespace, digest, payload) VALUES (?, ?, ?)",
                (*key, json.dumps(analysis.to_dict(), ensure_ascii=False, sort_keys=True)),
            )
        return analysis

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> AnalysisCache:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
