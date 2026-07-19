"""Frozen full-corpus indexes and generator round-trip evaluation."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from doc2query.evaluation.retrieval import (
    CORPUS_RETRIEVAL,
    CORPUS_ROUND_TRIP_CUTOFFS,
    corpus_round_trip_metrics,
    validate_recall_cutoffs,
)
from doc2query.text.cache import AnalysisCache
from doc2query.text.normalization import (
    SimplePolishNormalizer,
    SpacyPolishNormalizer,
    TextNormalizer,
)
from doc2query.utils.records import JsonlWriter, read_records, write_json

CORPUS_INDEX_SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class RankedDocument:
    doc_id: str
    score: float
    rank: int


@dataclass(frozen=True)
class CorpusSearchResult:
    documents: tuple[RankedDocument, ...]
    candidate_count: int
    effective_candidate_count: int
    possibly_ambiguous_query: bool


class CorpusIndex(Protocol):
    @property
    def metadata(self) -> Mapping[str, Any]: ...

    @property
    def candidate_count(self) -> int: ...

    def search(self, query: str, *, limit: int) -> CorpusSearchResult: ...

    def score_documents(self, query: str, doc_ids: Sequence[str]) -> dict[str, float]: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class BM25IndexConfig:
    k1: float = 1.2
    b: float = 0.75
    relevance_score_threshold: float = 1.0
    ambiguity_candidate_threshold: int = 20

    def __post_init__(self) -> None:
        if self.k1 <= 0 or not 0 <= self.b <= 1:
            raise ValueError("BM25 requires k1 > 0 and b between 0 and 1")
        if self.ambiguity_candidate_threshold < 1:
            raise ValueError("ambiguity_candidate_threshold must be positive")


def _normalizer_spec(normalizer: TextNormalizer) -> dict[str, str]:
    if isinstance(normalizer, SimplePolishNormalizer):
        return {"backend": "simple", "namespace": normalizer.cache_namespace}
    if isinstance(normalizer, SpacyPolishNormalizer):
        return {
            "backend": "spacy_pl",
            "model_name": normalizer.model_name,
            "namespace": normalizer.cache_namespace,
        }
    return {"backend": "custom", "namespace": normalizer.cache_namespace}


def _load_manifest_normalizer(spec: Mapping[str, Any]) -> TextNormalizer:
    backend = spec.get("backend")
    if backend == "simple":
        return SimplePolishNormalizer()
    if backend == "spacy_pl":
        return SpacyPolishNormalizer(str(spec["model_name"]))
    raise ValueError("custom BM25 normalizer must be supplied explicitly when loading the index")


def _document_payload(record: Mapping[str, Any]) -> tuple[str, str]:
    doc_id, text = record.get("doc_id"), record.get("text")
    if not isinstance(doc_id, str) or not doc_id or not isinstance(text, str) or not text:
        raise ValueError("corpus documents require non-empty string doc_id and text")
    return doc_id, text


def _corpus_digest_update(digest: Any, doc_id: str, text: str) -> None:
    payload = json.dumps(
        {"doc_id": doc_id, "text_sha256": hashlib.sha256(text.encode()).hexdigest()},
        sort_keys=True,
        separators=(",", ":"),
    )
    digest.update(payload.encode())
    digest.update(b"\n")


def backfill_candidate_pools(
    records: Sequence[dict[str, Any]],
    *,
    documents_path: Path,
    corpus_fingerprint: str,
    minimum_negative_count: int = 10,
) -> list[dict[str, Any]]:
    """Deterministically supplement short diagnostic pools from one frozen corpus."""
    if minimum_negative_count < 1:
        raise ValueError("minimum_negative_count must be positive")
    result = deepcopy(list(records))
    pending: list[tuple[dict[str, Any], set[str]]] = []
    for record in result:
        positives = record.get("positives", [])
        negatives = record.get("hard_negatives", [])
        if not isinstance(positives, list) or not isinstance(negatives, list):
            raise ValueError(
                "candidate-pool backfill requires canonical positives and hard_negatives"
            )
        excluded = {
            str(document["doc_id"])
            for document in [*positives, *negatives]
            if isinstance(document, dict) and "doc_id" in document
        }
        if len(negatives) < minimum_negative_count:
            pending.append((record, excluded))
    if not pending:
        return result
    for document in read_records(documents_path):
        doc_id, text = _document_payload(document)
        for record, excluded in pending:
            negatives = record["hard_negatives"]
            if len(negatives) >= minimum_negative_count or doc_id in excluded:
                continue
            metadata = dict(document.get("metadata", {}))
            metadata["candidate_pool_backfill"] = {
                "backfilled": True,
                "source": "same_frozen_corpus",
                "corpus_fingerprint": corpus_fingerprint,
            }
            negatives.append({"doc_id": doc_id, "text": text, "metadata": metadata})
            excluded.add(doc_id)
        if all(len(record["hard_negatives"]) >= minimum_negative_count for record, _ in pending):
            break
    incomplete = [
        str(record.get("example_id", ""))
        for record, _excluded in pending
        if len(record["hard_negatives"]) < minimum_negative_count
    ]
    if incomplete:
        raise ValueError(f"frozen corpus cannot backfill candidate pools for: {incomplete[:3]}")
    for record, _excluded in pending:
        record["candidate_pool_backfilled_count"] = sum(
            bool(
                isinstance(document.get("metadata"), dict)
                and document["metadata"].get("candidate_pool_backfill", {}).get("backfilled")
            )
            for document in record["hard_negatives"]
        )
    return result


def build_bm25_index(
    documents_path: Path,
    *,
    output_dir: Path,
    config: BM25IndexConfig,
    normalizer: TextNormalizer | None = None,
    analysis_cache_path: Path | None = None,
) -> dict[str, Any]:
    """Build an exact disk-backed BM25 index over cached Polish content lemmas."""
    if output_dir.exists():
        raise FileExistsError(f"corpus index output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    normalizer = normalizer or SimplePolishNormalizer()
    database_path = output_dir / "index.sqlite"
    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.executescript(
        """
        CREATE TABLE documents (
          ordinal INTEGER PRIMARY KEY,
          doc_id TEXT NOT NULL UNIQUE,
          token_count INTEGER NOT NULL,
          text_sha256 TEXT NOT NULL
        );
        CREATE TABLE postings (
          term TEXT NOT NULL,
          doc_ordinal INTEGER NOT NULL,
          term_frequency INTEGER NOT NULL,
          PRIMARY KEY(term, doc_ordinal)
        ) WITHOUT ROWID;
        """
    )
    corpus_digest = hashlib.sha256()
    previous_id: str | None = None
    document_count = 0
    total_tokens = 0
    cache = AnalysisCache(analysis_cache_path, normalizer) if analysis_cache_path else None

    def index_batch(records: Sequence[Mapping[str, Any]]) -> None:
        nonlocal document_count, previous_id, total_tokens
        payloads = [_document_payload(record) for record in records]
        texts = [text for _doc_id, text in payloads]
        analyses = (
            cache.analyze_many_uncommitted(texts)
            if cache
            else [normalizer.analyze(text) for text in texts]
        )
        for (doc_id, text), analysis in zip(payloads, analyses, strict=True):
            if previous_id is not None and doc_id <= previous_id:
                raise ValueError("frozen corpus must be sorted by unique doc_id")
            previous_id = doc_id
            counts = Counter(analysis.content_lemmas)
            token_count = sum(counts.values())
            text_digest = hashlib.sha256(text.encode()).hexdigest()
            connection.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?)",
                (document_count, doc_id, token_count, text_digest),
            )
            connection.executemany(
                "INSERT INTO postings VALUES (?, ?, ?)",
                ((term, document_count, frequency) for term, frequency in counts.items()),
            )
            _corpus_digest_update(corpus_digest, doc_id, text)
            document_count += 1
            total_tokens += token_count
        connection.commit()
        if cache:
            cache.commit()

    try:
        batch: list[Mapping[str, Any]] = []
        for record in read_records(documents_path):
            batch.append(record)
            if len(batch) == 10_000:
                index_batch(batch)
                batch = []
        if batch:
            index_batch(batch)
        if document_count == 0:
            raise ValueError("cannot build a corpus index from zero documents")
        connection.executescript(
            """
            CREATE INDEX idx_postings_doc ON postings(doc_ordinal);
            CREATE TABLE terms AS
              SELECT term, COUNT(*) AS document_frequency
              FROM postings GROUP BY term;
            CREATE UNIQUE INDEX idx_terms_term ON terms(term);
            """
        )
        connection.commit()
        if cache:
            cache.commit()
    finally:
        if cache:
            cache.close()
        connection.close()
    manifest = {
        "schema_version": CORPUS_INDEX_SCHEMA_VERSION,
        "protocol": CORPUS_RETRIEVAL,
        "backend": "bm25_sqlite",
        "implementation_license": "project-internal",
        "documents_path": str(documents_path.resolve()),
        "documents_sha256": sha256_file(documents_path),
        "document_fingerprint": corpus_digest.hexdigest(),
        "candidate_count": document_count,
        "average_document_length": total_tokens / document_count,
        "normalizer_namespace": normalizer.cache_namespace,
        "normalizer": _normalizer_spec(normalizer),
        "config": asdict(config),
        "database_file": database_path.name,
        "database_sha256": sha256_file(database_path),
    }
    manifest["index_fingerprint"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    write_json(output_dir / "manifest.json", manifest)
    return manifest


class BM25CorpusIndex:
    def __init__(self, index_dir: Path, *, normalizer: TextNormalizer | None = None) -> None:
        manifest_path = index_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema_version") != CORPUS_INDEX_SCHEMA_VERSION
            or manifest.get("backend") != "bm25_sqlite"
            or manifest.get("protocol") != CORPUS_RETRIEVAL
        ):
            raise ValueError(f"unsupported BM25 corpus index manifest: {manifest_path}")
        database_path = index_dir / str(manifest["database_file"])
        if sha256_file(database_path) != manifest["database_sha256"]:
            raise RuntimeError("BM25 corpus index fingerprint mismatch")
        self._manifest: dict[str, Any] = manifest
        raw_normalizer = manifest.get("normalizer", {})
        if not isinstance(raw_normalizer, dict):
            raise ValueError("invalid normalizer config in BM25 corpus manifest")
        self._normalizer = normalizer or _load_manifest_normalizer(raw_normalizer)
        if self._normalizer.cache_namespace != manifest["normalizer_namespace"]:
            raise ValueError("BM25 query normalizer does not match the frozen index")
        self._connection = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
        raw_config = manifest.get("config", {})
        if not isinstance(raw_config, dict):
            raise ValueError("invalid BM25 config in corpus manifest")
        self._config = BM25IndexConfig(**raw_config)

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self._manifest

    @property
    def candidate_count(self) -> int:
        return int(self._manifest["candidate_count"])

    def _query_terms(self, query: str) -> tuple[str, ...]:
        return tuple(sorted(set(self._normalizer.analyze(query).content_lemmas)))

    def _scored_sql(self, terms: Sequence[str], *, doc_filter: bool = False) -> str:
        placeholders = ", ".join("?" for _ in terms)
        document_count = self.candidate_count
        average_length = float(self._manifest["average_document_length"]) or 1.0
        k1, b = self._config.k1, self._config.b
        filter_clause = " AND d.doc_id IN ({doc_ids})" if doc_filter else ""
        return (
            "SELECT d.doc_id, SUM("
            f"(log(({document_count} - t.document_frequency + 0.5) / "
            "(t.document_frequency + 0.5) + 1.0)) * "
            f"(p.term_frequency * ({k1} + 1.0)) / "
            f"(p.term_frequency + {k1} * (1.0 - {b} + {b} * "
            f"d.token_count / {average_length}))"
            ") AS score "
            "FROM postings p "
            "JOIN terms t ON t.term = p.term "
            "JOIN documents d ON d.ordinal = p.doc_ordinal "
            f"WHERE p.term IN ({placeholders}){filter_clause} "
            "GROUP BY p.doc_ordinal"
        )

    def _top_matches(self, terms: Sequence[str], limit: int) -> list[tuple[str, float]]:
        if not terms:
            return []
        sql = self._scored_sql(terms) + " ORDER BY score DESC, d.doc_id ASC LIMIT ?"
        return [
            (str(doc_id), float(score))
            for doc_id, score in self._connection.execute(sql, (*terms, limit))
        ]

    def _effective_count(self, terms: Sequence[str]) -> int:
        threshold = self._config.relevance_score_threshold
        if threshold <= 0:
            return self.candidate_count
        if not terms:
            return 0
        sql = "SELECT COUNT(*) FROM (" + self._scored_sql(terms) + ") WHERE score >= ?"
        row = self._connection.execute(sql, (*terms, threshold)).fetchone()
        return int(row[0]) if row else 0

    def search(self, query: str, *, limit: int) -> CorpusSearchResult:
        validate_recall_cutoffs(self.candidate_count, (limit,))
        terms = self._query_terms(query)
        matches = self._top_matches(terms, limit)
        if len(matches) < limit:
            seen = {doc_id for doc_id, _score in matches}
            placeholders = ", ".join("?" for _ in seen)
            exclusion = f" WHERE doc_id NOT IN ({placeholders})" if seen else ""
            sql = f"SELECT doc_id FROM documents{exclusion} ORDER BY doc_id LIMIT ?"
            parameters: tuple[Any, ...] = (*sorted(seen), limit - len(matches))
            matches.extend((str(row[0]), 0.0) for row in self._connection.execute(sql, parameters))
        effective = self._effective_count(terms)
        return CorpusSearchResult(
            documents=tuple(
                RankedDocument(doc_id=doc_id, score=score, rank=rank)
                for rank, (doc_id, score) in enumerate(matches, 1)
            ),
            candidate_count=self.candidate_count,
            effective_candidate_count=effective,
            possibly_ambiguous_query=(
                effective == 0 or effective >= self._config.ambiguity_candidate_threshold
            ),
        )

    def score_documents(self, query: str, doc_ids: Sequence[str]) -> dict[str, float]:
        wanted = tuple(dict.fromkeys(str(doc_id) for doc_id in doc_ids))
        if not wanted:
            return {}
        doc_placeholders = ", ".join("?" for _ in wanted)
        present = {
            str(row[0])
            for row in self._connection.execute(
                f"SELECT doc_id FROM documents WHERE doc_id IN ({doc_placeholders})",
                wanted,
            )
        }
        missing = sorted(set(wanted) - present)
        if missing:
            raise KeyError(f"documents are absent from frozen corpus: {missing[:3]}")
        terms = self._query_terms(query)
        scores = dict.fromkeys(wanted, 0.0)
        if not terms:
            return scores
        sql = self._scored_sql(terms, doc_filter=True).format(doc_ids=doc_placeholders)
        for doc_id, score in self._connection.execute(sql, (*terms, *wanted)):
            scores[str(doc_id)] = float(score)
        return scores

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> BM25CorpusIndex:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


@dataclass(frozen=True)
class FrozenBiEncoderConfig:
    model_name_or_path: str
    revision: str
    license: str
    trust_remote_code: bool = False
    normalize_embeddings: bool = True
    batch_size: int = 64
    relevance_score_threshold: float = 0.5
    ambiguity_candidate_threshold: int = 20

    def __post_init__(self) -> None:
        if len(self.revision) != 40:
            raise ValueError("auxiliary bi-encoder revision must be a full 40-character commit")
        if not self.license.strip():
            raise ValueError("auxiliary bi-encoder license must be recorded")
        if self.trust_remote_code:
            raise ValueError("auxiliary bi-encoder must not require trust_remote_code")
        if self.batch_size < 1 or self.ambiguity_candidate_threshold < 1:
            raise ValueError("bi-encoder batch and ambiguity thresholds must be positive")


class TextEncoder(Protocol):
    def encode(self, texts: Sequence[str], *, batch_size: int) -> Any: ...


class SentenceTransformerTextEncoder:
    def __init__(self, config: FrozenBiEncoderConfig) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("install the retrieval dependency group") from exc
        self._config = config
        self._model = SentenceTransformer(
            config.model_name_or_path,
            revision=config.revision,
            trust_remote_code=config.trust_remote_code,
        )

    def encode(self, texts: Sequence[str], *, batch_size: int) -> Any:
        return self._model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=self._config.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )


def build_biencoder_index(
    documents_path: Path,
    *,
    output_dir: Path,
    config: FrozenBiEncoderConfig,
    encoder: TextEncoder | None = None,
) -> dict[str, Any]:
    """Build a frozen brute-force embedding index; FAISS is not required."""
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("install the data/retrieval dependency groups") from exc
    if output_dir.exists():
        raise FileExistsError(f"corpus index output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    encoder = encoder or SentenceTransformerTextEncoder(config)
    ids: list[str] = []
    texts: list[str] = []
    digest = hashlib.sha256()
    previous_id: str | None = None
    for record in read_records(documents_path):
        doc_id, text = _document_payload(record)
        if previous_id is not None and doc_id <= previous_id:
            raise ValueError("frozen corpus must be sorted by unique doc_id")
        previous_id = doc_id
        ids.append(doc_id)
        texts.append(text)
        _corpus_digest_update(digest, doc_id, text)
    if not ids:
        raise ValueError("cannot build a corpus index from zero documents")
    embeddings = np.asarray(encoder.encode(texts, batch_size=config.batch_size), dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(ids):
        raise ValueError("bi-encoder returned an invalid corpus embedding matrix")
    embeddings_path = output_dir / "embeddings.npy"
    np.save(embeddings_path, embeddings, allow_pickle=False)
    ids_path = output_dir / "doc_ids.jsonl"
    with JsonlWriter(ids_path) as writer:
        for doc_id in ids:
            writer.write({"doc_id": doc_id})
    manifest = {
        "schema_version": CORPUS_INDEX_SCHEMA_VERSION,
        "protocol": CORPUS_RETRIEVAL,
        "backend": "biencoder_bruteforce",
        "documents_path": str(documents_path.resolve()),
        "documents_sha256": sha256_file(documents_path),
        "document_fingerprint": digest.hexdigest(),
        "candidate_count": len(ids),
        "embedding_dimension": int(embeddings.shape[1]),
        "config": asdict(config),
        "embeddings_file": embeddings_path.name,
        "embeddings_sha256": sha256_file(embeddings_path),
        "doc_ids_file": ids_path.name,
        "doc_ids_sha256": sha256_file(ids_path),
    }
    manifest["index_fingerprint"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    write_json(output_dir / "manifest.json", manifest)
    return manifest


class BiEncoderCorpusIndex:
    def __init__(self, index_dir: Path, *, encoder: TextEncoder | None = None) -> None:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("install the data/retrieval dependency groups") from exc
        manifest_path = index_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema_version") != CORPUS_INDEX_SCHEMA_VERSION
            or manifest.get("backend") != "biencoder_bruteforce"
            or manifest.get("protocol") != CORPUS_RETRIEVAL
        ):
            raise ValueError(f"unsupported bi-encoder corpus index manifest: {manifest_path}")
        embeddings_path = index_dir / str(manifest["embeddings_file"])
        ids_path = index_dir / str(manifest["doc_ids_file"])
        if (
            sha256_file(embeddings_path) != manifest["embeddings_sha256"]
            or sha256_file(ids_path) != manifest["doc_ids_sha256"]
        ):
            raise RuntimeError("bi-encoder corpus index fingerprint mismatch")
        raw_config = manifest.get("config", {})
        if not isinstance(raw_config, dict):
            raise ValueError("invalid bi-encoder config in corpus manifest")
        self._config = FrozenBiEncoderConfig(**raw_config)
        self._encoder = encoder or SentenceTransformerTextEncoder(self._config)
        self._manifest: dict[str, Any] = manifest
        self._embeddings = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
        self._ids = tuple(str(row["doc_id"]) for row in read_records(ids_path))
        self._id_to_index = {doc_id: index for index, doc_id in enumerate(self._ids)}
        if len(self._ids) != self.candidate_count or self._embeddings.shape[0] != len(self._ids):
            raise RuntimeError("bi-encoder corpus index dimensions do not match its manifest")

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self._manifest

    @property
    def candidate_count(self) -> int:
        return int(self._manifest["candidate_count"])

    def _scores(self, query: str) -> Any:
        import numpy as np

        encoded = np.asarray(
            self._encoder.encode([query], batch_size=self._config.batch_size), dtype=np.float32
        )
        if encoded.shape != (1, self._embeddings.shape[1]):
            raise ValueError("bi-encoder returned an invalid query embedding")
        return self._embeddings @ encoded[0]

    def search(self, query: str, *, limit: int) -> CorpusSearchResult:
        import numpy as np

        validate_recall_cutoffs(self.candidate_count, (limit,))
        scores = self._scores(query)
        order = np.lexsort((np.asarray(self._ids), -scores))[:limit]
        effective = int(np.sum(scores >= self._config.relevance_score_threshold))
        return CorpusSearchResult(
            documents=tuple(
                RankedDocument(
                    doc_id=self._ids[int(index)],
                    score=float(scores[int(index)]),
                    rank=rank,
                )
                for rank, index in enumerate(order, 1)
            ),
            candidate_count=self.candidate_count,
            effective_candidate_count=effective,
            possibly_ambiguous_query=(
                effective == 0 or effective >= self._config.ambiguity_candidate_threshold
            ),
        )

    def score_documents(self, query: str, doc_ids: Sequence[str]) -> dict[str, float]:
        scores = self._scores(query)
        result = {}
        for doc_id in dict.fromkeys(str(value) for value in doc_ids):
            if doc_id not in self._id_to_index:
                raise KeyError(f"document is absent from frozen corpus: {doc_id}")
            result[doc_id] = float(scores[self._id_to_index[doc_id]])
        return result

    def close(self) -> None:
        return None


def load_corpus_index(index_dir: Path) -> CorpusIndex:
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    backend = manifest.get("backend")
    if backend == "bm25_sqlite":
        return BM25CorpusIndex(index_dir)
    if backend == "biencoder_bruteforce":
        return BiEncoderCorpusIndex(index_dir)
    raise ValueError(f"unsupported corpus index backend: {backend}")


def evaluate_round_trip_query(
    index: CorpusIndex,
    *,
    query: str,
    positive_doc_ids: Iterable[str],
    cutoffs: Sequence[int] = CORPUS_ROUND_TRIP_CUTOFFS,
) -> dict[str, Any]:
    """Evaluate one generated query against the complete frozen corpus."""
    normalized_cutoffs = validate_recall_cutoffs(index.candidate_count, cutoffs)
    positives = frozenset(str(value) for value in positive_doc_ids)
    if not positives:
        raise ValueError("corpus round-trip requires at least one positive document ID")
    result = index.search(query, limit=max(normalized_cutoffs))
    positive_ranks = [
        document.rank for document in result.documents if document.doc_id in positives
    ]
    positive_scores = index.score_documents(query, sorted(positives))
    best_positive = max(positive_scores.values())
    best_nonpositive = next(
        (document.score for document in result.documents if document.doc_id not in positives),
        None,
    )
    margin = best_positive - best_nonpositive if best_nonpositive is not None else None
    return {
        **corpus_round_trip_metrics(
            positive_ranks,
            candidate_count=result.candidate_count,
            cutoffs=normalized_cutoffs,
        ),
        "corpus_effective_candidate_count": result.effective_candidate_count,
        "corpus_margin_to_best_nonpositive": margin,
        "corpus_possibly_ambiguous_query": result.possibly_ambiguous_query,
        "corpus_best_positive_score": best_positive,
        "corpus_best_nonpositive_score": best_nonpositive,
    }
