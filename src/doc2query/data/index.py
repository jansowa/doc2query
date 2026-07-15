"""Disk-backed canonical document index for large retrieval datasets."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import unicodedata
from pathlib import Path
from typing import Any

from doc2query.utils.records import JsonParquetWriter, read_records, write_json
from doc2query.utils.tracking import collect_code_provenance


def normalize_document(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def _documents(record: dict[str, Any]) -> list[dict[str, Any]]:
    values = [*record.get("positives", []), *record.get("hard_negatives", [])]
    if not all(isinstance(value, dict) for value in values):
        raise ValueError("canonical record contains invalid document objects")
    return values


def build_document_index(
    input_path: Path,
    *,
    sqlite_path: Path,
    documents_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    if sqlite_path.exists():
        raise FileExistsError(f"document index already exists: {sqlite_path}")
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(sqlite_path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS documents (
          doc_id TEXT PRIMARY KEY,
          text TEXT NOT NULL,
          metadata_json TEXT NOT NULL,
          normalized_text TEXT NOT NULL,
          normalized_hash TEXT NOT NULL,
          char_length INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(normalized_hash);
        CREATE TABLE IF NOT EXISTS conflicts (
          doc_id TEXT NOT NULL,
          existing_hash TEXT NOT NULL,
          conflicting_hash TEXT NOT NULL,
          example_id TEXT NOT NULL
        );
        """
    )
    input_rows = occurrences = inserted = repeated = conflicts = 0
    dataset_fingerprint = hashlib.sha256()
    for record in read_records(input_path):
        input_rows += 1
        for document in _documents(record):
            occurrences += 1
            doc_id = str(document["doc_id"])
            text = str(document["text"])
            normalized = normalize_document(text)
            digest = hashlib.sha256(normalized.encode()).hexdigest()
            metadata_json = json.dumps(
                document.get("metadata", {}), ensure_ascii=False, sort_keys=True
            )
            current = connection.execute(
                "SELECT normalized_hash FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
            if current is None:
                connection.execute(
                    "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?)",
                    (doc_id, text, metadata_json, normalized, digest, len(text)),
                )
                inserted += 1
                dataset_fingerprint.update(doc_id.encode())
                dataset_fingerprint.update(digest.encode())
            elif str(current[0]) == digest:
                repeated += 1
            else:
                conflicts += 1
                connection.execute(
                    "INSERT INTO conflicts VALUES (?, ?, ?, ?)",
                    (doc_id, str(current[0]), digest, str(record.get("example_id", ""))),
                )
            if occurrences % 10_000 == 0:
                connection.commit()
    connection.commit()

    with JsonParquetWriter(documents_path) as writer:
        cursor = connection.execute(
            "SELECT doc_id, text, metadata_json, normalized_hash, char_length "
            "FROM documents ORDER BY doc_id"
        )
        for doc_id, text, metadata_json, digest, char_length in cursor:
            writer.write(
                {
                    "doc_id": doc_id,
                    "text": text,
                    "metadata": json.loads(metadata_json),
                    "normalized_hash": digest,
                    "char_length": char_length,
                }
            )
    report = {
        "input_records": input_rows,
        "document_occurrences": occurrences,
        "unique_document_ids": inserted,
        "repeated_occurrences": repeated,
        "conflicting_document_ids": conflicts,
        "conflict_examples": [
            {
                "doc_id": doc_id,
                "existing_hash": existing_hash,
                "conflicting_hash": conflicting_hash,
                "example_id": example_id,
            }
            for doc_id, existing_hash, conflicting_hash, example_id in connection.execute(
                "SELECT doc_id, existing_hash, conflicting_hash, example_id FROM conflicts LIMIT 50"
            )
        ],
        "document_fingerprint": dataset_fingerprint.hexdigest(),
        "sqlite_path": str(sqlite_path),
        "documents_path": str(documents_path),
        "code": collect_code_provenance(),
    }
    write_json(report_path, report)
    connection.close()
    return report
