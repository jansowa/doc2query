#!/usr/bin/env python3
"""Freeze the sorted corpus of documents referenced by the canonical train split."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

from doc2query.evaluation.corpus import sha256_file
from doc2query.utils.records import JsonParquetWriter, read_records, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--document-index", type=Path, required=True)
    parser.add_argument("--id-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.manifest.exists() or args.id_cache.exists():
        raise FileExistsError("frozen train-corpus outputs must not already exist")
    args.id_cache.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(args.id_cache)
    connection.execute("CREATE TABLE train_ids (doc_id TEXT PRIMARY KEY)")
    query_count = 0
    try:
        for record in read_records(args.train_input):
            metadata = record.get("metadata")
            if not isinstance(metadata, dict) or metadata.get("split") != "train":
                raise ValueError("train corpus accepts canonical train records only")
            documents = [*record.get("positives", []), *record.get("hard_negatives", [])]
            connection.executemany(
                "INSERT OR IGNORE INTO train_ids VALUES (?)",
                ((str(document["doc_id"]),) for document in documents),
            )
            query_count += 1
            if query_count % 1000 == 0:
                connection.commit()
        connection.commit()
        document_count = int(connection.execute("SELECT COUNT(*) FROM train_ids").fetchone()[0])
        connection.execute("ATTACH DATABASE ? AS source", (str(args.document_index),))
        found = int(
            connection.execute(
                "SELECT COUNT(*) FROM train_ids i JOIN source.documents d USING (doc_id)"
            ).fetchone()[0]
        )
        if found != document_count:
            raise ValueError(f"document index is missing {document_count - found} train documents")
        id_digest = hashlib.sha256()
        with JsonParquetWriter(args.output) as writer:
            cursor = connection.execute(
                """
                SELECT d.doc_id, d.text, d.metadata_json
                FROM train_ids i JOIN source.documents d USING (doc_id)
                ORDER BY d.doc_id
                """
            )
            for doc_id, text, metadata_json in cursor:
                id_digest.update(str(doc_id).encode())
                id_digest.update(b"\n")
                writer.write(
                    {
                        "doc_id": str(doc_id),
                        "text": str(text),
                        "metadata": json.loads(str(metadata_json)),
                    }
                )
    finally:
        connection.close()
    manifest = {
        "schema_version": 1,
        "artifact_id": "train-corpus-v1",
        "split": "train",
        "query_count": query_count,
        "document_count": document_count,
        "train_input_sha256": sha256_file(args.train_input),
        "document_index_sha256": sha256_file(args.document_index),
        "ordered_doc_ids_sha256": id_digest.hexdigest(),
        "documents_sha256": sha256_file(args.output),
        "tests_included": [],
    }
    manifest["artifact_fingerprint"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    write_json(args.manifest, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
