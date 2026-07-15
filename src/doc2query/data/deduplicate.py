"""Exact and banded-SimHash near-duplicate clustering backed by SQLite."""

from __future__ import annotations

import hashlib
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from doc2query.utils.records import JsonParquetWriter, write_json
from doc2query.utils.tracking import collect_code_provenance


def simhash64(text: str) -> int:
    tokens = text.split()
    shingles = [" ".join(tokens[index : index + 3]) for index in range(max(1, len(tokens) - 2))]
    if not shingles:
        shingles = [text]
    vector = [0] * 64
    for shingle in shingles:
        digest = int.from_bytes(hashlib.blake2b(shingle.encode(), digest_size=8).digest(), "big")
        for bit in range(64):
            vector[bit] += 1 if digest & (1 << bit) else -1
    return sum(1 << bit for bit, value in enumerate(vector) if value >= 0)


def _find(connection: sqlite3.Connection, node: str) -> str:
    path: list[str] = []
    current = node
    while True:
        row = connection.execute(
            "SELECT parent FROM parents WHERE doc_id = ?", (current,)
        ).fetchone()
        if row is None:
            connection.execute("INSERT INTO parents VALUES (?, ?)", (current, current))
            root = current
            break
        parent = str(row[0])
        if parent == current:
            root = current
            break
        path.append(current)
        current = parent
    for item in path:
        connection.execute("UPDATE parents SET parent = ? WHERE doc_id = ?", (root, item))
    return root


def _union(connection: sqlite3.Connection, left: str, right: str) -> None:
    left_root, right_root = _find(connection, left), _find(connection, right)
    if left_root == right_root:
        return
    root, child = sorted((left_root, right_root))
    connection.execute("UPDATE parents SET parent = ? WHERE doc_id = ?", (root, child))


def _hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def deduplicate_documents(
    index_path: Path,
    *,
    output_path: Path,
    report_path: Path,
    max_hamming_distance: int = 3,
    bands: int = 4,
    candidate_cap: int = 500,
) -> dict[str, Any]:
    if 64 % bands:
        raise ValueError("bands must divide 64")
    connection = sqlite3.connect(index_path)
    conflict_count = int(connection.execute("SELECT COUNT(*) FROM conflicts").fetchone()[0])
    if conflict_count:
        connection.close()
        raise ValueError(
            f"document index contains {conflict_count} conflicting doc_id values; "
            "resolve them first"
        )
    connection.executescript(
        """
        DROP TABLE IF EXISTS parents;
        DROP TABLE IF EXISTS simhash;
        DROP TABLE IF EXISTS lsh_bands;
        CREATE TABLE parents (doc_id TEXT PRIMARY KEY, parent TEXT NOT NULL);
        CREATE TABLE simhash (doc_id TEXT PRIMARY KEY, value_hex TEXT NOT NULL);
        CREATE TABLE lsh_bands (
          band INTEGER NOT NULL, band_key INTEGER NOT NULL, doc_id TEXT NOT NULL
        );
        CREATE INDEX idx_lsh_bands ON lsh_bands(band, band_key);
        """
    )
    exact_groups = 0
    exact_duplicates = 0
    cursor = connection.execute(
        "SELECT normalized_hash, MIN(doc_id), COUNT(*) FROM documents "
        "GROUP BY normalized_hash ORDER BY MIN(doc_id)"
    )
    for digest, representative, count in cursor:
        representative = str(representative)
        connection.execute(
            "INSERT OR IGNORE INTO parents VALUES (?, ?)", (representative, representative)
        )
        if int(count) > 1:
            exact_groups += 1
            exact_duplicates += int(count) - 1
        members = connection.execute(
            "SELECT doc_id FROM documents WHERE normalized_hash = ? ORDER BY doc_id", (digest,)
        ).fetchall()
        for (doc_id_value,) in members:
            doc_id = str(doc_id_value)
            connection.execute("INSERT OR IGNORE INTO parents VALUES (?, ?)", (doc_id, doc_id))
            _union(connection, representative, doc_id)
        text = connection.execute(
            "SELECT normalized_text FROM documents WHERE doc_id = ?", (representative,)
        ).fetchone()[0]
    connection.commit()

    band_bits = 64 // bands
    mask = (1 << band_bits) - 1
    near_edges = capped_queries = 0
    representative_cursor = connection.execute(
        "SELECT MIN(doc_id), MIN(normalized_text) FROM documents "
        "GROUP BY normalized_hash ORDER BY MIN(doc_id)"
    )
    for position, (doc_id_value, text_value) in enumerate(representative_cursor, start=1):
        doc_id, text = str(doc_id_value), str(text_value)
        value = simhash64(text)
        candidates: set[str] = set()
        for band in range(bands):
            key = (value >> (band * band_bits)) & mask
            rows = connection.execute(
                "SELECT doc_id FROM lsh_bands WHERE band = ? AND band_key = ? LIMIT ?",
                (band, key, candidate_cap + 1),
            ).fetchall()
            candidates.update(str(row[0]) for row in rows)
        if len(candidates) > candidate_cap:
            capped_queries += 1
            candidates = set(sorted(candidates)[:candidate_cap])
        for candidate in candidates:
            other_hex = connection.execute(
                "SELECT value_hex FROM simhash WHERE doc_id = ?", (candidate,)
            ).fetchone()
            if (
                other_hex is not None
                and _hamming(value, int(str(other_hex[0]), 16)) <= max_hamming_distance
            ):
                _union(connection, doc_id, candidate)
                near_edges += 1
        connection.execute("INSERT INTO simhash VALUES (?, ?)", (doc_id, f"{value:016x}"))
        for band in range(bands):
            key = (value >> (band * band_bits)) & mask
            connection.execute("INSERT INTO lsh_bands VALUES (?, ?, ?)", (band, key, doc_id))
        if position % 10_000 == 0:
            connection.commit()
    connection.commit()

    cluster_sizes: Counter[str] = Counter()
    with JsonParquetWriter(output_path) as writer:
        cursor = connection.execute(
            "SELECT doc_id, normalized_hash, normalized_text FROM documents ORDER BY doc_id"
        )
        document_count = 0
        for doc_id_value, digest, text in cursor:
            document_count += 1
            doc_id = str(doc_id_value)
            canonical = _find(connection, doc_id)
            cluster_sizes[canonical] += 1
            canonical_row = connection.execute(
                "SELECT normalized_hash, normalized_text FROM documents WHERE doc_id = ?",
                (canonical,),
            ).fetchone()
            same_hash = canonical_row is not None and str(canonical_row[0]) == str(digest)
            similarity = 1.0
            if not same_hash and canonical_row is not None:
                similarity = (
                    1.0 - _hamming(simhash64(str(text)), simhash64(str(canonical_row[1]))) / 64
                )
            writer.write(
                {
                    "doc_id": doc_id,
                    "canonical_doc_id": canonical,
                    "cluster_id": canonical,
                    "match_type": (
                        "canonical" if doc_id == canonical else "exact" if same_hash else "near"
                    ),
                    "similarity": similarity,
                }
            )
    largest = sorted(cluster_sizes.items(), key=lambda item: (-item[1], item[0]))[:50]
    largest_clusters = []
    for cluster_id, size in largest:
        members = []
        for (doc_id,) in connection.execute(
            "SELECT doc_id FROM parents WHERE parent = ? ORDER BY doc_id LIMIT 5", (cluster_id,)
        ):
            text_row = connection.execute(
                "SELECT text FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
            members.append({"doc_id": doc_id, "text_preview": str(text_row[0])[:300]})
        largest_clusters.append({"cluster_id": cluster_id, "size": size, "members": members})
    report = {
        "documents": document_count,
        "clusters": len(cluster_sizes),
        "exact_duplicate_groups": exact_groups,
        "exact_duplicate_documents": exact_duplicates,
        "near_duplicate_edges": near_edges,
        "candidate_cap_hits": capped_queries,
        "parameters": {
            "algorithm": "banded_simhash64",
            "max_hamming_distance": max_hamming_distance,
            "bands": bands,
            "candidate_cap": candidate_cap,
        },
        "largest_clusters": largest_clusters,
        "code": collect_code_provenance(),
    }
    write_json(report_path, report)
    connection.commit()
    connection.close()
    return report


def load_dedup_map(path: Path) -> dict[str, str]:
    from doc2query.utils.records import read_records

    return {str(row["doc_id"]): str(row["canonical_doc_id"]) for row in read_records(path)}
