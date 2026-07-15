"""Deterministic component-level splits with hard-negative leakage cleanup."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from doc2query.utils.records import JsonParquetWriter, read_records, write_json
from doc2query.utils.tracking import collect_code_provenance


@dataclass(frozen=True)
class SplitConfig:
    train_ratio: float = 0.90
    dev_ratio: float = 0.05
    test_ratio: float = 0.05
    seed: int = 42
    version: str = "v1"
    remove_cross_split_negatives: bool = True

    def __post_init__(self) -> None:
        total = self.train_ratio + self.dev_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-9 or min(self.train_ratio, self.dev_ratio, self.test_ratio) <= 0:
            raise ValueError("split ratios must be positive and sum to 1")

    @property
    def ratios(self) -> dict[str, float]:
        return {"train": self.train_ratio, "dev": self.dev_ratio, "test": self.test_ratio}


def _find(connection: sqlite3.Connection, node: str) -> str:
    row = connection.execute("SELECT parent FROM parents WHERE node = ?", (node,)).fetchone()
    if row is None:
        connection.execute("INSERT INTO parents VALUES (?, ?)", (node, node))
        return node
    parent = str(row[0])
    if parent == node:
        return node
    root = _find(connection, parent)
    connection.execute("UPDATE parents SET parent = ? WHERE node = ?", (root, node))
    return root


def _union(connection: sqlite3.Connection, left: str, right: str) -> None:
    left_root, right_root = _find(connection, left), _find(connection, right)
    if left_root == right_root:
        return
    root, child = sorted((left_root, right_root))
    connection.execute("UPDATE parents SET parent = ? WHERE node = ?", (root, child))


def _canonical(connection: sqlite3.Connection, doc_id: str) -> str:
    row = connection.execute(
        "SELECT canonical_doc_id FROM dedup WHERE doc_id = ?", (doc_id,)
    ).fetchone()
    return str(row[0]) if row is not None else doc_id


def _domain(record: dict[str, Any]) -> str:
    metadata = record.get("metadata")
    return str(metadata.get("domain", "unknown")) if isinstance(metadata, dict) else "unknown"


def _tie(seed: int, component: str, split: str) -> int:
    value = f"{seed}:{component}:{split}".encode()
    return int.from_bytes(hashlib.blake2b(value, digest_size=8).digest(), "big")


def _assign_components(connection: sqlite3.Connection, config: SplitConfig) -> dict[str, int]:
    total_queries = int(connection.execute("SELECT COUNT(*) FROM query_meta").fetchone()[0])
    domain_totals = {
        str(domain): int(count)
        for domain, count in connection.execute(
            "SELECT domain, COUNT(*) FROM query_meta GROUP BY domain"
        )
    }
    assigned: Counter[str] = Counter()
    assigned_domains: dict[str, Counter[str]] = defaultdict(Counter)
    component_rows = connection.execute(
        "SELECT component, COUNT(*) AS size FROM query_meta GROUP BY component "
        "ORDER BY size DESC, component"
    )
    for component_value, size_value in component_rows:
        component, size = str(component_value), int(size_value)
        domains = {
            str(domain): int(count)
            for domain, count in connection.execute(
                "SELECT domain, COUNT(*) FROM query_meta WHERE component = ? GROUP BY domain",
                (component,),
            )
        }
        candidates: list[tuple[float, int, str]] = []
        for split, ratio in config.ratios.items():
            global_target = max(total_queries * ratio, 1e-12)
            global_pressure = (assigned[split] + size) / global_target
            domain_pressure = 0.0
            for domain, count in domains.items():
                target = max(domain_totals[domain] * ratio, 1e-12)
                domain_pressure += (assigned_domains[domain][split] + count) / target
            candidates.append(
                (global_pressure + domain_pressure, _tie(config.seed, component, split), split)
            )
        split = min(candidates)[2]
        connection.execute("INSERT INTO component_split VALUES (?, ?)", (component, split))
        assigned[split] += size
        for domain, count in domains.items():
            assigned_domains[domain][split] += count
    return dict(assigned)


def build_splits(
    input_path: Path,
    dedup_path: Path,
    *,
    output_dir: Path,
    config: SplitConfig | None = None,
) -> dict[str, Any]:
    config = config or SplitConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "split_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"frozen split already exists: {manifest_path}")
    work_path = output_dir / "split_work.sqlite"
    connection = sqlite3.connect(work_path)
    connection.executescript(
        """
        CREATE TABLE dedup (doc_id TEXT PRIMARY KEY, canonical_doc_id TEXT NOT NULL);
        CREATE TABLE parents (node TEXT PRIMARY KEY, parent TEXT NOT NULL);
        CREATE TABLE query_meta (
          query_id TEXT PRIMARY KEY, domain TEXT NOT NULL, component TEXT
        );
        CREATE INDEX idx_query_component ON query_meta(component);
        CREATE TABLE component_split (component TEXT PRIMARY KEY, split TEXT NOT NULL);
        CREATE TABLE query_split (query_id TEXT PRIMARY KEY, split TEXT NOT NULL);
        CREATE TABLE positive_split (
          canonical_doc_id TEXT PRIMARY KEY, split TEXT NOT NULL
        );
        """
    )
    dedup_fingerprint = hashlib.sha256()
    for mapping in read_records(dedup_path):
        mapping_payload = json.dumps(mapping, sort_keys=True, separators=(",", ":"))
        dedup_fingerprint.update(mapping_payload.encode())
        connection.execute(
            "INSERT INTO dedup VALUES (?, ?)",
            (str(mapping["doc_id"]), str(mapping["canonical_doc_id"])),
        )
    connection.commit()

    input_fingerprint = hashlib.sha256()
    query_count = 0
    for record in read_records(input_path):
        query_id = str(record["example_id"])
        query_node = f"q:{query_id}"
        _find(connection, query_node)
        for positive in record["positives"]:
            canonical = _canonical(connection, str(positive["doc_id"]))
            _union(connection, query_node, f"d:{canonical}")
        connection.execute(
            "INSERT INTO query_meta(query_id, domain, component) VALUES (?, ?, NULL)",
            (query_id, _domain(record)),
        )
        input_fingerprint.update(
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        )
        query_count += 1
        if query_count % 10_000 == 0:
            connection.commit()
    connection.commit()
    for (query_id,) in connection.execute("SELECT query_id FROM query_meta ORDER BY query_id"):
        component = _find(connection, f"q:{query_id}")
        connection.execute(
            "UPDATE query_meta SET component = ? WHERE query_id = ?", (component, query_id)
        )
    connection.commit()
    assigned_counts = _assign_components(connection, config)
    connection.execute(
        "INSERT INTO query_split "
        "SELECT q.query_id, c.split FROM query_meta q JOIN component_split c USING(component)"
    )

    positive_leakage = 0
    for record in read_records(input_path):
        query_id = str(record["example_id"])
        split = str(
            connection.execute(
                "SELECT split FROM query_split WHERE query_id = ?", (query_id,)
            ).fetchone()[0]
        )
        for positive in record["positives"]:
            canonical = _canonical(connection, str(positive["doc_id"]))
            existing = connection.execute(
                "SELECT split FROM positive_split WHERE canonical_doc_id = ?", (canonical,)
            ).fetchone()
            if existing is not None and str(existing[0]) != split:
                positive_leakage += 1
            else:
                connection.execute(
                    "INSERT OR IGNORE INTO positive_split VALUES (?, ?)", (canonical, split)
                )
    connection.commit()
    if positive_leakage:
        connection.close()
        raise RuntimeError(
            f"split invariant violated: {positive_leakage} canonical positives cross splits"
        )

    output_counts: Counter[str] = Counter()
    removed_negatives: Counter[str] = Counter()
    below_ten: Counter[str] = Counter()
    writers = {
        split: JsonParquetWriter(output_dir / f"{split}.parquet")
        for split in ("train", "dev", "test")
    }
    try:
        for record in read_records(input_path):
            query_id = str(record["example_id"])
            split = str(
                connection.execute(
                    "SELECT split FROM query_split WHERE query_id = ?", (query_id,)
                ).fetchone()[0]
            )
            kept_negatives = []
            for negative in record["hard_negatives"]:
                canonical = _canonical(connection, str(negative["doc_id"]))
                positive_row = connection.execute(
                    "SELECT split FROM positive_split WHERE canonical_doc_id = ?", (canonical,)
                ).fetchone()
                cross_split = positive_row is not None and str(positive_row[0]) != split
                if config.remove_cross_split_negatives and cross_split:
                    removed_negatives[split] += 1
                else:
                    kept_negatives.append(negative)
            if len(kept_negatives) < 10:
                below_ten[split] += 1
            output_record = dict(record)
            output_record["hard_negatives"] = kept_negatives
            metadata = dict(record.get("metadata", {}))
            metadata.update({"split": split, "split_version": config.version})
            output_record["metadata"] = metadata
            writers[split].write(output_record)
            output_counts[split] += 1
    finally:
        for writer in writers.values():
            writer.close()

    with JsonParquetWriter(output_dir / "split_assignments.parquet") as writer:
        for query_id, split, component in connection.execute(
            "SELECT query_id, split, component FROM query_meta JOIN query_split USING(query_id) "
            "ORDER BY query_id"
        ):
            writer.write({"query_id": query_id, "split": split, "component_id": component})

    manifest = {
        "version": config.version,
        "seed": config.seed,
        "ratios": config.ratios,
        "input_fingerprint": input_fingerprint.hexdigest(),
        "dedup_fingerprint": dedup_fingerprint.hexdigest(),
        "query_count": query_count,
        "assigned_counts": assigned_counts,
        "output_counts": dict(output_counts),
        "positive_canonical_leakage": positive_leakage,
        "removed_cross_split_negatives": dict(removed_negatives),
        "records_below_ten_negatives_after_cleanup": dict(below_ten),
        "dedup_map": str(dedup_path),
        "assignment_artifact": str(output_dir / "split_assignments.parquet"),
        "code": collect_code_provenance(),
    }
    write_json(manifest_path, manifest)
    connection.close()
    return manifest
