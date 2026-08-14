#!/usr/bin/env python
"""Wytnij z zamrożonej kohorty Task 06 chudy plik pasaży do generacji tokenowej.

Skrypt jest read-only wobec kohorty i nie liczy żadnych sygnałów jakości. Celowo
**nie** przepisuje naturalnego `query` ani hard negatywów: agent generujący
kandydatów widzi dokładnie tyle, ile widzi lokalny generator w kontrakcie
`task06-candidate-execution-design-v1` (sam pasaż), więc nie ma ścieżki wycieku
gold query do wyjścia teachera.

Kolejność rekordów jest deterministyczna: sha256 po `cluster_id`, tak jak
quality-blind selection w kohortach same-prompt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from doc2query.data.focus_labels import focus_bucket, split_sentences

SLIM_CONTRACT = "task06-slim-passages-v1"


def _cluster_rank(cluster_id: str) -> str:
    return hashlib.sha256(cluster_id.encode("utf-8")).hexdigest()


def _positive_text(record: dict[str, Any]) -> str:
    positives = record.get("positives") or []
    if not positives:
        raise ValueError(f"record {record.get('example_id')!r} has no positives")
    text = positives[0].get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"record {record.get('example_id')!r} has an empty positive text")
    return text.strip()


def _positive_doc_id(record: dict[str, Any]) -> str | None:
    positives = record.get("positives") or []
    if not positives:
        return None
    doc_id = positives[0].get("doc_id")
    return doc_id if isinstance(doc_id, str) else None


def slim_records(records_path: Path, *, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with records_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            cluster_id = str(record["cluster_id"])
            passage = _positive_text(record)
            sentences = split_sentences(passage)
            sentence_count = len(sentences)
            rows.append(
                {
                    "cluster_id": cluster_id,
                    "example_id": str(record["example_id"]),
                    "split": str(record.get("split", "")),
                    "positive_doc_id": _positive_doc_id(record),
                    "passage": passage,
                    "passage_chars": len(passage),
                    "sentence_count": sentence_count,
                    "sentences": sentences,
                    "focus_buckets": [
                        focus_bucket(index, sentence_count) for index in range(sentence_count)
                    ],
                    "cluster_rank": _cluster_rank(cluster_id),
                }
            )

    rows.sort(key=lambda row: (row["cluster_rank"], row["cluster_id"]))
    for position, row in enumerate(rows):
        row["order_index"] = position
    if limit is not None:
        rows = rows[:limit]
    return rows


def write_shard_plan(rows: list[dict[str, Any]], *, shard_size: int) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for start in range(0, len(rows), shard_size):
        chunk = rows[start : start + shard_size]
        plan.append(
            {
                "shard_id": f"shard_{start // shard_size:03d}",
                "line_offset": start,
                "line_count": len(chunk),
                "first_cluster_id": chunk[0]["cluster_id"],
                "last_cluster_id": chunk[-1]["cluster_id"],
                "cluster_ids": [row["cluster_id"] for row in chunk],
            }
        )
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard-size", type=int, default=30)
    args = parser.parse_args()

    rows = slim_records(args.records, limit=args.limit)
    if not rows:
        raise SystemExit("no records were slimmed")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    args.output.write_text(payload, encoding="utf-8")

    plan = write_shard_plan(rows, shard_size=args.shard_size)
    manifest = {
        "contract": SLIM_CONTRACT,
        "source_records": str(args.records),
        "source_records_sha256": hashlib.sha256(args.records.read_bytes()).hexdigest(),
        "output": str(args.output),
        "output_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "record_count": len(rows),
        "order": "sha256_cluster_id_ascending",
        "natural_query_included": False,
        "hard_negatives_included": False,
        "quality_fields_included": [],
        "shard_size": args.shard_size,
        "shards": plan,
        "final_tests_used": [],
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"slimmed {len(rows)} passages -> {args.output}")
    print(f"shards: {len(plan)} x {args.shard_size} -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
