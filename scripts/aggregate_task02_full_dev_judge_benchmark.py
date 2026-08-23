#!/usr/bin/env python3
"""Zagreguj shardowany benchmark sędziów na frozen dev (Task 02) per pasmo.

Agregacja liczy się z **surowych** rekordów `scores.jsonl` wszystkich shardów
pasma, a nie ze średniej raportów shardowych — inaczej query-macro byłoby średnią
średnich o różnych mianownikach. Używa dokładnie tych samych funkcji
agregujących, co pojedynczy benchmark (`aggregate`, `aggregate_query_macro`,
`disagreement`), więc nie wprowadza własnej definicji metryki.

Nie zmienia żadnego progu, nie dotyka wag sędziów i nie czyta testów finalnych.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from doc2query.reranker.benchmark import aggregate, aggregate_query_macro, disagreement
from doc2query.reranker.infer import GroupScore
from doc2query.utils.records import read_records, write_json

TUPLE_FIELDS = ("document_scores", "negative_doc_ids", "source_en_negative_scores")
DROP_FIELDS = ("protocol", "slices")


def _group_score(row: dict[str, Any]) -> GroupScore:
    payload = {key: value for key, value in row.items() if key not in DROP_FIELDS}
    for field in TUPLE_FIELDS:
        if field in payload and payload[field] is not None:
            payload[field] = tuple(payload[field])
    return GroupScore(**payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("artifacts/task02/full_dev_judge_benchmark_v1"),
    )
    args = parser.parse_args()
    results = args.results_dir / "results"
    if not results.is_dir():
        raise SystemExit(f"brak katalogu wyników: {results}")

    by_band: dict[str, dict[str, list[GroupScore]]] = defaultdict(lambda: defaultdict(list))
    by_band_slices: dict[str, dict[str, dict[str, dict[str, list[GroupScore]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )
    shards_per_band: dict[str, list[str]] = defaultdict(list)
    for shard_dir in sorted(results.iterdir()):
        if not (shard_dir / "benchmark.json").is_file():
            continue
        band = shard_dir.name.split(".")[0]
        shards_per_band[band].append(shard_dir.name)
        for row in read_records(shard_dir / "scores.jsonl"):
            slices = dict(row.get("slices") or {})
            score = _group_score(row)
            by_band[band][score.judge].append(score)
            for dimension, value in slices.items():
                by_band_slices[band][score.judge][dimension][str(value)].append(score)

    summary: dict[str, Any] = {
        "schema_version": 1,
        "contract": "task02-full-dev-judge-benchmark-v1",
        "protocol_requires_hard_negatives": 10,
        "bands": {},
        "final_tests_used": [],
    }
    for band in sorted(by_band):
        judges = sorted(by_band[band])
        payload: dict[str, Any] = {
            "shard_count": len(shards_per_band[band]),
            "shards": sorted(shards_per_band[band]),
            "judges": {
                judge: {
                    "query_macro": aggregate_query_macro(by_band[band][judge]),
                    "pair_micro": aggregate(by_band[band][judge]),
                }
                for judge in judges
            },
            "slices": {
                judge: {
                    dimension: {
                        value: {
                            "query_macro": aggregate_query_macro(rows),
                            "pair_micro": aggregate(rows),
                        }
                        for value, rows in sorted(values.items())
                    }
                    for dimension, values in sorted(by_band_slices[band][judge].items())
                }
                for judge in judges
            },
            "note": "raw logits are never averaged across judges",
        }
        if len(judges) == 2:
            payload["disagreement"] = disagreement(
                by_band[band][judges[0]], by_band[band][judges[1]]
            )
        summary["bands"][band] = payload

    output = args.results_dir / "aggregate.json"
    write_json(output, summary)
    brief = {
        band: {
            "shards": rows["shard_count"],
            "judges": {
                judge: {
                    "query_count": values["query_macro"]["query_count"],
                    "pool_recall_at_1": round(values["query_macro"]["pool_recall_at_1"], 4),
                    "pool_mrr": round(values["query_macro"]["pool_mrr"], 4),
                    "pool_ndcg_at_10": round(values["query_macro"]["pool_ndcg_at_10"], 4),
                    "pool_negative_margin_rate": round(
                        values["query_macro"]["pool_negative_margin_rate"], 4
                    ),
                }
                for judge, values in rows["judges"].items()
            },
            "disagreement": {
                key: (round(value, 4) if isinstance(value, float) else value)
                for key, value in (rows.get("disagreement") or {}).items()
                if key != "disagreed_example_ids"
            },
        }
        for band, rows in summary["bands"].items()
    }
    print(json.dumps({"output": str(output), "summary": brief}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
