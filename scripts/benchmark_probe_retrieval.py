#!/usr/bin/env python3
"""Lightweight correctness and throughput benchmark for probe retrieval."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as functional

from doc2query.evaluation.dense_retrieval import ShardedEmbeddingIndex, exact_retrieval_batch
from doc2query.utils.records import write_json


def _reference_ranks(
    queries: torch.Tensor, corpus: torch.Tensor, positives: list[list[int]]
) -> list[list[int]]:
    result = []
    for query, rows in zip(queries, positives, strict=True):
        scores = query @ corpus.T
        result.append(
            [
                1
                + int((scores > scores[row]).sum().item())
                + int((scores[:row] == scores[row]).sum().item())
                for row in rows
            ]
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-size", type=int, default=20_000)
    parser.add_argument("--query-count", type=int, default=256)
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--shard-size", type=int, default=5_000)
    parser.add_argument("--query-batch-sizes", type=int, nargs="+", default=[8, 32, 128])
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(
        args.corpus_size,
        args.query_count,
        args.dimension,
        args.shard_size,
        args.threads,
        *args.query_batch_sizes,
    ) < 1:
        raise ValueError("all benchmark sizes must be positive")
    torch.set_num_threads(args.threads)
    generator = torch.Generator().manual_seed(args.seed)
    corpus = functional.normalize(
        torch.randn(args.corpus_size, args.dimension, generator=generator), dim=1
    )
    queries = functional.normalize(
        torch.randn(args.query_count, args.dimension, generator=generator), dim=1
    )
    positives = [[int((index * 7919) % args.corpus_size)] for index in range(args.query_count)]
    negatives = [
        [int((index * 7907 + offset + 1) % args.corpus_size) for offset in range(10)]
        for index in range(args.query_count)
    ]
    with tempfile.TemporaryDirectory(prefix="doc2query-retrieval-") as temporary:
        cache = Path(temporary)
        for shard_index, start in enumerate(range(0, args.corpus_size, args.shard_size)):
            torch.save(
                corpus[start : min(start + args.shard_size, args.corpus_size)],
                cache / f"chunk-{shard_index:05d}.pt",
            )
        index = ShardedEmbeddingIndex.load(
            cache, row_count=args.corpus_size, chunk_size=args.shard_size
        )
        started = time.perf_counter()
        reference = _reference_ranks(queries, corpus, positives)
        reference_seconds = time.perf_counter() - started
        measurements = []
        for batch_size in args.query_batch_sizes:
            started = time.perf_counter()
            observed: list[list[int]] = []
            for batch_start in range(0, args.query_count, batch_size):
                batch_end = min(batch_start + batch_size, args.query_count)
                retrieval_result = exact_retrieval_batch(
                    queries[batch_start:batch_end],
                    positive_rows=positives[batch_start:batch_end],
                    negative_rows=negatives[batch_start:batch_end],
                    index=index,
                    device=torch.device("cpu"),
                )
                observed.extend(retrieval_result.positive_ranks)
            elapsed = time.perf_counter() - started
            if observed != reference:
                raise RuntimeError("batched exact ranks differ from the legacy reference")
            measurement = {
                "query_batch_size": batch_size,
                "elapsed_seconds": elapsed,
                "queries_per_second": args.query_count / elapsed,
                "speedup_vs_legacy": reference_seconds / elapsed,
                "exact_rank_match": True,
            }
            measurements.append(measurement)
            print(json.dumps(measurement, sort_keys=True), flush=True)
    result = {
        "schema_version": 1,
        "backend": "torch_sharded_exact_ip",
        "approximate": False,
        "corpus_size": args.corpus_size,
        "query_count": args.query_count,
        "dimension": args.dimension,
        "shard_size": args.shard_size,
        "threads": args.threads,
        "seed": args.seed,
        "legacy": {
            "method": "one_full_in_memory_scan_per_query",
            "elapsed_seconds": reference_seconds,
            "queries_per_second": args.query_count / reference_seconds,
        },
        "measurements": measurements,
    }
    write_json(args.output, result)


if __name__ == "__main__":
    main()
