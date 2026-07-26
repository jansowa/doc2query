#!/usr/bin/env python3
"""Benchmark exact batched retrieval on existing probe corpus shards."""

from __future__ import annotations

import argparse
import bisect
import gc
import json
import time
from pathlib import Path
from typing import Any

import torch

from doc2query.evaluation.corpus import sha256_file
from doc2query.evaluation.datasets import load_frozen_records
from doc2query.evaluation.dense_retrieval import ShardedEmbeddingIndex, exact_retrieval_batch
from doc2query.evaluation.embedder_probe import (
    MeanPoolEncoder,
    _corpus_ids_from_cache_or_source,
    _encode_batched,
    _path_tree_fingerprint,
)
from doc2query.utils.records import write_json


def _positions(corpus_ids: list[str], values: list[str]) -> list[int]:
    result = []
    for doc_id in values:
        position = bisect.bisect_left(corpus_ids, doc_id)
        if position >= len(corpus_ids) or corpus_ids[position] != doc_id:
            raise ValueError(f"benchmark document is absent from the corpus: {doc_id}")
        result.append(position)
    return result


def main() -> None:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--subset", required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--query-count", type=int, default=512)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--parity-queries", type=int, default=8)
    parser.add_argument("--legacy-queries-per-second", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.query_count, args.encode_batch_size, args.parity_queries) < 1 or args.offset < 0:
        raise ValueError("benchmark sizes must be positive and offset non-negative")
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA benchmark requested but CUDA is unavailable")

    manifest = json.loads((args.embedding_cache / "manifest.json").read_text(encoding="utf-8"))
    identity = manifest.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("embedding cache manifest has no identity")
    if identity.get("corpus_sha256") != sha256_file(args.documents):
        raise ValueError("embedding cache corpus fingerprint mismatch")
    if identity.get("model_fingerprint") != _path_tree_fingerprint(args.model_path):
        raise ValueError("embedding cache model fingerprint mismatch")
    row_count, chunk_size = int(manifest["row_count"]), int(manifest["chunk_size"])
    index = ShardedEmbeddingIndex.load(
        args.embedding_cache, row_count=row_count, chunk_size=chunk_size
    )
    catalog_started = time.perf_counter()
    corpus_ids = _corpus_ids_from_cache_or_source(
        args.documents,
        cache_dir=args.embedding_cache,
        expected_count=row_count,
        expected_digest=str(identity["corpus_ids_sha256"]),
    )
    catalog_seconds = time.perf_counter() - catalog_started
    all_records = [
        record
        for record in load_frozen_records(args.frozen_manifest, args.subset)
        if record.get("positives")
    ]
    records = all_records[args.offset : args.offset + args.query_count]
    if len(records) != args.query_count:
        raise ValueError("benchmark slice is shorter than --query-count")

    loader: Any = getattr(AutoTokenizer, "from_" + "pretrained")
    encode_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = loader(args.model_path, trust_remote_code=False)
    model = MeanPoolEncoder(str(args.model_path), "main").to(encode_device).eval()
    encode_started = time.perf_counter()
    queries = _encode_batched(
        model,
        tokenizer,
        [str(record["query"]) for record in records],
        max_length=192,
        batch_size=args.encode_batch_size,
        device=encode_device,
        progress_stage="benchmark_encode_queries",
    )
    encode_seconds = time.perf_counter() - encode_started
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    positives = [
        _positions(corpus_ids, [str(value["doc_id"]) for value in record["positives"]])
        for record in records
    ]
    negatives = [
        _positions(corpus_ids, [str(value["doc_id"]) for value in record.get("hard_negatives", [])])
        for record in records
    ]
    score_device = torch.device(args.device)
    if score_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    retrieval_started = time.perf_counter()
    measured = exact_retrieval_batch(
        queries,
        positive_rows=positives,
        negative_rows=negatives,
        index=index,
        device=score_device,
    )
    if score_device.type == "cuda":
        torch.cuda.synchronize()
    retrieval_seconds = time.perf_counter() - retrieval_started

    parity_count = min(args.parity_queries, len(records))
    parity_started = time.perf_counter()
    cpu_reference = exact_retrieval_batch(
        queries[:parity_count],
        positive_rows=positives[:parity_count],
        negative_rows=negatives[:parity_count],
        index=index,
        device=torch.device("cpu"),
    )
    parity_seconds = time.perf_counter() - parity_started
    rank_match = measured.positive_ranks[:parity_count] == cpu_reference.positive_ranks
    win_match = (
        measured.hard_negative_win_rates[:parity_count]
        == cpu_reference.hard_negative_win_rates
    )
    if not rank_match or not win_match:
        raise RuntimeError("real-artifact CUDA/CPU exact parity audit failed")
    throughput = len(records) / retrieval_seconds
    result = {
        "schema_version": 1,
        "status": "measured",
        "scope": "runtime_only_not_probe_result",
        "backend": "torch_sharded_exact_ip",
        "approximate": False,
        "device": score_device.type,
        "model_path": str(args.model_path),
        "model_fingerprint": str(identity["model_fingerprint"]),
        "corpus_sha256": str(identity["corpus_sha256"]),
        "corpus_rows": row_count,
        "embedding_dimension": index.dimension,
        "embedding_shards": len(index.shards),
        "query_offset": args.offset,
        "query_count": len(records),
        "catalog_seconds": catalog_seconds,
        "query_encoding_seconds": encode_seconds,
        "retrieval_seconds": retrieval_seconds,
        "queries_per_second": throughput,
        "legacy_observed_queries_per_second": args.legacy_queries_per_second,
        "speedup_vs_observed_legacy": (
            throughput / args.legacy_queries_per_second
            if args.legacy_queries_per_second is not None
            else None
        ),
        "cpu_parity_audit": {
            "queries": parity_count,
            "seconds": parity_seconds,
            "rank_match": rank_match,
            "hard_negative_win_match": win_match,
        },
        "peak_vram_reserved_bytes": (
            torch.cuda.max_memory_reserved() if score_device.type == "cuda" else None
        ),
        "final_tests_used": [],
    }
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
