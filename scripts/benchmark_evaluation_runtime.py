#!/usr/bin/env python3
"""Benchmark frozen BM25 or the complete intrinsic scoring runtime on a real prefix."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from itertools import islice
from pathlib import Path
from typing import Any

import torch
import yaml

from doc2query.evaluation.corpus import BM25CorpusIndex, evaluate_round_trip_queries
from doc2query.evaluation.intrinsic import evaluate_intrinsic_records
from doc2query.reranker.base import FrozenRerankerConfig, PairScorer
from doc2query.reranker.load import load_frozen_reranker
from doc2query.utils.records import read_records, write_json


def _judge(path: Path, device: str) -> PairScorer:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"judge config must be a mapping: {path}")
    raw["device"] = device
    return load_frozen_reranker(FrozenRerankerConfig(**raw))


def _fingerprint(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--corpus-index", type=Path, required=True)
    parser.add_argument("--component", choices=("bm25", "full"), default="full")
    parser.add_argument("--primary-judge", type=Path)
    parser.add_argument("--shadow-judge", type=Path)
    parser.add_argument("--primary-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--shadow-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=64)
    parser.add_argument("--scoring-batch-size", type=int, default=16)
    parser.add_argument("--bm25-workers", type=int, nargs="+", default=[1, 2, 4, 6])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.offset < 0 or args.max_examples < 1 or args.scoring_batch_size < 1:
        raise ValueError("offset must be non-negative and sizes must be positive")
    if min(args.bm25_workers) < 1:
        raise ValueError("BM25 worker counts must be positive")
    rows = list(
        islice(read_records(args.generations), args.offset, args.offset + args.max_examples)
    )
    if not rows:
        raise ValueError("benchmark selected zero generation records")
    if args.output.exists():
        raise FileExistsError(args.output)
    artifact_root = args.output.with_suffix("")
    if args.component == "full" and artifact_root.exists():
        raise FileExistsError(artifact_root)
    primary = shadow = None
    if args.component == "full":
        if args.primary_judge is None:
            raise ValueError("full benchmark requires --primary-judge")
        primary = _judge(args.primary_judge, args.primary_device)
        shadow = _judge(args.shadow_judge, args.shadow_device) if args.shadow_judge else None
    sample_fingerprint = _fingerprint(rows)
    measurements = []
    with BM25CorpusIndex(args.corpus_index) as index:
        requests = [
            (str(row.get("generated", "")), (str(row["positive"]["doc_id"]),))
            for row in rows
        ]
        for workers in args.bm25_workers:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            if args.component == "bm25":
                results = evaluate_round_trip_queries(index, requests, workers=workers)
                checksum = sum(float(row["corpus_best_positive_score"]) for row in results)
            else:
                assert primary is not None
                destination = artifact_root / f"workers-{workers}"
                summary = evaluate_intrinsic_records(
                    rows,
                    primary=primary,
                    shadow=shadow,
                    output_dir=destination,
                    test_fingerprint=sample_fingerprint,
                    experiment_id=f"runtime-benchmark-workers-{workers}",
                    corpus_index=index,
                    scoring_batch_size=args.scoring_batch_size,
                    bm25_workers=workers,
                    progress_every=len(rows) + 1,
                )
                checksum = float(summary["reranker_margin"]["mean"])
            elapsed = time.perf_counter() - started
            measurement = {
                "bm25_workers": workers,
                "elapsed_seconds": elapsed,
                "records_per_second": len(rows) / elapsed,
                "checksum": checksum,
                "peak_vram_reserved_bytes": (
                    torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None
                ),
            }
            measurements.append(measurement)
            print(json.dumps(measurement, sort_keys=True), flush=True)
    result = {
        "schema_version": 1,
        "component": args.component,
        "generation_path": str(args.generations),
        "sample_offset": args.offset,
        "sample_count": len(rows),
        "sample_fingerprint": sample_fingerprint,
        "scoring_batch_size": args.scoring_batch_size,
        "primary_device": args.primary_device if primary is not None else None,
        "shadow_device": args.shadow_device if shadow is not None else None,
        "measurements": measurements,
    }
    write_json(args.output, result)


if __name__ == "__main__":
    main()
