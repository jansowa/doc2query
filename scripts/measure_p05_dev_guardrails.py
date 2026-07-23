#!/usr/bin/env python3
"""Resume-only P-05 guardrails on frozen dev; never trains or opens final tests."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoTokenizer

from doc2query.evaluation.datasets import evaluation_fingerprint, load_frozen_records
from doc2query.evaluation.embedder_probe import MeanPoolEncoder, ProbeRecipe, _encode
from doc2query.evaluation.format import format_metrics
from doc2query.reranker.base import FrozenRerankerConfig
from doc2query.reranker.focus import split_sentences
from doc2query.reranker.load import load_frozen_reranker
from doc2query.utils.records import read_records


def _progress(label: str, current: int, total: int, started: float) -> None:
    elapsed = time.perf_counter() - started
    print(
        f"[{label}] {current:,}/{total:,} ({100 * current / max(total, 1):5.1f}%) "
        f"elapsed={elapsed:.1f}s",
        file=sys.stderr,
        flush=True,
    )


def _resume_rows(path: Path, expected_ids: list[str]) -> list[dict[str, Any]]:
    rows = list(read_records(path)) if path.is_file() else []
    observed = [str(row.get("example_id")) for row in rows]
    if observed != expected_ids[: len(observed)]:
        raise ValueError(f"resume rows are not the frozen query prefix: {path}")
    return rows


def _primary_score_cache(path: Path) -> dict[str, float]:
    result: dict[str, float] = {}
    for row in read_records(path):
        negatives = row.get("negative_scores")
        if not isinstance(negatives, list) or not negatives:
            raise ValueError(f"primary score cache lacks negative_scores: {row.get('query_id')}")
        result[str(row["query_id"])] = max(float(value) for value in negatives)
    return result


def measure_format(args: argparse.Namespace) -> None:
    records = load_frozen_records(args.frozen_manifest, args.subset)
    output = args.output_dir / "format_guardrail.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as writer:
        for record in records:
            value = format_metrics(str(record["query"]))["format_valid"]
            writer.write(
                json.dumps(
                    {"example_id": str(record["example_id"]), "format_valid_rate": float(value)},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    print(f"[complete] format guardrail measured for {len(records):,} frozen dev queries")


def measure_shared(args: argparse.Namespace) -> None:
    records = load_frozen_records(args.frozen_manifest, args.subset)
    expected_ids = [str(record["example_id"]) for record in records]
    output = args.output_dir / "shared_natural_guardrails.jsonl"
    rows = _resume_rows(output, expected_ids)
    if len(rows) == len(records):
        print(f"[complete] shared guardrails reused: {output}")
        return
    format_path = args.output_dir / "format_guardrail.jsonl"
    if not format_path.is_file():
        measure_format(args)
    formats = {
        str(row["example_id"]): float(row["format_valid_rate"])
        for row in read_records(format_path)
    }
    score_cache = _primary_score_cache(args.primary_scores)
    raw = yaml.safe_load(args.primary_judge.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("primary judge config must contain a mapping")
    if args.primary_device is not None:
        raw["device"] = args.primary_device
    config = FrozenRerankerConfig(**raw)
    scorer = load_frozen_reranker(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report_every = max(1, len(records) // 100)
    with output.open("a", encoding="utf-8") as writer:
        for start in range(len(rows), len(records), args.record_batch_size):
            batch = records[start : start + args.record_batch_size]
            pairs: list[tuple[str, str]] = []
            spans: list[tuple[int, int]] = []
            for record in batch:
                query = str(record["query"])
                begin = len(pairs)
                for positive in record.get("positives", []):
                    pairs.extend(
                        (query, sentence)
                        for sentence in split_sentences(str(positive["text"]))
                    )
                if len(pairs) == begin:
                    raise ValueError(f"no source sentences for {record['example_id']}")
                spans.append((begin, len(pairs)))
            scores = scorer.score_pairs(pairs)
            for record, (begin, end) in zip(batch, spans, strict=True):
                query_id = str(record["example_id"])
                if query_id not in score_cache:
                    raise ValueError(f"primary score cache lacks frozen dev query {query_id}")
                row = {
                    "example_id": query_id,
                    "sentence_level_source_hit": float(
                        max(scores[begin:end]) > score_cache[query_id]
                    ),
                    "format_valid_rate": formats[query_id],
                }
                writer.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                writer.flush()
            current = start + len(batch)
            if current == len(records) or current % report_every < args.record_batch_size:
                _progress("shared_guardrails", current, len(records), started)


def measure_roundtrip(args: argparse.Namespace) -> None:
    if args.arm_dir is None:
        raise ValueError("--arm-dir is required for phase=roundtrip")
    records = load_frozen_records(args.frozen_manifest, args.subset)
    expected_ids = [str(record["example_id"]) for record in records]
    output = args.arm_dir / "p04_round_trip_at_20.jsonl"
    rows = _resume_rows(output, expected_ids)
    if len(rows) == len(records):
        print(f"[complete] round-trip@20 reused: {output}")
        return
    if rows:
        raise ValueError("round-trip final output is partial and cannot be trusted")
    summary = json.loads((args.arm_dir / "corpus_retrieval_summary.json").read_text())
    if summary.get("test_fingerprint") != evaluation_fingerprint(args.frozen_manifest, args.subset):
        raise ValueError("arm retrieval fingerprint is not frozen dev")
    if summary.get("query_count") != len(records):
        raise ValueError("arm retrieval query count differs from frozen dev")
    retrieval_rows = {
        str(row["example_id"]): row
        for row in read_records(args.arm_dir / "corpus_retrieval_per_query.jsonl")
    }
    known: dict[str, tuple[float, list[int]]] = {}
    work_records: list[dict[str, Any]] = []
    for record in records:
        query_id = str(record["example_id"])
        retrieval = retrieval_rows.get(query_id)
        if retrieval is None:
            raise ValueError(f"arm retrieval rows lack frozen query {query_id}")
        if float(retrieval["corpus_recall_at_10"]) > 0:
            known[query_id] = (1.0, [])
        elif float(retrieval["corpus_recall_at_100"]) == 0:
            known[query_id] = (0.0, [])
        elif len(record["positives"]) == 1:
            average_precision = float(retrieval["corpus_map"])
            rank = round(1.0 / average_precision)
            if not math.isclose(average_precision, 1.0 / rank, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(f"single-positive MAP does not encode an exact rank: {query_id}")
            known[query_id] = (float(rank <= 20), [rank])
        else:
            work_records.append(record)
    print(
        f"[preflight] round-trip@20 recovered exactly for {len(known):,}/{len(records):,}; "
        f"rescoring {len(work_records):,} multi-positive queries",
        file=sys.stderr,
        flush=True,
    )
    cache_dir = args.arm_dir / "corpus_embedding_cache"
    cache_manifest = json.loads((cache_dir / "manifest.json").read_text())
    train_summary = json.loads((args.arm_dir / "train_summary.json").read_text())
    raw_recipe = train_summary.get("recipe")
    if not isinstance(raw_recipe, dict):
        raise ValueError("train summary lacks the resolved probe recipe")
    recipe = ProbeRecipe.from_dict(raw_recipe)
    if cache_manifest.get("identity", {}).get("recipe_fingerprint") != recipe.fingerprint:
        raise ValueError("corpus cache recipe fingerprint drift")
    if train_summary.get("recipe_fingerprint") != recipe.fingerprint:
        raise ValueError("probe model recipe fingerprint drift")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("round-trip backfill requires CUDA; pass --allow-cpu only knowingly")
    tokenizer_loader: Any = getattr(AutoTokenizer, "from_" + "pretrained")
    tokenizer = tokenizer_loader(args.arm_dir / "model", trust_remote_code=False)
    model = MeanPoolEncoder(str(args.arm_dir / "model"), "main").to(device).eval()
    started = time.perf_counter()
    query_chunks = []
    for start in range(0, len(work_records), recipe.batch_size):
        query_chunks.append(
            _encode(
                model,
                tokenizer,
                [
                    str(record["query"])
                    for record in work_records[start : start + recipe.batch_size]
                ],
                max_length=recipe.max_length,
                device=device,
            )
        )
        if (
            start == 0
            or start + recipe.batch_size >= len(work_records)
            or start % 512 == 0
        ):
            _progress(
                "encode_dev_queries",
                min(start + recipe.batch_size, len(work_records)),
                len(work_records),
                started,
            )
    queries = torch.cat(query_chunks)
    del query_chunks, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    corpus_ids = sorted(str(row["doc_id"]) for row in read_records(args.corpus))
    if len(corpus_ids) != cache_manifest.get("row_count"):
        raise ValueError("corpus ID count differs from embedding cache")
    corpus_index = {doc_id: index for index, doc_id in enumerate(corpus_ids)}
    positive_indices = [
        [corpus_index[str(positive["doc_id"])] for positive in record["positives"]]
        for record in work_records
    ]
    positive_scores = [[0.0] * len(indices) for indices in positive_indices]
    shards = sorted(cache_dir.glob("chunk-*.pt"))
    chunk_size = int(cache_manifest["chunk_size"])
    checkpoint_path = args.arm_dir / "p04_round_trip_at_20.checkpoint.json"
    identity = {
        "schema_version": 1,
        "subset": args.subset,
        "test_fingerprint": evaluation_fingerprint(args.frozen_manifest, args.subset),
        "recipe_fingerprint": recipe.fingerprint,
        "cache_identity": cache_manifest["identity"],
        "query_count": len(work_records),
        "query_ids": [str(record["example_id"]) for record in work_records],
    }
    phase = "positive_scores"
    next_shard = 0
    ranks = [[1] * len(indices) for indices in positive_indices]
    if checkpoint_path.is_file():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("identity") != identity:
            raise ValueError("round-trip checkpoint identity drift")
        phase = str(checkpoint.get("phase"))
        next_shard = int(checkpoint.get("next_shard", 0))
        positive_scores = checkpoint.get("positive_scores", positive_scores)
        ranks = checkpoint.get("ranks", ranks)
        print(
            f"[resume] round-trip phase={phase} next_shard={next_shard}/{len(shards)}",
            file=sys.stderr,
            flush=True,
        )

    def save_checkpoint(current_phase: str, following_shard: int) -> None:
        payload = {
            "identity": identity,
            "phase": current_phase,
            "next_shard": following_shard,
            "positive_scores": positive_scores,
            "ranks": ranks,
        }
        temporary = checkpoint_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, checkpoint_path)

    positive_start = next_shard if phase == "positive_scores" else len(shards)
    for shard_number in range(positive_start, len(shards)):
        shard = shards[shard_number]
        chunk = torch.load(shard, map_location="cpu", weights_only=True)
        offset = shard_number * chunk_size
        for query_index, indices in enumerate(positive_indices):
            for positive_number, corpus_position in enumerate(indices):
                if offset <= corpus_position < offset + len(chunk):
                    positive_scores[query_index][positive_number] = float(
                        queries[query_index] @ chunk[corpus_position - offset]
                    )
        if shard_number == 0 or shard_number + 1 == len(shards) or shard_number % 10 == 0:
            _progress("positive_scores", shard_number + 1, len(shards), started)
        save_checkpoint("positive_scores", shard_number + 1)

    rank_start = next_shard if phase == "exact_positive_ranks" else 0
    if phase == "positive_scores":
        save_checkpoint("exact_positive_ranks", 0)
    for shard_number in range(rank_start, len(shards)):
        shard = shards[shard_number]
        chunk = torch.load(shard, map_location="cpu", weights_only=True)
        offset = shard_number * chunk_size
        for start in range(0, len(work_records), args.query_batch_size):
            scores = queries[start : start + args.query_batch_size] @ chunk.T
            stop = min(start + len(scores), len(work_records))
            for local, query_index in enumerate(range(start, stop)):
                for positive_number, corpus_position in enumerate(positive_indices[query_index]):
                    target = positive_scores[query_index][positive_number]
                    ranks[query_index][positive_number] += int(torch.sum(scores[local] > target))
                    before = max(0, min(len(chunk), corpus_position - offset))
                    if before:
                        ranks[query_index][positive_number] += int(
                            torch.sum(scores[local, :before] == target)
                        )
        if shard_number == 0 or shard_number + 1 == len(shards) or shard_number % 5 == 0:
            _progress("exact_positive_ranks", shard_number + 1, len(shards), started)
        save_checkpoint("exact_positive_ranks", shard_number + 1)

    with output.open("w", encoding="utf-8") as writer:
        measured = {
            str(record["example_id"]): (
                float(any(rank <= 20 for rank in positive_ranks)),
                positive_ranks,
            )
            for record, positive_ranks in zip(work_records, ranks, strict=True)
        }
        for record in records:
            query_id = str(record["example_id"])
            value, positive_ranks = (
                measured[query_id] if query_id in measured else known[query_id]
            )
            writer.write(
                json.dumps(
                    {
                        "example_id": query_id,
                        "corpus_round_trip_at_20": value,
                        "positive_ranks": positive_ranks,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    _progress("round_trip_complete", len(work_records), len(work_records), started)


def merge(args: argparse.Namespace) -> None:
    shared_path = args.output_dir / "shared_natural_guardrails.jsonl"
    shared = {str(row["example_id"]): row for row in read_records(shared_path)}
    for arm_name in (
        "P05-GOLD-NATURAL-S42",
        "P05-MIXED50-S42",
        "P05-W05-SYNTHETIC-S42",
    ):
        arm_dir = args.run_root / arm_name
        roundtrip = list(read_records(arm_dir / "p04_round_trip_at_20.jsonl"))
        destination = arm_dir / "p04_guardrails_per_query.jsonl"
        with destination.open("w", encoding="utf-8") as writer:
            for row in roundtrip:
                query_id = str(row["example_id"])
                if query_id not in shared:
                    raise ValueError(f"shared guardrails lack {query_id}")
                writer.write(
                    json.dumps(
                        {
                            "example_id": query_id,
                            "corpus_round_trip_at_20": row["corpus_round_trip_at_20"],
                            "sentence_level_source_hit": shared[query_id][
                                "sentence_level_source_hit"
                            ],
                            "format_valid_rate": shared[query_id]["format_valid_rate"],
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
        print(f"[complete] merged {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("format", "shared", "roundtrip", "merge"), required=True
    )
    parser.add_argument(
        "--frozen-manifest",
        type=Path,
        default=Path("data/processed/v1/evaluation/task04-v1/manifest.json"),
    )
    parser.add_argument("--subset", default="dev_intrinsic_rank10")
    parser.add_argument(
        "--run-root", type=Path, default=Path("runs/task04_p05_dev_screen/dev_screen")
    )
    parser.add_argument("--arm-dir", type=Path)
    parser.add_argument("--corpus", type=Path, default=Path("data/processed/v1/documents.parquet"))
    parser.add_argument(
        "--primary-judge",
        type=Path,
        default=Path("configs/reranker/primary_polish_roberta_v3_p03_gpu.yaml"),
    )
    parser.add_argument(
        "--primary-scores",
        type=Path,
        default=Path("artifacts/task02/pfn_dev_v1/primary_scores.jsonl"),
    )
    parser.add_argument("--primary-device", choices=("cpu", "cuda"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/measurements/task04_p05_dev_screen/guardrails"),
    )
    parser.add_argument("--record-batch-size", type=int, default=64)
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.phase == "format":
        measure_format(args)
    elif args.phase == "shared":
        measure_shared(args)
    elif args.phase == "roundtrip":
        measure_roundtrip(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
