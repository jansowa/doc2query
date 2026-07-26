#!/usr/bin/env python3
"""Generate deterministic S07 queries for the frozen P-05 common cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from doc2query.config import load_config
from doc2query.generation.batching import generate_text_batch
from doc2query.models.load_generator import load_generator, load_tokenizer
from doc2query.models.templates import render_prompt
from doc2query.utils.records import read_durable_jsonl_prefix, read_records, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generator-id", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    config = load_config(args.config)
    rows = list(read_records(args.input))
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        rows = rows[: args.limit]
    expected_ids = [str(row["example_id"]) for row in rows]
    report_path = args.output.with_suffix(".report.json")
    if args.output.is_file():
        existing = list(read_records(args.output))
        if [str(row["example_id"]) for row in existing] != expected_ids:
            raise RuntimeError("completed S07 generation artifact does not match input identity")
        if not report_path.is_file():
            raise RuntimeError("completed S07 generation artifact has no report")
        completed_report = json.loads(report_path.read_text(encoding="utf-8"))
        if completed_report.get("generator_id") != args.generator_id or completed_report.get(
            "examples"
        ) != len(rows):
            raise RuntimeError("completed S07 generation report has incompatible identity")
        print(json.dumps(completed_report, ensure_ascii=False, indent=2, sort_keys=True))
        return
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    completed: list[dict[str, Any]] = read_durable_jsonl_prefix(partial)
    completed_ids = [str(row["example_id"]) for row in completed]
    if completed_ids != expected_ids[: len(completed_ids)]:
        raise RuntimeError("S07 generation journal is not an exact input prefix")
    trajectory = {
        "schema_version": 1,
        "trajectory_version": "s07-probe-batched-v1",
        "batch_size": args.batch_size,
        "expected_ids_sha256": hashlib.sha256("\n".join(expected_ids).encode()).hexdigest(),
    }
    trajectory_path = partial.with_suffix(partial.suffix + ".resume.json")
    if trajectory_path.is_file():
        if json.loads(trajectory_path.read_text(encoding="utf-8")) != trajectory:
            raise RuntimeError("S07 probe generation resume trajectory mismatch")
    elif completed:
        raise RuntimeError("S07 probe generation journal has no trajectory identity")
    else:
        write_json(trajectory_path, trajectory)
    tokenizer = load_tokenizer(config)
    model, precision = load_generator(
        config, for_training=False, model_path=str(args.model_checkpoint)
    )
    model.eval()
    partial.parent.mkdir(parents=True, exist_ok=True)
    first_batch = (len(completed) // args.batch_size) * args.batch_size
    mode = {"do_sample": False, "num_return_sequences": 1}
    with partial.open("a", encoding="utf-8") as handle:
        for batch_start in range(first_batch, len(rows), args.batch_size):
            chunk = rows[batch_start : batch_start + args.batch_size]
            prompts = [
                render_prompt(str(row["passage"]), config.training.baseline) for row in chunk
            ]
            prompt_ids = [
                list(tokenizer.encode(prompt, add_special_tokens=True))[
                    : config.training.max_length
                ]
                for prompt in prompts
            ]
            generated = generate_text_batch(
                model,
                tokenizer,
                prompt_ids,
                mode=mode,
                max_new_tokens=config.generation.max_new_tokens,
            )
            for offset, (row, query) in enumerate(zip(chunk, generated, strict=True)):
                absolute_index = batch_start + offset
                if absolute_index < len(completed):
                    continue
                output = {
                    "candidate_index": 0,
                    "doc_id": str(row["doc_id"]),
                    "example_id": str(row["example_id"]),
                    "generated": query,
                    "generation_batch_size": len(chunk),
                    "generation_trajectory_version": "s07-probe-batched-v1",
                    "generator_id": args.generator_id,
                    "mode": "deterministic",
                    "pair_id": str(row["pair_id"]),
                    "passage": str(row["passage"]),
                    "query_source": "synthetic_s07",
                }
                handle.write(json.dumps(output, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            completed_count = min(batch_start + len(chunk), len(rows))
            if completed_count % args.progress_every < len(chunk) or completed_count == len(rows):
                print(f"[S07 generation] {completed_count}/{len(rows)}", flush=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    os.replace(partial, args.output)
    report = {
        "schema_version": 1,
        "status": "complete",
        "generator_id": args.generator_id,
        "examples": len(rows),
        "precision": precision.label,
        "generation_batch_size": args.batch_size,
        "generation_trajectory_version": "s07-probe-batched-v1",
        "final_tests_used": [],
        "output": str(args.output),
    }
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
