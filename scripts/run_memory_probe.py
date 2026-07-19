#!/usr/bin/env python3
"""Run isolated short training processes and report measured CUDA peaks."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from doc2query.config import load_config
from doc2query.utils.records import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[512, 768, 1024])
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/memory_probes"))
    args = parser.parse_args()
    config = load_config(args.config)
    root = args.output_dir / config.run.experiment_id
    reports: list[dict[str, Any]] = []
    for length in args.lengths:
        run_dir = root / f"length-{length}"
        summary_path = run_dir / "sft_summary.json"
        if summary_path.is_file() and (run_dir / "adapter").is_dir():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            reports.append(
                {
                    "max_length": length,
                    "status": "already_complete",
                    "returncode": 0,
                    "run_dir": str(run_dir),
                    "peak_vram_allocated_bytes": summary["peak_vram_allocated_bytes"],
                    "peak_vram_reserved_bytes": summary["peak_vram_reserved_bytes"],
                    "throughput_examples_per_second": summary["throughput_examples_per_second"],
                }
            )
            continue
        command = [
            sys.executable,
            str(Path(__file__).with_name("train_sft.py")),
            "--config",
            str(args.config),
            "--max-steps",
            str(args.steps),
            "--max-length",
            str(length),
            "--output-dir",
            str(run_dir),
            "--no-panel",
            "--resume-if-available",
        ]
        completed = subprocess.run(command, check=False, text=True, capture_output=True)
        entry: dict[str, Any] = {
            "max_length": length,
            "status": "ok" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "run_dir": str(run_dir),
        }
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            entry.update(
                {
                    "peak_vram_allocated_bytes": summary["peak_vram_allocated_bytes"],
                    "peak_vram_reserved_bytes": summary["peak_vram_reserved_bytes"],
                    "throughput_examples_per_second": summary["throughput_examples_per_second"],
                }
            )
        else:
            entry["stderr_tail"] = completed.stderr[-4000:]
        reports.append(entry)
    output = root / "memory_probe.json"
    write_json(output, {"experiment_id": config.run.experiment_id, "probes": reports})
    print(output)


if __name__ == "__main__":
    main()
