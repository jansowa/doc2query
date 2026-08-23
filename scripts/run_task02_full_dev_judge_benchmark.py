#!/usr/bin/env python3
"""Wznawialny benchmark obu zamrożonych sędziów na CAŁYM frozen dev (Task 02).

Domyka lukę notowaną w rejestrze: primary zmierzył pełny frozen dev, a shadow
tylko 775 wspólnych query bramki HN. Ten runner nie zmienia żadnego progu, nie
dotyka wag sędziów i **nie otwiera testów finalnych** — wejściem jest wyłącznie
`dev.parquet`, a ścieżka testowa jest odrzucana twardo.

Dwie własności, których brakowało, żeby run mógł iść bez nadzoru:

* **wznawialność** — wejście jest cięte na deterministyczne shardy; shard z
  gotowym `benchmark.json` jest pomijany, więc wyłączenie maszyny kosztuje
  najwyżej jeden shard, a dziennik `journal.jsonl` zapisuje każdy ukończony;
* **rozdzielone populacje** — frozen dev ma rekordy z 3 do 10 negatywami, a
  ranking na trzech negatywach nie jest porównywalny z rankingiem na dziesięciu.
  Shardy są więc grupowane w pasma liczby negatywów i nigdy nie mieszają ich w
  jednej liczbie. Agregacja per pasmo jest osobnym krokiem po runie.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from doc2query.reranker.commands import benchmark_main
from doc2query.utils.records import JsonlWriter

BANDS: tuple[tuple[str, int, int], ...] = (
    ("neg10", 10, 10),
    ("neg07_09", 7, 9),
    ("neg03_06", 0, 6),
)


def _band(count: int) -> str:
    for name, low, high in BANDS:
        if low <= count <= high:
            return name
    raise ValueError(f"unmapped negative count: {count}")


def _reject_final_test(path: Path) -> None:
    if "test" in path.name.lower():
        raise ValueError(f"{path}: ten benchmark nie otwiera testów finalnych")


def _materialize_shards(input_path: Path, work_dir: Path, shard_size: int) -> list[Path]:
    """Deterministycznie pokrój dev na shardy w obrębie pasma liczby negatywów."""
    _reject_final_test(input_path)
    by_band: dict[str, list[dict[str, Any]]] = {name: [] for name, _, _ in BANDS}
    frame = pd.read_parquet(input_path)
    for raw in frame["record_json"]:
        record = json.loads(raw)
        by_band[_band(len(record["hard_negatives"]))].append(record)
    shards: list[Path] = []
    work_dir.mkdir(parents=True, exist_ok=True)
    for name, _, _ in BANDS:
        rows = sorted(by_band[name], key=lambda row: str(row["example_id"]))
        for index in range(0, len(rows), shard_size):
            shard = work_dir / f"{name}.{index // shard_size:04d}.jsonl"
            if not shard.exists():
                with JsonlWriter(shard) as writer:
                    for row in rows[index : index + shard_size]:
                        writer.write(row)
            shards.append(shard)
    return shards


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/v1/dev.parquet"))
    parser.add_argument(
        "--judge-config",
        type=Path,
        action="append",
        default=None,
        help="Configi sędziów; domyślnie primary CUDA + shadow GPU bramki HN.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/task02/full_dev_judge_benchmark_v1"),
    )
    parser.add_argument("--shard-size", type=int, default=500)
    parser.add_argument("--max-shards", type=int, help="Limit shardów na jedno uruchomienie.")
    args = parser.parse_args()

    judges = args.judge_config or [
        Path("configs/reranker/primary_polish_roberta_v3_cuda.yaml"),
        Path("configs/reranker/shadow_bge_v2_m3_hn_gate_gpu.yaml"),
    ]
    work_dir = args.output_dir / "shards"
    shards = _materialize_shards(args.input, work_dir, args.shard_size)
    journal = args.output_dir / "journal.jsonl"
    done = 0
    started_run = time.perf_counter()
    for shard in shards:
        target = args.output_dir / "results" / shard.stem
        if (target / "benchmark.json").is_file():
            continue
        if args.max_shards is not None and done >= args.max_shards:
            break
        started = time.perf_counter()
        code = benchmark_main(
            [
                "--input",
                str(shard),
                "--output-dir",
                str(target),
                *[value for path in judges for value in ("--judge-config", str(path))],
            ]
        )
        if code != 0:
            raise SystemExit(f"benchmark shardu {shard.name} zwrócił {code}")
        elapsed = time.perf_counter() - started
        done += 1
        with journal.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "event": "shard_complete",
                        "shard": shard.name,
                        "band": shard.stem.split(".")[0],
                        "elapsed_seconds": round(elapsed, 3),
                        "final_tests_used": [],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            handle.flush()
        print(f"[{done}] {shard.name} gotowy w {elapsed:.1f} s", flush=True)

    remaining = sum(
        1
        for shard in shards
        if not (args.output_dir / "results" / shard.stem / "benchmark.json").is_file()
    )
    print(
        json.dumps(
            {
                "shards_total": len(shards),
                "shards_completed_this_run": done,
                "shards_remaining": remaining,
                "status": "complete" if remaining == 0 else "resumable_incomplete",
                "elapsed_seconds": round(time.perf_counter() - started_run, 1),
                "final_tests_used": [],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
