#!/usr/bin/env python3
"""Parowane porównanie dwóch runów probe per zapytanie (bez GPU).

Łączy `corpus_retrieval_per_query.jsonl` dwóch runów po `example_id` i liczy:
średnią różnicę metryki, win/tie/loss oraz parowany bootstrap CI (percentyle,
resampling zapytań). Parowanie per zapytanie kasuje błędy wspólne dla obu
ramion na tym samym zapytaniu — w tym „dziury" w etykietach (nieoznaczone
dokumenty relewantne) i wadliwe zapytania testowe — więc jest odporniejsze na
rzadkie etykiety MS MARCO niż porównanie średnich. Czysty postprocessing
istniejących wyników; żadnych nowych obliczeń modelowych.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from doc2query.utils.records import read_records


def _per_query(run_dir: Path, metric: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for row in read_records(run_dir / "corpus_retrieval_per_query.jsonl"):
        example_id = str(row["example_id"])
        if example_id in values:
            raise SystemExit(f"zduplikowany example_id {example_id} w {run_dir}")
        values[example_id] = float(row[metric])
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True, help="katalog runu kandydata")
    parser.add_argument("--reference", type=Path, required=True, help="katalog runu odniesienia")
    parser.add_argument("--metric", default="corpus_recall_at_10")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    candidate = _per_query(args.candidate, args.metric)
    reference = _per_query(args.reference, args.metric)
    if set(candidate) != set(reference):
        raise SystemExit(
            "zbiory zapytań się różnią: porównanie parowane wymaga identycznych example_id"
        )
    ids = sorted(candidate)
    deltas = [candidate[i] - reference[i] for i in ids]
    n = len(deltas)
    mean_delta = sum(deltas) / n
    wins = sum(1 for d in deltas if d > 0)
    losses = sum(1 for d in deltas if d < 0)

    rng = random.Random(args.seed)
    resampled = sorted(
        sum(deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(args.bootstrap_samples)
    )
    low = resampled[int(0.025 * args.bootstrap_samples)]
    high = resampled[int(0.975 * args.bootstrap_samples) - 1]

    result = {
        "schema_version": 1,
        "contract": "task07-probe-paired-comparison-v1",
        "metric": args.metric,
        "candidate": str(args.candidate),
        "reference": str(args.reference),
        "queries": n,
        "candidate_mean": sum(candidate[i] for i in ids) / n,
        "reference_mean": sum(reference[i] for i in ids) / n,
        "mean_delta": mean_delta,
        "wins": wins,
        "losses": losses,
        "ties": n - wins - losses,
        "bootstrap": {
            "samples": args.bootstrap_samples,
            "seed": args.seed,
            "ci95_low": low,
            "ci95_high": high,
            "significant": bool(low > 0.0 or high < 0.0),
        },
        "final_tests_used": [],
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
