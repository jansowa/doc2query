#!/usr/bin/env python
"""Zmierz wariancję między runami i częstość zapadnięć w obrębie JEDNEGO ramienia probe.

Motywacja (metrologia, nie eksperyment): kalibracja M-03 pokazała 4 zapadnięcia
na 22 runy oraz odchylenie par między seedami 0,0126 przy progu decyzyjnym
`+0,01`. Program mierzy więc efekty poniżej rozdzielczości własnego przyrządu.
Ten skrypt liczy własności **przyrządu**, dla identycznego wejścia i
identycznych hiperparametrów, gdzie różni tylko seed. Nie tworzy żadnej różnicy
między ramionami, więc nic nie może zostać wypromowane ani zdegradowane.

Do oceny zbieżności używa **tego samego** zamrożonego guardraila M-03
(`src/doc2query/evaluation/probe_convergence.py`), żeby liczby były porównywalne
z jego kalibracją.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path
from typing import Any

from doc2query.evaluation.probe_convergence import (
    ProbeRunMetrics,
    apply_convergence_guardrail,
    load_guardrail,
    read_probe_run,
)

GUARDRAIL = Path("configs/evaluation/task04_m03_probe_convergence_guardrail_v1.yaml")


def collect(run_root: Path, *, arm: str, pattern: str) -> list[ProbeRunMetrics]:
    runs: list[ProbeRunMetrics] = []
    for run_dir in sorted(run_root.glob(pattern)):
        if not (run_dir / "result.json").is_file():
            continue
        runs.append(read_probe_run(run_dir, arm=arm))
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("runs/task04_probe_variance_v1"))
    parser.add_argument("--pattern", default="PROBE-VAR-W06-*")
    parser.add_argument("--arm", default="W06")
    parser.add_argument("--guardrail", type=Path, default=GUARDRAIL)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/measurements/task04/probe_within_arm_variance_v1.json"),
    )
    args = parser.parse_args()

    guardrail = load_guardrail(args.guardrail)
    runs = collect(args.run_root, arm=args.arm, pattern=args.pattern)
    if len(runs) < 2:
        raise SystemExit(f"potrzebne co najmniej dwa ukończone runy, znaleziono {len(runs)}")

    signal_key = "corpus_recall_at_100"
    metric_key = "corpus_ndcg_at_10"
    verdicts = apply_convergence_guardrail(runs, guardrail)
    by_run = {verdict.run_id: verdict for verdict in verdicts}
    rows = [
        {
            "run": run.run_id,
            "seed": run.seed,
            signal_key: run.metrics[signal_key],
            metric_key: run.metrics[metric_key],
            "first_loss": run.first_loss,
            "last_loss": run.last_loss,
            "converged": by_run[run.run_id].converged,
            "applied_floor": by_run[run.run_id].applied_floor,
            "failure_reason": by_run[run.run_id].failure_reason,
        }
        for run in runs
    ]
    floor = st.median([float(verdict.applied_floor) for verdict in verdicts])
    random_floor = st.median([float(verdict.chance_floor) for verdict in verdicts])
    collapsed = [row["run"] for row in rows if not row["converged"]]
    converged_metrics = [float(row[metric_key]) for row in rows if row["converged"]]
    mean = st.mean(converged_metrics)
    sd = st.stdev(converged_metrics) if len(converged_metrics) > 1 else float("nan")
    results: dict[str, Any] = {
        "purpose": "instrument_metrology_single_arm",
        "guardrail": str(args.guardrail),
        "arm": args.arm,
        "run_count": len(rows),
        "runs": rows,
        "median_floor_applied": floor,
        "random_floor_reference": random_floor,
        "collapsed_runs": collapsed,
        "collapse_rate": len(collapsed) / len(rows),
        "converged_count": len(converged_metrics),
        f"{metric_key}_mean": mean,
        f"{metric_key}_sd": sd,
        f"{metric_key}_cv": sd / mean if mean else None,
        "half_width_95_at_n": {str(n): 1.96 * sd / math.sqrt(n) for n in (3, 5, 10, 20)},
        "promotion_threshold_reference": 0.01,
        "comparison_performed": False,
        "promotion_decided_here": False,
        "thresholds_calibrated_here": False,
        "final_tests_used": [],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"runów: {len(rows)}, zapadniętych: {len(collapsed)} ({results['collapse_rate']:.3f})")
    if collapsed:
        print(f"  zapadnięte: {collapsed}")
    print(f"{metric_key}: średnia={mean:.6f} sd={sd:.6f} cv={results[f'{metric_key}_cv']:.3f}")
    for n, hw in results["half_width_95_at_n"].items():
        print(f"  półszerokość 95% CI przy n={n}: {hw:.5f} (próg odniesienia 0,01)")
    print(f"zapisano {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
