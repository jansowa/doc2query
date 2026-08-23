#!/usr/bin/env python
"""Zmierz czułość i swoistość detektora zapadnięć wobec zamrożonego guardraila M-03.

Zakres raportowania zamroził amendment
`reports/decisions/task04_m03_in_run_collapse_shadow_mode_amendment_2026-08-22.md`
**przed** uruchomieniem serii: macierz pomyłek per reguła, odsetek zapadnięć z
przedziałem Cloppera-Pearsona, nieselekcjonowany rozkład metryki ramienia oraz
najwcześniejszy krok trafienia. Skrypt nie kalibruje żadnego progu.

Prawdą odniesienia jest werdykt `converged` zamrożonego guardraila M-03 policzony
post hoc na tej serii; predykcją jest to, czy reguła trafiłaby dwa razy z rzędu w
trakcie treningu. Seria ma jedno ramię (W06), więc nic nie może zostać
wypromowane. `final_tests_used=[]`.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from scipy.stats import beta

from doc2query.evaluation.probe_convergence import (
    apply_convergence_guardrail,
    load_guardrail,
    read_probe_run,
)

GUARDRAIL = Path("configs/evaluation/task04_m03_probe_convergence_guardrail_v1.yaml")
RETRIEVAL_RULE = "interim_recall_below_chance_floor"
LOSS_RULE = "loss_direction_non_decreasing"


def clopper_pearson(successes: int, total: int, confidence: float = 0.95) -> dict[str, float]:
    """Dokładny przedział dla proporcji; przy 0 lub n sukcesów granica jest domknięta."""
    alpha = 1.0 - confidence
    low = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, total - successes + 1))
    high = (
        1.0
        if successes == total
        else float(beta.ppf(1 - alpha / 2, successes + 1, total - successes))
    )
    return {"point": successes / total, "ci95_low": low, "ci95_high": high}


def first_consecutive_hit(flags: Sequence[bool], steps: Sequence[int], required: int) -> int | None:
    """Krok, na którym reguła trafiłaby po `required` kolejnych trafieniach."""
    streak = 0
    for flag, step in zip(flags, steps, strict=True):
        streak = streak + 1 if flag else 0
        if streak >= required:
            return step
    return None


def confusion(predicted: Sequence[bool], collapsed: Sequence[bool]) -> dict[str, Any]:
    true_positive = sum(p and c for p, c in zip(predicted, collapsed, strict=True))
    false_positive = sum(p and not c for p, c in zip(predicted, collapsed, strict=True))
    false_negative = sum(not p and c for p, c in zip(predicted, collapsed, strict=True))
    true_negative = sum(not p and not c for p, c in zip(predicted, collapsed, strict=True))
    positives = true_positive + false_negative
    negatives = true_negative + false_positive
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "sensitivity": clopper_pearson(true_positive, positives) if positives else None,
        "false_alarm_rate": clopper_pearson(false_positive, negatives) if negatives else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root", type=Path, default=Path("runs/task04_probe_shadow_collapse_v1")
    )
    parser.add_argument("--guardrail", type=Path, default=GUARDRAIL)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/measurements/task04/shadow_collapse_detector_v1/summary.json"),
    )
    args = parser.parse_args()

    run_dirs = sorted(path for path in args.run_root.iterdir() if (path / "result.json").is_file())
    if not run_dirs:
        raise SystemExit(f"brak ukończonych runów pod {args.run_root}")
    guardrail = load_guardrail(args.guardrail)
    verdicts = {
        verdict.run_id: verdict
        for verdict in apply_convergence_guardrail(
            [read_probe_run(run_dir, arm="W06") for run_dir in run_dirs], guardrail
        )
    }
    required = 2  # z zamrożonego kontraktu detekcji; nie jest tu dobierane

    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        interim = [
            json.loads(line)
            for line in (run_dir / "training_interim_evaluation.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line
        ]
        steps = [int(row["step"]) for row in interim]
        below = [bool(row["below_floor"]) for row in interim]
        rising = [bool(row["loss_non_decreasing"]) for row in interim]
        verdict = verdicts[run_dir.name]
        detection = result.get("collapse_detection") or {}
        rows.append(
            {
                "run": run_dir.name,
                "seed": result["training"]["recipe"]["seed"],
                "mode": detection.get("mode"),
                "corpus_recall_at_100": result["corpus_retrieval"]["metrics"][
                    "corpus_recall_at_100"
                ],
                "corpus_ndcg_at_10": result["corpus_retrieval"]["metrics"]["corpus_ndcg_at_10"],
                "converged": verdict.converged,
                "applied_floor": verdict.applied_floor,
                "interim_steps": steps,
                "interim_recall": [row["train_holdin_recall_at_100"] for row in interim],
                "interim_below_floor": below,
                "interim_loss_non_decreasing": rising,
                "retrieval_rule_hit_step": first_consecutive_hit(below, steps, required),
                "loss_rule_hit_step": first_consecutive_hit(rising, steps, required),
                "single_hit_retrieval_step": first_consecutive_hit(below, steps, 1),
                "interim_seconds_total": sum(float(row.get("seconds", 0.0)) for row in interim),
                "training_seconds": result["training"]["elapsed_seconds"],
                "index_build_seconds": result["corpus_retrieval"]["index_build_seconds"],
            }
        )

    collapsed = [not row["converged"] for row in rows]
    retrieval_predicted = [row["retrieval_rule_hit_step"] is not None for row in rows]
    loss_predicted = [row["loss_rule_hit_step"] is not None for row in rows]
    either_predicted = [a or b for a, b in zip(retrieval_predicted, loss_predicted, strict=True)]
    single_hit_predicted = [row["single_hit_retrieval_step"] is not None for row in rows]

    converged_ndcg = [row["corpus_ndcg_at_10"] for row in rows if row["converged"]]
    mean = st.mean(converged_ndcg)
    sd = st.stdev(converged_ndcg) if len(converged_ndcg) > 1 else float("nan")
    summary = {
        "contract": "task04-m03-in-run-collapse-detection-v1",
        "mode": "shadow_observe_only",
        "adr": ("reports/decisions/task04_m03_in_run_collapse_shadow_mode_amendment_2026-08-22.md"),
        "purpose": "detector_operating_characteristics_single_arm",
        "arm": "W06",
        "run_count": len(rows),
        "runs": rows,
        "collapse_rate": clopper_pearson(sum(collapsed), len(rows)),
        "collapsed_runs": [row["run"] for row, flag in zip(rows, collapsed, strict=True) if flag],
        "confusion_matrix": {
            "retrieval_floor_rule": confusion(retrieval_predicted, collapsed),
            "loss_direction_rule": confusion(loss_predicted, collapsed),
            "either_rule_as_deployed": confusion(either_predicted, collapsed),
            "retrieval_floor_single_hit_counterfactual": confusion(single_hit_predicted, collapsed),
        },
        "unselected_metric_distribution": {
            "converged_run_count": len(converged_ndcg),
            "mean": mean,
            "sd": sd,
            "min": min(converged_ndcg),
            "max": max(converged_ndcg),
            "half_width_95_at_n": {str(n): 1.96 * sd / math.sqrt(n) for n in (3, 5, 10, 20)},
            "superiority_threshold_reference": 0.01,
        },
        "detection_cost_seconds_per_run": st.mean([row["interim_seconds_total"] for row in rows]),
        "thresholds_calibrated_here": False,
        "comparison_performed": False,
        "promotion_decided_here": False,
        "task07_training_authorized": False,
        "final_tests_used": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rate = summary["collapse_rate"]
    print(
        f"runów {len(rows)}, zapadnięć {sum(collapsed)} "
        f"({rate['point']:.3f}, CI [{rate['ci95_low']:.3f}, {rate['ci95_high']:.3f}])"
    )
    for name, matrix in summary["confusion_matrix"].items():
        print(
            f"  {name}: TP {matrix['true_positive']} FP {matrix['false_positive']} "
            f"FN {matrix['false_negative']} TN {matrix['true_negative']}"
        )
    print(f"ndcg@10 zbieżnych: średnia {mean:.6f} sd {sd:.6f}")
    for n, half in summary["unselected_metric_distribution"]["half_width_95_at_n"].items():
        print(f"  półszerokość 95% CI przy n={n}: {half:.5f} (próg odniesienia 0,01)")
    print(f"zapisano {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
