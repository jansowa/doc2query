#!/usr/bin/env python
"""Zmierz kryteria akceptacji A1-A5 detekcji zapadnięcia probe w trakcie runu.

Kryteria zamroził ADR
`reports/decisions/task04_m03_in_run_collapse_detection_v1.md` **przed**
uruchomieniem pierwszego runu; ten skrypt wyłącznie je liczy i nie kalibruje
żadnego progu. Zbieżność runów ukończonych ocenia zamrożony guardrail M-03,
zastosowany do nowej serii bez zmiany progów.

Skrypt niczego nie promuje: seria ma jedno ramię (W06), więc nie powstaje żadna
różnica między ramionami. `final_tests_used=[]`.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path
from typing import Any

from doc2query.evaluation.probe_convergence import (
    apply_convergence_guardrail,
    load_guardrail,
    read_probe_run,
)

GUARDRAIL = Path("configs/evaluation/task04_m03_probe_convergence_guardrail_v1.yaml")
FROZEN_REFERENCE = Path("runs/task04_probe_variance_v1/PROBE-VAR-W06-4.5B-S47")
DETECTION_ARTIFACTS = (
    "training_loss_curve.jsonl",
    "training_interim_evaluation.jsonl",
    "collapse_detection_journal.jsonl",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: oczekiwano obiektu JSON")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _run_record(run_dir: Path) -> dict[str, Any]:
    result = _read_json(run_dir / "result.json")
    training = result["training"]
    retrieval = result["corpus_retrieval"]
    detection = result.get("collapse_detection")
    interim = _read_jsonl(run_dir / "training_interim_evaluation.jsonl")
    curve = _read_jsonl(run_dir / "training_loss_curve.jsonl")
    attempts = list(detection["attempts"]) if detection else []
    collapsed = [row for row in attempts if row.get("outcome") == "collapsed"]
    return {
        "run": run_dir.name,
        "detection_enabled": detection is not None,
        "requested_seed": (detection or {}).get("requested_seed", training["recipe"]["seed"]),
        "effective_seed": training["recipe"]["seed"],
        "attempt_count": (detection or {}).get("attempt_count", 1),
        "detection_count": (detection or {}).get("detection_count", 0),
        "collapsed_attempts": [
            {
                "seed": row.get("seed"),
                "rule": row.get("rule"),
                "detected_at_step": row.get("detected_at_step"),
                "train_holdin_recall_at_100": row.get("train_holdin_recall_at_100"),
                "floor": row.get("floor"),
                "seconds": row.get("seconds"),
            }
            for row in collapsed
        ],
        "interim_checks": len(interim),
        "interim_seconds_total": sum(float(row.get("seconds", 0.0)) for row in interim),
        "interim_recall_by_step": {
            str(row["step"]): row["train_holdin_recall_at_100"] for row in interim
        },
        "loss_curve_steps": len(curve),
        "first_loss": training["first_loss"],
        "last_loss": training["last_loss"],
        "training_seconds": training["elapsed_seconds"],
        "index_build_seconds": retrieval["index_build_seconds"],
        "corpus_recall_at_100": retrieval["metrics"]["corpus_recall_at_100"],
        "corpus_ndcg_at_10": retrieval["metrics"]["corpus_ndcg_at_10"],
        "train_summary_keys": sorted(training),
        "result_keys": sorted(result),
        "detection_artifacts_present": sorted(
            name for name in DETECTION_ARTIFACTS if (run_dir / name).exists()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root", type=Path, default=Path("runs/task04_probe_inrun_collapse_v1")
    )
    parser.add_argument("--guardrail", type=Path, default=GUARDRAIL)
    parser.add_argument("--frozen-reference", type=Path, default=FROZEN_REFERENCE)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/measurements/task04/in_run_collapse_detection_v1/summary.json"),
    )
    args = parser.parse_args()

    run_dirs = sorted(path for path in args.run_root.iterdir() if (path / "result.json").is_file())
    if not run_dirs:
        raise SystemExit(f"brak ukończonych runów pod {args.run_root}")
    records = [_run_record(run_dir) for run_dir in run_dirs]

    guardrail = load_guardrail(args.guardrail)
    verdicts = apply_convergence_guardrail(
        [read_probe_run(run_dir, arm="W06") for run_dir in run_dirs], guardrail
    )
    convergence = {verdict.run_id: verdict.model_dump(mode="json") for verdict in verdicts}
    for record in records:
        record["converged"] = convergence[record["run"]]["converged"]
        record["applied_floor"] = convergence[record["run"]]["applied_floor"]

    reference = _read_json(args.frozen_reference / "result.json")
    control = next(
        (record for record in records if not record["detection_enabled"]),
        None,
    )
    enabled = [record for record in records if record["detection_enabled"]]
    healthy = [record for record in enabled if record["requested_seed"] in (47, 48, 51)]
    collapsing = [record for record in enabled if record["requested_seed"] == 50]

    a1: dict[str, Any] = {"status": "not_measured", "reason": "brak runu kontrolnego"}
    if control is not None:
        a1 = {
            "run": control["run"],
            "train_summary_keys_match": control["train_summary_keys"]
            == sorted(reference["training"]),
            "result_keys_match": control["result_keys"] == sorted(reference),
            "no_detection_artifacts": control["detection_artifacts_present"] == [],
        }
        a1["status"] = (
            "pass" if all(value is True for key, value in a1.items() if key != "run") else "fail"
        )

    a2: dict[str, Any] = {"status": "not_measured", "reason": "brak runu seeda 50"}
    if collapsing:
        record = collapsing[0]
        steps = [row["detected_at_step"] for row in record["collapsed_attempts"]]
        a2 = {
            "run": record["run"],
            "detection_count": record["detection_count"],
            "detected_at_steps": steps,
            "detected_by_step_768": bool(steps) and max(steps) <= 768,
            "effective_seed": record["effective_seed"],
            "accepted_run_converged": record["converged"],
            "saved_seconds_estimate": sum(
                float(row["seconds"] or 0.0) for row in record["collapsed_attempts"]
            ),
        }
        a2["status"] = (
            "pass"
            if a2["detection_count"] >= 1
            and a2["detected_by_step_768"]
            and a2["accepted_run_converged"]
            else "fail"
        )

    a3 = {
        "runs": {record["run"]: record["detection_count"] for record in healthy},
        "healthy_seed_count": len(healthy),
        "status": "pass"
        if healthy and all(record["detection_count"] == 0 for record in healthy)
        else "fail"
        if healthy
        else "not_measured",
    }

    a4: dict[str, Any] = {"status": "not_measured", "reason": "brak pary seeda 47"}
    enabled_47 = next((record for record in enabled if record["requested_seed"] == 47), None)
    if control is not None and enabled_47 is not None:
        reference_training = reference["training"]
        reference_metrics = reference["corpus_retrieval"]["metrics"]
        comparisons = {}
        for metric, control_value, enabled_value, frozen_value in (
            (
                "first_loss",
                control["first_loss"],
                enabled_47["first_loss"],
                reference_training["first_loss"],
            ),
            (
                "last_loss",
                control["last_loss"],
                enabled_47["last_loss"],
                reference_training["last_loss"],
            ),
            (
                "corpus_ndcg_at_10",
                control["corpus_ndcg_at_10"],
                enabled_47["corpus_ndcg_at_10"],
                reference_metrics["corpus_ndcg_at_10"],
            ),
        ):
            detection_delta = abs(enabled_value - control_value)
            nondeterminism_delta = abs(control_value - frozen_value)
            comparisons[metric] = {
                "frozen_2026_08_17": frozen_value,
                "control_off": control_value,
                "detection_on": enabled_value,
                "detection_minus_control": detection_delta,
                "control_minus_frozen": nondeterminism_delta,
                "within_nondeterminism": detection_delta <= nondeterminism_delta,
            }
        a4 = {
            "metrics": comparisons,
            "status": "pass"
            if all(value["within_nondeterminism"] for value in comparisons.values())
            else "fail",
        }

    a5 = {
        "runs": {
            record["run"]: {
                "attempt_count": record["attempt_count"],
                "requested_seed": record["requested_seed"],
                "effective_seed": record["effective_seed"],
            }
            for record in enabled
        },
        "status": "pass"
        if enabled
        and all(record["attempt_count"] >= 1 and record["interim_checks"] > 0 for record in enabled)
        else "fail"
        if enabled
        else "not_measured",
    }

    ndcg = [record["corpus_ndcg_at_10"] for record in records if record["converged"]]
    summary = {
        "contract": "task04-m03-in-run-collapse-detection-v1",
        "adr": "reports/decisions/task04_m03_in_run_collapse_detection_v1.md",
        "guardrail": str(args.guardrail),
        "purpose": "validation_of_in_run_collapse_detection_single_arm",
        "arm": "W06",
        "run_count": len(records),
        "runs": records,
        "convergence_verdicts": convergence,
        "corpus_ndcg_at_10_mean_converged": st.mean(ndcg) if ndcg else None,
        "corpus_ndcg_at_10_sd_converged": st.stdev(ndcg) if len(ndcg) > 1 else None,
        "acceptance": {"A1": a1, "A2": a2, "A3": a3, "A4": a4, "A5": a5},
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
    for name, verdict in summary["acceptance"].items():
        print(f"{name}: {verdict['status']}")
    for record in records:
        print(
            f"  {record['run']}: seed {record['requested_seed']}->{record['effective_seed']}, "
            f"prób {record['attempt_count']}, wykryć {record['detection_count']}, "
            f"ndcg@10 {record['corpus_ndcg_at_10']:.6f}, zbieżny {record['converged']}"
        )
    print(f"zapisano {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
