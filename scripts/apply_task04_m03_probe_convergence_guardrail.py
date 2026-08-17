#!/usr/bin/env python3
"""Zastosuj zamrożony guardrail zbieżności probe (M-03) do istniejących runów.

Skrypt nic nie trenuje i niczego nie promuje: czyta zakończone artefakty probe,
oznacza runy niezbieżne według sygnału retrievalowego i liczy statystykę
decyzyjną na sparowanych różnicach per-seed. Wynik jest diagnostyką.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from doc2query.evaluation.probe_convergence import (
    apply_convergence_guardrail,
    convergence_report,
    load_guardrail,
    load_run_group,
    paired_seed_comparison,
)

ARMS = {"HYBRID": "hybrid", "W06": "w06"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-group",
        action="append",
        required=True,
        help="Katalog grupy runów; opcjonalny sufiks '::FRAGMENT' zawęża grupę do "
        "runów o tej nazwie (tak dzieli się sweep budżetu na jedno porównanie na budżet).",
    )
    parser.add_argument(
        "--guardrail",
        type=Path,
        default=Path("configs/evaluation/task04_m03_probe_convergence_guardrail_v1.yaml"),
    )
    parser.add_argument("--variant-arm", default="hybrid")
    parser.add_argument("--anchor-arm", default="w06")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    guardrail = load_guardrail(args.guardrail)
    payload: dict[str, object] = {
        "guardrail_id": guardrail.guardrail_id,
        "guardrail": guardrail.model_dump(mode="json"),
        "groups": [],
        "final_tests_used": [],
    }
    groups: list[dict[str, object]] = []
    for specification in args.run_group:
        raw_root, _, include = str(specification).partition("::")
        root = Path(raw_root)
        runs = load_run_group(root, ARMS, include=include)
        verdicts = apply_convergence_guardrail(runs, guardrail)
        variant = [run for run in runs if run.arm == args.variant_arm]
        anchor = [run for run in runs if run.arm == args.anchor_arm]
        groups.append(
            {
                "run_group": str(specification),
                "convergence": convergence_report(verdicts, group_id=root.name),
                "comparison": paired_seed_comparison(variant, anchor, guardrail),
                "runs": [run.model_dump(mode="json") for run in runs],
            }
        )
    payload["groups"] = groups
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
