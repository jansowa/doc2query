#!/usr/bin/env python3
"""Zmierz podaż osi A po certyfikacji odpowiadalności (wejście projektowe dla ADR V2-03).

Łączy zamrożone kohorty z werdyktami przyjętego sędziego i liczy, ile grup ma
jednocześnie certyfikowany `chosen` (czysty wg polityki i werdykt `yes`) oraz defektowy
`rejected` osi A (werdykt `no` albo brak round-tripu @100), przy zachowaniu zamrożonego
ograniczenia różnorodności. Nie buduje par i nie zamraża progów.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from doc2query.preferences.axis_a_supply import run_axis_a_supply

AUTHORIZED = (1, 2, 3)
REST = tuple(range(4, 12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("authorized", "rest", "all"), default="authorized")
    parser.add_argument("--cohort-dir", type=Path, action="append")
    parser.add_argument("--journal", type=Path, action="append", required=True)
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("configs/preferences/task06_tentative_pair_policy_v1_1.yaml"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    indices = {"authorized": AUTHORIZED, "rest": REST, "all": AUTHORIZED + REST}[args.scope]
    cohorts = args.cohort_dir or [
        Path(f"artifacts/task06/same_prompt_expansion_v{index}") for index in indices
    ]
    output = args.output or Path(f"reports/measurements/task06/axis_a_supply_v1/{args.scope}.json")
    report = run_axis_a_supply(
        cohort_dirs=cohorts,
        journal_paths=args.journal,
        policy_path=args.policy,
        output_path=output,
    )
    print(json.dumps(report["pooled"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"\nraport: {output}")


if __name__ == "__main__":
    main()
