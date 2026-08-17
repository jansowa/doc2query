#!/usr/bin/env python3
"""V2-00: zmierz podaż par defektowych w zamrożonych kohortach same-prompt.

Skrypt czyta pola jakości jawnie (zadeklarowane wejście projektowe polityki v2),
ale nie buduje żadnej pary i nie zamraża żadnego progu — punkty cięcia osi B są
raportowane w kilku kandydujących kwantylach, a wybór jednego z nich należy do
prospektywnego ADR V2-03.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from doc2query.preferences.defect_inventory import run_inventory

DEFAULT_COHORTS = [
    Path(f"artifacts/task06/same_prompt_expansion_v{index}") for index in range(1, 12)
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort-dir",
        type=Path,
        action="append",
        help="Katalog zamrożonej kohorty (domyślnie wszystkie v1-v11).",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("configs/preferences/task06_tentative_pair_policy_v1_1.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/measurements/task06/defect_inventory_v1/summary.json"),
    )
    args = parser.parse_args()

    report = run_inventory(
        cohort_dirs=args.cohort_dir or DEFAULT_COHORTS,
        policy_path=args.policy,
        output_path=args.output,
    )
    compact = {key: report[key] for key in ("overlap_cut_candidates", "pooled")}
    print(json.dumps(compact, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
