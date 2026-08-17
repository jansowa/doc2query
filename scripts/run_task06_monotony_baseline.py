#!/usr/bin/env python3
"""Zmierz baseline monotonii (oś D) na zamrożonych kohortach same-prompt.

Zadeklarowane wejście projektowe: czyta wygenerowane teksty jawnie, ale nie zamraża
żadnego progu, nie buduje par i niczego nie promuje. Wynik zasila M-05 (tryb
produkcyjny bez selektora) i set-level komponent nagrody GRPO (notatka V2-07).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from doc2query.evaluation.query_monotony import run_monotony_baseline

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
        "--output",
        type=Path,
        default=Path("reports/measurements/task06/monotony_baseline_v1/summary.json"),
    )
    parser.add_argument(
        "--limit-per-cohort",
        type=int,
        help="Tylko do szybkiego przebiegu kontrolnego; artefakt notuje limit.",
    )
    args = parser.parse_args()

    report = run_monotony_baseline(
        cohort_dirs=args.cohort_dir or DEFAULT_COHORTS,
        output_path=args.output,
        limit_per_cohort=args.limit_per_cohort,
    )
    compact = {
        cohort: {
            "pooled_word_length": value["pooled"]["word_length"],
            "pooled_first_word": {
                key: value["pooled"]["first_word"][key]
                for key in ("distinct", "top1_share", "top5_share", "normalized_entropy")
            },
            "set_level_per_group": value["set_level_per_group"],
        }
        for cohort, value in report["cohorts"].items()
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
