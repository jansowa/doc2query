#!/usr/bin/env python3
"""Zbuduj tentative pary chosen/rejected Task 06 z zamrożonej polityki.

Skrypt jest cienką powłoką nad `doc2query.preferences.pair_policy`: nie ustala
żadnego progu, nie ładuje modelu i nie autoryzuje treningu DPO. Buduje wyłącznie
kohorty wymienione w `authorized_cohorts` zamrożonego configu.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from doc2query.preferences.pair_policy import build_tentative_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort-dir",
        type=Path,
        action="append",
        required=True,
        help="Katalog zamrożonej kohorty same-prompt (można podać wielokrotnie).",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("configs/preferences/task06_tentative_pair_policy_v1.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Nadpisz katalog wyjściowy (domyślnie <kohorta>/tentative_pairs).",
    )
    args = parser.parse_args()
    if args.output_dir is not None and len(args.cohort_dir) != 1:
        parser.error("--output-dir wolno podać tylko dla jednej kohorty")

    manifests = [
        build_tentative_pairs(
            cohort_dir=cohort_dir,
            policy_path=args.policy,
            output_dir=args.output_dir,
        ).model_dump(mode="json")
        for cohort_dir in args.cohort_dir
    ]
    print(json.dumps(manifests, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
