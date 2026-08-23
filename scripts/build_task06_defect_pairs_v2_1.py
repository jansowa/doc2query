#!/usr/bin/env python3
"""Zbuduj jednoosiowe pary defektowe (polityka v2.1) z zamrożonych kohort.

Skrypt jest cienką powłoką nad `doc2query.preferences.pair_policy_v2_1`: nie
ustala żadnego progu, nie ładuje modelu, nie generuje werdyktów (czyta je z
przypiętych journali sędziego) i nie autoryzuje treningu DPO. Buduje wyłącznie
kohorty wymienione w `authorized_cohorts` zamrożonego configu. Zamknięte
artefakty polityki v2.0 pozostają nietknięte — wyjście idzie do osobnych
katalogów `defect_pairs_v2_1`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from doc2query.preferences.pair_policy_v2_1 import build_defect_pairs_v2_1

DEFAULT_JOURNAL = Path(
    "artifacts/task06/answerability_verdicts/verdicts_pool_authorized_single.jsonl"
)


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
        "--journal",
        type=Path,
        action="append",
        help="Journal werdyktów sędziego odpowiadalności (musi być przypięty w polityce).",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("configs/preferences/task06_defect_pair_policy_v2_1.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Nadpisz katalog wyjściowy (domyślnie <kohorta>/defect_pairs_v2_1).",
    )
    args = parser.parse_args()
    if args.output_dir is not None and len(args.cohort_dir) != 1:
        parser.error("--output-dir wolno podać tylko dla jednej kohorty")

    journals = args.journal or [DEFAULT_JOURNAL]
    manifests = [
        build_defect_pairs_v2_1(
            cohort_dir=cohort_dir,
            policy_path=args.policy,
            journal_paths=journals,
            output_dir=args.output_dir,
        ).model_dump(mode="json")
        for cohort_dir in args.cohort_dir
    ]
    print(json.dumps(manifests, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
