#!/usr/bin/env python3
"""Złóż pary v3 z werdyktów turnieju i pełnych rekordów kohort (etap 3).

Nie woła żadnego API. Do zbioru trafiają wyłącznie pary jednomyślne 6/6, a guardy
czystości i formatu są liczone PONOWNIE z pełnych rekordów — flagom z pakietu
turniejowego nie wierzymy na słowo. Oba warianty strony `rejected` (`bottom` i
`near_miss`) są zapisywane osobno i żaden nie jest promowany bez pomiaru.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from doc2query.preferences.pair_assembly_v3 import assemble_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-dir", type=Path, action="append", required=True)
    parser.add_argument(
        "--tournament-dir", type=Path, default=Path("artifacts/task06/v3_tournament_v1")
    )
    parser.add_argument(
        "--bundle-dir", type=Path, default=Path("artifacts/task06/v3_tournament_bundle_v1")
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("configs/preferences/task06_defect_pair_policy_v2_1.yaml"),
        help="Config, z którego brane są wyłącznie guardy czystości i formatu.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/task06/v3_pairs_v1"))
    parser.add_argument(
        "--variant", choices=("bottom", "near_miss", "both"), default="both"
    )
    args = parser.parse_args()

    variants = ("bottom", "near_miss") if args.variant == "both" else (args.variant,)
    manifests = {}
    for variant in variants:
        manifest = assemble_pairs(
            cohort_dirs=args.cohort_dir,
            tournament_dir=args.tournament_dir,
            bundle_dir=args.bundle_dir,
            policy_path=args.policy,
            output_dir=args.output_root / variant,
            variant=variant,  # type: ignore[arg-type]
        )
        manifests[variant] = manifest.model_dump(mode="json")
    print(json.dumps(manifests, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
