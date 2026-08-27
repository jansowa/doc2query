#!/usr/bin/env python3
"""Wytwórz wejścia Task 07 z par v3 i spakuj je zamrożonym packagerem.

Dwa etapy w jednym wywołaniu: budowa sześciu artefaktów (preferencje train/dev,
continued-SFT train/dev, wagi) oraz przepuszczenie ich przez `package_task07_inputs`,
który niczego nie liczy i tylko sprawdza spójność, pokrycie 1:1, kolejność, brak
leakage klastrów i fingerprinty.

Nie ładuje modelu, nie tokenizuje i nie autoryzuje treningu.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from doc2query.preferences.handoff import package_task07_inputs
from doc2query.preferences.handoff_v3 import artifact_paths, build_handoff


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs",
        type=Path,
        default=Path("artifacts/task06/v3_pairs_v1/bottom/pairs.jsonl"),
    )
    parser.add_argument(
        "--handoff-dir", type=Path, default=Path("artifacts/task07/handoff_v3_bottom/inputs")
    )
    parser.add_argument(
        "--packaged-dir", type=Path, default=Path("artifacts/task07/handoff_v3_bottom/packaged")
    )
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--dev-every", type=int, default=10)
    args = parser.parse_args()

    summary = build_handoff(
        pairs_path=args.pairs,
        output_dir=args.handoff_dir,
        seed=args.seed,
        dev_every=args.dev_every,
    )
    manifest = package_task07_inputs(
        **artifact_paths(args.handoff_dir),  # type: ignore[arg-type]
        output_dir=args.packaged_dir,
    )
    print(
        json.dumps(
            {"handoff": summary, "packaged": manifest.model_dump(mode="json")},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
