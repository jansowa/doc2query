#!/usr/bin/env python3
"""Zamroź label-free pakiet PULI KANDYDATÓW do certyfikacji odpowiadalności.

Prekalkulacja pomiaru per kandydat, nie polityka: certyfikuje pulę, z której polityka v2
będzie mogła dobrać pary, dokładnie tak jak zamrożony scoring policzył `pool_margin` dla
tych samych kandydatów, zanim jakakolwiek polityka par istniała. Nie buduje par, nie
zamraża progów, niczego nie porządkuje.

Domyślnie obejmuje **kohorty autoryzowane** dla par v2 (v1+v2+v3). Kohorty v4-v11 są
osobnym pakietem, bo budowa par z nich nie jest jeszcze autoryzowana.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from doc2query.preferences.answerability_remote import candidate_pool_items, write_packet

AUTHORIZED = [1, 2, 3]
REST = list(range(4, 12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope",
        choices=("authorized", "rest", "all"),
        default="authorized",
        help="authorized = v1+v2+v3 (kohorty autoryzowane dla par v2), rest = v4-v11.",
    )
    parser.add_argument("--cohort-dir", type=Path, action="append")
    parser.add_argument("--packet-dir", type=Path)
    args = parser.parse_args()

    if args.cohort_dir:
        cohorts = args.cohort_dir
    else:
        indices = {"authorized": AUTHORIZED, "rest": REST, "all": AUTHORIZED + REST}[args.scope]
        cohorts = [Path(f"artifacts/task06/same_prompt_expansion_v{index}") for index in indices]
    packet_dir = args.packet_dir or Path(f"artifacts/task06/answerability_pool_{args.scope}_v1")

    items = candidate_pool_items(cohorts)
    manifest = write_packet(items, packet_dir)
    manifest = manifest | {"scope": args.scope, "cohorts": [path.name for path in cohorts]}
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
