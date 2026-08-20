#!/usr/bin/env python3
"""Zamroź ślepy, deterministyczny eksport par v2 do audytu dual-LLM.

Skrypt nie wywołuje żadnego API. Publikuje trzy rozdzielone pliki (ślepe pary,
klucz odślepiający, pełne rekordy próbki) oraz manifest z zobowiązaniem do
orientacji A/B podjętym przed jakąkolwiek oceną. Próbka jest bilansowana kwotą
osi z zamrożonego configu; niedobór kwoty jest raportowany, nigdy nadrabiany
poluzowaniem progu.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from doc2query.preferences.pair_audit_export_v2 import (
    export_blind_defect_audit_sample,
    verify_orientation_commitments,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair-dir",
        type=Path,
        action="append",
        required=True,
        help="Katalog defect_pairs_v2 jednej kohorty (można podać wielokrotnie).",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("configs/preferences/task06_defect_pair_policy_v2.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/task06/preference_audit_v3_defect_pairs"),
    )
    args = parser.parse_args()

    manifest = export_blind_defect_audit_sample(
        pair_dirs=args.pair_dir,
        policy_path=args.policy,
        output_dir=args.output_dir,
    )
    verified = verify_orientation_commitments(args.output_dir)
    print(
        json.dumps(
            {
                "manifest": manifest.model_dump(mode="json"),
                "orientation_commitments_verified": verified,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
