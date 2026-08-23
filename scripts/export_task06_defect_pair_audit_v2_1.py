#!/usr/bin/env python3
"""Zamroź ślepy, deterministyczny eksport par v2.1 do audytu dual-LLM.

Skrypt nie wywołuje żadnego API. Publikuje trzy rozdzielone pliki (ślepe pary,
klucz odślepiający, pełne rekordy próbki) oraz manifest z zobowiązaniem do
orientacji A/B podjętym przed jakąkolwiek oceną. Próbka jest stratyfikowana po
kohorcie, jednowartościowej etykiecie defektu i żądanej formie; niedobór jest
raportowany, nigdy nadrabiany poluzowaniem progu. `--exclude-export-dir`
wyklucza pary już użyte w innej komórce (rozłączność komórki kotwic).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from doc2query.preferences.pair_audit_export_v2_1 import (
    export_blind_defect_audit_sample_v2_1,
    verify_orientation_commitments_v2_1,
)
from doc2query.utils.records import read_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair-dir",
        type=Path,
        action="append",
        required=True,
        help="Katalog defect_pairs_v2_1 jednej kohorty (można podać wielokrotnie).",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("configs/preferences/task06_defect_pair_policy_v2_1.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/task06/preference_audit_v4_defect_pairs_v2_1"),
    )
    parser.add_argument(
        "--exclude-export-dir",
        type=Path,
        action="append",
        help="Eksport, którego pary mają zostać wykluczone z tej próbki.",
    )
    args = parser.parse_args()

    excluded: list[str] = []
    for export_dir in args.exclude_export_dir or []:
        excluded.extend(
            str(row["pair_id"]) for row in read_records(export_dir / "machine_key.jsonl")
        )

    manifest = export_blind_defect_audit_sample_v2_1(
        pair_dirs=args.pair_dir,
        policy_path=args.policy,
        output_dir=args.output_dir,
        excluded_pair_ids=excluded,
    )
    verified = verify_orientation_commitments_v2_1(args.output_dir)
    print(
        json.dumps(
            {
                "manifest": manifest.model_dump(mode="json"),
                "orientation_commitments_verified": verified,
                "excluded_pair_ids": len(excluded),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
