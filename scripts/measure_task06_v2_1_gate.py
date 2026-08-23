#!/usr/bin/env python3
"""Policz bramkę V2.1-05 na ukończonym audycie par v2.1.

Skrypt nie wywołuje API, nie zmienia artefaktów audytu i nie zna żadnego progu —
wszystkie czyta z zamrożonej polityki. Odmawia policzenia czegokolwiek, dopóki
audyt nie ma statusu `complete` i dopóki każda para nie ma ocen obu sędziów, żeby
bramki nie dało się podejrzeć po pierwszym oknie dziennego budżetu.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from doc2query.preferences.defect_pair_gate_v2_1 import measure_gate
from doc2query.utils.records import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=Path("artifacts/task06/preference_audit_v4_defect_pairs_v2_1"),
    )
    parser.add_argument("--audit-dir", type=Path, default=None)
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("configs/preferences/task06_defect_pair_policy_v2_1.yaml"),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    audit_dir = args.audit_dir or args.export_dir / "groq_dual_llm"
    result = measure_gate(
        export_dir=args.export_dir, audit_dir=audit_dir, policy_path=args.policy
    )
    output = args.output or audit_dir / "gate_v2_1_05.json"
    write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
