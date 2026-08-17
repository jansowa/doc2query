#!/usr/bin/env python3
"""Skalibruj proxy odpowiadalności v1 na etykietach konsensusu sędziów Groq.

Protokół, przestrzeń reguł i kryteria akceptacji zamraża
`reports/decisions/task06_answerability_proxy_v1.md` **przed** tym uruchomieniem.
Skrypt niczego nie dostraja poza połową fit i odczytuje holdout dokładnie raz.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from doc2query.preferences.answerability_proxy import calibrate_answerability_proxy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=Path("artifacts/task06/preference_audit_v2"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/task06/answerability_proxy_v1"),
    )
    args = parser.parse_args()
    result = calibrate_answerability_proxy(export_dir=args.export_dir, output_dir=args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
