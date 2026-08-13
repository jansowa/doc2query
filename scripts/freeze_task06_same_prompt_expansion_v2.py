#!/usr/bin/env python3
"""Freeze the additional same-prompt Task 06 cohort on CPU, without generating anything."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from doc2query.preferences.same_prompt_cohort import freeze_same_prompt_expansion_cohort


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = freeze_same_prompt_expansion_cohort(args.config, args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
