#!/usr/bin/env python3
"""Freeze the prospective P06-T train-only sample and blind review artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from doc2query.evaluation.translation_audit import freeze_translation_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/v1/train.parquet"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/task03/p06_t/translation_audit_v1")
    )
    parser.add_argument(
        "--registry-manifest",
        type=Path,
        default=Path("configs/evaluation/p06_translation_audit_v1_manifest.json"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stratum-size", type=int, default=75)
    args = parser.parse_args()
    manifest = freeze_translation_audit(
        args.input,
        output_dir=args.output_dir,
        seed=args.seed,
        stratum_size=args.stratum_size,
        registry_manifest_path=args.registry_manifest,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
