#!/usr/bin/env python3
"""Package frozen Task 06 train/dev data into the model-free Task 07 contract."""

import argparse
import json
from pathlib import Path

from doc2query.preferences.handoff import package_task07_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preference-train", type=Path, required=True)
    parser.add_argument("--preference-dev", type=Path, required=True)
    parser.add_argument("--continued-sft-train", type=Path, required=True)
    parser.add_argument("--continued-sft-dev", type=Path, required=True)
    parser.add_argument("--weight-manifest", type=Path, required=True)
    parser.add_argument("--weight-records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = package_task07_inputs(
        preference_train_path=args.preference_train,
        preference_dev_path=args.preference_dev,
        continued_sft_train_path=args.continued_sft_train,
        continued_sft_dev_path=args.continued_sft_dev,
        weight_manifest_path=args.weight_manifest,
        weight_records_path=args.weight_records,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
