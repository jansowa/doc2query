#!/usr/bin/env python3
"""Plan matched-budget DPO controls from precomputed token-length evidence."""

import argparse
import json
from pathlib import Path

from validate_dpo_dataset import add_dataset_arguments, dataset_kwargs

from doc2query.training.dpo import plan_dpo_controls, validate_dpo_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_arguments(parser)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--token-length-manifest", type=Path, required=True)
    parser.add_argument("--token-length-records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = validate_dpo_dataset(**dataset_kwargs(args))
    manifest = plan_dpo_controls(
        config_path=args.config,
        token_length_manifest_path=args.token_length_manifest,
        token_length_records_path=args.token_length_records,
        output_path=args.output,
        dataset=dataset,
    )
    print(json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
