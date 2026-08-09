#!/usr/bin/env python3
"""Prepare an atomic model-free Task 07 launch bundle from frozen evidence."""

import argparse
import json
from pathlib import Path

from validate_dpo_dataset import add_dataset_arguments, dataset_kwargs

from doc2query.training.launch import prepare_task07_launch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_arguments(parser)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--token-length-manifest", type=Path, required=True)
    parser.add_argument("--token-length-records", type=Path, required=True)
    parser.add_argument("--reference-logprob-manifest", type=Path, required=True)
    parser.add_argument("--reference-logprob-records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_task07_launch(
        **dataset_kwargs(args),
        plan_path=args.plan,
        token_length_manifest_path=args.token_length_manifest,
        token_length_records_path=args.token_length_records,
        reference_logprob_manifest_path=args.reference_logprob_manifest,
        reference_logprob_records_path=args.reference_logprob_records,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
