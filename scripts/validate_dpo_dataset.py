#!/usr/bin/env python3
"""Validate frozen Task 06 preference and control datasets without loading models."""

import argparse
import json
from pathlib import Path

from doc2query.training.dpo import validate_dpo_dataset


def add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task06-manifest", type=Path, required=True)
    parser.add_argument("--preference-train", type=Path, required=True)
    parser.add_argument("--preference-dev", type=Path, required=True)
    parser.add_argument("--continued-sft-train", type=Path, required=True)
    parser.add_argument("--continued-sft-dev", type=Path, required=True)
    parser.add_argument("--weighted-sft-train", type=Path, required=True)
    parser.add_argument("--weighted-sft-dev", type=Path, required=True)


def dataset_kwargs(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "task06_manifest_path": args.task06_manifest,
        "preference_train_path": args.preference_train,
        "preference_dev_path": args.preference_dev,
        "continued_sft_train_path": args.continued_sft_train,
        "continued_sft_dev_path": args.continued_sft_dev,
        "weighted_sft_train_path": args.weighted_sft_train,
        "weighted_sft_dev_path": args.weighted_sft_dev,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_arguments(parser)
    args = parser.parse_args()
    dataset = validate_dpo_dataset(**dataset_kwargs(args))
    print(
        json.dumps(
            {
                "status": "validated_not_trained",
                "train_preferences": len(dataset.preference_train),
                "dev_preferences": len(dataset.preference_dev),
                "model_loading_performed": False,
                "training_started": False,
                "final_tests_used": [],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
