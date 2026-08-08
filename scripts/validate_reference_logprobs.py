#!/usr/bin/env python3
"""Validate precomputed reference logprobs against a frozen DPO plan."""

import argparse
import json
from pathlib import Path

from validate_dpo_dataset import add_dataset_arguments, dataset_kwargs

from doc2query.training.dpo import validate_dpo_dataset, validate_reference_logprobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_arguments(parser)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    dataset = validate_dpo_dataset(**dataset_kwargs(args))
    rows = validate_reference_logprobs(
        records_path=args.records,
        manifest_path=args.manifest,
        plan_path=args.plan,
        dataset=dataset,
    )
    print(
        json.dumps(
            {
                "status": "validated_precomputed_not_computed",
                "record_count": len(rows),
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
