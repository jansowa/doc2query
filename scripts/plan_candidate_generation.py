#!/usr/bin/env python3
"""Materialize a model-free Task 06 candidate-generation request plan."""

import argparse
import json
from pathlib import Path

from doc2query.preferences.planning import write_generation_plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dedup-map", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            write_generation_plan(
                args.source, args.dedup_map, args.config, args.output, args.manifest
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
