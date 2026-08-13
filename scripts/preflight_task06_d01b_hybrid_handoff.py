#!/usr/bin/env python3
"""Verify the owner-approved D01b Hybrid handoff without loading models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from doc2query.preferences.hybrid_handoff import preflight_hybrid_handoff
from doc2query.utils.records import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = preflight_hybrid_handoff(args.config)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
