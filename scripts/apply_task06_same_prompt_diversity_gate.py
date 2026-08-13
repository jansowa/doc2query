#!/usr/bin/env python3
"""Apply the frozen same-prompt diversity gate without scoring, ranking, or pairing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from doc2query.preferences.diversity_gate import apply_same_prompt_diversity_gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--generations-summary", type=Path)
    parser.add_argument("--generations-identity", type=Path)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    generations: Path = args.generations
    summary = args.generations_summary or generations.with_suffix(
        generations.suffix + ".summary.json"
    )
    identity = args.generations_identity or generations.with_suffix(
        generations.suffix + ".identity.json"
    )
    manifest = apply_same_prompt_diversity_gate(
        generations_path=generations,
        generations_summary_path=summary,
        generations_identity_path=identity,
        policy_path=args.policy,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
