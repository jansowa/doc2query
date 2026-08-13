#!/usr/bin/env python3
"""Validate frozen Task 07 post-run evidence without comparing or selecting arms."""

import argparse
import json
from pathlib import Path

from doc2query.training.comparison_preflight import prepare_task07_comparison_preflight


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--task06-handoff-manifest", type=Path, required=True)
    parser.add_argument("--task06-selection-preflight-manifest", type=Path, required=True)
    parser.add_argument("--task07-launch-manifest", type=Path, required=True)
    parser.add_argument(
        "--outcome-manifest",
        type=Path,
        action="append",
        required=True,
        help="Explicit Task07ArmOutcomeEvidenceManifest; repeat for every arm and seed.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_task07_comparison_preflight(
        protocol_manifest_path=args.protocol_manifest,
        task06_handoff_manifest_path=args.task06_handoff_manifest,
        task06_selection_preflight_manifest_path=args.task06_selection_preflight_manifest,
        task07_launch_manifest_path=args.task07_launch_manifest,
        outcome_manifest_paths=args.outcome_manifest,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
