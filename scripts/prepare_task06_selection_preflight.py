#!/usr/bin/env python3
"""Validate frozen Task 06 selection inputs without scoring, ranking, or pair building."""

import argparse
import json
from pathlib import Path

from doc2query.preferences.selection_preflight import prepare_preference_selection_preflight


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-evidence", type=Path, required=True)
    parser.add_argument("--candidate-evidence-manifest", type=Path, required=True)
    parser.add_argument("--selection-policy-manifest", type=Path, required=True)
    parser.add_argument(
        "--calibration-manifest",
        type=Path,
        action="append",
        required=True,
        help="Externally prepared component calibration manifest; repeat for all 7 components.",
    )
    parser.add_argument("--human-calibration-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_preference_selection_preflight(
        candidate_evidence_path=args.candidate_evidence,
        candidate_evidence_manifest_path=args.candidate_evidence_manifest,
        policy_manifest_path=args.selection_policy_manifest,
        calibration_manifest_paths=args.calibration_manifest,
        human_manifest_path=args.human_calibration_manifest,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
