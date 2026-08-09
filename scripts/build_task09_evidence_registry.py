#!/usr/bin/env python3
"""Build a model-free Task 09 evidence registry from explicit frozen manifests."""

import argparse
import json
from pathlib import Path

from doc2query.evaluation.evidence_registry import (
    CampaignEvidenceRequirements,
    build_campaign_evidence_registry,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence-manifest",
        type=Path,
        action="append",
        required=True,
        help="Explicit ExperimentEvidenceManifest path; repeat for every run.",
    )
    parser.add_argument(
        "--requirements",
        type=Path,
        required=True,
        help="Versioned JSON requirements for seeds, metrics and artifact roles.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.requirements.read_text(encoding="utf-8"))
    requirements = CampaignEvidenceRequirements.model_validate(raw)
    bundle = build_campaign_evidence_registry(
        evidence_manifest_paths=args.evidence_manifest,
        requirements=requirements,
        output_dir=args.output_dir,
    )
    print(json.dumps(bundle, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
