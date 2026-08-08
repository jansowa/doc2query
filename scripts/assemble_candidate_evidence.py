#!/usr/bin/env python3
"""Assemble frozen Task 06 evidence without model scoring, calibration or ranking."""

import argparse
import json
from pathlib import Path

from doc2query.preferences.evidence import assemble_candidate_evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--shadow", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--lexical", type=Path, required=True)
    parser.add_argument("--focus", type=Path, required=True)
    parser.add_argument("--style", type=Path, required=True)
    parser.add_argument("--format-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--primary-judge-id", required=True)
    parser.add_argument("--primary-judge-revision", required=True)
    parser.add_argument("--shadow-judge-id", required=True)
    parser.add_argument("--shadow-judge-revision", required=True)
    parser.add_argument("--margin-tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    manifest = assemble_candidate_evidence(
        requests_path=args.requests,
        candidates_path=args.candidates,
        primary_path=args.primary,
        shadow_path=args.shadow,
        corpus_path=args.corpus,
        lexical_path=args.lexical,
        focus_path=args.focus,
        style_path=args.style,
        format_path=args.format_evidence,
        output_path=args.output,
        manifest_path=args.manifest,
        primary_judge_id=args.primary_judge_id,
        primary_judge_revision=args.primary_judge_revision,
        shadow_judge_id=args.shadow_judge_id,
        shadow_judge_revision=args.shadow_judge_revision,
        margin_tolerance=args.margin_tolerance,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
