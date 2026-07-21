#!/usr/bin/env python3
"""Read-only campaign completion auditor."""

from __future__ import annotations

import argparse
from pathlib import Path

from doc2query.evaluation.campaign_audit import audit_campaign, write_campaign_audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--status", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--check-complete", action="store_true")
    args = parser.parse_args()
    report = audit_campaign(args.root, status_path=args.status)
    write_campaign_audit(report, json_path=args.json_output, markdown_path=args.markdown_output)
    counts = report["state_counts"]
    print(
        f"complete={str(bool(report['complete'])).lower()} "
        f"completed={counts['completed']}/{report['expected_arm_count']}"
    )
    return 1 if args.check_complete and not report["complete"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
