#!/usr/bin/env python3
"""Plan or run quota-safe Groq labelling of Task 05 audit samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from doc2query.evaluation.groq_audits import (
    build_request_plan,
    load_api_key,
    load_groq_contract,
    plan_summary,
    run_groq_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/evaluation/task05_groq_llm_audit_v2.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/task05/groq_llm_audit_v2"),
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--max-new-requests-per-model", type=int)
    parser.add_argument(
        "--allow-ambiguous-resend",
        action="store_true",
        help="Jawnie zezwól na ponowienie requestu przerwanego bez zapisanej odpowiedzi.",
    )
    args = parser.parse_args()
    if args.plan_only:
        config = load_groq_contract(args.config)
        result = plan_summary(build_request_plan(config))
    else:
        result = run_groq_audit(
            args.config,
            output_dir=args.output_dir,
            api_key=load_api_key(args.env_file),
            max_new_requests_per_model=args.max_new_requests_per_model,
            allow_ambiguous_resend=args.allow_ambiguous_resend,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
