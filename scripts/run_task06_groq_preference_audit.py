#!/usr/bin/env python3
"""Zaplanuj, uruchom lub przeanalizuj ślepy audyt dual-LLM par Task 06.

Domyślnie `--plan-only`: skrypt nie wykonuje żadnego wywołania API bez jawnej
komendy operatora. Klucz czytany jest wyłącznie z lokalnego `.env` (pole
`api_key`) i nigdy nie jest logowany.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from doc2query.preferences.groq_pair_audit import (
    analyze_pair_audit,
    load_api_key,
    run_pair_audit,
)
from doc2query.preferences.llm_audit import (
    build_dual_llm_request_plan,
    load_llm_audit_config,
    plan_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/preferences/task06_groq_preference_audit_v1.json"),
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=Path("artifacts/task06/preference_audit_v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/task06/preference_audit_v1/groq_dual_llm"),
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--max-new-requests-per-model", type=int)
    parser.add_argument(
        "--allow-ambiguous-resend",
        action="store_true",
        help="Jawnie zezwól na ponowienie requestu przerwanego bez zapisanej odpowiedzi.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Wymagane, aby wykonać jakiekolwiek wywołanie API.",
    )
    args = parser.parse_args()

    if args.analyze_only:
        result = analyze_pair_audit(export_dir=args.export_dir, output_dir=args.output_dir)
    elif args.plan_only or not args.execute:
        config = load_llm_audit_config(args.config)
        plan = build_dual_llm_request_plan(config, args.export_dir / "blind_pairs.jsonl")
        result = plan_summary(plan) | {"executed": False, "requires": "--execute"}
    else:
        result = run_pair_audit(
            args.config,
            export_dir=args.export_dir,
            output_dir=args.output_dir,
            api_key=load_api_key(args.env_file),
            max_new_requests_per_model=args.max_new_requests_per_model,
            allow_ambiguous_resend=args.allow_ambiguous_resend,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
