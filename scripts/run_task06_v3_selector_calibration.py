#!/usr/bin/env python3
"""Kalibracja selektora v3 na klasach o etykietach znanych z konstrukcji.

Wymagane przez §6 ADR `task06_judge_selected_pair_policy_v3.md` PRZED zamrożeniem
progu agregacji i przed zbudowaniem pierwszej pary v3. Skrypt nie buduje par, nie
ustala progu i nie autoryzuje treningu — zbiera werdykty i liczy czystość, obciążenie
pozycyjne oraz krzywą czystość/wydajność.

Adres serwera i klucz API są parametrami wywołania; nie ma ich w repozytorium.
Domyślnie `--plan-only`: bez `--execute` nie leci ani jedno wywołanie API.

Przykład:
  uv run python scripts/run_task06_v3_selector_calibration.py \
    --base-url https://twoj-serwer/v1 --api-key TWOJ_KLUCZ \
    --model qwen3.8-27b-fp8 --execute
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from doc2query.preferences.pair_selector_v3 import (
    RUBRICS,
    analyze_calibration,
    calibration_items,
    endpoint_from_args,
    plan_summary,
    run_pairwise,
    write_report,
)

CORPUS = Path("artifacts/task06/reward_validation_corpus_v1/corpus.jsonl")
PASSAGES = Path("artifacts/task06/reward_validation_corpus_v1/passages.slim.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="np. https://host/v1")
    parser.add_argument("--api-key", help="Klucz API; alternatywnie zmienna środowiskowa.")
    parser.add_argument("--api-key-env", default="QWEN_API_KEY")
    parser.add_argument(
        "--allow-no-auth",
        action="store_true",
        help="Serwer bez uwierzytelniania (typowe dla lokalnego vLLM).",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--passages", type=Path, default=PASSAGES)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/task06/v3_selector_calibration_v1"),
    )
    parser.add_argument("--limit", type=int, help="Ogranicz liczbę par (smoke).")
    parser.add_argument("--max-completion-tokens", type=int, default=512)
    parser.add_argument(
        "--reasoning",
        action="store_true",
        help="Pozwól sędziemu myśleć przed werdyktem (osobne ramię, osobny journal).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Ile requestów jednocześnie; vLLM robi continuous batching (domyślnie 8).",
    )
    parser.add_argument("--execute", action="store_true", help="Wymagane, by wołać API.")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()

    items = calibration_items(args.corpus, args.passages)
    if args.limit is not None:
        items = items[: args.limit]
    arm = "reasoning" if args.reasoning else "direct"
    output_dir = args.output_dir / arm
    journal_path = output_dir / "judgments.journal.jsonl"

    if args.analyze_only:
        report = analyze_calibration(journal_path)
        write_report(output_dir / "calibration_report.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return

    plan = plan_summary(items, list(RUBRICS)) | {
        "arm": arm,
        "journal": str(journal_path),
        "concurrency": args.concurrency,
    }
    if not args.execute:
        print(json.dumps(plan | {"executed": False, "requires": "--execute"},
                         ensure_ascii=False, indent=2, sort_keys=True))
        return

    endpoint = endpoint_from_args(
        base_url=args.base_url,
        api_key=args.api_key,
        api_key_env=args.api_key_env,
        model=args.model,
        allow_reasoning=args.reasoning,
        max_completion_tokens=args.max_completion_tokens,
        allow_no_auth=args.allow_no_auth,
    )
    print(json.dumps(plan, ensure_ascii=False, sort_keys=True), flush=True)
    summary = run_pairwise(
        items=items,
        endpoint=endpoint,
        journal_path=journal_path,
        concurrency=args.concurrency,
    )
    write_report(output_dir / "run_summary.json", summary)
    report = analyze_calibration(journal_path)
    write_report(output_dir / "calibration_report.json", report)
    print(json.dumps({"run": summary, "calibration": report},
                     ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
