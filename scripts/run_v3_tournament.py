#!/usr/bin/env python3
"""Rozegraj turniej selektora v3 na maszynie z serwerem sędziego.

Czyta chudy pakiet turniejowy, rozgrywa w każdej grupie pojedynczą eliminację
(najlepszy, najgorszy, drugi od końca) i potwierdza obie finałowe pary pełnym
ensemble sześciu głosów. Journal jest cache'em porównań kluczowanym parą kandydatów,
więc wznowienie nie powtarza ani jednego wywołania.

Nie buduje par: składanie z pełną proweniencją odbywa się na maszynie z artefaktami.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from doc2query.preferences.pair_policy_v3 import load_bundle, run_tournaments
from doc2query.preferences.pair_selector_v3 import (
    JudgeApiError,
    endpoint_from_args,
    probe_endpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key")
    parser.add_argument("--api-key-env", default="QWEN_API_KEY")
    parser.add_argument("--allow-no-auth", action="store_true")
    parser.add_argument(
        "--bundle-dir", type=Path, default=Path("artifacts/task06/v3_tournament_bundle_v1")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/task06/v3_tournament_v1")
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-completion-tokens", type=int, default=512)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--limit", type=int, help="Ogranicz liczbę grup (smoke).")
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Wymagane, by wołać API.")
    args = parser.parse_args()

    groups = load_bundle(args.bundle_dir / "tournament_bundle.jsonl")
    if args.limit is not None:
        groups = groups[: args.limit]
    plan = {
        "groups": len(groups),
        "candidates": sum(len(g.candidates) for g in groups),
        "admissible_pool_mean": round(
            sum(len(g.ranked_pool()) for g in groups) / max(1, len(groups)), 2
        ),
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
        allow_reasoning=False,
        max_completion_tokens=args.max_completion_tokens,
        allow_no_auth=args.allow_no_auth,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(plan, ensure_ascii=False, sort_keys=True), flush=True)
    if not args.skip_probe:
        try:
            print(json.dumps({"preflight": probe_endpoint(endpoint)}, ensure_ascii=False,
                             sort_keys=True), flush=True)
        except (JudgeApiError, ValueError) as exc:
            summary = exc.summary(1200) if isinstance(exc, JudgeApiError) else str(exc)
            print(f"PREFLIGHT NIE PRZESZEDŁ — nie startuję turnieju.\n  przyczyna: {summary}")
            raise SystemExit(2) from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = run_tournaments(
        groups=groups,
        endpoint=endpoint,
        journal_path=args.output_dir / "comparisons.journal.jsonl",
        output_path=args.output_dir / "tournament_outcomes.jsonl",
        concurrency=args.concurrency,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
