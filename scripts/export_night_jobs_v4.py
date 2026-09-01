#!/usr/bin/env python3
"""Paczka v4 dla serwera: niezależny recheck answerability flag pełnego audytu.

Pełny audyt SFT oznaczył pary `nieodpowiadalne`, ale przegląd próbek dał
precyzję ~60-80% — za mało, by na samych flagach oprzeć filtr puli (usunięcie
fałszywej flagi wyrzuca dobrą parę). Ten pakiet buduje wejście do zadania
`chosen_recheck` (niezależny prompt answerability, inna instrukcja niż audyt):

1. wszystkie pary z flagą `odpowiadalne=false` — para trafia na listę filtra
   tylko przy ZGODNYM werdykcie obu promptów (flaga ∧ recheck=false);
2. kontrolna losowa próbka par czystych (seed 20260901) — do oszacowania,
   ile wad audyt przepuszcza (FN rate), zanim zapadnie decyzja o filtrze.

To pomiar; żadna para nie jest usuwana. `final_tests_used=[]`.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from doc2query.utils.records import JsonlWriter, read_records, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--journal",
        type=Path,
        default=Path("artifacts/task06/night_jobs_v3/verdicts/night_jobs.journal.jsonl"),
    )
    parser.add_argument(
        "--pool",
        type=Path,
        default=Path("artifacts/task06/night_jobs_v3/input/sft_full_pool.jsonl"),
    )
    parser.add_argument("--clean-sample-size", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/task06/night_jobs_v4/input")
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"wyjście już istnieje: {args.output_dir}")

    flagged: list[str] = []
    clean: list[str] = []
    for row in read_records(args.journal):
        key = str(row["key"])
        if not key.startswith("sft_full_audit::"):
            continue
        verdict = row.get("verdict", {})
        answerable = verdict.get("odpowiadalne")
        if not isinstance(answerable, bool):
            continue
        pid = key.split("::", 1)[1]
        (clean if answerable else flagged).append(pid)

    rng = random.Random(args.seed)
    control = sorted(rng.sample(clean, min(args.clean_sample_size, len(clean))))

    pool = {str(row["id"]): row for row in read_records(args.pool)}
    args.output_dir.mkdir(parents=True)
    written = {"recheck_flagged": 0, "recheck_control_clean": 0}
    with JsonlWriter(args.output_dir / "dropped_groups.jsonl") as writer:
        for group, ids in (("flagged", sorted(flagged)), ("control_clean", control)):
            for pid in ids:
                item = pool.get(pid)
                if item is None:
                    continue
                writer.write(
                    {
                        "id": f"{group}::{pid}",
                        "passage": str(item["passage"]),
                        "chosen": str(item["query"]),
                    }
                )
                written[f"recheck_{group}"] += 1

    manifest = {
        "schema_version": 1,
        "contract": "task06-night-jobs-v4-input-v1",
        "purpose": "answerability_recheck_of_full_audit_flags_plus_clean_control",
        "source_journal": str(args.journal),
        "items_per_job": written,
        "estimated_calls": sum(written.values()),
        "clean_sample_seed": args.seed,
        "pairs_built": 0,
        "final_tests_used": [],
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
