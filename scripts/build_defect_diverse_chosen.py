#!/usr/bin/env python3
"""Kohorta DIVCH: te same pary defect, ale `chosen` różnicowany wewnątrz grupy.

ADR `task07_anti_collapse_v1` §1.2: w kohorcie defect grupa wystawia do trzech
par o identycznym `chosen`, co uczy „jednej odpowiedzi na pasaż" i sprzyja
kolapsowi generacji. Ta kohorta niczego nie dogenerowuje — w grupie o k parach
klasy dostają **różne** `chosen` spośród kandydatów już zweryfikowanych przez
serwer (klasyfikacja `ok` + answerability TAK) oraz zwycięzcy turnieju. Rotacja
jest deterministyczna: klasy posortowane alfabetycznie, kandydaci po
`candidate_id`, przydział cyklicznie. Grupa z jednym kandydatem zostaje bez
zmian.

Uczciwość metadanych: `chosen_components` pochodzą od zwycięzcy turnieju i po
podmianie NIE opisują nowego `chosen` — pole zostaje (wymaga go packager), ale
para dostaje `chosen_components_source`, a ramię weighted SFT NIE może być na
tej kohorcie trenowane (wagi z `pool_margin` byłyby fałszywe). ADR ogranicza
DIVCH do ramienia DPO.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from doc2query.preferences.defect_pairs_v1 import load_journal
from doc2query.training.dpo import normalize_task06_query
from doc2query.utils.records import JsonlWriter, read_records, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs",
        type=Path,
        default=Path("artifacts/task06/defect_pairs_v1/pairs_trainable.jsonl"),
    )
    parser.add_argument(
        "--groups",
        type=Path,
        default=Path("artifacts/task06/defect_pipeline_v1/input/groups.jsonl"),
    )
    parser.add_argument(
        "--journal",
        type=Path,
        default=Path("artifacts/task06/defect_pipeline_v1/verdicts/verdicts.journal.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/task06/defect_diverse_v1"),
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"wyjście już istnieje: {args.output_dir}")

    journal = load_journal(args.journal)
    groups = {str(row["group_id"]): dict(row) for row in read_records(args.groups)}
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_records(args.pairs):
        by_group[str(row["group_id"])].append(dict(row))

    counters: dict[str, int] = {
        "groups": 0,
        "pairs": 0,
        "chosen_substituted": 0,
        "collision_fallback": 0,
    }
    args.output_dir.mkdir(parents=True)
    with JsonlWriter(args.output_dir / "pairs_trainable.jsonl") as writer:
        for gid in sorted(by_group):
            group = groups[gid]
            pairs = sorted(by_group[gid], key=lambda row: str(row["defect_class"]))
            counters["groups"] += 1

            # Pula chosen: zwycięzca + kandydaci ok/answerable TAK, deterministycznie.
            pool: list[dict[str, str]] = [
                {
                    "candidate_id": str(group["chosen"]["candidate_id"]),
                    "query": str(group["chosen"]["query"]),
                }
            ]
            for candidate in sorted(group["others"], key=lambda c: str(c["candidate_id"])):
                cid = str(candidate["candidate_id"])
                classify = journal.get(f"{gid}::classify::{cid}")
                answerable = journal.get(f"{gid}::answerable::{cid}")
                if (
                    classify is not None
                    and str(classify.get("verdict", {}).get("class")) == "ok"
                    and answerable is not None
                    and bool(answerable.get("verdict", {}).get("answerable"))
                ):
                    pool.append({"candidate_id": cid, "query": str(candidate["query"])})

            for index, pair in enumerate(pairs):
                # Kandydat nie może być identyczny z rejected TEJ pary po
                # normalizacji Task 06 (walidator handoffu to odrzuca); przy
                # kolizji bierzemy następnego z puli, a gdy wszyscy kolidują —
                # oryginalny chosen pary, który przeszedł walidację kohorty defect.
                rejected_norm = normalize_task06_query(str(pair["rejected"]))
                assigned = None
                for offset in range(len(pool)):
                    candidate = pool[(index + offset) % len(pool)]
                    if normalize_task06_query(candidate["query"]) != rejected_norm:
                        assigned = candidate
                        break
                if assigned is None:
                    assigned = {
                        "candidate_id": str(pair["chosen_candidate_id"]),
                        "query": str(pair["chosen"]),
                    }
                    counters["collision_fallback"] += 1
                substituted = assigned["query"] != str(pair["chosen"])
                counters["pairs"] += 1
                counters["chosen_substituted"] += int(substituted)
                writer.write(
                    {
                        **pair,
                        "pair_id": hashlib.sha256(
                            f"{gid}::{pair['defect_class']}::divch".encode()
                        ).hexdigest()[:32],
                        "chosen": assigned["query"],
                        "chosen_candidate_id": assigned["candidate_id"],
                        "chosen_components_source": (
                            "group_winner_not_this_candidate" if substituted else "group_winner"
                        ),
                        "cohort_variant": "defect_diverse_v1",
                        "source_pair_id_defect": str(pair["pair_id"]),
                    }
                )

    summary = {
        "schema_version": 1,
        "contract": "task06-defect-diverse-chosen-v1",
        "adr": "reports/decisions/task07_anti_collapse_v1.md",
        "counters": counters,
        "weighted_sft_forbidden": True,
        "task07_training_authorized": False,
        "final_tests_used": [],
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
