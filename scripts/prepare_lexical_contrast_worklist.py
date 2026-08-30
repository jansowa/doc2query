#!/usr/bin/env python3
"""Wskaż grupy, w których klasa `lexical_contrast` wymaga jeszcze generacji.

ADR `task06_defect_pair_pipeline_v1.md` §6 zamroził tę klasę: `rejected` to
kandydat `not_answerable` o **maksymalnym** pokryciu lematycznym ≥0,6, `chosen`
to kandydat `ok` z answerability TAK o **minimalnym** pokryciu ≤0,4. Ten skrypt
liczy pokrycia lematyczne (stanza, UPOS treściowe) dla kandydatów, o których
serwer już się wypowiedział, i dzieli grupy na trzy kubełki:

* `ready` — obie strony znalezione wśród kandydatów studenckich, nic nie trzeba
  generować (para powstanie lokalnie, bez ani jednego wywołania modelu);
* `needs_rejected` — jest `chosen` ≤0,4, brakuje negatywu ≥0,6; do serwera idzie
  mutacja `not_answerable` **zachowująca keywordy pasażu**;
* `impossible` — brak kandydata `ok` poniżej progu; grupa nie wystawia pary tej
  klasy (fail-closed §6.3, bez parafraz teachera po stronie `chosen`).

Nie tworzy par i nie woła modelu — produkuje listę pracy i pomiar rozkładu.
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from doc2query.preferences.defect_pairs_v1 import load_journal
from doc2query.utils.records import JsonlWriter, read_records, write_json

CONTENT_UPOS = frozenset({"NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM", "X"})
REJECTED_MIN_COVERAGE = 0.6  # ADR §6
CHOSEN_MAX_COVERAGE = 0.4  # ADR §6


def build_lemmatizer() -> Any:
    import stanza

    return stanza.Pipeline(
        "pl", processors="tokenize,pos,lemma", verbose=False, tokenize_no_ssplit=True
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
        default=Path("artifacts/task06/lexical_contrast_v1"),
    )
    parser.add_argument("--limit", type=int, default=0, help="0 = wszystkie grupy")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"wyjście już istnieje: {args.output_dir}")

    nlp = build_lemmatizer()

    @lru_cache(maxsize=200_000)
    def lemmas(text: str) -> frozenset[str]:
        doc = nlp(text.replace("\n", " "))
        return frozenset(
            word.lemma.lower()
            for sentence in doc.sentences
            for word in sentence.words
            if word.upos in CONTENT_UPOS and word.lemma
        )

    def coverage(query: str, passage_lemmas: frozenset[str]) -> float:
        query_lemmas = lemmas(query)
        if not query_lemmas:
            return 0.0
        return len(query_lemmas & passage_lemmas) / len(query_lemmas)

    journal = load_journal(args.journal)
    groups = [dict(row) for row in read_records(args.groups)]
    if args.limit:
        groups = groups[: args.limit]

    args.output_dir.mkdir(parents=True)
    worklist_path = args.output_dir / "worklist.jsonl"
    ready_path = args.output_dir / "ready.jsonl"
    counters: dict[str, int] = {}

    def bump(name: str) -> None:
        counters[name] = counters.get(name, 0) + 1

    with JsonlWriter(worklist_path) as worklist, JsonlWriter(ready_path) as ready:
        for index, group in enumerate(groups):
            gid = str(group["group_id"])
            if journal.get(f"{gid}::group_status") is None:
                bump("group_without_status")
                continue
            chosen_row = journal.get(f"{gid}::answerable::chosen")
            if chosen_row is None or not bool(chosen_row.get("verdict", {}).get("answerable")):
                bump("group_chosen_not_answerable")
                continue
            passage_lemmas = lemmas(str(group["passage"]))

            ok_candidates: list[dict[str, Any]] = []
            hard_negatives: list[dict[str, Any]] = []
            for candidate in group["others"]:
                verdict = journal.get(f"{gid}::classify::{candidate['candidate_id']}")
                if verdict is None:
                    continue
                label = str(verdict.get("verdict", {}).get("class", ""))
                answerable = journal.get(f"{gid}::answerable::{candidate['candidate_id']}")
                answerable_value = (
                    bool(answerable.get("verdict", {}).get("answerable"))
                    if answerable is not None
                    else None
                )
                value = coverage(str(candidate["query"]), passage_lemmas)
                row = {
                    "candidate_id": str(candidate["candidate_id"]),
                    "query": str(candidate["query"]),
                    "lemma_coverage": round(value, 4),
                    "answerable": answerable_value,
                }
                if label == "ok" and answerable_value is True and value <= CHOSEN_MAX_COVERAGE:
                    ok_candidates.append(row)
                if (
                    label == "not_answerable"
                    and answerable_value is False
                    and value >= (REJECTED_MIN_COVERAGE)
                ):
                    hard_negatives.append(row)

            # Zwycięzca turnieju też bywa niskopokryciowy — jest legalnym `chosen`.
            winner_coverage = coverage(str(group["chosen"]["query"]), passage_lemmas)
            if winner_coverage <= CHOSEN_MAX_COVERAGE:
                ok_candidates.append(
                    {
                        "candidate_id": str(group["chosen"]["candidate_id"]),
                        "query": str(group["chosen"]["query"]),
                        "lemma_coverage": round(winner_coverage, 4),
                        "answerable": True,
                    }
                )

            if not ok_candidates:
                bump("impossible_no_low_coverage_chosen")
                continue
            best_chosen = min(ok_candidates, key=lambda row: float(row["lemma_coverage"]))
            if hard_negatives:
                best_rejected = max(hard_negatives, key=lambda row: float(row["lemma_coverage"]))
                ready.write(
                    {
                        "group_id": gid,
                        "preference_id": str(group["preference_id"]),
                        "chosen": best_chosen,
                        "rejected": best_rejected,
                        "source": "mined_organic",
                    }
                )
                bump("ready")
            else:
                worklist.write(
                    {
                        "group_id": gid,
                        "preference_id": str(group["preference_id"]),
                        "passage": str(group["passage"]),
                        "controls": str(group["controls"]),
                        "form": str(group["form"]),
                        "chosen": best_chosen,
                        "needed": "keyword_preserving_not_answerable",
                    }
                )
                bump("needs_rejected")
            if (index + 1) % 200 == 0:
                print(f"[lexical] {index + 1}/{len(groups)}", flush=True)

    summary = {
        "schema_version": 1,
        "contract": "task06-lexical-contrast-worklist-v1",
        "adr": "reports/decisions/task06_defect_pair_pipeline_v1.md",
        "lemmatizer": "stanza-pl",
        "thresholds": {
            "rejected_min_coverage": REJECTED_MIN_COVERAGE,
            "chosen_max_coverage": CHOSEN_MAX_COVERAGE,
        },
        "groups_seen": len(groups),
        "counters": dict(sorted(counters.items())),
        "pairs_built": 0,
        "task07_training_authorized": False,
        "final_tests_used": [],
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
