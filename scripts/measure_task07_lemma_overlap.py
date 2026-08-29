#!/usr/bin/env python3
"""Zmierz pokrycie leksykalne par na lematach (stanza pl), bez stopwordów.

Fundament pod klasę par `lexical_contrast` z projektu pipeline'u: `chosen`
odpowiadalne przy minimalnym pokryciu słownictwa pasażu, `rejected` z wysokim
pokryciem bez odpowiedzi. Zanim ADR zamrozi progi, trzeba znać rozkład pokrycia
na lematach — miara powierzchniowa bez lematyzacji zaniża pokrycie przez polską
fleksję, więc progi z niej byłyby zawyżone.

Pomiar diagnostyczny: nie zmienia par, nie filtruje, nie buduje kohort.
Lematyzer: stanza (rekomendacja właściciela), UPOS content words
(NOUN/PROPN/VERB/ADJ/ADV/NUM/X), reszta odpada jako stopwordy strukturalne.
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
from functools import lru_cache
from pathlib import Path
from typing import Any

from doc2query.utils.records import read_records, write_json

CONTENT_UPOS = frozenset({"NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM", "X"})
BUCKETS = ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01))


def build_lemmatizer() -> Any:
    import stanza

    return stanza.Pipeline(
        "pl", processors="tokenize,pos,lemma", verbose=False, tokenize_no_ssplit=True
    )


def passage_of(prompt: str) -> str:
    if "Pasaż:\n" not in prompt:
        return prompt
    return prompt.split("Pasaż:\n", 1)[1].split("\n\nZapytanie:", 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs",
        type=Path,
        action="append",
        default=None,
    )
    parser.add_argument("--limit", type=int, default=0, help="0 = wszystkie pary")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/measurements/task07/lemma_overlap_v1/bottom.json"),
    )
    args = parser.parse_args()
    if args.pairs is None:
        packaged = Path("artifacts/task07/handoff_v3_bottom/packaged")
        args.pairs = [packaged / "preference_train.jsonl", packaged / "preference_dev.jsonl"]

    nlp = build_lemmatizer()

    @lru_cache(maxsize=100_000)
    def lemmas(text: str) -> frozenset[str]:
        doc = nlp(text.replace("\n", " "))
        return frozenset(
            word.lemma.lower()
            for sentence in doc.sentences
            for word in sentence.words
            if word.upos in CONTENT_UPOS and word.lemma
        )

    def coverage(query: str, passage: frozenset[str]) -> float:
        query_lemmas = lemmas(query)
        if not query_lemmas:
            return 0.0
        return len(query_lemmas & passage) / len(query_lemmas)

    rows = [dict(row) for path in args.pairs for row in read_records(path)]
    if args.limit:
        rows = rows[: args.limit]
    chosen_cov: list[float] = []
    rejected_cov: list[float] = []
    for index, row in enumerate(rows):
        passage = lemmas(passage_of(str(row["prompt"])))
        chosen_cov.append(coverage(str(row["chosen"]), passage))
        rejected_cov.append(coverage(str(row["rejected"]), passage))
        if (index + 1) % 200 == 0:
            print(f"[lemma] {index + 1}/{len(rows)}", flush=True)

    def histogram(values: list[float]) -> dict[str, float]:
        return {
            f"[{low};{high if high <= 1 else 1})": round(
                sum(1 for value in values if low <= value < high) / len(values), 4
            )
            for low, high in BUCKETS
        }

    total = len(rows)
    trap_supply = sum(
        1
        for chosen, rejected in zip(chosen_cov, rejected_cov, strict=True)
        if chosen <= 0.5 and rejected >= 0.75
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "contract": "task07-lemma-overlap-diagnostic-v1",
        "lemmatizer": "stanza-pl",
        "content_upos": sorted(CONTENT_UPOS),
        "sources": [str(path) for path in args.pairs],
        "pairs": total,
        "chosen": {
            "median": round(stats.median(chosen_cov), 4),
            "mean": round(stats.fmean(chosen_cov), 4),
            "histogram": histogram(chosen_cov),
        },
        "rejected": {
            "median": round(stats.median(rejected_cov), 4),
            "mean": round(stats.fmean(rejected_cov), 4),
            "histogram": histogram(rejected_cov),
        },
        "lexical_trap_natural_supply": {
            "definition": "chosen<=0.5 i rejected>=0.75 (progi robocze, nie zamrożone)",
            "count": trap_supply,
            "share": round(trap_supply / total, 4),
        },
        "pairs_modified": 0,
        "final_tests_used": [],
    }
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
