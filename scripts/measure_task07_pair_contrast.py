#!/usr/bin/env python3
"""Zdiagnozuj, na jakiej osi stoi kontrast w parach preferencyjnych.

Pytanie jest jedno: czy `chosen` bije `rejected` **jakością zapytania**, czy tylko
tym, że jest o tym pasażu. Jeśli tym drugim, to DPO uczy się głównie trzymania
tematu, a tego punkt startowy w dużej mierze już umie — i taki wynik trzeba
umieć zinterpretować, a nie tłumaczyć po fakcie.

Miara jest gruba i celowo model-free: udział słów treściowych zapytania (≥4 znaki),
które **dokładnie** występują w pasażu, bez lematyzacji. Polska fleksja sprawia, że
ta miara **zaniża** pokrycie obu stron symetrycznie; służy do porównania stron, nie
do orzekania o ugruntowaniu pojedynczej pary.

Skrypt nie filtruje, nie odrzuca i nie zmienia żadnej pary — tylko liczy.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as stats
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from doc2query.utils.records import read_records, write_json

MIN_WORD_LENGTH = 4
GROUNDED = 0.6
UNGROUNDED = 0.34
QUESTION_START = re.compile(r"^(co to|czym|jak |jakie|jaki |jaka |ile |kto |kim |gdzie|kiedy|czy )")


def content_words(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).lower()
    return [
        word
        for word in re.findall(r"\w+", normalized, flags=re.UNICODE)
        if len(word) >= MIN_WORD_LENGTH
    ]


def passage_of(prompt: str) -> str:
    if "Pasaż:\n" not in prompt:
        return prompt
    return prompt.split("Pasaż:\n", 1)[1].split("\n\nZapytanie:", 1)[0]


def prompt_field(prompt: str, name: str) -> str:
    match = re.search(rf"^{name}: (.+)$", prompt, flags=re.MULTILINE)
    return match.group(1).strip() if match else "?"


def _coverage(words: list[str], passage: set[str]) -> float:
    if not words:
        return 0.0
    return sum(word in passage for word in words) / len(words)


def measure(paths: list[Path]) -> dict[str, Any]:
    rows = [dict(row) for path in paths for row in read_records(path)]
    if not rows:
        raise SystemExit(f"brak par w {paths}")
    chosen_cov: list[float] = []
    rejected_cov: list[float] = []
    chosen_len: list[int] = []
    rejected_len: list[int] = []
    forms: Counter[str] = Counter()
    first_word: Counter[str] = Counter()
    first_word_rejected: Counter[str] = Counter()
    question_form = 0
    for row in rows:
        prompt = str(row["prompt"])
        passage = set(content_words(passage_of(prompt)))
        chosen = content_words(str(row["chosen"]))
        rejected = content_words(str(row["rejected"]))
        chosen_cov.append(_coverage(chosen, passage))
        rejected_cov.append(_coverage(rejected, passage))
        chosen_len.append(len(chosen))
        rejected_len.append(len(rejected))
        forms[prompt_field(prompt, "Forma")] += 1
        words = str(row["chosen"]).lower().split()
        first_word[words[0] if words else ""] += 1
        rejected_words = str(row["rejected"]).lower().split()
        first_word_rejected[rejected_words[0] if rejected_words else ""] += 1
        question_form += bool(QUESTION_START.match(str(row["chosen"]).lower()))

    total = len(rows)

    def share(predicate: Any) -> float:
        return round(
            sum(1 for pair in zip(chosen_cov, rejected_cov, strict=True) if predicate(*pair))
            / total,
            4,
        )

    return {
        "schema_version": 1,
        "contract": "task07-pair-contrast-diagnostic-v1",
        "sources": [str(path) for path in paths],
        "pairs": total,
        "requested_form": dict(forms.most_common()),
        "passage_coverage": {
            "chosen_median": round(stats.median(chosen_cov), 4),
            "chosen_mean": round(stats.fmean(chosen_cov), 4),
            "rejected_median": round(stats.median(rejected_cov), 4),
            "rejected_mean": round(stats.fmean(rejected_cov), 4),
        },
        "shares": {
            "rejected_ungrounded": share(lambda _chosen, rejected: rejected < UNGROUNDED),
            "both_grounded": share(
                lambda chosen, rejected: chosen >= GROUNDED and rejected >= GROUNDED
            ),
            "chosen_fully_in_passage": round(
                sum(1 for value in chosen_cov if value >= 0.999) / total, 4
            ),
            "chosen_question_form": round(question_form / total, 4),
        },
        "definicja_prefix": {
            "chosen": round(first_word.get("definicja", 0) / total, 4),
            "rejected": round(first_word_rejected.get("definicja", 0) / total, 4),
        },
        "length_content_words": {
            "chosen_median": stats.median(chosen_len),
            "rejected_median": stats.median(rejected_len),
            "chosen_at_most_three": round(sum(1 for value in chosen_len if value <= 3) / total, 4),
        },
        "top_first_words_chosen": dict(first_word.most_common(8)),
        "measure_notes": {
            "min_word_length": MIN_WORD_LENGTH,
            "lemmatization": False,
            "bias": "zaniża pokrycie obu stron symetrycznie (fleksja)",
        },
        "pairs_modified": 0,
        "final_tests_used": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, action="append", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = measure(args.pairs)
    result["label"] = args.label
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
