"""Lokalne składanie par z wadami z verdictów serwera (ADR task06_defect_pair_pipeline_v1).

Serwer tylko pyta model; **tu** zapadają rozstrzygnięcia. Wszystkie filtry
deterministyczne są liczone od nowa na tekstach, a nie przepisywane z runnera —
verdicty LLM są danymi wejściowymi, nie wyrocznią. W szczególności
`copy_phrasing` rozstrzyga wyłącznie najdłuższy wspólny ciąg słów z pasażem
(ADR §2: klasyfikator LLM nadgorliwie flaguje krótkie frazy).

Wynik ma schemat zgodny z parami v3, więc `build_task07_handoff_v3.py --pairs …`
działa bez zmian; dodatkowe pola (`defect_class`, `negative_population`,
`pair_class`) jadą obok i służą do slice'ów.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from doc2query.utils.records import JsonlWriter, read_records, write_json

CONTRACT = "task06-defect-pairs-v1"
ADR = "reports/decisions/task06_defect_pair_pipeline_v1.md"
# Progi zamrożone w ADR §4; zmiana wymaga nowego ADR.
LCS_MIN = 5
EQUIVALENCE_JACCARD = 0.6
LENGTH_RANGE = (2, 24)
LENGTH_RATIO = (0.4, 2.5)
ENTITY_ANCHOR_MIN_LENGTH = 5
MIN_CONTENT_WORD = 4
DEFECT_CLASSES = ("copy_phrasing", "not_answerable", "too_general", "answer_leak")
QUESTION_WORDS = frozenset(
    {
        "co",
        "czy",
        "czym",
        "gdzie",
        "ile",
        "jak",
        "jaka",
        "jaki",
        "jakie",
        "kiedy",
        "kim",
        "kto",
        "dlaczego",
        "który",
        "która",
        "które",
    }
)


def words(text: str) -> list[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFKC", str(text)).lower())


def content_words(text: str) -> list[str]:
    return [word for word in words(text) if len(word) >= MIN_CONTENT_WORD]


def jaccard(first: str, second: str) -> float:
    left, right = set(words(first)), set(words(second))
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


# Polskie słowa funkcyjne dłuższe niż próg `MIN_CONTENT_WORD`: bez nich pokrycie
# powierzchniowe systematycznie zaniża wynik pytań pełnych („jaka jest ...").
# Amendment 2026-08-29 do ADR §2; lista zamrożona razem z nim.
FUNCTION_WORDS = frozenset(
    {
        "albo",
        "aby",
        "będzie",
        "czym",
        "gdzie",
        "jaka",
        "jaki",
        "jakie",
        "jakim",
        "jest",
        "kiedy",
        "które",
        "który",
        "która",
        "lub",
        "może",
        "oraz",
        "przez",
        "tego",
        "wiele",
        "względem",
        "zostać",
    }
)


def content_words_filtered(text: str) -> list[str]:
    return [word for word in content_words(text) if word not in FUNCTION_WORDS]


def coverage(query: str, passage: str) -> float:
    """Pokrycie słów treściowych zapytania w pasażu, bez słów funkcyjnych."""
    tokens = content_words_filtered(query)
    if not tokens:
        return 0.0
    passage_words = set(content_words(passage))
    return sum(token in passage_words for token in tokens) / len(tokens)


def shares_entity_anchor(query: str, passage: str) -> bool:
    """Czy zapytanie dzieli z pasażem kotwicę encji (długie słowo albo liczbę).

    Zastępuje próg udziałowy: przy 2-4 słowach treściowych udział jest ziarnisty
    i „ta sama encja, brakujący atrybut" — czyli intencja klasy `not_answerable` —
    ląduje na 1/3, poniżej dowolnego rozsądnego progu. Kotwica encji wyraża ten
    wymóg wprost, zamiast przybliżać go ułamkiem.
    """
    passage_tokens = set(content_words(passage))
    for token in content_words_filtered(query):
        if token not in passage_tokens:
            continue
        if len(token) >= ENTITY_ANCHOR_MIN_LENGTH or any(char.isdigit() for char in token):
            return True
    return False


def longest_common_run(query: str, passage: str) -> int:
    """Najdłuższy wspólny ciąg słów zapytania i pasażu (miara kopiowania)."""
    left, right = words(query), words(passage)
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    best = 0
    for token in left:
        current = [0] * (len(right) + 1)
        for position, other in enumerate(right, start=1):
            if token == other:
                current[position] = previous[position - 1] + 1
                best = max(best, current[position])
        previous = current
    return best


def form_violation(query: str, form: str) -> bool:
    tokens = words(query)
    if not tokens:
        return True
    looks_question = tokens[0] in QUESTION_WORDS or str(query).strip().endswith("?")
    if form == "keyword_query":
        return looks_question
    if form == "full_question":
        return not looks_question
    return False


def deterministic_reject(query: str, chosen: str, form: str) -> str | None:
    """Filtry ADR §4 niezależne od klasy; `None` = kandydat przechodzi."""
    count = len(words(query))
    if not (LENGTH_RANGE[0] <= count <= LENGTH_RANGE[1]):
        return "length"
    if not (LENGTH_RATIO[0] <= count / max(1, len(words(chosen))) <= LENGTH_RATIO[1]):
        return "length_ratio"
    if jaccard(query, chosen) > EQUIVALENCE_JACCARD:
        return "equivalent_to_chosen"
    if form_violation(query, form):
        return "form_violation"
    return None


def class_reject(defect: str, *, query: str, passage: str, answerable: bool | None) -> str | None:
    """Wymagania klasowe ADR §2; `copy_phrasing` rozstrzyga wyłącznie LCS."""
    run = longest_common_run(query, passage)
    if defect == "copy_phrasing":
        if run < LCS_MIN:
            return "lcs_too_short"
        if answerable is not True:
            return "not_answerable_for_copy_class"
        return None
    if run >= LCS_MIN:
        return "reclassified_as_copy_phrasing"
    if defect == "not_answerable":
        if answerable is not False:
            return "answerable_true"
        if not shares_entity_anchor(query, passage):
            return "off_topic"
        return None
    if defect == "answer_leak" and answerable is not True:
        return "not_answerable"
    return None


def load_journal(path: Path) -> dict[str, dict[str, Any]]:
    """Odczytaj trwały prefiks journala; niepełna ostatnia linia jest pomijana."""
    rows: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").split("\n"):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            break
        rows[str(row["key"])] = row
    return rows


def _answerable(journal: Mapping[str, Any], gid: str, item: str) -> bool | None:
    row = journal.get(f"{gid}::answerable::{item}")
    if row is None:
        return None
    value = row.get("verdict", {}).get("answerable")
    return bool(value) if isinstance(value, bool) else None


def _pair_id(group_id: str, defect: str) -> str:
    return hashlib.sha256(f"{group_id}::{defect}".encode()).hexdigest()[:32]


def assemble_defect_pairs(
    *,
    groups_path: Path,
    journal_path: Path,
    v3_pairs_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Złóż pary z wadami; jedna para na (grupę, klasę), wszystkie filtry od nowa."""
    if output_dir.exists():
        raise FileExistsError(f"wyjście już istnieje: {output_dir}")
    journal = load_journal(journal_path)
    v3_by_preference = {str(row["pair_id"]): dict(row) for row in read_records(v3_pairs_path)}
    groups: list[dict[str, Any]] = [dict(row) for row in read_records(groups_path)]

    counters: dict[str, int] = {}

    def bump(name: str) -> None:
        counters[name] = counters.get(name, 0) + 1

    pairs: list[dict[str, Any]] = []
    for group in groups:
        gid = str(group["group_id"])
        passage = str(group["passage"])
        chosen = str(group["chosen"]["query"])
        form = str(group["form"])
        template = v3_by_preference.get(str(group["preference_id"]))
        if template is None:
            bump("group_without_v3_template")
            continue
        chosen_answerable = _answerable(journal, gid, "chosen")
        if chosen_answerable is not True:
            bump(
                "group_chosen_not_answerable" if chosen_answerable is False else "group_no_verdict"
            )
            continue

        # Kandydaci: organiczni z klasyfikacji, syntetyczni z mutacji.
        candidates: list[dict[str, Any]] = []
        for candidate in group["others"]:
            row = journal.get(f"{gid}::classify::{candidate['candidate_id']}")
            if row is None:
                continue
            label = str(row.get("verdict", {}).get("class", ""))
            if label in DEFECT_CLASSES:
                candidates.append(
                    {
                        "candidate_id": str(candidate["candidate_id"]),
                        "query": str(candidate["query"]),
                        "defect_class": label,
                        "negative_population": "mined_organic",
                    }
                )
        for defect in DEFECT_CLASSES:
            row = journal.get(f"{gid}::mutate::{defect}")
            if row is None:
                continue
            query = str(row.get("verdict", {}).get("query", "")).strip()
            if query:
                candidates.append(
                    {
                        "candidate_id": f"{gid}::mutated::{defect}",
                        "query": query,
                        "defect_class": defect,
                        "negative_population": "mutated_synthetic",
                    }
                )

        # ADR §3: organiczne mają pierwszeństwo, jedna para na (grupę, klasę).
        chosen_per_class: dict[str, dict[str, Any]] = {}
        for candidate in sorted(
            candidates, key=lambda row: (row["negative_population"] != "mined_organic",)
        ):
            defect = str(candidate["defect_class"])
            if defect in chosen_per_class:
                continue
            query = str(candidate["query"])
            reject = deterministic_reject(query, chosen, form)
            if reject is not None:
                bump(f"reject_{reject}")
                continue
            answerable = _answerable(journal, gid, str(candidate["candidate_id"]))
            if answerable is None:
                bump("reject_no_answerability_verdict")
                continue
            reject = class_reject(defect, query=query, passage=passage, answerable=answerable)
            if reject is not None:
                bump(f"reject_{reject}")
                continue
            confirm = journal.get(f"{gid}::confirm::{candidate['candidate_id']}")
            if confirm is None:
                bump("reject_no_confirmation")
                continue
            if not bool(confirm.get("unanimous_chosen")):
                bump("reject_not_unanimous")
                continue
            chosen_per_class[defect] = {**candidate, "answerable": answerable}

        for defect, candidate in sorted(chosen_per_class.items()):
            pairs.append(
                {
                    **{
                        key: template[key]
                        for key in (
                            "prompt",
                            "passage",
                            "passage_cluster_id",
                            "doc_id",
                            "cohort_id",
                            "chosen",
                            "chosen_candidate_id",
                            "chosen_components",
                            "requested_form",
                            "requested_intent",
                        )
                        if key in template
                    },
                    "pair_id": _pair_id(gid, defect),
                    "group_id": gid,
                    "source_pair_id": str(group["preference_id"]),
                    "rejected": str(candidate["query"]),
                    "rejected_candidate_id": str(candidate["candidate_id"]),
                    "defect_class": defect,
                    "negative_population": str(candidate["negative_population"]),
                    "pair_class": "defect",
                    "rejected_measurements": {
                        "passage_coverage_surface": round(
                            coverage(str(candidate["query"]), passage), 4
                        ),
                        "longest_common_run": longest_common_run(str(candidate["query"]), passage),
                        "jaccard_to_chosen": round(jaccard(str(candidate["query"]), chosen), 4),
                        "answerable": bool(candidate["answerable"]),
                    },
                    "adr": ADR,
                    "final_tests_used": [],
                }
            )
            bump(f"kept_{defect}")
            bump(f"kept_{candidate['negative_population']}")

    output_dir.mkdir(parents=True)
    pairs_path = output_dir / "pairs.jsonl"
    with JsonlWriter(pairs_path) as writer:
        for row in pairs:
            writer.write(row)

    summary: dict[str, Any] = {
        "schema_version": 1,
        "contract": CONTRACT,
        "adr": ADR,
        "groups_seen": len(groups),
        "pairs": len(pairs),
        "groups_with_pairs": len({row["group_id"] for row in pairs}),
        "counters": dict(sorted(counters.items())),
        "thresholds": {
            "lcs_min": LCS_MIN,
            "equivalence_jaccard": EQUIVALENCE_JACCARD,
            "length_range": list(LENGTH_RANGE),
            "length_ratio": list(LENGTH_RATIO),
            "entity_anchor_min_length": ENTITY_ANCHOR_MIN_LENGTH,
        },
        "task07_training_authorized": False,
        "final_tests_used": [],
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def shortcut_audit(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Bramka ADR §7.2: czy cechy powierzchniowe rozdzielają strony pary.

    Regresja logistyczna na długości, interpunkcji, pokryciu pasażu i pierwszym
    słowie. AUC wysokie oznacza, że para daje się rozstrzygnąć bez czytania
    treści — czyli DPO może wyuczyć się skrótu zamiast wady.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict

    def features(query: str, passage: str) -> list[float]:
        tokens = words(query)
        return [
            float(len(tokens)),
            float(len(str(query))),
            float(str(query).count("?")),
            float(sum(char in ",.;:-()" for char in str(query))),
            coverage(query, passage),
            float(tokens[0] in QUESTION_WORDS) if tokens else 0.0,
        ]

    rows: list[list[float]] = []
    labels: list[int] = []
    for pair in pairs:
        passage = str(pair.get("passage", ""))
        rows.append(features(str(pair["chosen"]), passage))
        labels.append(1)
        rows.append(features(str(pair["rejected"]), passage))
        labels.append(0)
    if len(set(labels)) < 2 or len(rows) < 20:
        return {"auc": None, "note": "za mało danych na audyt"}
    matrix = np.asarray(rows, dtype=float)
    target = np.asarray(labels, dtype=int)
    model = LogisticRegression(max_iter=1000)
    scores = cross_val_predict(model, matrix, target, cv=5, method="predict_proba")[:, 1]
    return {
        "auc": round(float(roc_auc_score(target, scores)), 4),
        "pairs": len(pairs),
        "features": [
            "word_count",
            "char_count",
            "question_marks",
            "punctuation",
            "passage_coverage",
            "starts_with_question_word",
        ],
        "threshold_blocking": 0.80,
    }


__all__ = [
    "ADR",
    "CONTRACT",
    "assemble_defect_pairs",
    "class_reject",
    "coverage",
    "deterministic_reject",
    "jaccard",
    "longest_common_run",
    "shares_entity_anchor",
    "shortcut_audit",
]
