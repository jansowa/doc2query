"""Axis-D monotony baseline: how uniform are the generator's queries as a population?

Problem P3 of the pair-policy v2 specification ("questions are always the same length and
start with the same words") is explicitly **outside** preference pairs — it is a property
of the *population* of generated queries, which a pairwise objective cannot express.  Its
proper homes are the generation controls, the mandatory selector-free production-mode
evaluation (M-05, Task 07) and a set-level diversity component of the GRPO reward
(Task 08, note V2-07).  All three need the same thing first: a **baseline** of what the
current generator actually produces.

This module measures exactly that and nothing else:

* per-slice distributions of ``word_length`` and ``character_length``;
* the distribution of **first words** with concentration statistics (top-1/3/5 share,
  normalized entropy, distinct count) — the direct measurement of "always starts the same";
* per-group **set-level** distinct-1/distinct-2 over the same-prompt group, i.e. the
  candidate GRPO set-level component computed on the set of queries per passage.

Slices are the frozen generation controls (`form`, `length`, `intent`) plus the cohort, so
the report also answers whether a control separates its own distribution at all.

It is a **declared design input**: it reads generated text openly, freezes no threshold,
gates nothing, builds no pair and promotes nothing.  Tokenization is deliberately plain
(regex word split, lowercased, no lemmatization) because no Polish spaCy model is pinned
here; the report says so rather than pretending to lemma-level diversity.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from doc2query.evaluation.retrieval import distribution
from doc2query.training.dpo import file_sha256
from doc2query.utils.records import read_records, write_json

MONOTONY_CONTRACT = "task06-query-monotony-baseline-v1"
MONOTONY_STATUS = "design_input_measured_no_thresholds"
TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
TOP_FIRST_WORDS = 10


def tokenize(query: str) -> list[str]:
    """Plain lowercased word split; no lemmatization is claimed anywhere in this module."""
    return TOKEN_PATTERN.findall(query.lower())


def first_word(query: str) -> str | None:
    tokens = tokenize(query)
    return tokens[0] if tokens else None


def distinct_n(queries: Sequence[str], n: int) -> float | None:
    """Distinct-n over the concatenated n-grams of a query set (set-level diversity)."""
    grams: list[tuple[str, ...]] = []
    for query in queries:
        tokens = tokenize(query)
        grams.extend(tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1))
    if not grams:
        return None
    return len(set(grams)) / len(grams)


@dataclass
class SliceAccumulator:
    """Streaming accumulator for one slice; keeps counts, not rows."""

    word_lengths: list[float] = field(default_factory=list)
    character_lengths: list[float] = field(default_factory=list)
    first_words: Counter[str] = field(default_factory=Counter)
    queries: int = 0
    empty_token_queries: int = 0

    def add(self, query: str, word_length: float, character_length: float) -> None:
        self.queries += 1
        self.word_lengths.append(word_length)
        self.character_lengths.append(character_length)
        token = first_word(query)
        if token is None:
            self.empty_token_queries += 1
        else:
            self.first_words[token] += 1

    def summary(self) -> dict[str, Any]:
        total = sum(self.first_words.values())
        ordered = self.first_words.most_common()
        shares = [count / total for _, count in ordered] if total else []
        entropy = -sum(share * math.log(share) for share in shares if share > 0)
        distinct = len(self.first_words)
        return {
            "queries": self.queries,
            "word_length": distribution(self.word_lengths),
            "character_length": distribution(self.character_lengths),
            "first_word": {
                "distinct": distinct,
                "queries_without_tokens": self.empty_token_queries,
                "top1_share": shares[0] if shares else None,
                "top3_share": sum(shares[:3]) if shares else None,
                "top5_share": sum(shares[:5]) if shares else None,
                "entropy_nats": entropy if shares else None,
                # Znormalizowana entropia: 1.0 = rozkład jednostajny po zaobserwowanych
                # pierwszych słowach, 0.0 = zawsze to samo słowo.
                "normalized_entropy": (
                    entropy / math.log(distinct)
                    if distinct > 1
                    else (0.0 if distinct == 1 else None)
                ),
                "top": [
                    {"token": token, "count": count, "share": count / total}
                    for token, count in ordered[:TOP_FIRST_WORDS]
                ],
            },
        }


def _control(row: Mapping[str, Any]) -> Mapping[str, Any]:
    control = row.get("control")
    if not isinstance(control, Mapping):
        raise ValueError(f"scored candidate {row.get('evaluation_id')} has no control block")
    return control


def accumulate_cohort(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Stream one cohort into slice accumulators plus per-group set-level diversity."""
    slices: dict[tuple[str, str], SliceAccumulator] = {}
    pooled = SliceAccumulator()
    group_queries: dict[str, list[str]] = {}
    for row in rows:
        query = str(row["generated"])
        word_length = float(row["word_length"])
        character_length = float(row["character_length"])
        control = _control(row)
        pooled.add(query, word_length, character_length)
        for dimension in ("form", "length", "intent"):
            value = control.get(dimension)
            key = (dimension, "none" if value is None else str(value))
            slices.setdefault(key, SliceAccumulator()).add(query, word_length, character_length)
        group_queries.setdefault(str(row["evaluation_group_id"]), []).append(query)
    distinct_1 = [
        value for queries in group_queries.values() if (value := distinct_n(queries, 1)) is not None
    ]
    distinct_2 = [
        value for queries in group_queries.values() if (value := distinct_n(queries, 2)) is not None
    ]
    return {
        "pooled": pooled.summary(),
        "by_control": {
            dimension: {
                value: accumulator.summary()
                for (other, value), accumulator in sorted(slices.items())
                if other == dimension
            }
            for dimension in ("form", "length", "intent")
        },
        "set_level_per_group": {
            "groups": len(group_queries),
            "distinct_1": distribution(distinct_1),
            "distinct_2": distribution(distinct_2),
        },
    }


def run_monotony_baseline(
    *, cohort_dirs: Iterable[Path], output_path: Path, limit_per_cohort: int | None = None
) -> dict[str, Any]:
    """Measure the baseline over frozen cohorts and publish one versioned JSON artifact."""
    cohorts: dict[str, dict[str, Any]] = {}
    inputs: dict[str, str] = {}
    for cohort_dir in cohort_dirs:
        scoring_path = cohort_dir / "d01_controlled" / "scoring" / "per_generation.jsonl"
        if not scoring_path.is_file():
            raise ValueError(f"missing scored cohort: {scoring_path}")
        rows = read_records(scoring_path)
        if limit_per_cohort is not None:
            rows = (row for index, row in enumerate(rows) if index < limit_per_cohort)
        cohorts[cohort_dir.name] = accumulate_cohort(rows)
        inputs[cohort_dir.name] = file_sha256(scoring_path)
    if not cohorts:
        raise ValueError("the monotony baseline needs at least one cohort")
    report: dict[str, Any] = {
        "schema": "task06-query-monotony-baseline-result-v1",
        "contract": MONOTONY_CONTRACT,
        "status": MONOTONY_STATUS,
        "role": "design_input_for_m05_and_grpo_set_level_reward",
        "tokenization": "lowercased regex word split, no lemmatization, no pinned spaCy model",
        "thresholds_frozen_here": False,
        "pairs_built": False,
        "limit_per_cohort": limit_per_cohort,
        "inputs_sha256": dict(sorted(inputs.items())),
        "cohorts": cohorts,
        "task07_training_authorized": False,
        "final_tests_used": [],
    }
    write_json(output_path, report)
    return report
