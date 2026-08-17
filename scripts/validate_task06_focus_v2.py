#!/usr/bin/env python3
"""V2-02: porównaj focus_v2 z zamrożonym v1 na korpusie o etykietach z konstrukcji.

Skrypt niczego nie zmienia w zamrożonych artefaktach: liczy etykiety v1 i v2
obok siebie na tych samych 360 zapytaniach klas `good_specific`/`wrong_focus`
korpusu walidacyjnego nagrody i raportuje zgodność z deklaracją, abstencję oraz
statystyki segmentacji 180 pasaży. Wynik jest pomiarem komponentu, nie zmianą
żadnego progu.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from doc2query.data.focus_labels import assign_focus, split_sentences
from doc2query.data.focus_labels_v2 import FOCUS_V2_VERSION, assign_focus_v2, split_sentences_v2
from doc2query.training.dpo import file_sha256
from doc2query.utils.records import read_records, write_json

FOCUS_CLASSES = ("good_specific", "wrong_focus")


def _passages(cohort_records: Path) -> dict[str, str]:
    passages: dict[str, str] = {}
    for row in read_records(cohort_records):
        positives = row.get("positives") or []
        if positives:
            passages[str(row["example_id"])] = str(positives[0]["text"])
    if not passages:
        raise ValueError(f"no passages found in {cohort_records}")
    return passages


def _evaluate(rows: list[dict[str, Any]], passages: dict[str, str]) -> dict[str, Any]:
    """Score both label variants against the constructed declarations.

    Semantyka klas jest asymetryczna: dla `good_specific` deklaracja to focus,
    o który zapytanie faktycznie pyta (sukces = zgodność), a dla `wrong_focus`
    deklaracja to focus **żądany**, który zapytanie celowo łamie (sukces =
    wykryta niezgodność). Rekordy z `declared_focus_bucket = None`
    (`degenerate_single_sentence`) są wyłączone z mianownika, jak w P3.
    """
    per_variant: dict[str, dict[str, Counter[str]]] = {
        "v1": {label: Counter() for label in FOCUS_CLASSES},
        "v2": {label: Counter() for label in FOCUS_CLASSES},
    }
    changed = 0
    degenerate = 0
    for row in rows:
        declared_raw = row.get("declared_focus_bucket")
        if declared_raw is None:
            degenerate += 1
            continue
        passage = passages[str(row["example_id"])]
        declared = str(declared_raw)
        label = str(row["label"])
        outcomes = {}
        for variant, assigner in (("v1", assign_focus), ("v2", assign_focus_v2)):
            assignment = assigner(str(row["query"]), passage)
            if assignment.bucket is None:
                outcome = "abstained"
            elif assignment.bucket == declared:
                outcome = "match"
            else:
                outcome = "mismatch"
            per_variant[variant][label][outcome] += 1
            outcomes[variant] = (assignment.bucket, outcome)
        if outcomes["v1"] != outcomes["v2"]:
            changed += 1

    summary: dict[str, Any] = {
        "changed_assignments": changed,
        "query_count": len(rows),
        "degenerate_single_sentence_excluded": degenerate,
    }
    for variant, labels in per_variant.items():
        block: dict[str, Any] = {}
        for label, counts in labels.items():
            total = sum(counts.values())
            decided = counts["match"] + counts["mismatch"]
            success = counts["match"] if label == "good_specific" else counts["mismatch"]
            block[label] = {
                "total": total,
                "match": counts["match"],
                "mismatch": counts["mismatch"],
                "abstained": counts["abstained"],
                "abstention_rate": counts["abstained"] / total if total else None,
                "success_definition": (
                    "match_declared" if label == "good_specific" else "detect_violation"
                ),
                "overall_success_rate": success / total if total else None,
                "success_rate_on_decided": success / decided if decided else None,
            }
        summary[variant] = block
    return summary


def _segmentation(passages: dict[str, str]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for variant, splitter in (("v1", split_sentences), ("v2", split_sentences_v2)):
        counts = [len(splitter(text)) for text in passages.values()]
        letterless = sum(
            1
            for text in passages.values()
            for sentence in splitter(text)
            if not any(character.isalpha() for character in sentence)
        )
        stats[variant] = {
            "passages": len(counts),
            "total_sentences": sum(counts),
            "mean_sentences": sum(counts) / len(counts),
            "single_sentence_passages": sum(1 for value in counts if value == 1),
            "two_sentence_passages": sum(1 for value in counts if value == 2),
            "letterless_sentences": letterless,
        }
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("artifacts/task06/reward_validation_corpus_v1/corpus.jsonl"),
    )
    parser.add_argument(
        "--cohort-records",
        type=Path,
        default=Path("artifacts/task06/candidate_pilot_v1/cohort.records.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/measurements/task06/focus_v2_validation/summary.json"),
    )
    args = parser.parse_args()

    rows = [
        row for row in read_records(args.corpus) if str(row.get("label")) in FOCUS_CLASSES
    ]
    if not rows:
        raise ValueError("the corpus holds no focus-labeled classes")
    passages = _passages(args.cohort_records)
    missing = sorted({str(row["example_id"]) for row in rows} - set(passages))
    if missing:
        raise ValueError(f"corpus references passages absent from the cohort: {missing[:3]}")

    report: dict[str, Any] = {
        "schema_version": 1,
        "contract": "task06-focus-v2-validation-v1",
        "status": "component_measured_frozen_artifacts_untouched",
        "focus_v2_version": FOCUS_V2_VERSION,
        "inputs": {
            "corpus_sha256": file_sha256(args.corpus),
            "cohort_records_sha256": file_sha256(args.cohort_records),
        },
        "focus_agreement": _evaluate(rows, {k: passages[k] for k in passages}),
        "segmentation": _segmentation(
            {str(row["example_id"]): passages[str(row["example_id"])] for row in rows}
        ),
        "v1_labels_modified": False,
        "final_tests_used": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
