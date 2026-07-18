"""Blind A/B export and categorical inter-rater agreement."""

from __future__ import annotations

import csv
import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

from doc2query.utils.records import JsonlWriter, read_records, write_json

QUESTIONS = (
    "answerable_from_passage",
    "natural_query",
    "retrieval_useful",
    "not_overcopied",
    "does_not_reveal_answer",
    "preference",
    "target_passage_fragment",
)


def _by_example(path: Path, *, mode: str) -> dict[str, dict[str, Any]]:
    return {
        str(row["example_id"]): row
        for row in read_records(path)
        if str(row.get("mode", "deterministic")) == mode and int(row.get("candidate_index", 0)) == 0
    }


def export_blind_ab(
    left_path: Path,
    right_path: Path,
    *,
    output_jsonl: Path,
    output_csv: Path,
    seed: int = 42,
    max_examples: int = 300,
    mode: str = "deterministic",
) -> dict[str, Any]:
    left, right = _by_example(left_path, mode=mode), _by_example(right_path, mode=mode)
    shared = sorted(left.keys() & right.keys())[:max_examples]
    if not shared:
        raise ValueError("human export requires shared example IDs")
    rng = random.Random(seed)
    rows = []
    for index, example_id in enumerate(shared):
        first, second = left[example_id], right[example_id]
        swapped = bool(rng.getrandbits(1))
        a, b = (second, first) if swapped else (first, second)
        rows.append(
            {
                "panel_id": f"panel-{index:04d}",
                "passage": first.get("positive", {}).get("text", first.get("passage", "")),
                "query_a": a["generated"],
                "query_b": b["generated"],
                **{question: "" for question in QUESTIONS},
                "hidden_experiment_a": hashlib.sha256(
                    str(a.get("experiment_id", "left" if not swapped else "right")).encode()
                ).hexdigest()[:16],
                "hidden_experiment_b": hashlib.sha256(
                    str(b.get("experiment_id", "right" if not swapped else "left")).encode()
                ).hexdigest()[:16],
                "hidden_example_id": example_id,
            }
        )
    with JsonlWriter(output_jsonl) as writer:
        for row in rows:
            writer.write(row)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        visible = [
            "panel_id",
            "passage",
            "query_a",
            "query_b",
            *QUESTIONS,
            "hidden_experiment_a",
            "hidden_experiment_b",
            "hidden_example_id",
        ]
        csv_writer = csv.DictWriter(handle, fieldnames=visible)
        csv_writer.writeheader()
        csv_writer.writerows(rows)
    return {
        "count": len(rows),
        "seed": seed,
        "mode": mode,
        "jsonl": str(output_jsonl),
        "csv": str(output_csv),
    }


def cohen_kappa(left: list[str], right: list[str]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    labels = set(left) | set(right)
    observed = fmean(a == b for a, b in zip(left, right, strict=True))
    expected = sum(
        (left.count(label) / len(left)) * (right.count(label) / len(right)) for label in labels
    )
    return (observed - expected) / (1 - expected) if expected < 1 else None


def fleiss_kappa(items: list[list[str]]) -> float | None:
    if not items or any(len(item) < 2 for item in items):
        return None
    rater_count = len(items[0])
    if any(len(item) != rater_count for item in items):
        raise ValueError("Fleiss kappa requires a constant number of raters")
    labels = sorted({label for item in items for label in item})
    agreement = fmean(
        sum(count * (count - 1) for count in Counter(item).values())
        / (rater_count * (rater_count - 1))
        for item in items
    )
    totals = Counter(label for item in items for label in item)
    expected = sum((totals[label] / (len(items) * rater_count)) ** 2 for label in labels)
    return (agreement - expected) / (1 - expected) if expected < 1 else None


def import_ratings(path: Path, *, output_path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "rater_id" not in rows[0]:
        raise ValueError("ratings CSV requires rater_id and panel_id")
    report: dict[str, Any] = {"rating_rows": len(rows), "questions": {}}
    for question in QUESTIONS[:-1]:
        by_item: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for row in rows:
            value = str(row.get(question, "")).strip()
            if value:
                by_item[str(row["panel_id"])].append((str(row["rater_id"]), value))
        complete = [values for values in by_item.values() if len(values) >= 2]
        rater_ids = sorted({rater for values in complete for rater, _ in values})
        if len(rater_ids) == 2:
            paired = [
                {rater: value for rater, value in values}
                for values in complete
                if {rater for rater, _ in values} == set(rater_ids)
            ]
            agreement = cohen_kappa(
                [value[rater_ids[0]] for value in paired],
                [value[rater_ids[1]] for value in paired],
            )
            method = "cohen_kappa"
        else:
            agreement = fleiss_kappa([[value for _, value in values] for values in complete])
            method = "fleiss_kappa"
        report["questions"][question] = {
            "method": method,
            "agreement": agreement,
            "item_count": len(complete),
            "rater_count": len(rater_ids),
        }
    write_json(output_path, report)
    return report
