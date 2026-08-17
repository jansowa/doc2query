#!/usr/bin/env python
"""Porównaj kohortę teachera ze studentem na dokładnie tym samym promptcie.

Kryteria są zapisane prospektywnie w ADR
`reports/decisions/task06_claude_teacher_ablation_v1.md`:

- odsetek pasaży, w których **najlepszy** kandydat teachera ma wyższy primary
  score niż najlepszy kandydat studenta na tej samej kontrolce;
- to samo według shadow (niezależna kontrola) i corpus round-trip;
- rozkład `copy_density` i `format_valid` teachera vs studenta;
- odsetek przypadków, w których primary i shadow są niezgodne co do kierunku.

ADR nie zapisał progu awansu, bo ablacja ma być przesłanką, nie bramką. Skrypt
raportuje więc liczby i **nie** orzeka o awansie ani nie buduje par.

Porównanie odbywa się wyłącznie w obrębie par (pasaż, kontrolka) o identycznym
`prompt_sha256`; wszystko inne jest odrzucane, żeby nie porównywać różnych zadań.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

PRIMARY = "pool_positive_score"
SHADOW = "shadow_score"
ROUND_TRIP = "corpus_round_trip_at_20"


def _control_key(row: dict[str, Any]) -> tuple[str, str, str]:
    control = row.get("control") or {}
    return (
        str(control.get("form", "")),
        str(control.get("intent", "")),
        str(control.get("focus_bucket", "")),
    )


def load_side(
    path: Path, wanted: set[str] | None = None
) -> dict[tuple[str, tuple[str, str, str]], list[dict[str, Any]]]:
    groups: dict[tuple[str, tuple[str, str, str]], list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            example_id = str(row["example_id"])
            if wanted is not None and example_id not in wanted:
                continue
            groups[(example_id, _control_key(row))].append(row)
    return groups


def _rate(hits: int, total: int) -> float | None:
    return hits / total if total else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher",
        type=Path,
        default=Path("artifacts/task06/teacher_claude_v1/scoring/per_generation.jsonl"),
    )
    parser.add_argument(
        "--student",
        type=Path,
        default=Path(
            "artifacts/task06/same_prompt_expansion_v3/d01_controlled/scoring/per_generation.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/task06/teacher_claude_v1/comparison_vs_student_v3.json"),
    )
    args = parser.parse_args()

    teacher = load_side(args.teacher)
    wanted = {key[0] for key in teacher}
    student = load_side(args.student, wanted=wanted)

    shared = sorted(set(teacher) & set(student))
    prompt_mismatch = 0
    comparable: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []
    for key in shared:
        teacher_rows, student_rows = teacher[key], student[key]
        prompts = {str(row["prompt_sha256"]) for row in teacher_rows + student_rows}
        if len(prompts) != 1:
            prompt_mismatch += 1
            continue
        comparable.append((teacher_rows, student_rows))

    primary_wins = shadow_wins = round_trip_wins = 0
    primary_ties = shadow_ties = round_trip_ties = 0
    direction_disagreement = 0
    teacher_copy: list[float] = []
    student_copy: list[float] = []
    teacher_valid: list[float] = []
    student_valid: list[float] = []
    teacher_primary: list[float] = []
    student_primary: list[float] = []

    for teacher_rows, student_rows in comparable:
        best_teacher = max(float(row[PRIMARY]) for row in teacher_rows)
        best_student = max(float(row[PRIMARY]) for row in student_rows)
        teacher_primary.append(best_teacher)
        student_primary.append(best_student)
        if best_teacher > best_student:
            primary_wins += 1
        elif best_teacher == best_student:
            primary_ties += 1

        best_teacher_shadow = max(float(row[SHADOW]) for row in teacher_rows)
        best_student_shadow = max(float(row[SHADOW]) for row in student_rows)
        if best_teacher_shadow > best_student_shadow:
            shadow_wins += 1
        elif best_teacher_shadow == best_student_shadow:
            shadow_ties += 1

        if (best_teacher > best_student) != (best_teacher_shadow > best_student_shadow):
            direction_disagreement += 1

        best_teacher_rt = max(float(row[ROUND_TRIP]) for row in teacher_rows)
        best_student_rt = max(float(row[ROUND_TRIP]) for row in student_rows)
        if best_teacher_rt > best_student_rt:
            round_trip_wins += 1
        elif best_teacher_rt == best_student_rt:
            round_trip_ties += 1

        teacher_copy.extend(float(row["copy_density"]) for row in teacher_rows)
        student_copy.extend(float(row["copy_density"]) for row in student_rows)
        teacher_valid.extend(float(bool(row["format_valid"])) for row in teacher_rows)
        student_valid.extend(float(bool(row["format_valid"])) for row in student_rows)

    total = len(comparable)
    results: dict[str, Any] = {
        "contract": "task06-claude-teacher-ablation-v1",
        "adr": "reports/decisions/task06_claude_teacher_ablation_v1.md",
        "teacher_scoring": str(args.teacher),
        "student_scoring": str(args.student),
        "comparable_prompt_groups": total,
        "prompt_mismatch_groups_skipped": prompt_mismatch,
        "primary": {
            "teacher_better_rate": _rate(primary_wins, total),
            "ties": primary_ties,
            "mean_best_teacher": mean(teacher_primary) if teacher_primary else None,
            "mean_best_student": mean(student_primary) if student_primary else None,
        },
        "shadow_control": {
            "teacher_better_rate": _rate(shadow_wins, total),
            "ties": shadow_ties,
        },
        "corpus_round_trip": {
            "teacher_better_rate": _rate(round_trip_wins, total),
            "ties": round_trip_ties,
        },
        "primary_shadow_direction_disagreement_rate": _rate(direction_disagreement, total),
        "copy_density": {
            "teacher_mean": mean(teacher_copy) if teacher_copy else None,
            "student_mean": mean(student_copy) if student_copy else None,
        },
        "format_valid_rate": {
            "teacher": mean(teacher_valid) if teacher_valid else None,
            "student": mean(student_valid) if student_valid else None,
        },
        "promotion_threshold": None,
        "promotion_decided_here": False,
        "pairs_built": False,
        "final_tests_used": [],
    }

    args.output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"grup porównywalnych (identyczny prompt): {total}, odrzuconych: {prompt_mismatch}")
    print(f"primary teacher lepszy: {results['primary']['teacher_better_rate']}")
    print(f"shadow teacher lepszy: {results['shadow_control']['teacher_better_rate']}")
    print(
        f"corpus round-trip teacher lepszy: {results['corpus_round_trip']['teacher_better_rate']}"
    )
    disagreement = results["primary_shadow_direction_disagreement_rate"]
    print(f"niezgodność kierunku primary/shadow: {disagreement}")
    print(
        f"copy_density teacher/student: {results['copy_density']['teacher_mean']:.4f} / "
        f"{results['copy_density']['student_mean']:.4f}"
    )
    print(
        f"format_valid teacher/student: {results['format_valid_rate']['teacher']:.4f} / "
        f"{results['format_valid_rate']['student']:.4f}"
    )
    print(f"zapisano {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
