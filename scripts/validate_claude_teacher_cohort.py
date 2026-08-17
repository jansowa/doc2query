#!/usr/bin/env python
"""Zwaliduj i scal shardy kohorty teachera `teacher_claude_v1`.

Skrypt sprawdza kontrakt z `configs/preferences/task06_claude_teacher_ablation_v1.yaml`:
komplet kontrolek i kandydatur na pasaż, jednolinijkowość, zgodność formy,
różnorodność w obrębie kontrolki oraz przynależność klastrów do zamrożonej
kohorty. Nie liczy żadnych sygnałów jakości — scoring jest odroczony do GPU
i wykonuje go zamrożony kontrakt Task 06.

Uruchamiany wielokrotnie: raportuje, które shardy są kompletne, a które trzeba
wygenerować ponownie, więc przerwana praca kosztuje najwyżej jeden shard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

_TOKEN = re.compile(r"\w+", re.UNICODE)

REQUIRED_FIELDS = (
    "cluster_id",
    "example_id",
    "order_index",
    "control_id",
    "form",
    "intent",
    "focus_bucket",
    "candidate_index",
    "query",
    "author_model",
)


def _normalized(query: str) -> str:
    return " ".join(token.lower() for token in _TOKEN.findall(query))


def _word_count(query: str) -> int:
    return len(_TOKEN.findall(query))


def validate_shard(path: Path, controls: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    problems: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                problems.append(f"linia {line_number}: niepoprawny JSON ({error})")
                continue
            missing = [field for field in REQUIRED_FIELDS if field not in row]
            if missing:
                problems.append(f"linia {line_number}: brakuje pól {missing}")
                continue
            query = str(row["query"])
            control_id = str(row["control_id"])
            control = controls.get(control_id)
            if control is None:
                problems.append(f"linia {line_number}: nieznana kontrolka {control_id!r}")
                continue
            if not query.strip():
                problems.append(f"linia {line_number}: puste query")
            if "\n" in query or "\r" in query:
                problems.append(f"linia {line_number}: query nie jest jednolinijkowe")
            if len(query) > 320:
                problems.append(f"linia {line_number}: query dłuższe niż 320 znaków")
            if str(row["form"]) != control["form"]:
                problems.append(f"linia {line_number}: form niezgodna z kontrolką {control_id}")
            if str(row["intent"]) != control["intent"]:
                problems.append(f"linia {line_number}: intent niezgodny z kontrolką {control_id}")
            if control["form"] == "full_question" and not query.rstrip().endswith("?"):
                problems.append(f"linia {line_number}: full_question bez znaku zapytania")
            if control["form"] == "keyword_query":
                if query.rstrip().endswith("?"):
                    problems.append(f"linia {line_number}: keyword_query ze znakiem zapytania")
                if not 2 <= _word_count(query) <= 6:
                    problems.append(
                        f"linia {line_number}: keyword_query ma {_word_count(query)} słów"
                    )
            rows.append(row)

    per_control: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        per_control[(str(row["cluster_id"]), str(row["control_id"]))].append(row)
    duplicate_groups = 0
    incomplete_groups = 0
    for (cluster_id, control_id), group in sorted(per_control.items()):
        if len(group) != 4:
            incomplete_groups += 1
            problems.append(f"{cluster_id}/{control_id}: {len(group)} kandydatur zamiast 4")
        normalized = {_normalized(str(row["query"])) for row in group}
        if len(normalized) != len(group):
            duplicate_groups += 1
            problems.append(f"{cluster_id}/{control_id}: powtórzone query po normalizacji")

    clusters = Counter(str(row["cluster_id"]) for row in rows)
    return {
        "shard": path.name,
        "record_count": len(rows),
        "cluster_count": len(clusters),
        "clusters_with_wrong_record_count": sorted(
            cluster for cluster, count in clusters.items() if count != 16
        ),
        "control_group_count": len(per_control),
        "incomplete_control_groups": incomplete_groups,
        "duplicate_control_groups": duplicate_groups,
        "problems": problems,
        "complete": len(rows) == 400 and not problems,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/preferences/task06_claude_teacher_ablation_v1.yaml"),
    )
    parser.add_argument(
        "--merge", action="store_true", help="scal kompletne shardy do candidates.jsonl"
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    controls = {item["control_id"]: item for item in config["controls"]}
    generation = config["generation"]
    shard_dir = Path(generation["shard_dir"])
    shard_dir.mkdir(parents=True, exist_ok=True)

    cohort_clusters: set[str] = set()
    with Path(config["cohort"]["passages"]).open(encoding="utf-8") as handle:
        for line in handle:
            cohort_clusters.add(str(json.loads(line)["cluster_id"]))

    reports: list[dict[str, Any]] = []
    merged_rows: list[dict[str, Any]] = []
    for path in sorted(shard_dir.glob("shard_*.jsonl")):
        report = validate_shard(path, controls)
        outside = sorted({str(row["cluster_id"]) for row in report["rows"]} - cohort_clusters)
        if outside:
            report["problems"].append(f"klastry poza kohortą: {outside[:5]}")
            report["complete"] = False
        rows = report.pop("rows")
        if report["complete"]:
            merged_rows.extend(rows)
        reports.append(report)

    expected = int(generation["shard_count"])
    present = {report["shard"] for report in reports}
    missing = [
        f"shard_{index:03d}.jsonl"
        for index in range(expected)
        if f"shard_{index:03d}.jsonl" not in present
    ]
    incomplete = [report["shard"] for report in reports if not report["complete"]]

    summary: dict[str, Any] = {
        "contract": config["contract"],
        "adr": config["adr"],
        "author_model": config["teacher"]["author_model"],
        "pinned_weights": config["teacher"]["pinned_weights"],
        "expected_shard_count": expected,
        "expected_record_count": generation["expected_record_count"],
        "present_shard_count": len(reports),
        "complete_shard_count": sum(1 for report in reports if report["complete"]),
        "record_count_in_complete_shards": len(merged_rows),
        "missing_shards": missing,
        "incomplete_shards": incomplete,
        "shard_reports": reports,
        "scoring_performed": False,
        "quality_fields_used": [],
        "diversity_gate_applied_to_teacher": False,
        "pairs_built": False,
        "final_tests_used": [],
    }

    if args.merge and merged_rows:
        merged_rows.sort(
            key=lambda row: (
                int(row["order_index"]),
                str(row["control_id"]),
                int(row["candidate_index"]),
            )
        )
        payload = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in merged_rows
        )
        merged_path = Path(generation["merged"])
        merged_path.write_text(payload, encoding="utf-8")
        summary["merged"] = str(merged_path)
        summary["merged_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        summary["merged_record_count"] = len(merged_rows)

    summary_path = shard_dir.parent / "cohort.validation.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        f"shardy: {summary['complete_shard_count']}/{expected} kompletne, "
        f"rekordów {len(merged_rows)}/{generation['expected_record_count']}"
    )
    if missing:
        print(f"brakuje: {', '.join(missing)}")
    for report in reports:
        if not report["complete"]:
            head = "; ".join(report["problems"][:3])
            print(f"NIEKOMPLETNY {report['shard']} ({report['record_count']} rekordów): {head}")
    print(f"zapisano {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
