#!/usr/bin/env python3
"""Eksport wejścia pipeline'u wad: grupy z kandydatami dla runnera serwerowego.

Jedna linia = jedna grupa handoffu bottom: pasaż, tekst kontrolek z promptu,
`chosen` (zwycięzca turnieju), obecny `rejected` i pozostali kandydaci
studenccy. Runner (`scripts/task06_defect_pipeline_remote.py`) niczego więcej
nie potrzebuje. Eksport jest deterministyczny; manifest niesie SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from doc2query.utils.records import JsonlWriter, read_records, write_json


def controls_text(prompt: str) -> str:
    lines = [
        line
        for line in prompt.split("\n")
        if re.match(r"^(Forma|Intencja|Docelowy fragment|Długość):", line)
    ]
    return "\n".join(lines)


def form_of(prompt: str) -> str:
    match = re.search(r"^Forma: (.+)$", prompt, flags=re.MULTILINE)
    return match.group(1).strip() if match else "?"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path("artifacts/task06/v3_tournament_bundle_v1/tournament_bundle.jsonl"),
    )
    parser.add_argument(
        "--packaged-dir",
        type=Path,
        default=Path("artifacts/task07/handoff_v3_bottom/packaged"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/task06/defect_pipeline_v1/input")
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"wyjście już istnieje: {args.output_dir}")

    pairs = {
        str(row["chosen_candidate_id"]): dict(row)
        for name in ("preference_train.jsonl", "preference_dev.jsonl")
        for row in read_records(args.packaged_dir / name)
    }
    groups_path = args.output_dir / "groups.jsonl"
    args.output_dir.mkdir(parents=True)
    count = 0
    with JsonlWriter(groups_path) as writer:
        for group in read_records(args.bundle):
            candidates = list(group["candidates"])
            hit = next(
                (c for c in candidates if str(c["candidate_id"]) in pairs),
                None,
            )
            if hit is None:
                continue
            pair = pairs[str(hit["candidate_id"])]
            prompt = str(pair["prompt"])
            writer.write(
                {
                    "group_id": str(group["group_id"]),
                    "preference_id": str(pair["preference_id"]),
                    "passage": str(group["passage"]),
                    "controls": controls_text(prompt),
                    "form": form_of(prompt),
                    "chosen": {
                        "candidate_id": str(pair["chosen_candidate_id"]),
                        "query": str(pair["chosen"]),
                    },
                    "current_rejected": {
                        "candidate_id": str(pair["rejected_candidate_id"]),
                        "query": str(pair["rejected"]),
                    },
                    "others": [
                        {"candidate_id": str(c["candidate_id"]), "query": str(c["query"])}
                        for c in candidates
                        if str(c["candidate_id"])
                        not in (
                            str(pair["chosen_candidate_id"]),
                            str(pair["rejected_candidate_id"]),
                        )
                    ],
                }
            )
            count += 1
    digest = hashlib.sha256(groups_path.read_bytes()).hexdigest()
    write_json(
        args.output_dir / "manifest.json",
        {
            "schema_version": 1,
            "contract": "task06-defect-pipeline-input-v1",
            "adr": "reports/decisions/task06_defect_pair_pipeline_v1.md",
            "groups": count,
            "records": {"path": groups_path.name, "sha256": digest},
            "final_tests_used": [],
        },
    )
    print(json.dumps({"groups": count, "sha256": digest[:16]}, indent=2))


if __name__ == "__main__":
    main()
