#!/usr/bin/env python3
"""Wyeksportuj chudy pakiet turniejowy v3 z zamrożonych kohort.

Pakiet zawiera wyłącznie to, co sędzia ma prawo zobaczyć: pasaż i teksty kandydatów
plus flagi dopuszczalności policzone tanimi guardami. Nie zawiera score'ów, marginesów,
round-tripu, werdyktów odpowiadalności ani niczego, co zdradza wybór automatu — §3 ADR
v3 nakłada ślepość. Dzięki temu na maszynę z serwerem jedzie kilkanaście MB, a nie 380.

Ten skrypt nie woła żadnego API i nie buduje par.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from doc2query.evaluation.d01_usefulness import _copy_risk as copy_risk_flag
from doc2query.preferences.pair_policy import (
    _Candidate,
    _format_admissible,
    _load_gate,
    _load_scoring,
)
from doc2query.preferences.pair_policy_v2_1 import load_defect_pair_policy_v2_1
from doc2query.preferences.pair_policy_v3 import BUNDLE_CONTRACT
from doc2query.training.dpo import canonical_fingerprint, file_sha256
from doc2query.utils.records import JsonlWriter, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-dir", type=Path, action="append", required=True)
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("configs/preferences/task06_defect_pair_policy_v2_1.yaml"),
        help="Config, z którego brane są tylko guardy formatu i copy_risk.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/task06/v3_tournament_bundle_v1")
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"pakiet już istnieje: {args.output_dir}")

    policy = load_defect_pair_policy_v2_1(args.policy)
    thresholds = policy.copy_risk.thresholds()
    round_trip_field = policy.corpus_round_trip.chosen_required_field
    args.output_dir.mkdir(parents=True)
    bundle_path = args.output_dir / "tournament_bundle.jsonl"
    sources: dict[str, Any] = {}
    groups = 0
    candidates = 0
    admissible_chosen = 0

    with JsonlWriter(bundle_path) as writer:
        for cohort_dir in args.cohort_dir:
            scoring_dir = cohort_dir / "d01_controlled" / "scoring"
            gate, gate_verdicts = _load_gate(cohort_dir / "diversity_gate")
            rows = _load_scoring(
                scoring_dir / "per_generation.jsonl", scoring_dir / "summary.json", gate
            )
            sources[cohort_dir.name] = {
                "scoring_sha256": file_sha256(scoring_dir / "per_generation.jsonl"),
                "gate_verdicts_sha256": gate.verdicts.sha256,
            }
            grouped: dict[str, list[_Candidate]] = {}
            for row in rows:
                grouped.setdefault(str(row["evaluation_group_id"]), []).append(
                    _Candidate(
                        candidate_id=str(row["evaluation_id"]),
                        candidate_index=int(row["candidate_index"]),
                        query=str(row["generated"]),
                        row=row,
                    )
                )
            for group_id in sorted(grouped):
                verdict = gate_verdicts.get(group_id)
                if verdict is None or not bool(verdict["eligible"]):
                    continue
                representatives = {
                    str(value) for value in verdict["representative_candidate_ids"]
                }
                members = [c for c in grouped[group_id] if c.candidate_id in representatives]
                if len(members) < 2:
                    continue
                payload = []
                for member in sorted(members, key=lambda c: c.candidate_index):
                    format_ok = _format_admissible(member)
                    clean = (
                        format_ok
                        and member.number(round_trip_field) >= 1.0
                        and not copy_risk_flag(member.row, thresholds)
                    )
                    payload.append(
                        {
                            "candidate_id": member.candidate_id,
                            "candidate_index": member.candidate_index,
                            "query": member.query,
                            "admissible_as_chosen": bool(clean),
                            "admissible_as_rejected": bool(format_ok),
                        }
                    )
                    candidates += 1
                    admissible_chosen += int(bool(clean))
                writer.write(
                    {
                        "group_id": group_id,
                        "cohort_id": cohort_dir.name,
                        "passage": str(members[0].row["positive"]["text"]),
                        "candidates": payload,
                    }
                )
                groups += 1

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "contract": BUNDLE_CONTRACT,
        "policy_id": policy.policy_id,
        "sources": dict(sorted(sources.items())),
        "group_count": groups,
        "candidate_count": candidates,
        "admissible_as_chosen": admissible_chosen,
        "bundle": {
            "path": bundle_path.name,
            "sha256": file_sha256(bundle_path),
            "record_count": groups,
        },
        "contains_scores": False,
        "contains_automatic_choice": False,
        "final_tests_used": [],
    }
    manifest["manifest_fingerprint"] = canonical_fingerprint(manifest)
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
