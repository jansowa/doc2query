#!/usr/bin/env python
"""Zmaterializuj wejścia scoringu dla kohort pisanych tokenami modelu.

Zamienia `candidates.jsonl` kohorty teachera albo `corpus.jsonl` korpusu
walidacyjnego nagrody na rekordy generacji w **zamrożonym** schemacie Task 06,
żeby oceniał je dokładnie ten sam pipeline co kandydatów lokalnego generatora
(`evaluate_intrinsic_records`: primary builder, shadow kontrola, corpus
round-trip).

Skrypt jest deterministyczny i idempotentny: przy ponownym uruchomieniu albo
zapisuje bajtowo ten sam plik, albo — gdy istniejący plik ma inną treść —
kończy się błędem zamiast po cichu unieważnić trwający dziennik scoringu.
ADR: reports/decisions/task06_llm_cohort_gpu_scoring_amendment_2026-08-16.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from doc2query.models.templates import render_controlled_prompt
from doc2query.schemas import FocusMode, QueryControl, QueryForm, QueryIntent

TEACHER_CONTRACT = "task06-claude-teacher-ablation-v1"
REWARD_CONTRACT = "task06-reward-validation-corpus-v1"


def _load_cohort_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records[str(record["cluster_id"])] = record
    return records


def _control(form: str, intent: str, focus_bucket: str | None) -> QueryControl:
    return QueryControl(
        form=QueryForm(form),
        intent=QueryIntent(intent),
        intent_applicable=None,
        focus_mode=FocusMode.BUCKET if focus_bucket else FocusMode.NONE,
        focus_bucket=focus_bucket,  # type: ignore[arg-type]
        length="medium",
    )


def _base_record(
    cohort_record: dict[str, Any],
    *,
    generated: str,
    evaluation_id: str,
    evaluation_group_id: str,
    experiment_id: str,
    prompt: str,
    mode: str,
    control: QueryControl | None,
    extra: dict[str, Any],
) -> dict[str, Any]:
    positive = cohort_record["positives"][0]
    record: dict[str, Any] = {
        "author_model": "claude-opus-5[1m]",
        "candidate_index": extra.pop("candidate_index"),
        "doc_id": str(positive["doc_id"]),
        "evaluation_group_id": evaluation_group_id,
        "evaluation_id": evaluation_id,
        "example_id": str(cohort_record["example_id"]),
        "experiment_id": experiment_id,
        "final_tests_used": [],
        "generated": generated,
        "generation_config": {"decoding": "not_applicable_api_teacher"},
        "hard_negatives": cohort_record["hard_negatives"],
        "metadata": cohort_record.get("metadata", {}),
        "mode": mode,
        "positive": positive,
        "positive_count": len(cohort_record["positives"]),
        "positives": cohort_record["positives"],
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "reference": str(cohort_record["query"]),
        "seed": None,
        "source_example_id": str(cohort_record.get("source_example_id", "")),
        "token_logprobs": None,
    }
    if control is not None:
        record["control"] = control.model_dump(mode="json")
        record["requested_form"] = control.form.value
        record["requested_intent"] = control.intent.value
        record["requested_focus"] = control.focus_bucket
        record["requested_focus_bucket"] = control.focus_bucket
    record.update(extra)
    return record


def build_teacher_records(
    candidates_path: Path, cohort_records: dict[str, dict[str, Any]], *, experiment_id: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with candidates_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            candidate = json.loads(line)
            cluster_id = str(candidate["cluster_id"])
            cohort_record = cohort_records.get(cluster_id)
            if cohort_record is None:
                raise SystemExit(f"klaster {cluster_id} spoza zamrożonej kohorty źródłowej")
            control = _control(
                str(candidate["form"]), str(candidate["intent"]), str(candidate["focus_bucket"])
            )
            prompt = render_controlled_prompt(str(cohort_record["positives"][0]["text"]), control)
            control_id = str(candidate["control_id"])
            candidate_index = int(candidate["candidate_index"])
            rows.append(
                _base_record(
                    cohort_record,
                    generated=str(candidate["query"]),
                    evaluation_id=f"{cohort_record['example_id']}::teacher::{control_id}::{candidate_index}",
                    evaluation_group_id=f"task06-teacher::{cohort_record['example_id']}::{control_id}",
                    experiment_id=experiment_id,
                    prompt=prompt,
                    mode="teacher_ablation_controlled",
                    control=control,
                    extra={
                        "candidate_index": candidate_index,
                        "control_id": control_id,
                        "focus_fit": candidate.get("focus_fit"),
                        "intent_fit": candidate.get("intent_fit"),
                        "order_index": int(candidate["order_index"]),
                        "passage_quality_note": candidate.get("passage_quality_note"),
                    },
                )
            )
    return rows


def build_reward_records(
    corpus_path: Path, cohort_records: dict[str, dict[str, Any]], *, experiment_id: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with corpus_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            cluster_id = str(item["cluster_id"])
            cohort_record = cohort_records.get(cluster_id)
            if cohort_record is None:
                raise SystemExit(f"klaster {cluster_id} spoza zamrożonej kohorty źródłowej")
            slot = int(item["slot"])
            # Korpus diagnostyczny nie powstał z promptu generacyjnego; prompt jest
            # etykietą grupy, żeby schemat rekordu pozostał kompletny i jawny.
            prompt = f"reward-validation-corpus-v1::{cluster_id}"
            rows.append(
                _base_record(
                    cohort_record,
                    generated=str(item["query"]),
                    evaluation_id=f"{cohort_record['example_id']}::rvc::{slot}",
                    evaluation_group_id=f"task06-reward-validation::{cohort_record['example_id']}",
                    experiment_id=experiment_id,
                    prompt=prompt,
                    mode="reward_validation_corpus",
                    control=None,
                    extra={
                        "candidate_index": slot,
                        "construction_label": str(item["label"]),
                        "construction_sublabel": item.get("sublabel"),
                        "declared_focus_bucket": item.get("declared_focus_bucket"),
                        "declared_form": str(item["declared_form"]),
                        "order_index": int(item["order_index"]),
                        "slot": slot,
                    },
                )
            )
    return rows


def write_idempotent(path: Path, rows: list[dict[str, Any]]) -> tuple[str, bool]:
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing == payload:
            return digest, False
        raise SystemExit(
            f"{path} istnieje z inną treścią; nie nadpisuję, bo unieważniłoby to dziennik scoringu"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)
    return digest, True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=("teacher", "reward"))
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--cohort-records", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--experiment-id", required=True)
    args = parser.parse_args()

    cohort_records = _load_cohort_records(args.cohort_records)
    if args.kind == "teacher":
        rows = build_teacher_records(
            args.source, cohort_records, experiment_id=args.experiment_id
        )
    else:
        rows = build_reward_records(args.source, cohort_records, experiment_id=args.experiment_id)
    if not rows:
        raise SystemExit("brak rekordów do zmaterializowania")

    digest, written = write_idempotent(args.output, rows)
    manifest = {
        "contract": TEACHER_CONTRACT if args.kind == "teacher" else REWARD_CONTRACT,
        "adr": "reports/decisions/task06_llm_cohort_gpu_scoring_amendment_2026-08-16.md",
        "kind": args.kind,
        "source": str(args.source),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "cohort_records": str(args.cohort_records),
        "experiment_id": args.experiment_id,
        "record_count": len(rows),
        "group_count": len({row["evaluation_group_id"] for row in rows}),
        "output": str(args.output),
        "output_sha256": digest,
        "scoring_performed": False,
        "final_tests_used": [],
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    state = "zapisano" if written else "bez zmian (identyczny)"
    print(f"{state}: {args.output} ({len(rows)} rekordów, sha256={digest[:16]}…)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
