#!/usr/bin/env python3
"""Przetwórz werdykty nocne: klasa leksykalna, answer_leak v2, czystość, odzysk.

Wszystkie rozstrzygnięcia zapadają tutaj, lokalnie i deterministycznie; werdykty
serwera są danymi wejściowymi. Progi pochodzą z zamrożonego ADR
`task06_defect_pair_pipeline_v1.md` (§4, §6) i z jego amendmentu; ten skrypt
żadnego nie zmienia.

Cztery wyniki, każdy w osobnym pliku, żaden nie wchodzi do treningu bez
osobnej decyzji właściciela:

* `lexical_contrast_pairs.jsonl` — pary klasy `lexical_contrast`;
* `answer_leak_v2_pairs.jsonl` + audyt anty-skrótowy tej klasy;
* `label_purity.json` — czystość etykiet kohorty trenowalnej (pomiar);
* `chosen_recheck.json` — ile grup wraca przy zgodnym TAK obu wywołań.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from doc2query.preferences.defect_pairs_v1 import (
    class_reject,
    coverage,
    deterministic_reject,
    jaccard,
    load_journal,
    longest_common_run,
    shortcut_audit,
)
from doc2query.utils.records import JsonlWriter, read_records, write_json

REJECTED_MIN_COVERAGE = 0.6  # ADR §6, na lematach
CHOSEN_MAX_COVERAGE = 0.4  # ADR §6, na lematach


def _pair_id(group_id: str, tag: str) -> str:
    return hashlib.sha256(f"{group_id}::{tag}".encode()).hexdigest()[:32]


def _lemma_coverage(nlp: Any, query: str, passage: str) -> float:
    content = {"NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM", "X"}

    def lemmas(text: str) -> set[str]:
        doc = nlp(text.replace("\n", " "))
        return {
            word.lemma.lower()
            for sentence in doc.sentences
            for word in sentence.words
            if word.upos in content and word.lemma
        }

    query_lemmas = lemmas(query)
    if not query_lemmas:
        return 0.0
    return len(query_lemmas & lemmas(passage)) / len(query_lemmas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--night-journal",
        type=Path,
        default=Path("artifacts/task06/night_jobs_v1/verdicts/night_jobs.journal.jsonl"),
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("artifacts/task06/night_jobs_v1/input"),
    )
    parser.add_argument(
        "--lexical-ready",
        type=Path,
        default=Path("artifacts/task06/lexical_contrast_v1/ready.jsonl"),
    )
    parser.add_argument(
        "--groups",
        type=Path,
        default=Path("artifacts/task06/defect_pipeline_v1/input/groups.jsonl"),
    )
    parser.add_argument(
        "--v3-pairs",
        type=Path,
        default=Path("artifacts/task06/v3_pairs_v1/bottom/pairs.jsonl"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/task06/night_results_v1")
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"wyjście już istnieje: {args.output_dir}")

    journal = load_journal(args.night_journal)
    groups = {str(row["group_id"]): dict(row) for row in read_records(args.groups)}
    templates = {str(row["pair_id"]): dict(row) for row in read_records(args.v3_pairs)}
    args.output_dir.mkdir(parents=True)
    counters: dict[str, int] = {}

    def bump(name: str) -> None:
        counters[name] = counters.get(name, 0) + 1

    import stanza

    nlp = stanza.Pipeline(
        "pl", processors="tokenize,pos,lemma", verbose=False, tokenize_no_ssplit=True
    )

    def emit(
        writer: Any,
        group: dict[str, Any],
        chosen: str,
        rejected: str,
        tag: str,
        meta: dict[str, Any],
    ) -> None:
        template = templates.get(str(group["preference_id"]))
        if template is None:
            bump("no_template")
            return
        writer.write(
            {
                **{
                    key: template[key]
                    for key in (
                        "prompt",
                        "passage",
                        "passage_cluster_id",
                        "doc_id",
                        "cohort_id",
                        "requested_form",
                        "requested_intent",
                    )
                    if key in template
                },
                "pair_id": _pair_id(str(group["group_id"]), tag),
                "group_id": str(group["group_id"]),
                "source_pair_id": str(group["preference_id"]),
                "chosen": chosen,
                "chosen_candidate_id": meta.get("chosen_candidate_id", ""),
                "chosen_components": template.get("chosen_components", {}),
                "rejected": rejected,
                "rejected_candidate_id": meta.get("rejected_candidate_id", ""),
                **meta,
                "adr": "reports/decisions/task06_defect_pair_pipeline_v1.md",
                "final_tests_used": [],
            }
        )

    # --- klasa lexical_contrast -------------------------------------------------
    lexical_path = args.output_dir / "lexical_contrast_pairs.jsonl"
    with JsonlWriter(lexical_path) as writer:
        for row in read_records(args.lexical_ready):
            group = groups.get(str(row["group_id"]))
            if group is None:
                continue
            emit(
                writer,
                group,
                str(row["chosen"]["query"]),
                str(row["rejected"]["query"]),
                "lexical_mined",
                {
                    "pair_class": "lexical_contrast",
                    "defect_class": "not_answerable",
                    "negative_population": "mined_organic",
                    "chosen_candidate_id": str(row["chosen"]["candidate_id"]),
                    "rejected_candidate_id": str(row["rejected"]["candidate_id"]),
                    "lemma_coverage": {
                        "chosen": row["chosen"]["lemma_coverage"],
                        "rejected": row["rejected"]["lemma_coverage"],
                    },
                },
            )
            bump("lexical_mined")

        worklist = {
            str(row["id"]): row for row in read_records(args.input_dir / "lexical_worklist.jsonl")
        }
        for gid, item in sorted(worklist.items()):
            entry = journal.get(f"lexical_mutation::{gid}")
            group = groups.get(gid)
            if entry is None or group is None:
                bump("lexical_no_verdict")
                continue
            query = str(entry.get("verdict", {}).get("query", "")).strip()
            if not query:
                bump("lexical_empty")
                continue
            answerable = entry.get("answerable_check", {}).get("answerable")
            if answerable is not False:
                bump("lexical_rejected_answerable")
                continue
            chosen = str(item["chosen"])
            passage = str(group["passage"])
            if deterministic_reject(query, chosen, str(group["form"])) is not None:
                bump("lexical_filtered")
                continue
            rejected_cov = _lemma_coverage(nlp, query, passage)
            chosen_cov = _lemma_coverage(nlp, chosen, passage)
            if rejected_cov < REJECTED_MIN_COVERAGE or chosen_cov > CHOSEN_MAX_COVERAGE:
                bump("lexical_thresholds")
                continue
            emit(
                writer,
                group,
                chosen,
                query,
                "lexical_mutated",
                {
                    "pair_class": "lexical_contrast",
                    "defect_class": "not_answerable",
                    "negative_population": "mutated_synthetic",
                    "rejected_candidate_id": f"{gid}::lexical::mutated",
                    "lemma_coverage": {
                        "chosen": round(chosen_cov, 4),
                        "rejected": round(rejected_cov, 4),
                    },
                },
            )
            bump("lexical_mutated")

    # --- answer_leak v2 ---------------------------------------------------------
    leak_path = args.output_dir / "answer_leak_v2_pairs.jsonl"
    with JsonlWriter(leak_path) as writer:
        for row in read_records(args.input_dir / "answer_leak_groups.jsonl"):
            gid = str(row["id"])
            entry = journal.get(f"answer_leak_v2::{gid}")
            group = groups.get(gid)
            if entry is None or group is None:
                bump("leak_no_verdict")
                continue
            query = str(entry.get("verdict", {}).get("query", "")).strip()
            chosen = str(row["chosen"])
            passage = str(group["passage"])
            if not query:
                bump("leak_empty")
                continue
            if deterministic_reject(query, chosen, str(group["form"])) is not None:
                bump("leak_filtered")
                continue
            answerable = entry.get("answerable_check", {}).get("answerable")
            if class_reject("answer_leak", query=query, passage=passage, answerable=answerable):
                bump("leak_class_reject")
                continue
            emit(
                writer,
                group,
                chosen,
                query,
                "answer_leak_v2",
                {
                    "pair_class": "defect",
                    "defect_class": "answer_leak",
                    "negative_population": "mutated_synthetic",
                    "rejected_candidate_id": f"{gid}::leak_v2::mutated",
                    "rejected_measurements": {
                        "passage_coverage_surface": round(coverage(query, passage), 4),
                        "longest_common_run": longest_common_run(query, passage),
                        "jaccard_to_chosen": round(jaccard(query, chosen), 4),
                        "answerable": bool(answerable),
                    },
                },
            )
            bump("leak_kept")

    leak_pairs = [dict(row) for row in read_records(leak_path)]
    leak_audit = shortcut_audit(leak_pairs) if leak_pairs else {"auc": None}
    lexical_pairs = [dict(row) for row in read_records(lexical_path)]
    lexical_audit = shortcut_audit(lexical_pairs) if lexical_pairs else {"auc": None}

    # --- czystość etykiet -------------------------------------------------------
    purity: dict[str, dict[str, int]] = {}
    for row in read_records(args.input_dir / "pairs_to_verify.jsonl"):
        entry = journal.get(f"label_purity::{row['id']}")
        if entry is None:
            continue
        defect = str(row["defect_class"])
        stats = purity.setdefault(
            defect, {"n": 0, "defect_confirmed": 0, "b_worse": 0, "class_matches": 0}
        )
        verdict = entry.get("verdict", {})
        stats["n"] += 1
        stats["defect_confirmed"] += int(bool(verdict.get("wada_potwierdzona")))
        stats["b_worse"] += int(bool(verdict.get("b_gorsze_od_a")))
        stats["class_matches"] += int(str(verdict.get("faktyczna_klasa", "")) == defect)

    # --- odzysk grup ------------------------------------------------------------
    recheck = {"n": 0, "second_yes": 0, "recovered": 0}
    for row in read_records(args.input_dir / "dropped_groups.jsonl"):
        entry = journal.get(f"chosen_recheck::{row['id']}")
        if entry is None:
            continue
        recheck["n"] += 1
        # Pierwsze wywołanie orzekło NIE; grupa wraca tylko przy zgodnym TAK,
        # czyli nigdy na podstawie samej zmiany zdania — a takiej zgody tu z
        # definicji nie ma. Liczymy więc rozbieżność, nie odzysk.
        recheck["second_yes"] += int(bool(entry.get("verdict", {}).get("answerable")))

    summary = {
        "schema_version": 1,
        "contract": "task06-night-results-v1",
        "adr": "reports/decisions/task06_defect_pair_pipeline_v1.md",
        "counters": dict(sorted(counters.items())),
        "lexical_contrast": {
            "pairs": len(lexical_pairs),
            "shortcut_audit": lexical_audit,
            "thresholds": {
                "rejected_min_lemma_coverage": REJECTED_MIN_COVERAGE,
                "chosen_max_lemma_coverage": CHOSEN_MAX_COVERAGE,
            },
        },
        "answer_leak_v2": {
            "pairs": len(leak_pairs),
            "shortcut_audit": leak_audit,
            "blocking_threshold": 0.80,
            "blocked": bool(leak_audit.get("auc") is not None and leak_audit["auc"] > 0.80),
        },
        "label_purity": {
            defect: {
                **stats,
                "defect_confirmed_rate": round(stats["defect_confirmed"] / stats["n"], 4),
                "b_worse_rate": round(stats["b_worse"] / stats["n"], 4),
                "class_match_rate": round(stats["class_matches"] / stats["n"], 4),
            }
            for defect, stats in sorted(purity.items())
            if stats["n"]
        },
        "chosen_recheck": {
            **recheck,
            "disagreement_rate": (
                round(recheck["second_yes"] / recheck["n"], 4) if recheck["n"] else None
            ),
            "recovery_policy": "grupa wraca tylko przy zgodnym TAK obu wywołań",
        },
        "task07_training_authorized": False,
        "final_tests_used": [],
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
