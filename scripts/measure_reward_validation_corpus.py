#!/usr/bin/env python
"""Scal shardy korpusu walidacyjnego nagrody i zmierz prerejestrowane predykcje P1-P7.

Predykcje i progi są zamrożone przed pomiarem w
`configs/rewards/reward_validation_corpus_v1.yaml` oraz ADR
`reports/decisions/task06_reward_validation_corpus_v1.md`. Skrypt ich nie
kalibruje: czyta progi, liczy wskaźniki i raportuje przejście lub nieprzejście.

Etykiety klas pochodzą z konstrukcji autora zapytań, nie od żadnego sędziego.
Pomiar jest w całości CPU-only; P8 (primary/shadow/corpus) jest odroczone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from doc2query.data.focus_labels import assign_focus
from doc2query.data.invert import query_style
from doc2query.evaluation.diversity import pairwise_lemma_jaccards
from doc2query.evaluation.format import format_metrics
from doc2query.preferences.diversity_gate import evaluate_group, load_gate_policy
from doc2query.rewards.lexical import lexical_metrics
from doc2query.text.normalization import SimplePolishNormalizer
from doc2query.training.dpo import normalize_task06_query

REQUIRED_FIELDS = (
    "cluster_id",
    "example_id",
    "order_index",
    "slot",
    "label",
    "declared_form",
    "declared_focus_bucket",
    "query",
    "construction_note",
    "author_model",
)

SLOT_LABELS = {
    0: "good_specific",
    1: "good_alternative",
    2: "near_duplicate_of_good",
    3: "too_general",
    4: "ungrounded",
    5: "copy_verbatim",
    6: "wrong_focus",
    7: "wrong_form",
}


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def load_shards(shard_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(shard_dir.glob("shard_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                missing = [field for field in REQUIRED_FIELDS if field not in row]
                if missing:
                    raise ValueError(f"{path.name}:{line_number} brakuje pól {missing}")
                query = str(row["query"])
                if not query.strip():
                    raise ValueError(f"{path.name}:{line_number} puste query")
                if "\n" in query or "\r" in query:
                    raise ValueError(f"{path.name}:{line_number} query nie jest jednolinijkowe")
                if len(query) > 320:
                    raise ValueError(f"{path.name}:{line_number} query dłuższe niż 320 znaków")
                slot = int(row["slot"])
                if SLOT_LABELS[slot] != str(row["label"]):
                    raise ValueError(
                        f"{path.name}:{line_number} slot {slot} ma etykietę {row['label']!r}"
                    )
                row["shard"] = path.name
                rows.append(row)
    if not rows:
        raise SystemExit(f"brak shardów w {shard_dir}")
    return rows


def group_rows(rows: list[dict[str, Any]]) -> dict[str, dict[int, dict[str, Any]]]:
    groups: dict[str, dict[int, dict[str, Any]]] = {}
    for row in rows:
        cluster = str(row["cluster_id"])
        slot = int(row["slot"])
        if slot in groups.setdefault(cluster, {}):
            raise ValueError(f"klaster {cluster} ma zduplikowany slot {slot}")
        groups[cluster][slot] = row
    return groups


def annotate(rows: list[dict[str, Any]], passages: dict[str, str]) -> None:
    normalizer = SimplePolishNormalizer()
    for row in rows:
        passage = passages[str(row["cluster_id"])]
        query = str(row["query"])
        metrics = lexical_metrics(normalizer.analyze(query), normalizer.analyze(passage))
        focus = assign_focus(query, passage)
        row["_lexical"] = metrics.to_dict()
        row["_format"] = format_metrics(query)
        row["_query_style"] = query_style(query)
        row["_focus_bucket"] = focus.bucket
        row["_focus_confidence"] = focus.confidence


def _gate_rows(group: dict[int, dict[str, Any]], cluster_id: str) -> list[dict[str, Any]]:
    prompt = f"reward-validation-corpus-v1::{cluster_id}"
    prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    example_id = str(group[0]["example_id"])
    return [
        {
            "evaluation_group_id": f"rvc-v1::{cluster_id}",
            "evaluation_id": f"rvc-v1::{cluster_id}::{slot}",
            "example_id": example_id,
            "doc_id": cluster_id,
            "candidate_index": slot,
            "generated": str(row["query"]),
            "prompt": prompt,
            "prompt_sha256": prompt_sha,
            "metadata": {"split": "train"},
            "generation_config": {"temperature": None, "top_p": None},
            "seed": None,
        }
        for slot, row in sorted(group.items())
    ]


def measure(
    groups: dict[str, dict[int, dict[str, Any]]],
    *,
    gate_policy_path: Path,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    policy = load_gate_policy(gate_policy_path)

    p1_hits = p1_total = 0
    p2_hits = p2_total = 0
    p3_wrong_hits = p3_wrong_total = 0
    p3_good_hits = p3_good_total = 0
    p4_wrong_hits = p4_wrong_total = 0
    p4_good_hits = p4_good_total = 0
    p5_hits = p5_total = 0
    p6_hits = p6_total = 0
    p7_hits = p7_total = 0
    gate_failures: Counter[str] = Counter()
    complete_groups = 0

    for cluster_id, group in sorted(groups.items()):
        if len(group) != 8:
            continue
        complete_groups += 1

        copy_values = {slot: row["_lexical"]["copy_density"] for slot, row in group.items()}
        longest = {slot: row["_lexical"]["longest_copied_ngram"] for slot, row in group.items()}
        p1_total += 1
        others_copy = [value for slot, value in copy_values.items() if slot != 5]
        others_longest = [value for slot, value in longest.items() if slot != 5]
        if copy_values[5] > max(others_copy) and longest[5] > max(others_longest):
            p1_hits += 1

        p2_total += 1
        if (
            group[3]["_lexical"]["content_jaccard"] < group[0]["_lexical"]["content_jaccard"]
            and group[3]["_lexical"]["entity_preservation"]
            < group[0]["_lexical"]["entity_preservation"]
        ):
            p2_hits += 1

        if group[6]["declared_focus_bucket"] is not None:
            p3_wrong_total += 1
            if group[6]["_focus_bucket"] != group[6]["declared_focus_bucket"]:
                p3_wrong_hits += 1
        if group[0]["declared_focus_bucket"] is not None:
            p3_good_total += 1
            if group[0]["_focus_bucket"] == group[0]["declared_focus_bucket"]:
                p3_good_hits += 1

        p4_wrong_total += 1
        if not group[7]["_format"]["format_valid"]:
            p4_wrong_hits += 1
        for slot in (0, 1, 3):
            p4_good_total += 1
            if group[slot]["_format"]["format_valid"]:
                p4_good_hits += 1

        for slot in (0, 1, 3):
            p5_total += 1
            if group[slot]["_query_style"] == group[slot]["declared_form"]:
                p5_hits += 1

        p6_total += 1
        left = normalize_task06_query(str(group[0]["query"]))
        right = normalize_task06_query(str(group[2]["query"]))
        jaccard = pairwise_lemma_jaccards([str(group[0]["query"]), str(group[2]["query"])])[0]
        if left == right or jaccard >= policy.normalization.near_duplicate_lemma_jaccard:
            p6_hits += 1

        verdict = evaluate_group(_gate_rows(group, cluster_id), policy)
        p7_total += 1
        if verdict.eligible:
            p7_hits += 1
        else:
            for reason in verdict.failure_reasons:
                gate_failures[reason] += 1

    results = {
        "complete_groups": complete_groups,
        "P1_copy_density_argmax_group_rate": _rate(p1_hits, p1_total),
        "P2_too_general_below_good_group_rate": _rate(p2_hits, p2_total),
        "P3_wrong_focus_mismatch_rate": _rate(p3_wrong_hits, p3_wrong_total),
        "P3_good_specific_match_rate": _rate(p3_good_hits, p3_good_total),
        "P4_wrong_form_invalid_rate": _rate(p4_wrong_hits, p4_wrong_total),
        "P4_good_valid_rate": _rate(p4_good_hits, p4_good_total),
        "P5_declared_form_agreement_rate": _rate(p5_hits, p5_total),
        "P6_near_duplicate_collapse_rate": _rate(p6_hits, p6_total),
        "P7_group_gate_pass_rate": _rate(p7_hits, p7_total),
        "P7_gate_failure_reasons": dict(sorted(gate_failures.items())),
        "denominators": {
            "P1": p1_total,
            "P2": p2_total,
            "P3_wrong_focus": p3_wrong_total,
            "P3_good_specific": p3_good_total,
            "P4_wrong_form": p4_wrong_total,
            "P4_good": p4_good_total,
            "P5": p5_total,
            "P6": p6_total,
            "P7": p7_total,
        },
    }

    checks = {
        "P1": ("P1_copy_density_argmax_group_rate", "P1_copy_density_argmax_group_rate_min"),
        "P2": ("P2_too_general_below_good_group_rate", "P2_too_general_below_good_group_rate_min"),
        "P3_wrong_focus": ("P3_wrong_focus_mismatch_rate", "P3_wrong_focus_mismatch_rate_min"),
        "P3_good_specific": ("P3_good_specific_match_rate", "P3_good_specific_match_rate_min"),
        "P4_wrong_form": ("P4_wrong_form_invalid_rate", "P4_wrong_form_invalid_rate_min"),
        "P4_good": ("P4_good_valid_rate", "P4_good_valid_rate_exact"),
        "P5": ("P5_declared_form_agreement_rate", "P5_declared_form_agreement_rate_min"),
        "P6": ("P6_near_duplicate_collapse_rate", "P6_near_duplicate_collapse_rate_min"),
        "P7": ("P7_group_gate_pass_rate", "P7_group_gate_pass_rate_min"),
    }
    verdicts: dict[str, Any] = {}
    for name, (metric_key, threshold_key) in checks.items():
        value = results[metric_key]
        threshold = thresholds[threshold_key]
        verdicts[name] = {
            "value": value,
            "threshold": threshold,
            "passed": value is not None and value >= threshold,
        }
    results["prediction_verdicts"] = verdicts
    results["all_predictions_passed"] = all(item["passed"] for item in verdicts.values())
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/rewards/reward_validation_corpus_v1.yaml")
    )
    parser.add_argument(
        "--gate-policy",
        type=Path,
        default=Path("configs/preferences/task06_same_prompt_diversity_gate_v1.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    corpus_config = config["corpus"]
    shard_dir = Path(corpus_config["shard_dir"])
    output_dir = args.output_dir or shard_dir.parent

    passages_path = Path(corpus_config["passages"])
    passages: dict[str, str] = {}
    with passages_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            passages[str(record["cluster_id"])] = str(record["passage"])

    rows = load_shards(shard_dir)
    groups = group_rows(rows)
    unknown = sorted(set(groups) - set(passages))
    if unknown:
        raise SystemExit(f"shardy zawierają klastry poza kohortą: {unknown[:5]}")
    annotate(rows, passages)

    results = measure(
        groups,
        gate_policy_path=args.gate_policy,
        thresholds=config["predictions"]["thresholds"],
    )

    merged_path = Path(corpus_config["merged"])
    payload_rows = [
        {key: value for key, value in row.items() if not key.startswith("_")} for row in rows
    ]
    payload_rows.sort(key=lambda row: (int(row["order_index"]), int(row["slot"])))
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in payload_rows
    )
    merged_path.write_text(payload, encoding="utf-8")

    label_counts = Counter(str(row["label"]) for row in rows)
    manifest = {
        "contract": config["contract"],
        "adr": config["adr"],
        "author_model": config["author"]["model_id"],
        "pinned_weights": config["author"]["pinned_weights"],
        "human_evidence_claimed": config["author"]["human_evidence_claimed"],
        "record_count": len(rows),
        "expected_record_count": corpus_config["expected_record_count"],
        "complete_group_count": results["complete_groups"],
        "label_counts": dict(sorted(label_counts.items())),
        "merged": str(merged_path),
        "merged_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "gate_policy": str(args.gate_policy),
        "gate_policy_sha256": hashlib.sha256(args.gate_policy.read_bytes()).hexdigest(),
        "measurement": results,
        "gpu_deferred_predictions": config["predictions"]["gpu_deferred"],
        "thresholds_recalibrated": False,
        "final_tests_used": [],
    }
    measurement_path = output_dir / "measurement.json"
    measurement_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"rekordów: {len(rows)} (oczekiwano {corpus_config['expected_record_count']})")
    print(f"pełnych grup: {results['complete_groups']}")
    for name, item in results["prediction_verdicts"].items():
        status = "PASS" if item["passed"] else "FAIL"
        print(f"{status} {name}: {item['value']} (próg {item['threshold']})")
    print(f"zapisano {merged_path} i {measurement_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
