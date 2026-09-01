#!/usr/bin/env python3
"""Materializacja wejść probe dla ramion Task 07 (przecięcie slotów, HN0/drop).

Preregistrowana konsekwencja kolapsu generacji
(`reports/measurements/task07_generation_collapse_2026-08-31.md`, §3): budżet
probe jest równany **przecięciem slotów wypełnionych przez wszystkie ramiona**,
żeby kolaps nie wyciekał do wyniku probe jako różnica objętości danych. Slot to
(doc_id, form, intent); duplikaty tekstu zapytania NIE wykluczają slotu — są
prawdziwym wyjściem ramienia i probe ma je zobaczyć.

Przepis na negatywy jest zamrożony i identyczny z Task 05
(`materialize_prospective_probe_inputs`): surowe `hard_negatives` dziedziczone
z zamrożonego dev (`dev_intrinsic`), filtr HN0 prymarnym sędzią z kalibracją
`artifacts/task02/pfn_dev_v1/calibration.json`, polityka `drop`. Filtr jest
zależny od zapytania, więc liczony per ramię; slot wchodzi do budżetu tylko,
gdy w KAŻDYM ramieniu przeżył ≥1 negatyw (uogólnienie polityki
`dual_arm_group_intersection_hn0_filter_drop` na N ramion).

`final_tests_used=[]` — ewaluacja probe zostaje na dev (`dev_intrinsic_rank10`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from doc2query.evaluation.datasets import load_frozen_records
from doc2query.evaluation.embedder_probe import ProbeRecipe
from doc2query.reranker.base import FrozenRerankerConfig
from doc2query.reranker.load import load_frozen_reranker
from doc2query.utils.records import JsonlWriter, read_records, write_json

Slot = tuple[str, str, str]

SCORING_CHUNK = 2048


def _slot_id(slot: Slot) -> str:
    doc_id, form, intent = slot
    return f"{doc_id}::{form}::{intent}"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gen-root", type=Path, default=Path("runs/task07_probe_gen_v1"))
    parser.add_argument(
        "--frozen-manifest",
        type=Path,
        default=Path("data/processed/v1/evaluation/task04-v1/manifest.json"),
    )
    parser.add_argument(
        "--probe-recipe", type=Path, default=Path("configs/evaluation/probe_v1.yaml")
    )
    parser.add_argument(
        "--primary-judge-config",
        type=Path,
        default=Path("configs/reranker/primary_polish_roberta_v3_p03_gpu.yaml"),
    )
    parser.add_argument(
        "--passage-example-map",
        type=Path,
        default=Path(
            "artifacts/task05/d01b_prospective_1_5b_v3/probe_inputs/selected_hybrid.jsonl"
        ),
        help="Zamrożony artefakt Task 05 z mapą source_passage_id -> source_example_id.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/task07/probe_inputs_v1"))
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"wyjście już istnieje: {args.output_dir}")

    recipe_raw = yaml.safe_load(args.probe_recipe.read_text(encoding="utf-8"))
    recipe = ProbeRecipe.from_dict(recipe_raw)
    if (
        recipe.negative_recipe.strategy != "hn0_filter"
        or recipe.negative_recipe.false_negative_policy != "drop"
    ):
        raise SystemExit("wejścia probe wymagają zamrożonego przepisu HN0+filter/drop")
    calibration = recipe.negative_recipe.load_calibration()
    if calibration is None:
        raise SystemExit("brak zamrożonej kalibracji dev dla HN0")
    judge_raw = yaml.safe_load(args.primary_judge_config.read_text(encoding="utf-8"))
    judge_config = FrozenRerankerConfig(**judge_raw)
    if (
        judge_config.name_or_path != calibration.primary_judge_name
        or judge_config.revision != calibration.primary_judge_revision
    ):
        raise SystemExit("konfiguracja sędziego nie zgadza się z proweniencją kalibracji")

    # --- 1. Sloty wypełnione per ramię -----------------------------------------
    arms: dict[str, dict[Slot, str]] = {}
    for arm_dir in sorted(p for p in args.gen_root.iterdir() if p.is_dir()):
        generated = arm_dir / "generated.jsonl"
        if not generated.is_file():
            continue
        slots: dict[Slot, str] = {}
        for row in read_records(generated):
            query = str(row.get("generated", "")).strip()
            if not query:
                continue
            slot = (
                str(row["doc_id"]),
                str(row["control"]["form"]),
                str(row["control"]["intent"]),
            )
            if slot in slots:
                raise SystemExit(f"zduplikowany slot {slot} w ramieniu {arm_dir.name}")
            slots[slot] = query
        if slots:
            arms[arm_dir.name] = slots
    if len(arms) < 2:
        raise SystemExit("przecięcie slotów wymaga co najmniej dwóch ramion")

    filled_intersection: set[Slot] = set.intersection(*(set(s) for s in arms.values()))

    # --- 2. Zamrożony dev po example_id; mapa pasaż->example z artefaktu Task 05 -
    example_by_doc: dict[str, str] = {}
    for row in read_records(args.passage_example_map):
        doc_id = str(row["source_passage_id"])
        example_id = str(row["source_example_id"])
        if example_by_doc.setdefault(doc_id, example_id) != example_id:
            raise SystemExit(f"niejednoznaczna mapa pasaż->example dla {doc_id}")
    dev_by_example = {
        str(record["example_id"]): record
        for record in load_frozen_records(args.frozen_manifest, "dev_intrinsic")
    }
    dev_by_doc: dict[str, dict[str, Any]] = {}
    for slot_doc in {slot[0] for slot in filled_intersection}:
        example_id = example_by_doc.get(slot_doc)
        record = dev_by_example.get(example_id) if example_id is not None else None
        if record is None:
            continue
        # Pozytyw = dokładnie ten pasaż, na którym generowano zapytania.
        matching = [
            dict(positive)
            for positive in record.get("positives", [])
            if str(positive["doc_id"]) == slot_doc
        ]
        if len(matching) != 1:
            continue
        dev_by_doc[slot_doc] = {**record, "positives": matching}
    missing = sorted({slot[0] for slot in filled_intersection} - set(dev_by_doc))
    if missing:
        raise SystemExit(f"pasaże bez rekordu dev: {missing[:5]} (łącznie {len(missing)})")

    # --- 3. Filtr HN0 per ramię (zależny od zapytania) --------------------------
    print(
        f"[probe-inputs] ramion: {len(arms)}, slotów w przecięciu wypełnień: "
        f"{len(filled_intersection)}; ładuję sędziego na {judge_config.device}",
        flush=True,
    )
    scorer = load_frozen_reranker(judge_config)
    ordered_slots = sorted(filled_intersection)
    retained_by_arm: dict[str, dict[Slot, list[dict[str, Any]]]] = {}
    for arm, slots in sorted(arms.items()):
        pairs: list[tuple[str, str]] = []
        offsets: list[tuple[Slot, int, int]] = []
        for slot in ordered_slots:
            negatives = dev_by_doc[slot[0]]["hard_negatives"]
            start = len(pairs)
            pairs.extend((slots[slot], str(negative["text"])) for negative in negatives)
            offsets.append((slot, start, len(pairs)))
        scores: list[float] = []
        for start in range(0, len(pairs), SCORING_CHUNK):
            scores.extend(scorer.score_pairs(pairs[start : start + SCORING_CHUNK]))
        if len(scores) != len(pairs):
            raise SystemExit("scoring sędziego zwrócił złą liczbę wyników")
        retained: dict[Slot, list[dict[str, Any]]] = {}
        for slot, start, end in offsets:
            negatives = dev_by_doc[slot[0]]["hard_negatives"]
            kept = [
                dict(negative)
                for negative, score in zip(negatives, scores[start:end], strict=True)
                if float(score) < calibration.threshold
            ]
            if kept:
                retained[slot] = kept
        retained_by_arm[arm] = retained
        print(f"[probe-inputs] {arm}: slotów z ≥1 negatywem po HN0: {len(retained)}", flush=True)

    eligible: list[Slot] = sorted(
        set.intersection(*(set(kept) for kept in retained_by_arm.values()))
    )
    if not eligible:
        raise SystemExit("puste przecięcie slotów po filtrze HN0")

    # --- 4. Zapis w formacie Task 05 --------------------------------------------
    args.output_dir.mkdir(parents=True)
    arm_summaries: dict[str, dict[str, Any]] = {}
    for arm in sorted(arms):
        output = args.output_dir / f"{arm}.jsonl"
        with JsonlWriter(output) as writer:
            for slot in eligible:
                record = dev_by_doc[slot[0]]
                positive = dict(record["positives"][0])
                pair_id = f"T07-PROBE-{arm.upper()}-S42:{_slot_id(slot)}"
                writer.write(
                    {
                        "pair_id": pair_id,
                        "example_id": pair_id,
                        "query": arms[arm][slot],
                        "generated": arms[arm][slot],
                        "mode": "deterministic",
                        "candidate_index": 0,
                        "positive": positive,
                        "positives": [positive],
                        "hard_negatives": retained_by_arm[arm][slot],
                        "source_example_id": str(record["example_id"]),
                        "source_passage_id": slot[0],
                        "generator_experiment_id": f"T07-PROBE-{arm.upper()}-S42",
                    }
                )
        arm_summaries[arm] = {
            "path": str(output),
            "sha256": _file_sha256(output),
            "filled_slots": len(arms[arm]),
            "slots_with_negatives_after_hn0": len(retained_by_arm[arm]),
        }

    manifest = {
        "schema_version": 1,
        "contract": "task07-probe-inputs-v1",
        "policy": "multi_arm_slot_intersection_hn0_filter_drop",
        "preregistration": "reports/measurements/task07_generation_collapse_2026-08-31.md",
        "slot_key": ["doc_id", "form", "intent"],
        "arms": arm_summaries,
        "filled_slot_intersection": len(filled_intersection),
        "eligible_slots": len(eligible),
        "pairs_per_arm": len(eligible),
        "negative_recipe": {
            "strategy": recipe.negative_recipe.strategy,
            "false_negative_policy": recipe.negative_recipe.false_negative_policy,
            "calibration_threshold": calibration.threshold,
            "primary_judge_name": calibration.primary_judge_name,
            "primary_judge_revision": calibration.primary_judge_revision,
        },
        "frozen_manifest": str(args.frozen_manifest),
        "dev_set": "dev_intrinsic",
        "final_tests_used": [],
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(
        json.dumps({k: v for k, v in manifest.items() if k != "arms"}, ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    main()
