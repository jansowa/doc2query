"""Handoff par v3 do kontraktu Task 07: preferencje, continued-SFT i wagi.

Packager Task 07 (`preferences/handoff.py`) niczego nie liczy — przyjmuje **zamrożone**
sześć artefaktów i sprawdza ich spójność. Ten moduł je wytwarza z par v3 i podejmuje
dokładnie dwie decyzje, których kontrakt wymaga, a pary v3 same nie zawierają:

**1. `score_margin` jest stałą 1.0 i tak jest zapisane.** Kontrakt wymaga wartości
dodatniej, a v3 nie ma stopniowanego marginesu: wszystkie pary przechodzą przy
jednomyślności 6/6, więc głosy są stałe. Wstawienie tu marginesu primary byłoby
przemyceniem sygnału, który polityka v3 świadomie wyrzuciła z selekcji, a w 10,3% par
miałoby **ujemny** znak, czyli wskazywało odwrotną stronę. Pole jest
więc jawnie bezwładne i nie udaje informacji.

**2. Wagi ramienia score-weighted pochodzą z rangi percentylowej `pool_margin` strony
`chosen`.** Bez tego trzecie obowiązkowe ramię Task 07 byłoby identyczne z continued
SFT, czyli puste. Margines primary jest tu użyty **wyłącznie jako waga w ramieniu
kontrolnym** i nigdy jako sygnał selekcji; zakres 0,01-17,59 (mediana 4,67) daje realny
gradient, a ranga percentylowa mapuje go na [0,5; 1,5] o średniej 1,0.

Podział train/dev jest deterministyczny po haszu klastra pasażu. Klastry są w parach v3
unikalne z konstrukcji, więc rozłączność train/dev jest strukturalna, nie wynikowa.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from doc2query.training.dpo import (
    canonical_fingerprint,
    file_sha256,
    ordered_ids_fingerprint,
)
from doc2query.utils.records import JsonlWriter, read_records, write_json

DATASET_ID = "task06-judge-selected-pairs-v3"
SELECTION_POLICY_ID = "task06-judge-selected-pair-policy-v3"
SELECTION_ADR = Path("reports/decisions/task06_judge_selected_pair_policy_v3.md")
WEIGHT_POLICY_ID = "task07-chosen-primary-margin-percentile-v1"
WEIGHT_CONTRACT = "task06-score-weight-assignments-for-task07-v1"
# Stała z §1 docstringa: v3 nie ma stopniowanego marginesu.
CONSTANT_SCORE_MARGIN = 1.0
WEIGHT_RANGE = (0.5, 1.5)


def weight_policy() -> dict[str, Any]:
    """Zamrożona treść polityki wag; fingerprint liczy się z tego słownika."""
    return {
        "weight_policy_id": WEIGHT_POLICY_ID,
        "signal": "chosen_components.pool_margin",
        "transform": "percentile_rank_within_dataset",
        "range": list(WEIGHT_RANGE),
        "role": "control_arm_weighting_only_never_selection",
        "rationale": (
            "Bez stopniowanej wagi trzecie obowiązkowe ramię Task 07 byłoby identyczne "
            "z continued SFT. Margines primary jest jedynym stopniowanym sygnałem "
            "niezależnym od sędziego, który wybrał parę."
        ),
    }


def dev_split_assignment(passage_cluster_id: str, *, seed: int, dev_every: int) -> bool:
    """Deterministyczny podział po haszu klastra; klastry są unikalne z konstrukcji."""
    digest = hashlib.sha256(f"{seed}:task07-dev:{passage_cluster_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % dev_every == 0


def _percentile_weights(values: Sequence[float]) -> list[float]:
    """Ranga percentylowa przemapowana na zamrożony zakres; remisy dzielą rangę."""
    low, high = WEIGHT_RANGE
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        shared = (position + end) / 2 / max(1, len(values) - 1)
        for index in order[position : end + 1]:
            ranks[index] = shared
        position = end + 1
    return [low + (high - low) * rank for rank in ranks]


def build_handoff(
    *,
    pairs_path: Path,
    output_dir: Path,
    selection_adr: Path = SELECTION_ADR,
    seed: int = 20260827,
    dev_every: int = 10,
) -> dict[str, Any]:
    """Wytwórz sześć artefaktów wejściowych Task 07 z par v3."""
    if output_dir.exists():
        raise FileExistsError(f"handoff już istnieje: {output_dir}")
    pairs = list(read_records(pairs_path))
    if not pairs:
        raise ValueError(f"{pairs_path}: brak par")
    provenance = {
        "dataset_id": DATASET_ID,
        "dataset_fingerprint": file_sha256(pairs_path),
        "selection_policy_id": SELECTION_POLICY_ID,
        "selection_policy_fingerprint": file_sha256(selection_adr),
    }
    policy = weight_policy()
    policy_fingerprint = canonical_fingerprint(policy)

    margins = [float(row["chosen_components"]["pool_margin"]) for row in pairs]
    weights = _percentile_weights(margins)
    train: list[dict[str, Any]] = []
    dev: list[dict[str, Any]] = []
    weight_by_id: dict[str, float] = {}
    for row, weight in zip(pairs, weights, strict=True):
        cluster = str(row["passage_cluster_id"])
        split = "dev" if dev_split_assignment(cluster, seed=seed, dev_every=dev_every) else "train"
        preference_id = str(row["pair_id"])
        weight_by_id[preference_id] = weight
        record = {
            "preference_id": preference_id,
            "prompt": str(row["prompt"]),
            "chosen": str(row["chosen"]),
            "rejected": str(row["rejected"]),
            "score_margin": CONSTANT_SCORE_MARGIN,
            "chosen_candidate_id": str(row["chosen_candidate_id"]),
            "rejected_candidate_id": str(row["rejected_candidate_id"]),
            "passage_id": str(row["doc_id"]),
            "passage_cluster_id": cluster,
            "split": split,
            "provenance": dict(provenance),
        }
        (dev if split == "dev" else train).append(record)
    if not train or not dev:
        raise ValueError("podział musi dać niepuste train i dev")

    output_dir.mkdir(parents=True)
    paths: dict[str, Path] = {}
    for name, rows in (("preference_train", train), ("preference_dev", dev)):
        path = output_dir / f"{name}.jsonl"
        with JsonlWriter(path) as writer:
            for row in rows:
                writer.write(row)
        paths[name] = path
    for name, rows in (("continued_sft_train", train), ("continued_sft_dev", dev)):
        path = output_dir / f"{name}.jsonl"
        with JsonlWriter(path) as writer:
            for row in rows:
                writer.write(
                    {
                        "preference_id": row["preference_id"],
                        "prompt": row["prompt"],
                        "completion": row["chosen"],
                        "candidate_id": row["chosen_candidate_id"],
                        "passage_id": row["passage_id"],
                        "passage_cluster_id": row["passage_cluster_id"],
                        "split": row["split"],
                        "provenance": dict(provenance),
                    }
                )
        paths[name] = path

    ordered = train + dev
    weight_path = output_dir / "weight_records.jsonl"
    with JsonlWriter(weight_path) as writer:
        for row in ordered:
            writer.write(
                {
                    "preference_id": row["preference_id"],
                    "split": row["split"],
                    "sample_weight": weight_by_id[row["preference_id"]],
                    **provenance,
                    "weight_policy_id": WEIGHT_POLICY_ID,
                    "weight_policy_fingerprint": policy_fingerprint,
                }
            )
    paths["weight_records"] = weight_path

    weight_manifest: dict[str, Any] = {
        "contract": WEIGHT_CONTRACT,
        "records": {
            "path": weight_path.name,
            "sha256": file_sha256(weight_path),
            "record_count": len(ordered),
        },
        **provenance,
        "weight_policy_id": WEIGHT_POLICY_ID,
        "weight_policy_fingerprint": policy_fingerprint,
        "ordered_preference_ids_fingerprint": ordered_ids_fingerprint(
            [row["preference_id"] for row in ordered]
        ),
        "final_tests_used": [],
    }
    weight_manifest["artifact_fingerprint"] = canonical_fingerprint(weight_manifest)
    manifest_path = output_dir / "weight_manifest.json"
    write_json(manifest_path, weight_manifest)
    paths["weight_manifest"] = manifest_path

    summary: dict[str, Any] = {
        "schema_version": 1,
        "contract": "task07-handoff-inputs-from-v3-pairs-v1",
        "pairs_path": str(pairs_path),
        "dataset_id": DATASET_ID,
        "dataset_fingerprint": provenance["dataset_fingerprint"],
        "selection_policy_fingerprint": provenance["selection_policy_fingerprint"],
        "weight_policy": policy,
        "weight_policy_fingerprint": policy_fingerprint,
        "score_margin_is_constant": True,
        "score_margin_value": CONSTANT_SCORE_MARGIN,
        "split_seed": seed,
        "dev_every": dev_every,
        "train_count": len(train),
        "dev_count": len(dev),
        "weight_min": min(weights),
        "weight_max": max(weights),
        "artifacts": {name: path.name for name, path in sorted(paths.items())},
        "task07_training_authorized": False,
        "final_tests_used": [],
    }
    write_json(output_dir / "handoff_summary.json", summary)
    return summary


def artifact_paths(handoff_dir: Path) -> Mapping[str, Path]:
    """Ścieżki w kolejności, jakiej oczekuje `package_task07_inputs`."""
    return {
        "preference_train_path": handoff_dir / "preference_train.jsonl",
        "preference_dev_path": handoff_dir / "preference_dev.jsonl",
        "continued_sft_train_path": handoff_dir / "continued_sft_train.jsonl",
        "continued_sft_dev_path": handoff_dir / "continued_sft_dev.jsonl",
        "weight_manifest_path": handoff_dir / "weight_manifest.json",
        "weight_records_path": handoff_dir / "weight_records.jsonl",
    }


__all__ = [
    "CONSTANT_SCORE_MARGIN",
    "DATASET_ID",
    "SELECTION_POLICY_ID",
    "WEIGHT_POLICY_ID",
    "WEIGHT_RANGE",
    "artifact_paths",
    "build_handoff",
    "dev_split_assignment",
    "weight_policy",
]
