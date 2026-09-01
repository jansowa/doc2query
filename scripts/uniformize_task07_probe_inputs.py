#!/usr/bin/env python3
"""Ujednolicenie K zapytań na pasaż w wejściach probe Task 07 (kontrakt P-04).

`train_probe` egzekwuje jednolite K zapytań na pasaż (P-04 comparison budget),
a przecięcie slotów daje zmienne K. Ten krok wybiera K maksymalizujące budżet
par (K x liczba pasaży z ≥K slotami), odrzuca pasaże z < K slotami i w pasażach
z nadmiarem zostawia pierwsze K slotów po posortowaniu identyfikatorów slotów.
Reguła jest deterministyczna i ślepa na ramiona: zbiór slotów po przecięciu
jest identyczny we wszystkich ramionach, więc każdemu ramieniu zostaje
dokładnie ten sam podzbiór slotów.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from doc2query.utils.records import JsonlWriter, read_records, write_json


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path("artifacts/task07/probe_inputs_v1"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/task07/probe_inputs_v1_uniform_k")
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"wyjście już istnieje: {args.output_dir}")
    source_manifest = json.loads((args.inputs / "manifest.json").read_text(encoding="utf-8"))
    arms = sorted(source_manifest["arms"])

    # Zbiór slotów jest wspólny — wyznacz wybór na pierwszym ramieniu.
    reference = list(read_records(args.inputs / f"{arms[0]}.jsonl"))
    slots_by_doc: dict[str, list[str]] = defaultdict(list)
    for row in reference:
        slot_id = str(row["example_id"]).split(":", 1)[1]
        slots_by_doc[str(row["source_passage_id"])].append(slot_id)
    counts = {doc: len(slots) for doc, slots in slots_by_doc.items()}
    budgets = {
        k: k * sum(1 for count in counts.values() if count >= k)
        for k in range(1, max(counts.values()) + 1)
    }
    uniform_k = max(budgets, key=lambda k: (budgets[k], k))
    kept_slot_ids = {
        slot_id
        for doc, slots in slots_by_doc.items()
        if len(slots) >= uniform_k
        for slot_id in sorted(slots)[:uniform_k]
    }

    args.output_dir.mkdir(parents=True)
    arm_summaries: dict[str, dict[str, object]] = {}
    pairs_per_arm: set[int] = set()
    for arm in arms:
        output = args.output_dir / f"{arm}.jsonl"
        kept = 0
        with JsonlWriter(output) as writer:
            for row in read_records(args.inputs / f"{arm}.jsonl"):
                slot_id = str(row["example_id"]).split(":", 1)[1]
                if slot_id in kept_slot_ids:
                    writer.write(dict(row))
                    kept += 1
        pairs_per_arm.add(kept)
        arm_summaries[arm] = {"path": str(output), "sha256": _file_sha256(output), "pairs": kept}
    if len(pairs_per_arm) != 1 or pairs_per_arm != {len(kept_slot_ids)}:
        raise SystemExit(f"budżet nie jest równy między ramionami: {sorted(pairs_per_arm)}")

    manifest = {
        **source_manifest,
        "contract": "task07-probe-inputs-uniform-k-v1",
        "derived_from": {
            "path": str(args.inputs),
            "manifest_sha256": _file_sha256(args.inputs / "manifest.json"),
        },
        "uniform_k": uniform_k,
        "k_distribution_before": {
            str(k): sum(1 for count in counts.values() if count == k) for k in sorted(budgets)
        },
        "budgets_considered": {str(k): budgets[k] for k in sorted(budgets)},
        "passages": sum(1 for count in counts.values() if count >= uniform_k),
        "pairs_per_arm": len(kept_slot_ids),
        "eligible_slots": len(kept_slot_ids),
        "arms": arm_summaries,
        "final_tests_used": [],
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {key: manifest[key] for key in ("uniform_k", "passages", "pairs_per_arm")},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
