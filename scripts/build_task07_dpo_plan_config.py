#!/usr/bin/env python3
"""Wytwórz config planu DPO (Task 07) z policzonych, a nie wpisanych, tożsamości.

`DPOPlanConfig` wymaga fingerprintów stosu modelowego, a `plan_dpo_controls`
porównuje tożsamość tokenizatora z manifestem długości tokenów. Wpisywanie tych
wartości ręcznie byłoby deklaracją bez pokrycia, dlatego ten skrypt:

1. liczy `artifact_fingerprint` bazy z **treści** snapshotu HF (sha256 każdego
   pliku, posortowane) — a nie z nazwy repozytorium i rewizji;
2. liczy `adapter_fingerprint` z treści katalogu adaptera SFT;
3. **kopiuje** tożsamość tokenizatora z zamrożonego manifestu długości tokenów,
   bo kontrakt wymaga równości, a manifest jest wcześniejszy;
4. liczy budżet tokenów i liczbę kroków optymalizatora z tego samego manifestu,
   więc `target_*` w configu są pomiarem, nie życzeniem.

Hiperparametry (beta, LR, loss) pochodzą z `tasks/07_dpo_training.md` §Konfiguracja
startowa. `max_length`/`max_prompt_length` są wyprowadzone z danych: maksimum
`prompt+chosen` to 547 tokenów, maksimum promptu 540, więc 768/704 gwarantuje
**zero truncacji** — logprob referencji policzony na obciętym prompcie byłby
logprobem innego warunkowania.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

from doc2query.training.dpo import canonical_fingerprint
from doc2query.utils.records import read_records

# tasks/07_dpo_training.md §Konfiguracja startowa
BETA = 0.1
LEARNING_RATE = 1.0e-5
PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
# Wyprowadzone z pomiaru task07-model-free-token-lengths-v1 (p99 335, max 547).
MAX_LENGTH = 768
MAX_PROMPT_LENGTH = 704


def _directory_fingerprint(directory: Path, kind: str) -> str:
    """Fingerprint treści katalogu: sha256 każdego pliku, posortowane po nazwie."""
    if not directory.is_dir():
        raise SystemExit(f"brak katalogu {kind}: {directory}")
    files: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        files[str(path.relative_to(directory))] = digest.hexdigest()
    if not files:
        raise SystemExit(f"katalog {kind} jest pusty: {directory}")
    return str(canonical_fingerprint({"kind": kind, "files": files}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-id", default="task07-dpo-plan-v3-bottom-s42")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--base-snapshot",
        type=Path,
        default=Path(
            ".cache/huggingface/hub/models--speakleash--Bielik-4.5B-v3.0-Instruct"
            "/snapshots/4b1220a9d745bdd874c44347075ef25484ef322b"
        ),
    )
    parser.add_argument("--adapter", type=Path, default=Path("runs/D01-4.5B-STYLE-50K-S42/adapter"))
    parser.add_argument("--adapter-id", default="D01-4.5B-STYLE-50K-S42")
    parser.add_argument(
        "--token-length-manifest",
        type=Path,
        default=Path(
            "artifacts/task07/handoff_v3_bottom/token_lengths/token_lengths.manifest.json"
        ),
    )
    parser.add_argument(
        "--token-length-records",
        type=Path,
        default=Path("artifacts/task07/handoff_v3_bottom/token_lengths/token_lengths.jsonl"),
    )
    parser.add_argument(
        "--weight-manifest",
        type=Path,
        default=Path("artifacts/task07/handoff_v3_bottom/inputs/weight_manifest.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"config planu już istnieje: {args.output}")

    token_manifest = json.loads(args.token_length_manifest.read_text(encoding="utf-8"))
    tokenizer = dict(token_manifest["tokenizer"])
    weight_manifest = json.loads(args.weight_manifest.read_text(encoding="utf-8"))

    train_rows = [row for row in read_records(args.token_length_records) if row["split"] == "train"]
    if not train_rows:
        raise SystemExit("kohorta treningowa jest pusta")
    dpo_pair_tokens = sum(
        int(row["prompt_chosen_tokens"]) + int(row["prompt_rejected_tokens"]) for row in train_rows
    )
    effective_batch = PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    optimizer_steps = math.ceil(len(train_rows) / effective_batch)

    base_fingerprint = _directory_fingerprint(args.base_snapshot, "base_model_snapshot")
    adapter_fingerprint = _directory_fingerprint(args.adapter, "sft_adapter")
    stack: dict[str, Any] = {
        "base_model": {
            "model_id": tokenizer["tokenizer_id"],
            "revision": tokenizer["revision"],
            "artifact_fingerprint": base_fingerprint,
        },
        "sft_adapter": {
            "adapter_id": args.adapter_id,
            "adapter_revision": str(args.adapter),
            "adapter_fingerprint": adapter_fingerprint,
            "base_model_fingerprint": base_fingerprint,
        },
        "tokenizer": tokenizer,
    }
    config: dict[str, Any] = {
        "plan_id": args.plan_id,
        "seeds": [args.seed],
        "beta": BETA,
        "loss_type": "sigmoid",
        "learning_rate": LEARNING_RATE,
        "max_length": MAX_LENGTH,
        "max_prompt_length": MAX_PROMPT_LENGTH,
        # Budżet dopasowany: jedno przejście po kohorcie w ramieniu DPO.
        "target_token_budget": dpo_pair_tokens,
        "target_optimizer_steps": optimizer_steps,
        "start_model": stack,
        # Kontrakt wymaga, by referencja była dokładnie modelem startowym.
        "reference_model": stack,
        "weight_policy_id": str(weight_manifest["weight_policy_id"]),
        "weight_policy_fingerprint": str(weight_manifest["weight_policy_fingerprint"]),
        "arms": ["dpo", "continued_sft", "score_weighted_continued_sft"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "train_examples": len(train_rows),
                "target_token_budget": dpo_pair_tokens,
                "target_optimizer_steps": optimizer_steps,
                "base_model_fingerprint": base_fingerprint,
                "adapter_fingerprint": adapter_fingerprint,
                "model_loading_performed": False,
                "training_started": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
