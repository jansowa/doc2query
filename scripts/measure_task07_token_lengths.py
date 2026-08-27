#!/usr/bin/env python3
"""Policz długości tokenowe par Task 07 prawdziwym tokenizatorem (bez ładowania modelu).

Kontrakt `task07-model-free-token-lengths-v1` istniał z walidatorem, ale bez producenta,
więc launch preflight nie miał czym się nakarmić. Skrypt tokenizuje prompt, `chosen` i
`rejected`, zapisuje rekordy w kolejności datasetu i wystawia manifest z tożsamością
tokenizatora. Poza tokenizatorem nie ładuje niczego — manifest nosi
`model_loading_performed: false` i to jest prawda, nie deklaracja.

Wynik ma też zastosowanie praktyczne: percentyle długości mówią, jaki `max_length`
wybrać do DPO, zamiast zgadywać.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from doc2query.config import load_config
from doc2query.models.load_generator import load_tokenizer
from doc2query.training.dpo import (
    TokenizerIdentity,
    canonical_fingerprint,
    file_sha256,
    ordered_ids_fingerprint,
    validate_dpo_dataset,
    validate_token_length_evidence,
)
from doc2query.utils.records import JsonlWriter, read_records, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preference-train",
        type=Path,
        default=Path("artifacts/task07/handoff_v3_bottom/inputs/preference_train.jsonl"),
    )
    parser.add_argument(
        "--preference-dev",
        type=Path,
        default=Path("artifacts/task07/handoff_v3_bottom/inputs/preference_dev.jsonl"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/d01_4_5b_style_50k_s42.yaml"),
    )
    parser.add_argument(
        "--packaged-dir",
        type=Path,
        default=Path("artifacts/task07/handoff_v3_bottom/packaged"),
        help="Katalog spakowanego datasetu Task 07; służy do walidacji spójności.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/task07/handoff_v3_bottom/token_lengths"),
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"wyjście już istnieje: {args.output_dir}")

    config = load_config(args.config)
    tokenizer = load_tokenizer(config)

    def count(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    rows: list[dict[str, Any]] = []
    dataset_fingerprint = ""
    for path, split in ((args.preference_train, "train"), (args.preference_dev, "dev")):
        for record in read_records(path):
            dataset_fingerprint = str(record["provenance"]["dataset_fingerprint"])
            prompt = count(str(record["prompt"]))
            chosen = count(str(record["chosen"]))
            rejected = count(str(record["rejected"]))
            rows.append(
                {
                    "preference_id": str(record["preference_id"]),
                    "split": split,
                    "prompt_tokens": prompt,
                    "chosen_tokens": chosen,
                    "rejected_tokens": rejected,
                    "prompt_chosen_tokens": prompt + chosen,
                    "prompt_rejected_tokens": prompt + rejected,
                }
            )

    args.output_dir.mkdir(parents=True)
    records_path = args.output_dir / "token_lengths.jsonl"
    with JsonlWriter(records_path) as writer:
        for row in rows:
            writer.write(row)

    tokenizer_fingerprint = canonical_fingerprint(
        {
            "tokenizer_id": config.model.name_or_path,
            "revision": config.model.revision,
            "vocab_size": int(getattr(tokenizer, "vocab_size", 0)),
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
            "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        }
    )
    manifest: dict[str, Any] = {
        "contract": "task07-model-free-token-lengths-v1",
        "records": {
            "path": records_path.name,
            "sha256": file_sha256(records_path),
            "record_count": len(rows),
        },
        "tokenizer": {
            "tokenizer_id": config.model.name_or_path,
            "revision": config.model.revision,
            "tokenizer_fingerprint": tokenizer_fingerprint,
        },
        "dataset_fingerprint": dataset_fingerprint,
        "ordered_preference_ids_fingerprint": ordered_ids_fingerprint(
            [row["preference_id"] for row in rows]
        ),
        "model_loading_performed": False,
        "final_tests_used": [],
    }
    manifest["artifact_fingerprint"] = canonical_fingerprint(manifest)
    manifest_path = args.output_dir / "token_lengths.manifest.json"
    write_json(manifest_path, manifest)
    # Walidacja fail-closed przez zamrożony walidator: kolejność, pokrycie, splity,
    # tożsamość tokenizatora i fingerprint datasetu muszą się zgadzać z packagerem.
    packaged = args.packaged_dir
    dataset = validate_dpo_dataset(
        task06_manifest_path=packaged / "manifest.json",
        preference_train_path=packaged / "preference_train.jsonl",
        preference_dev_path=packaged / "preference_dev.jsonl",
        continued_sft_train_path=packaged / "continued_sft_train.jsonl",
        continued_sft_dev_path=packaged / "continued_sft_dev.jsonl",
        weighted_sft_train_path=packaged / "weighted_sft_train.jsonl",
        weighted_sft_dev_path=packaged / "weighted_sft_dev.jsonl",
    )
    validate_token_length_evidence(
        manifest_path,
        records_path,
        dataset,
        TokenizerIdentity.model_validate(manifest["tokenizer"]),
    )

    def percentiles(key: str) -> dict[str, int]:
        values = sorted(int(row[key]) for row in rows)
        return {
            "min": values[0],
            "p50": values[len(values) // 2],
            "p95": values[int(0.95 * (len(values) - 1))],
            "p99": values[int(0.99 * (len(values) - 1))],
            "max": values[-1],
        }

    summary = {
        "records": len(rows),
        "prompt_tokens": percentiles("prompt_tokens"),
        "chosen_tokens": percentiles("chosen_tokens"),
        "rejected_tokens": percentiles("rejected_tokens"),
        "prompt_chosen_tokens": percentiles("prompt_chosen_tokens"),
        "mean_prompt_chosen": round(
            statistics.mean(row["prompt_chosen_tokens"] for row in rows), 1
        ),
        "model_loading_performed": False,
        "final_tests_used": [],
    }
    write_json(args.output_dir / "token_length_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
