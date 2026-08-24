#!/usr/bin/env python3
"""Memory probe DPO na prawdziwym stosie: baza 4-bit + adapter SFT, dwie fazy.

AGENTS.md §12 wymaga krótkiego memory probe przed każdym runem, a Task 07 nie ma
zmierzonego ani jednego. Skrypt mierzy **osobno** obie fazy runtime'u DPO:

1. precompute logprobów referencji (model zamrożony, `no_grad`),
2. kroki treningowe polityki (adapter trainable, gradienty i optymalizator).

Używa **syntetycznych** par o kontrolowanej długości, więc nie potrzebuje żadnych
danych preferencyjnych, nie dotyka par v2.1 ani żadnej bramki i nie jest treningiem
selekcyjnym: to pomiar wykonalności i kosztu. Nie zmienia `task07_training_authorized`.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import torch

from doc2query.config import load_config
from doc2query.models.load_generator import load_generator, load_tokenizer
from doc2query.training.dpo import PreferenceRecord
from doc2query.training.dpo_runtime import (
    load_reference_logprobs,
    precompute_reference_logprobs,
    run_dpo_steps,
)


def _synthetic_records(count: int, passage_words: int) -> list[PreferenceRecord]:
    """Pary o powtarzalnej, kontrolowanej długości promptu — mierzymy pamięć, nie jakość."""
    passage = " ".join(f"slowo{index}" for index in range(passage_words))
    rows = []
    for index in range(count):
        rows.append(
            PreferenceRecord.model_validate(
                {
                    "preference_id": f"probe-{index}",
                    "prompt": (
                        "Wygeneruj jedno polskie zapytanie wyszukiwawcze.\n\n"
                        f"Pasaz:\n{passage}\n\nZapytanie:"
                    ),
                    "chosen": f"jakie objawy wywoluje wirus numer {index}",
                    "rejected": f"ile kosztuje bilet numer {index} do miasta",
                    "score_margin": 1.0,
                    "chosen_candidate_id": f"c-{index}",
                    "rejected_candidate_id": f"r-{index}",
                    "passage_id": f"doc-{index}",
                    "passage_cluster_id": f"cluster-{index}",
                    "split": "train",
                    "provenance": {
                        "dataset_id": "task07-dpo-memory-probe",
                        "dataset_fingerprint": "0" * 64,
                        "selection_policy_id": "task07-dpo-memory-probe",
                        "selection_policy_fingerprint": "0" * 64,
                    },
                }
            )
        )
    return rows


def _reset_peak() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _peak_gib() -> float | None:
    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / 1024**3, 3)


def _load_policy(config: Any, adapter: Path, *, trainable: bool) -> Any:
    from peft import PeftModel

    model, _precision = load_generator(config, for_training=trainable)
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=trainable)
    if trainable:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/d01_4_5b_style_50k_s42.yaml"),
    )
    parser.add_argument("--adapter", type=Path, default=Path("runs/D01-4.5B-STYLE-50K-S42/adapter"))
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--passage-words", type=int, default=180)
    parser.add_argument("--max-length", type=int, action="append")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/task07/dpo_memory_probe_v1/probe.json"),
    )
    args = parser.parse_args()
    lengths = args.max_length or [512, 768]

    config = load_config(args.config)
    tokenizer = load_tokenizer(config)
    records = _synthetic_records(args.pairs, args.passage_words)
    result: dict[str, Any] = {
        "schema_version": 1,
        "contract": "task07-dpo-memory-probe-v1",
        "config": str(args.config),
        "adapter": str(args.adapter),
        "base_model": config.model.name_or_path,
        "quantization_4bit": config.quantization.load_in_4bit,
        "pairs": args.pairs,
        "passage_words": args.passage_words,
        "gpu": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        ),
        "phases": {},
        "task07_training_authorized": False,
        "final_tests_used": [],
    }

    for max_length in lengths:
        _reset_peak()
        reference_model = _load_policy(config, args.adapter, trainable=False)
        loaded_peak = _peak_gib()
        started = time.perf_counter()
        output_dir = args.output.parent / f"reference_len{max_length}"
        precompute = precompute_reference_logprobs(
            records=records,
            model=reference_model,
            tokenizer=tokenizer,
            output_dir=output_dir,
            max_length=max_length,
            dataset_fingerprint="0" * 64,
            plan_fingerprint="0" * 64,
            reference_model={
                "base": {"name": config.model.name_or_path, "revision": config.model.revision},
                "adapter": str(args.adapter),
            },
            tokenizer_fingerprint="0" * 64,
            progress_every=0,
        )
        precompute_seconds = time.perf_counter() - started
        precompute_peak = _peak_gib()
        del reference_model
        _reset_peak()

        policy = _load_policy(config, args.adapter, trainable=True)
        reference = load_reference_logprobs(Path(precompute["manifest_path"]))
        started = time.perf_counter()
        summary = run_dpo_steps(
            records=records[: args.steps],
            reference=reference,
            model=policy,
            tokenizer=tokenizer,
            beta=args.beta,
            learning_rate=1e-5,
            max_length=max_length,
            batch_size=1,
            gradient_accumulation_steps=1,
            max_steps=args.steps,
        )
        train_seconds = time.perf_counter() - started
        train_peak = _peak_gib()
        del policy
        _reset_peak()

        result["phases"][str(max_length)] = {
            "model_load_peak_gib": loaded_peak,
            "precompute_peak_gib": precompute_peak,
            "precompute_seconds_per_pair": round(precompute_seconds / max(1, args.pairs), 3),
            "precompute_prompt_truncated": precompute["prompt_truncated_count"],
            "train_peak_gib": train_peak,
            "train_seconds_per_step": round(train_seconds / max(1, summary["steps"]), 3),
            "train_steps": summary["steps"],
            "final_loss": summary["final_loss"],
            "mean_reward_accuracy": summary["mean_reward_accuracy"],
        }
        phase = result["phases"][str(max_length)]
        print(
            f"[probe] max_length={max_length} | precompute {precompute_peak} GiB, "
            f"{phase['precompute_seconds_per_pair']} s/para | trening {train_peak} GiB, "
            f"{phase['train_seconds_per_step']} s/krok",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
