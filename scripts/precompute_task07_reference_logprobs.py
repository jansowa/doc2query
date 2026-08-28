#!/usr/bin/env python3
"""Policz logprob referencji dla kohorty treningowej Task 07 (forward-only).

To **nie jest trening**: model referencyjny (baza 4-bit + zamrożony adapter SFT
D01) liczy `log p(chosen|prompt)` i `log p(rejected|prompt)` pod `no_grad`, bez
optymalizatora i bez zapisu adaptera. `task07_training_authorized` pozostaje
`false` i ten skrypt tego nie zmienia — precompute jest wejściem, nie decyzją.

Kolejność fail-closed:

1. walidacja datasetu i planu **przed** dotknięciem GPU;
2. **przeliczenie** fingerprintów stosu modelowego z treści plików i wymóg
   równości z planem — manifest ma stwierdzać zmierzoną tożsamość, nie skopiowaną;
3. precompute wznawialny po journalu, w kolejności `preference_train`;
4. walidacja wyniku zamrożonym `validate_reference_logprobs`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from build_task07_dpo_plan_config import _directory_fingerprint
from validate_dpo_dataset import add_dataset_arguments, dataset_kwargs

from doc2query.config import load_config
from doc2query.training.dpo import (
    validate_dpo_dataset,
    validate_dpo_plan,
    validate_reference_logprobs,
)
from doc2query.training.dpo_runtime import precompute_reference_logprobs


def _verify_stack(plan: Any, base_snapshot: Path, adapter: Path, config: Any) -> None:
    """Odmów policzenia referencji na innym stosie, niż zamroził plan."""
    stack = plan.reference_model
    if config.model.name_or_path != stack.base_model.model_id:
        raise SystemExit(
            f"config ładuje {config.model.name_or_path}, plan wymaga {stack.base_model.model_id}"
        )
    if config.model.revision != stack.base_model.revision:
        raise SystemExit(
            f"config ładuje rewizję {config.model.revision}, "
            f"plan wymaga {stack.base_model.revision}"
        )
    measured_base = _directory_fingerprint(base_snapshot, "base_model_snapshot")
    if measured_base != stack.base_model.artifact_fingerprint:
        raise SystemExit(
            "fingerprint wag bazy nie zgadza się z planem: "
            f"zmierzono {measured_base}, plan {stack.base_model.artifact_fingerprint}"
        )
    measured_adapter = _directory_fingerprint(adapter, "sft_adapter")
    if measured_adapter != stack.sft_adapter.adapter_fingerprint:
        raise SystemExit(
            "fingerprint adaptera SFT nie zgadza się z planem: "
            f"zmierzono {measured_adapter}, plan {stack.sft_adapter.adapter_fingerprint}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_arguments(parser)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/d01_4_5b_style_50k_s42.yaml"),
    )
    parser.add_argument("--adapter", type=Path, default=Path("runs/D01-4.5B-STYLE-50K-S42/adapter"))
    parser.add_argument(
        "--base-snapshot",
        type=Path,
        default=Path(
            ".cache/huggingface/hub/models--speakleash--Bielik-4.5B-v3.0-Instruct"
            "/snapshots/4b1220a9d745bdd874c44347075ef25484ef322b"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()

    dataset = validate_dpo_dataset(**dataset_kwargs(args))
    plan = validate_dpo_plan(args.plan)
    if dataset.provenance.dataset_fingerprint != plan.dataset_fingerprint:
        raise SystemExit("dataset nie jest tym, dla którego zamrożono plan")
    config = load_config(args.config)
    _verify_stack(plan, args.base_snapshot, args.adapter, config)

    # Import dopiero po walidacjach: bez GPU nie ma sensu ładować torcha modelowego.
    import torch
    from peft import PeftModel

    from doc2query.models.load_generator import load_generator, load_tokenizer

    tokenizer = load_tokenizer(config)
    model, precision = load_generator(config, for_training=False)
    model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False)

    records = dataset.preference_train
    started = time.perf_counter()
    result = precompute_reference_logprobs(
        records=records,
        model=model,
        tokenizer=tokenizer,
        output_dir=args.output_dir,
        max_length=plan.max_length,
        dataset_fingerprint=plan.dataset_fingerprint,
        plan_fingerprint=plan.plan_fingerprint,
        reference_model=plan.reference_model.model_dump(mode="json"),
        tokenizer_fingerprint=plan.reference_model.tokenizer.tokenizer_fingerprint,
        progress_every=args.progress_every,
    )
    elapsed = time.perf_counter() - started

    validated = validate_reference_logprobs(
        records_path=args.output_dir / "reference_logprobs.jsonl",
        manifest_path=Path(result["manifest_path"]),
        plan_path=args.plan,
        dataset=dataset,
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "contract": "task07-reference-logprob-run-v1",
        "plan_id": plan.plan_id,
        "plan_fingerprint": plan.plan_fingerprint,
        "max_length": plan.max_length,
        "precision": str(precision),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "peak_gib": (
            round(torch.cuda.max_memory_allocated() / 1024**3, 3)
            if torch.cuda.is_available()
            else None
        ),
        "record_count": result["record_count"],
        "computed_records": result["computed_records"],
        "resumed_records": result["resumed_records"],
        "prompt_truncated_count": result["prompt_truncated_count"],
        "validated_records": len(validated),
        "seconds_total": round(elapsed, 1),
        "seconds_per_pair": round(elapsed / max(1, result["computed_records"]), 3),
        "training_started": False,
        "adapter_written": False,
        "task07_training_authorized": False,
        "final_tests_used": [],
    }
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
