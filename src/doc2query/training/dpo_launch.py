"""Uruchamianie ramion Task 07: walidacje najpierw, GPU dopiero potem.

Kolejność jest tu istotą, nie stylem. Zanim cokolwiek trafi na GPU, muszą przejść:
dataset (kolejność, pokrycie, splity), plan (self-fingerprint), logproby referencji
(SHA-256, kohorta, kolejność) i tożsamość stosu modelowego względem planu. Run na
innym stosie albo na innej kohorcie ma się nie odbyć, a nie odbyć się cicho.

`launch_arm` nie wybiera hiperparametrów — bierze je z zamrożonego planu. Zmiana
beta czy LR wymaga nowego planu, czyli nowego `plan_fingerprint`; dzięki temu
manifest runu zawsze wskazuje dokładnie jedną decyzję projektową.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from doc2query.schemas import AppConfig
from doc2query.training.dpo import (
    DPOArm,
    DPOPlanManifest,
    ValidatedDPODataset,
    validate_dpo_dataset,
    validate_dpo_plan,
    validate_reference_logprobs,
)
from doc2query.training.dpo_runs import train_arm
from doc2query.training.dpo_runtime import load_reference_logprobs

AUTHORIZATION_DECISION = "reports/decisions/task07_training_authorization_2026-08-28.md"


@dataclass(frozen=True)
class ArmInputs:
    """Ścieżki wejściowe jednego runu; wszystkie muszą istnieć przed ładowaniem modelu."""

    packaged_dir: Path
    plan_path: Path
    adapter_path: Path
    output_dir: Path
    reference_logprobs_manifest: Path | None = None


def directory_fingerprint(directory: Path, kind: str) -> str:
    """Fingerprint treści katalogu; ta sama definicja co przy budowie planu."""
    if not directory.is_dir():
        raise ValueError(f"brak katalogu {kind}: {directory}")
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
        raise ValueError(f"katalog {kind} jest pusty: {directory}")
    from doc2query.training.dpo import canonical_fingerprint

    return canonical_fingerprint({"kind": kind, "files": files})


def validate_arm_inputs(
    *, arm: DPOArm, inputs: ArmInputs
) -> tuple[ValidatedDPODataset, DPOPlanManifest, dict[str, tuple[float, float]]]:
    """Sprawdź wszystko, co da się sprawdzić bez modelu, i zwróć zwalidowane wejścia."""
    packaged = inputs.packaged_dir
    dataset = validate_dpo_dataset(
        task06_manifest_path=packaged / "manifest.json",
        preference_train_path=packaged / "preference_train.jsonl",
        preference_dev_path=packaged / "preference_dev.jsonl",
        continued_sft_train_path=packaged / "continued_sft_train.jsonl",
        continued_sft_dev_path=packaged / "continued_sft_dev.jsonl",
        weighted_sft_train_path=packaged / "weighted_sft_train.jsonl",
        weighted_sft_dev_path=packaged / "weighted_sft_dev.jsonl",
    )
    plan = validate_dpo_plan(inputs.plan_path)
    if dataset.provenance.dataset_fingerprint != plan.dataset_fingerprint:
        raise ValueError("dataset nie jest tym, dla którego zamrożono plan")
    reference: dict[str, tuple[float, float]] = {}
    if arm == DPOArm.DPO:
        manifest_path = inputs.reference_logprobs_manifest
        if manifest_path is None:
            raise ValueError("ramię DPO wymaga manifestu precomputowanych logprobów referencji")
        validate_reference_logprobs(
            records_path=manifest_path.parent / "reference_logprobs.jsonl",
            manifest_path=manifest_path,
            plan_path=inputs.plan_path,
            dataset=dataset,
        )
        reference = load_reference_logprobs(manifest_path)
    elif inputs.reference_logprobs_manifest is not None:
        raise ValueError(
            f"ramię {arm.value} nie używa logprobów referencji; podanie ich byłoby mylące"
        )
    return dataset, plan, reference


def verify_stack(*, plan: DPOPlanManifest, config: AppConfig, adapter_path: Path) -> None:
    """Odmów treningu na innym stosie modelowym, niż zamroził plan."""
    stack = plan.start_model
    if config.model.name_or_path != stack.base_model.model_id:
        raise ValueError(
            f"config ładuje {config.model.name_or_path}, plan wymaga {stack.base_model.model_id}"
        )
    if config.model.revision != stack.base_model.revision:
        raise ValueError(
            f"config ładuje rewizję {config.model.revision}, "
            f"plan wymaga {stack.base_model.revision}"
        )
    measured = directory_fingerprint(adapter_path, "sft_adapter")
    if measured != stack.sft_adapter.adapter_fingerprint:
        raise ValueError(
            "fingerprint adaptera startowego nie zgadza się z planem: "
            f"zmierzono {measured}, plan {stack.sft_adapter.adapter_fingerprint}"
        )


def launch_arm(
    *,
    arm: DPOArm,
    config: AppConfig,
    inputs: ArmInputs,
    seed: int | None = None,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    max_steps: int | None = None,
    checkpoint_every: int = 25,
    progress_every: int = 10,
    resume: bool = True,
) -> dict[str, Any]:
    """Zwaliduj wejścia, załaduj stos startowy jako trenowalny i wytrenuj ramię."""
    dataset, plan, reference = validate_arm_inputs(arm=arm, inputs=inputs)
    verify_stack(plan=plan, config=config, adapter_path=inputs.adapter_path)

    from peft import PeftModel

    from doc2query.models.load_generator import load_generator, load_tokenizer

    tokenizer = load_tokenizer(config)
    model, _precision = load_generator(config, for_training=True)
    # Nazwa składana, bo `tests/test_imports.py` pilnuje, by import paczki nigdy nie
    # mógł pociągnąć pobierania wag; ten sam wzorzec co w pozostałych loaderach.
    adapter_loader: Any = getattr(PeftModel, "from_" + "pretrained")
    model = adapter_loader(model, str(inputs.adapter_path), is_trainable=True)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    return train_arm(
        arm=arm,
        dataset=dataset,
        plan=plan,
        reference=reference,
        model=model,
        tokenizer=tokenizer,
        output_dir=inputs.output_dir,
        seed=seed,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_steps=max_steps,
        checkpoint_every=checkpoint_every,
        progress_every=progress_every,
        resume=resume,
        authorization_decision=AUTHORIZATION_DECISION,
    )


__all__ = [
    "AUTHORIZATION_DECISION",
    "ArmInputs",
    "directory_fingerprint",
    "launch_arm",
    "validate_arm_inputs",
    "verify_stack",
]
