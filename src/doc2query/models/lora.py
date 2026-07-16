"""Architecture-aware LoRA target discovery and coverage validation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import torch

from doc2query.schemas import LoraConfig as LoraSettings


@dataclass(frozen=True)
class TrainableParameterStats:
    trainable: int
    total: int
    ratio: float


def discover_linear_target_modules(model: torch.nn.Module) -> list[str]:
    """Discover reusable leaf names from actual linear modules in the loaded architecture."""
    targets: set[str] = set()
    for full_name, module in model.named_modules():
        if not full_name or full_name.endswith("lm_head"):
            continue
        class_name = module.__class__.__name__.lower()
        if isinstance(module, torch.nn.Linear) or class_name.startswith("linear"):
            targets.add(full_name.rsplit(".", 1)[-1])
    if not targets:
        raise RuntimeError("LoRA target discovery found no linear modules")
    return sorted(targets)


def trainable_parameter_stats(model: torch.nn.Module) -> TrainableParameterStats:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    ratio = trainable / total if total else 0.0
    return TrainableParameterStats(trainable=trainable, total=total, ratio=ratio)


def validate_lora_coverage(
    model: torch.nn.Module,
    *,
    target_modules: list[str],
    minimum_target_modules: int,
    expected_layer_patterns: list[str],
) -> TrainableParameterStats:
    """Fail early when adapter targets or expected attention/MLP layer families are absent."""
    if len(target_modules) < minimum_target_modules:
        raise RuntimeError(
            f"LoRA found {len(target_modules)} target module types; "
            f"at least {minimum_target_modules} are required"
        )
    trainable_names = [name for name, value in model.named_parameters() if value.requires_grad]
    if not trainable_names:
        raise RuntimeError("LoRA created no trainable parameters")
    missing = [
        pattern
        for pattern in expected_layer_patterns
        if not any(re.search(pattern, name, flags=re.IGNORECASE) for name in trainable_names)
    ]
    if missing:
        raise RuntimeError(f"LoRA did not cover expected layer patterns: {missing}")
    stats = trainable_parameter_stats(model)
    if not 0.0 < stats.ratio < 0.25:
        raise RuntimeError(f"implausible trainable parameter ratio for LoRA: {stats.ratio:.6f}")
    return stats


def attach_lora(
    model: Any,
    settings: LoraSettings,
) -> tuple[Any, list[str], TrainableParameterStats]:
    """Attach a causal-LM LoRA adapter using explicit or discovered targets."""
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("install the training dependency group to use LoRA") from exc
    targets = (
        discover_linear_target_modules(model)
        if settings.target_modules == "auto"
        else sorted(set(settings.target_modules))
    )
    if not targets:
        raise RuntimeError("LoRA target_modules cannot be empty")
    peft_config = LoraConfig(
        r=settings.r,
        lora_alpha=settings.alpha,
        lora_dropout=settings.dropout,
        target_modules=targets,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    adapted: Any = get_peft_model(model, peft_config)
    stats = validate_lora_coverage(
        adapted,
        target_modules=targets,
        minimum_target_modules=settings.minimum_target_modules,
        expected_layer_patterns=settings.expected_layer_patterns,
    )
    return adapted, targets, stats
