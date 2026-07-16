"""Per-example completion-only weighted SFT objective."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as functional
from transformers import Trainer

from doc2query.training.data import IGNORE_INDEX


def weighted_completion_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sample_weight: torch.Tensor,
) -> torch.Tensor:
    """Mean completion NLL per example, followed by a normalized weighted mean."""
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    token_loss = functional.cross_entropy(
        shifted_logits.transpose(1, 2),
        shifted_labels,
        ignore_index=IGNORE_INDEX,
        reduction="none",
    )
    mask = shifted_labels.ne(IGNORE_INDEX)
    counts = mask.sum(dim=1).clamp_min(1)
    per_example = (token_loss * mask).sum(dim=1) / counts
    weights = sample_weight.to(device=per_example.device, dtype=per_example.dtype)
    return (per_example * weights).sum() / weights.sum().clamp_min(torch.finfo(weights.dtype).eps)


class WeightedSFTTrainer(Trainer):
    """Transformers Trainer that applies weights after completion-only token averaging."""

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        del num_items_in_batch
        sample_weight = inputs.pop("sample_weight")
        outputs = model(**inputs)
        loss = weighted_completion_loss(outputs.logits, inputs["labels"], sample_weight)
        return (loss, outputs) if return_outputs else loss
