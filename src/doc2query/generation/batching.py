"""Reusable padded generation batches for causal and encoder-decoder models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import torch


class IndependentSeedSamplingProcessor:
    """Sample each batch row from its own RNG stream, then force greedy selection.

    Transformers sampling normally shares one global generator across a batch.
    Independent generators keep sampling RNG streams separate while retaining the
    standard temperature/top-p transform. Model logits can still differ slightly
    with padding or batch shape, so the batch size remains part of run identity.
    """

    def __init__(self, seeds: Sequence[int], *, temperature: float, top_p: float) -> None:
        if not seeds:
            raise ValueError("seeded sampling requires at least one seed")
        if temperature <= 0 or not 0 < top_p <= 1:
            raise ValueError("seeded sampling requires temperature > 0 and 0 < top_p <= 1")
        self.seeds = tuple(int(seed) for seed in seeds)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self._generators: list[torch.Generator] | None = None
        from transformers.generation.logits_process import (
            TemperatureLogitsWarper,
            TopPLogitsWarper,
        )

        self._temperature_warper = TemperatureLogitsWarper(self.temperature)
        self._top_p_warper = TopPLogitsWarper(self.top_p)

    def _initialize(self, device: torch.device) -> list[torch.Generator]:
        generators = [torch.Generator(device=device).manual_seed(seed) for seed in self.seeds]
        self._generators = generators
        return generators

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if scores.shape[0] != len(self.seeds):
            raise ValueError("seed count differs from generation batch rows")
        generators = self._generators or self._initialize(scores.device)
        if any(generator.device != scores.device for generator in generators):
            raise ValueError("seeded sampling processor cannot change device mid-generation")
        warped = self._temperature_warper(input_ids, scores)
        warped = self._top_p_warper(input_ids, warped)
        probabilities = torch.nn.functional.softmax(warped, dim=-1)
        sampled = torch.stack(
            [
                torch.multinomial(probabilities[row], 1, generator=generator).squeeze(0)
                for row, generator in enumerate(generators)
            ]
        )
        forced = torch.full_like(scores, -torch.inf)
        forced.scatter_(1, sampled[:, None], 0.0)
        return forced


def pad_token_sequences(
    sequences: Sequence[Sequence[int]],
    *,
    pad_token_id: int,
    device: torch.device,
    padding_side: Literal["left", "right"],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad token ID sequences and return their attention mask."""
    if not sequences:
        raise ValueError("cannot pad an empty generation batch")
    if padding_side not in {"left", "right"}:
        raise ValueError("padding_side must be left or right")
    width = max(len(sequence) for sequence in sequences)
    if width < 1:
        raise ValueError("generation prompts cannot be empty")
    input_ids = torch.full((len(sequences), width), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    for row, sequence in enumerate(sequences):
        length = len(sequence)
        start = width - length if padding_side == "left" else 0
        stop = start + length
        input_ids[row, start:stop] = torch.tensor(sequence, dtype=torch.long, device=device)
        attention_mask[row, start:stop] = 1
    return input_ids, attention_mask


def generate_text_batch(
    model: Any,
    tokenizer: Any,
    prompt_ids: Sequence[Sequence[int]],
    *,
    mode: Mapping[str, Any],
    max_new_tokens: int,
) -> list[str]:
    """Generate prompt-major decoded completions for one padded model batch."""
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("generation tokenizer requires pad_token_id and eos_token_id")
    encoder_decoder = bool(getattr(getattr(model, "config", None), "is_encoder_decoder", False))
    device = next(model.parameters()).device
    input_ids, attention_mask = pad_token_sequences(
        prompt_ids,
        pad_token_id=int(tokenizer.pad_token_id),
        device=device,
        padding_side="right" if encoder_decoder else "left",
    )
    candidates = int(mode["num_return_sequences"])
    if candidates < 1:
        raise ValueError("num_return_sequences must be positive")
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": bool(mode["do_sample"]),
        "num_return_sequences": candidates,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if mode["do_sample"]:
        kwargs.update(
            temperature=float(mode["temperature"]),
            top_p=float(mode["top_p"]),
        )
    with torch.inference_mode():
        sequences = model.generate(**kwargs)
    expected = len(prompt_ids) * candidates
    if len(sequences) != expected:
        raise RuntimeError(f"generation returned {len(sequences)}/{expected} sequences")
    prompt_width = input_ids.shape[1]
    return [
        tokenizer.decode(
            sequence if encoder_decoder else sequence[prompt_width:],
            skip_special_tokens=True,
        ).strip()
        for sequence in sequences
    ]


def generate_text_batch_seeded(
    model: Any,
    tokenizer: Any,
    prompt_ids: Sequence[Sequence[int]],
    *,
    seeds: Sequence[int],
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> list[str]:
    """Generate one sampled completion per prompt with independent RNG streams."""
    if len(prompt_ids) != len(seeds):
        raise ValueError("seeded generation requires one seed per prompt")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("generation tokenizer requires pad_token_id and eos_token_id")
    encoder_decoder = bool(getattr(getattr(model, "config", None), "is_encoder_decoder", False))
    device = next(model.parameters()).device
    input_ids, attention_mask = pad_token_sequences(
        prompt_ids,
        pad_token_id=int(tokenizer.pad_token_id),
        device=device,
        padding_side="right" if encoder_decoder else "left",
    )
    processor = IndependentSeedSamplingProcessor(
        seeds,
        temperature=temperature,
        top_p=top_p,
    )
    with torch.inference_mode():
        sequences = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            logits_processor=[processor],
        )
    if len(sequences) != len(prompt_ids):
        raise RuntimeError(
            f"seeded generation returned {len(sequences)}/{len(prompt_ids)} sequences"
        )
    prompt_width = input_ids.shape[1]
    return [
        tokenizer.decode(
            sequence if encoder_decoder else sequence[prompt_width:],
            skip_special_tokens=True,
        ).strip()
        for sequence in sequences
    ]


def is_cuda_oom(error: BaseException) -> bool:
    """Recognize Torch and wrapped CUDA out-of-memory exceptions."""
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        "out of memory" in str(error).casefold() and "cuda" in str(error).casefold()
    )
