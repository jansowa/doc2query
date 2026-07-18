"""Fixed-panel generation for SFT checkpoints and prompting baselines."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from doc2query.models.templates import render_prompt
from doc2query.schemas import AppConfig
from doc2query.utils.records import JsonlWriter


def select_fixed_panel(records: list[dict[str, Any]], size: int = 100) -> list[dict[str, Any]]:
    """Select the same lexical pair-id panel independently of input ordering."""
    return sorted(records, key=lambda item: str(item.get("pair_id", item.get("doc_id", ""))))[:size]


def generate_panel(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    *,
    output_path: Path,
    config: AppConfig,
) -> dict[str, Any]:
    panel = select_fixed_panel(records)
    model.eval()
    started = time.perf_counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with JsonlWriter(output_path) as writer, torch.inference_mode():
        for record in panel:
            prompt = render_prompt(str(record["passage"]), config.training.baseline)
            prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
            if len(prompt_ids) > config.training.max_length:
                prefix = min(config.training.min_prompt_tokens, config.training.max_length)
                suffix = config.training.max_length - prefix
                prompt_ids = prompt_ids[:prefix] + (prompt_ids[-suffix:] if suffix else [])
            encoded = {
                "input_ids": torch.tensor([prompt_ids], dtype=torch.long),
                "attention_mask": torch.ones((1, len(prompt_ids)), dtype=torch.long),
            }
            device = next(model.parameters()).device
            encoded = {key: value.to(device) for key, value in encoded.items()}
            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": config.generation.max_new_tokens,
                "do_sample": config.generation.do_sample,
                "num_return_sequences": config.generation.num_return_sequences,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if config.generation.do_sample:
                generation_kwargs.update(
                    temperature=config.generation.temperature,
                    top_p=config.generation.top_p,
                )
            output = model.generate(**encoded, **generation_kwargs)
            for candidate_index, sequence in enumerate(output):
                generated_ids = sequence[encoded["input_ids"].shape[1] :]
                generated = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                writer.write(
                    {
                        "pair_id": record.get("pair_id"),
                        "doc_id": record.get("doc_id"),
                        "split": record.get("split"),
                        "passage": record.get("passage"),
                        "prompt": prompt,
                        "reference": record.get("query"),
                        "candidate_index": candidate_index,
                        "decoding": "sampling" if config.generation.do_sample else "greedy",
                        "generated": generated,
                        "format_valid": bool(generated and "\n" not in generated),
                    }
                )
    elapsed = time.perf_counter() - started
    return {
        "examples": len(panel),
        "generations": len(panel) * config.generation.num_return_sequences,
        "decoding": "sampling" if config.generation.do_sample else "greedy",
        "seconds": elapsed,
        "examples_per_second": len(panel) / elapsed if elapsed else None,
        "output_path": str(output_path),
    }
