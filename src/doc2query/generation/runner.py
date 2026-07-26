"""One-command controlled generation over canonical inverted records."""

from __future__ import annotations

import time
from itertools import cycle
from pathlib import Path
from typing import Any

from doc2query.data.focus_labels import split_sentences
from doc2query.data.style_labels import intent_applicable
from doc2query.generation.batching import generate_text_batch
from doc2query.generation.controlled import generate_query_set
from doc2query.models.load_generator import load_generator, load_tokenizer
from doc2query.schemas import AppConfig, FocusMode, QueryControl
from doc2query.utils.records import JsonlWriter, read_records, write_json
from doc2query.utils.reproducibility import set_seed
from doc2query.utils.tracking import collect_code_provenance


def _control_matrix(config: AppConfig, passage: str) -> list[QueryControl]:
    sentence_count = len(split_sentences(passage))
    axes = cycle(
        (form, intent, mode)
        for form in config.generation.forms
        for intent in config.generation.intents
        for mode in config.generation.focus_modes
    )
    controls: list[QueryControl] = []
    for index in range(config.generation.target_query_count):
        form, intent, mode = next(axes)
        kwargs: dict[str, Any] = {}
        if mode == FocusMode.BUCKET:
            kwargs["focus_bucket"] = ("beginning", "middle", "end")[index % 3]
        elif mode in {FocusMode.MARKED_SENTENCE, FocusMode.SENTENCE_ID}:
            if sentence_count == 0:
                mode = FocusMode.NONE
            else:
                kwargs["focus_sentence_id"] = index % sentence_count
        applicability = intent_applicable(intent, passage)
        if applicability is False:
            continue
        controls.append(
            QueryControl(
                form=form,
                intent=intent,
                intent_applicable=applicability,
                focus_mode=mode,
                **kwargs,
            )
        )
    return controls


def run_controlled_generation(
    config: AppConfig,
    *,
    output_path: Path | None = None,
    adapter_path: Path | None = None,
) -> dict[str, Any]:
    """Generate a bounded, deduplicated query set per passage and persist provenance."""
    if config.data.input_path is None:
        raise ValueError("controlled generation requires a materialized local data.input_path")
    destination = output_path or config.run.output_dir / "generation" / "controlled.jsonl"
    if destination.exists():
        raise FileExistsError(f"generation artifact already exists: {destination}")
    records = list(read_records(config.data.input_path))
    if config.data.max_eval_examples is not None:
        records = records[: config.data.max_eval_examples]
    tokenizer = load_tokenizer(config)
    model, precision = load_generator(config, for_training=False)
    if adapter_path is not None:
        try:
            from peft import PeftModel
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("install PEFT to load a generator adapter") from exc
        adapter_loader: Any = getattr(PeftModel, "from_" + "pretrained")
        model = adapter_loader(model, adapter_path)
    model.eval()
    set_seed(config.run.seed)
    def backend(prompt: str, seed: int) -> str:
        set_seed(seed)
        prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
        if len(prompt_ids) > config.training.max_length:
            prefix = min(config.training.min_prompt_tokens, config.training.max_length)
            suffix = config.training.max_length - prefix
            prompt_ids = prompt_ids[:prefix] + (prompt_ids[-suffix:] if suffix else [])
        mode: dict[str, Any] = {
            "do_sample": config.generation.do_sample,
            "num_return_sequences": 1,
        }
        if config.generation.do_sample:
            mode.update(
                temperature=config.generation.temperature,
                top_p=config.generation.top_p,
            )
        return generate_text_batch(
            model,
            tokenizer,
            [prompt_ids],
            mode=mode,
            max_new_tokens=config.generation.max_new_tokens,
        )[0]

    started = time.perf_counter()
    attempts = duplicates = invalid = generated = exhausted_groups = 0
    with JsonlWriter(destination) as writer:
        for record_index, record in enumerate(records):
            passage = str(record.get("passage", "")).strip()
            if not passage:
                raise ValueError("generation input records require a non-empty passage")
            batch = generate_query_set(
                passage,
                _control_matrix(config, passage),
                backend,
                seed=config.run.seed + record_index * 1000,
                max_attempts_per_query=config.generation.max_attempts_per_query,
            )
            attempts += batch.attempts
            duplicates += batch.duplicate_outputs
            invalid += batch.invalid_outputs
            exhausted_groups += batch.exhausted
            for item in batch.queries:
                generated += 1
                writer.write(
                    {
                        "pair_id": record.get("pair_id"),
                        "doc_id": record.get("doc_id"),
                        "passage": passage,
                        "reference": record.get("query"),
                        "generated": item.text,
                        "control": item.control.model_dump(mode="json"),
                        "seed": item.seed,
                        "attempt": item.attempt,
                    }
                )
    elapsed = time.perf_counter() - started
    summary = {
        "experiment_id": config.run.experiment_id,
        "records": len(records),
        "generated": generated,
        "attempts": attempts,
        "duplicate_outputs": duplicates,
        "invalid_outputs": invalid,
        "exhausted_groups": exhausted_groups,
        "seconds": elapsed,
        "precision": precision.label,
        "adapter_path": str(adapter_path) if adapter_path is not None else None,
        "output_path": str(destination),
        "code": collect_code_provenance(),
    }
    write_json(destination.with_suffix(".summary.json"), summary)
    return summary
