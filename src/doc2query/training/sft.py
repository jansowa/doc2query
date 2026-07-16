"""End-to-end ordinary, balanced, and weighted QLoRA SFT orchestration."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import torch
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

from doc2query.models.load_generator import (
    PrecisionSelection,
    load_generator,
    load_tokenizer,
    resolved_optimizer,
)
from doc2query.models.lora import attach_lora
from doc2query.schemas import AppConfig
from doc2query.training.data import (
    CompletionOnlyCollator,
    PreparedDatasets,
    PromptCompletionDataset,
    prepare_datasets,
)
from doc2query.training.panel import generate_panel
from doc2query.training.weighted_sft import WeightedSFTTrainer
from doc2query.utils.records import write_json
from doc2query.utils.reproducibility import set_seed
from doc2query.utils.tracking import write_run_manifest

_RESUME_IDENTITY_NAME = "resume_identity.json"
_CHECKPOINT_REQUIRED_FILES = ("trainer_state.json", "optimizer.pt", "scheduler.pt")


class AtomicCheckpointMixin:
    """Write each checkpoint under a hidden staging directory before one rename."""

    def _save_checkpoint(self, model: torch.nn.Module, trial: Any) -> None:
        if trial is not None:
            raise RuntimeError("atomic checkpointing does not support hyperparameter-search trials")
        trainer = self
        assert isinstance(trainer, Trainer)
        if trainer.args.output_dir is None:
            raise RuntimeError("Trainer output_dir is required for checkpointing")
        actual_root = Path(trainer.args.output_dir)
        staging_root = actual_root / f".checkpoint-staging-{uuid.uuid4().hex}"
        checkpoint_name = f"checkpoint-{trainer.state.global_step}"
        original_output = trainer.args.output_dir
        try:
            trainer.args.output_dir = str(staging_root)
            super()._save_checkpoint(model, trial)  # type: ignore[misc]
        finally:
            trainer.args.output_dir = original_output
        staged = staging_root / checkpoint_name
        destination = actual_root / checkpoint_name
        if not staged.is_dir():
            raise RuntimeError(f"staged checkpoint was not created: {staged}")
        if destination.exists():
            raise RuntimeError(f"refusing to replace existing checkpoint: {destination}")
        actual_root.mkdir(parents=True, exist_ok=True)
        os.replace(staged, destination)
        shutil.rmtree(staging_root, ignore_errors=True)
        if trainer.state.best_model_checkpoint and str(staging_root) in str(
            trainer.state.best_model_checkpoint
        ):
            trainer.state.best_model_checkpoint = str(destination)
            trainer.state.save_to_json(str(destination / "trainer_state.json"))
        checkpoints = sorted(
            actual_root.glob("checkpoint-*"),
            key=lambda path: int(path.name.rsplit("-", 1)[-1]),
        )
        save_limit = trainer.args.save_total_limit
        while save_limit is not None and len(checkpoints) > save_limit:
            removable = checkpoints.pop(0)
            if str(removable) != trainer.state.best_model_checkpoint:
                shutil.rmtree(removable)


class CompletionOnlySFTTrainer(AtomicCheckpointMixin, Trainer):
    """Ordinary SFT control that discards metadata but uses completion-only labels."""

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        inputs.pop("sample_weight", None)
        return super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )


class AtomicWeightedSFTTrainer(AtomicCheckpointMixin, WeightedSFTTrainer):
    """Weighted SFT with the same atomic checkpoint contract as ordinary SFT."""


def _training_arguments(
    config: AppConfig, *, has_eval: bool, precision: PrecisionSelection
) -> TrainingArguments:
    evaluation = "steps" if has_eval else "no"
    kwargs: dict[str, Any] = {
        "output_dir": str(config.run.output_dir),
        "per_device_train_batch_size": config.training.per_device_train_batch_size,
        "per_device_eval_batch_size": config.training.per_device_train_batch_size,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "learning_rate": config.training.learning_rate,
        "num_train_epochs": config.training.num_train_epochs,
        "max_steps": config.training.max_steps,
        "gradient_checkpointing": config.training.gradient_checkpointing,
        "bf16": precision.bf16,
        "fp16": precision.fp16,
        "optim": resolved_optimizer(config),
        "lr_scheduler_type": config.training.lr_scheduler_type,
        "warmup_ratio": config.training.warmup_ratio,
        "weight_decay": config.training.weight_decay,
        "logging_steps": config.training.logging_steps,
        "logging_first_step": True,
        "eval_steps": config.training.eval_steps if has_eval else None,
        "save_strategy": "steps",
        "save_steps": config.training.save_steps,
        "save_total_limit": config.training.save_total_limit,
        "load_best_model_at_end": bool(config.training.early_stopping_metric and has_eval),
        "metric_for_best_model": config.training.early_stopping_metric,
        "greater_is_better": False if config.training.early_stopping_metric else None,
        "remove_unused_columns": False,
        "dataloader_num_workers": config.training.dataloader_num_workers,
        "report_to": [],
        "seed": config.run.seed,
        "data_seed": config.run.seed,
        "use_cpu": not torch.cuda.is_available(),
        "skip_memory_metrics": False,
    }
    parameter_names = inspect.signature(TrainingArguments).parameters
    strategy_name = "eval_strategy" if "eval_strategy" in parameter_names else "evaluation_strategy"
    kwargs[strategy_name] = evaluation
    return TrainingArguments(**kwargs)


def _save_adapter_atomically(model: Any, tokenizer: Any, destination: Path) -> Path:
    if destination.exists():
        adapter_weights = destination / "adapter_model.safetensors"
        legacy_weights = destination / "adapter_model.bin"
        if (destination / "adapter_config.json").is_file() and (
            adapter_weights.is_file() or legacy_weights.is_file()
        ):
            return destination
        raise RuntimeError(f"existing adapter is incomplete: {destination}")
    staging = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
    staging.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(staging, safe_serialization=True)
    tokenizer.save_pretrained(staging)
    os.replace(staging, destination)
    return destination


def _loss_summary(log_history: list[dict[str, Any]]) -> dict[str, float | None]:
    losses = [float(item["loss"]) for item in log_history if "loss" in item]
    eval_losses = [float(item["eval_loss"]) for item in log_history if "eval_loss" in item]
    return {
        "first_train_loss": losses[0] if losses else None,
        "last_train_loss": losses[-1] if losses else None,
        "last_eval_loss": eval_losses[-1] if eval_losses else None,
    }


def _effective_config(
    config: AppConfig,
    *,
    max_steps: int | None,
    max_length: int | None,
    output_dir: Path | None,
) -> AppConfig:
    training_updates: dict[str, Any] = {}
    if max_steps is not None:
        training_updates["max_steps"] = max_steps
    if max_length is not None:
        training_updates["max_length"] = max_length
    training = config.training.model_copy(update=training_updates)
    run = config.run.model_copy(update={"output_dir": output_dir}) if output_dir else config.run
    return config.model_copy(update={"training": training, "run": run})


def _resume_identity(config: AppConfig, dataset_fingerprint: str) -> dict[str, Any]:
    training = config.training.model_dump(mode="json", exclude={"resume_if_available"})
    for field in (
        "logging_steps",
        "eval_steps",
        "save_steps",
        "save_total_limit",
        "dataloader_num_workers",
        "early_stopping_patience",
    ):
        training.pop(field, None)
    contract: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": config.run.experiment_id,
        "seed": config.run.seed,
        "dataset_fingerprint": dataset_fingerprint,
        "model": config.model.model_dump(mode="json"),
        "quantization": config.quantization.model_dump(mode="json"),
        "lora": config.lora.model_dump(mode="json"),
        "training": training,
    }
    canonical = json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {**contract, "signature": hashlib.sha256(canonical.encode()).hexdigest()}


def _write_resume_identity(path: Path, identity: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(identity, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _checkpoint_step(path: Path) -> int | None:
    prefix = "checkpoint-"
    if not path.is_dir() or not path.name.startswith(prefix):
        return None
    suffix = path.name.removeprefix(prefix)
    return int(suffix) if suffix.isdigit() else None


def checkpoint_is_complete(path: Path) -> bool:
    """Return true only for checkpoints sufficient to restore the full training state."""
    if _checkpoint_step(path) is None:
        return False
    if not all((path / name).is_file() for name in _CHECKPOINT_REQUIRED_FILES):
        return False
    adapter_present = (path / "adapter_model.safetensors").is_file() or (
        path / "adapter_model.bin"
    ).is_file()
    rng_present = any(path.glob("rng_state*.pth"))
    return adapter_present and rng_present


def find_latest_complete_checkpoint(run_dir: Path) -> Path | None:
    """Select the highest complete checkpoint and ignore interrupted staging writes."""
    candidates = [
        (step, path)
        for path in run_dir.glob("checkpoint-*")
        if (step := _checkpoint_step(path)) is not None and checkpoint_is_complete(path)
    ]
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def _resolve_resume_checkpoint(
    run_dir: Path,
    *,
    identity: dict[str, Any],
    resume_if_available: bool,
) -> Path | None:
    identity_path = run_dir / _RESUME_IDENTITY_NAME
    existing_entries = list(run_dir.iterdir()) if run_dir.exists() else []
    if not resume_if_available:
        if existing_entries:
            raise RuntimeError(
                f"output directory is not empty: {run_dir}; use --resume-if-available "
                "or choose a new output directory"
            )
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_resume_identity(identity_path, identity)
        return None

    run_dir.mkdir(parents=True, exist_ok=True)
    if identity_path.is_file():
        previous = json.loads(identity_path.read_text(encoding="utf-8"))
        if previous.get("signature") != identity["signature"]:
            raise RuntimeError(
                "resume identity mismatch: model, dataset, seed, or trajectory-defining "
                f"training settings changed for {run_dir}"
            )
    elif existing_entries:
        raise RuntimeError(
            f"cannot safely infer resume state in {run_dir}: {_RESUME_IDENTITY_NAME} is missing"
        )
    else:
        _write_resume_identity(identity_path, identity)

    checkpoint_dirs = [path for path in run_dir.glob("checkpoint-*") if path.is_dir()]
    latest = find_latest_complete_checkpoint(run_dir)
    if checkpoint_dirs and latest is None:
        raise RuntimeError(f"only incomplete checkpoints were found in {run_dir}")
    return latest


def run_sft(
    config: AppConfig,
    *,
    max_steps: int | None = None,
    max_length: int | None = None,
    output_dir: Path | None = None,
    generate_fixed_panel: bool = True,
    resume_if_available: bool = False,
) -> dict[str, Any]:
    """Train one reproducible SFT adapter and persist metrics/provenance locally."""
    effective = _effective_config(
        config, max_steps=max_steps, max_length=max_length, output_dir=output_dir
    )
    if effective.data.input_path is None:
        raise ValueError("SFT requires a materialized Task 01 input_path")
    set_seed(effective.run.seed)
    prepared: PreparedDatasets = prepare_datasets(
        effective.data.input_path,
        eval_path=effective.data.eval_path,
        train_split=effective.data.train_split,
        eval_split=effective.data.eval_split,
        baseline=effective.training.baseline,
        strategy=effective.training.strategy,
        weight_min=effective.training.weight_min,
        weight_max=effective.training.weight_max,
        seed=effective.run.seed,
        batch_size=effective.training.per_device_train_batch_size,
        max_train_examples=effective.data.max_train_examples,
        max_eval_examples=effective.data.max_eval_examples,
    )
    dataset_fingerprint = effective.data.fingerprint or prepared.fingerprint
    resume_enabled = resume_if_available or effective.training.resume_if_available
    identity = _resume_identity(effective, dataset_fingerprint)
    resume_checkpoint = _resolve_resume_checkpoint(
        effective.run.output_dir,
        identity=identity,
        resume_if_available=resume_enabled,
    )
    summary_path = effective.run.output_dir / "sft_summary.json"
    adapter_path = effective.run.output_dir / "adapter"
    if resume_enabled and summary_path.is_file() and adapter_path.is_dir():
        loaded = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise RuntimeError(f"invalid SFT summary: {summary_path}")
        completed_summary: dict[str, Any] = loaded
        completed_summary["resume"] = {
            "enabled": True,
            "checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
            "status": "already_complete",
        }
        return completed_summary
    write_json(effective.run.output_dir / "example_weights.json", prepared.weight_report)
    tokenizer = load_tokenizer(effective)
    model, precision = load_generator(effective, for_training=True)
    model, targets, parameter_stats = attach_lora(model, effective.lora)
    collator = CompletionOnlyCollator(
        tokenizer,
        max_length=effective.training.max_length,
        max_completion_tokens=effective.training.max_completion_tokens,
        min_prompt_tokens=effective.training.min_prompt_tokens,
        pad_to_max_length=effective.training.pad_to_max_length,
    )
    train_dataset = PromptCompletionDataset(prepared.train)
    eval_dataset = PromptCompletionDataset(prepared.evaluation) if prepared.evaluation else None
    arguments = _training_arguments(
        effective, has_eval=eval_dataset is not None, precision=precision
    )
    callbacks: list[Any] = []
    if effective.training.early_stopping_metric and eval_dataset is not None:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=effective.training.early_stopping_patience
            )
        )
    trainer_class = (
        AtomicWeightedSFTTrainer
        if effective.training.strategy == "weighted"
        else CompletionOnlySFTTrainer
    )
    trainer = trainer_class(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = trainer.train(
        resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None
    )
    elapsed = time.perf_counter() - started
    peak_allocated = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    peak_reserved = torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0
    adapter_path = _save_adapter_atomically(trainer.model, tokenizer, adapter_path)
    panel_report: dict[str, Any] | None = None
    if generate_fixed_panel:
        panel_source = prepared.evaluation or prepared.train
        panel_report = generate_panel(
            trainer.model,
            tokenizer,
            panel_source,
            output_path=effective.run.output_dir / "panel_generations.jsonl",
            config=effective,
        )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": effective.run.experiment_id,
        "strategy": effective.training.strategy,
        "baseline": effective.training.baseline,
        "model": effective.model.model_dump(mode="json"),
        "precision": precision.label,
        "target_modules": targets,
        "trainable_parameters": parameter_stats.trainable,
        "total_parameters": parameter_stats.total,
        "trainable_ratio": parameter_stats.ratio,
        "train_examples": len(prepared.train),
        "eval_examples": len(prepared.evaluation),
        "dataset_fingerprint": dataset_fingerprint,
        "train_metrics": result.metrics,
        "loss": _loss_summary(trainer.state.log_history),
        "global_step": trainer.state.global_step,
        "elapsed_seconds": elapsed,
        "throughput_examples_per_second": result.metrics.get("train_samples_per_second"),
        "peak_vram_allocated_bytes": peak_allocated,
        "peak_vram_reserved_bytes": peak_reserved,
        "panel": panel_report,
        "probe_embedder_score": None,
        "intrinsic_metrics": None,
        "adapter_path": str(adapter_path),
        "resume": {
            "enabled": resume_enabled,
            "checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
            "status": "resumed" if resume_checkpoint else "started_from_scratch",
        },
    }
    write_json(summary_path, summary)
    artifacts = {
        "adapter": str(adapter_path),
        "summary": str(summary_path),
    }
    if panel_report is not None:
        artifacts["panel"] = str(effective.run.output_dir / "panel_generations.jsonl")
    write_run_manifest(
        effective.run.output_dir,
        experiment_id=effective.run.experiment_id,
        seed=effective.run.seed,
        config=effective,
        dataset_fingerprint=str(summary["dataset_fingerprint"]),
        artifacts=artifacts,
    )
    return summary


def compare_run_summaries(paths: list[Path], output_path: Path) -> Path:
    """Create a compact Markdown table without inventing absent Task 04 metrics."""
    headers = [
        "experiment",
        "model",
        "strategy",
        "last loss",
        "intrinsic",
        "peak VRAM GiB",
        "examples/s",
        "probe embedder",
    ]
    rows: list[list[str]] = []
    for path in paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            [
                str(value["experiment_id"]),
                str(value["model"]["name_or_path"]),
                str(value["strategy"]),
                str(value["loss"].get("last_train_loss")),
                str(value.get("intrinsic_metrics")),
                f"{float(value.get('peak_vram_reserved_bytes', 0)) / 2**30:.3f}",
                str(value.get("throughput_examples_per_second")),
                str(value.get("probe_embedder_score")),
            ]
        )
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
