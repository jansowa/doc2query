"""One-command checkpoint generation, intrinsic scoring and report assembly."""

from __future__ import annotations

import gc
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import yaml

from doc2query.config import load_config
from doc2query.evaluation.corpus import backfill_candidate_pools, load_corpus_index
from doc2query.evaluation.datasets import evaluation_fingerprint, load_frozen_records
from doc2query.evaluation.intrinsic import evaluate_intrinsic_records
from doc2query.evaluation.report import build_generator_report
from doc2query.models.load_generator import load_generator, load_tokenizer
from doc2query.models.templates import render_prompt
from doc2query.reranker.base import FrozenRerankerConfig, PairScorer
from doc2query.reranker.load import load_frozen_reranker
from doc2query.schemas import AppConfig
from doc2query.utils.records import JsonlWriter, read_records, write_json
from doc2query.utils.reproducibility import set_seed
from doc2query.utils.tracking import collect_code_provenance

DETERMINISTIC = {
    "mode": "deterministic",
    "do_sample": False,
    "temperature": None,
    "top_p": None,
    "num_return_sequences": 1,
    "max_new_tokens": 64,
}
DIVERSE = {
    "mode": "diverse",
    "do_sample": True,
    "temperature": 0.8,
    "top_p": 0.95,
    "num_return_sequences": 4,
    "max_new_tokens": 64,
}


def _generation_id(experiment_id: str, mode: dict[str, Any]) -> str:
    payload = json.dumps(mode, sort_keys=True, separators=(",", ":"))
    suffix = hashlib.sha256(payload.encode()).hexdigest()[:10]
    return f"{experiment_id}.{mode['mode']}.{suffix}"


def _prompt_ids(tokenizer: Any, passage: str, config: AppConfig) -> list[int]:
    prompt = render_prompt(passage, config.training.baseline)
    result = list(tokenizer.encode(prompt, add_special_tokens=False))
    if len(result) <= config.training.max_length:
        return result
    prefix = min(config.training.min_prompt_tokens, config.training.max_length)
    suffix = config.training.max_length - prefix
    return result[:prefix] + (result[-suffix:] if suffix else [])


def _expand_source(record: dict[str, Any]) -> dict[str, Any]:
    positives = record.get("positives")
    negatives = record.get("hard_negatives")
    if not isinstance(positives, list) or not positives:
        raise ValueError("frozen generator record has no positive")
    if not isinstance(negatives, list) or len(negatives) < 10:
        raise ValueError("frozen generator panel must contain at least 10 hard negatives")
    positive = sorted(positives, key=lambda value: str(value["doc_id"]))[0]
    return {
        "example_id": str(record["example_id"]),
        "positive": positive,
        "hard_negatives": negatives,
        "positive_count": len(positives),
        "reference": str(record["query"]),
        "metadata": record.get("metadata", {}),
    }


def generate_evaluation_queries(
    config: AppConfig,
    records: list[dict[str, Any]],
    *,
    adapter_path: Path | None,
    output_path: Path,
    modes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"generation artifact already exists: {output_path}")
    modes = modes or [DETERMINISTIC, DIVERSE]
    set_seed(config.run.seed)
    tokenizer = load_tokenizer(config)
    model, precision = load_generator(config, for_training=False)
    if adapter_path is not None:
        from peft import PeftModel

        adapter_loader: Any = getattr(PeftModel, "from_" + "pretrained")
        model = adapter_loader(model, adapter_path, is_trainable=False)
    model.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    generation_count = 0
    with JsonlWriter(output_path) as writer, torch.inference_mode():
        for source in records:
            expanded = _expand_source(source)
            passage = str(expanded["positive"]["text"])
            prompt_ids = _prompt_ids(tokenizer, passage, config)
            encoded = torch.tensor(
                [prompt_ids], dtype=torch.long, device=next(model.parameters()).device
            )
            attention = torch.ones_like(encoded)
            for mode in modes:
                seed_offset = int(
                    hashlib.sha256(
                        f"{config.run.seed}:{expanded['example_id']}:{mode['mode']}".encode()
                    ).hexdigest()[:8],
                    16,
                )
                torch.manual_seed(seed_offset)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed_offset)
                kwargs: dict[str, Any] = {
                    "input_ids": encoded,
                    "attention_mask": attention,
                    "max_new_tokens": int(mode["max_new_tokens"]),
                    "do_sample": bool(mode["do_sample"]),
                    "num_return_sequences": int(mode["num_return_sequences"]),
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                }
                if mode["do_sample"]:
                    kwargs.update(
                        temperature=float(mode["temperature"]),
                        top_p=float(mode["top_p"]),
                    )
                generated_sequences = model.generate(**kwargs)
                generation_run_id = _generation_id(config.run.experiment_id, mode)
                for candidate_index, sequence in enumerate(generated_sequences):
                    completion_ids = sequence[encoded.shape[1] :]
                    generated = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
                    writer.write(
                        {
                            **expanded,
                            "experiment_id": config.run.experiment_id,
                            "generation_run_id": generation_run_id,
                            "evaluation_id": (
                                f"{expanded['example_id']}::{mode['mode']}::{candidate_index}"
                            ),
                            "mode": mode["mode"],
                            "candidate_index": candidate_index,
                            "generation_config": mode,
                            "generated": generated,
                        }
                    )
                    generation_count += 1
    elapsed = time.perf_counter() - started
    return {
        "status": "measured",
        "precision": precision.label,
        "source_examples": len(records),
        "generation_count": generation_count,
        "elapsed_seconds": elapsed,
        "generations_per_second": generation_count / elapsed if elapsed else None,
        "peak_vram_allocated_bytes": (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        ),
        "peak_vram_reserved_bytes": (
            torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None
        ),
        "output_path": str(output_path),
        "modes": modes,
    }


def _judge_config(path: Path, device: str | None) -> FrozenRerankerConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("judge config must be a mapping")
    if device is not None:
        raw["device"] = device
    return FrozenRerankerConfig(**raw)


def _release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _near_duplicate_sizes(dedup_map_path: Path, wanted_doc_ids: set[str]) -> dict[str, int]:
    doc_clusters: dict[str, str] = {}
    for row in read_records(dedup_map_path):
        doc_id = str(row["doc_id"])
        if doc_id in wanted_doc_ids:
            doc_clusters[doc_id] = str(row["cluster_id"])
    wanted_clusters = set(doc_clusters.values())
    counts: Counter[str] = Counter()
    for row in read_records(dedup_map_path):
        cluster = str(row["cluster_id"])
        if cluster in wanted_clusters:
            counts[cluster] += 1
    return {doc_id: counts[cluster] for doc_id, cluster in doc_clusters.items()}


def score_generation_artifact(
    generations_path: Path,
    *,
    primary_config: Path,
    shadow_config: Path | None,
    judge_device: str | None,
    output_dir: Path,
    test_fingerprint: str,
    experiment_id: str,
    corpus_index_path: Path | None = None,
) -> dict[str, Any]:
    generation_records = list(read_records(generations_path))
    dedup_map = Path("data/processed/v1/dedup_map.parquet")
    if dedup_map.is_file():
        wanted = {str(row.get("positive", {}).get("doc_id", "")) for row in generation_records}
        cluster_sizes = _near_duplicate_sizes(dedup_map, wanted)
        for row in generation_records:
            metadata = dict(row.get("metadata", {}))
            doc_id = str(row.get("positive", {}).get("doc_id", ""))
            metadata["near_duplicate_cluster_size"] = cluster_sizes.get(doc_id, "unknown")
            row["metadata"] = metadata
    primary: PairScorer = load_frozen_reranker(_judge_config(primary_config, judge_device))
    shadow: PairScorer | None = (
        load_frozen_reranker(_judge_config(shadow_config, judge_device)) if shadow_config else None
    )
    corpus_index = load_corpus_index(corpus_index_path) if corpus_index_path is not None else None
    try:
        return evaluate_intrinsic_records(
            generation_records,
            primary=primary,
            shadow=shadow,
            output_dir=output_dir,
            test_fingerprint=test_fingerprint,
            experiment_id=experiment_id,
            corpus_index=corpus_index,
        )
    finally:
        if corpus_index is not None:
            corpus_index.close()


def run_checkpoint_evaluation(
    config_path: Path,
    *,
    frozen_manifest: Path,
    subset: str,
    output_dir: Path,
    adapter_path: Path | None = None,
    primary_config: Path | None = None,
    shadow_config: Path | None = None,
    judge_device: str | None = None,
    max_examples: int | None = None,
    generations_path: Path | None = None,
    generation_only: bool = False,
    corpus_index_path: Path | None = None,
) -> dict[str, Any]:
    """Generate two decoding modes, score them, and build all cheap report artifacts."""
    config = load_config(config_path)
    test_fingerprint = evaluation_fingerprint(frozen_manifest, subset)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = load_frozen_records(frozen_manifest, subset)
    if max_examples is not None:
        selected = selected[:max_examples]
    if any(len(record.get("hard_negatives", [])) < 10 for record in selected):
        if corpus_index_path is None:
            raise ValueError(
                "short candidate pools require --corpus-index for deterministic backfill"
            )
        corpus_manifest = json.loads(
            (corpus_index_path / "manifest.json").read_text(encoding="utf-8")
        )
        selected = backfill_candidate_pools(
            selected,
            documents_path=Path(str(corpus_manifest["documents_path"])),
            corpus_fingerprint=str(corpus_manifest["document_fingerprint"]),
        )
    local_generations = generations_path or output_dir / "generations.jsonl"
    generation_report_path = output_dir / "generation_report.json"
    if generations_path is None and not local_generations.exists():
        generation_report = generate_evaluation_queries(
            config,
            selected,
            adapter_path=adapter_path or config.run.output_dir / "adapter",
            output_path=local_generations,
        )
        write_json(generation_report_path, generation_report)
        _release_cuda()
    elif generation_report_path.exists():
        generation_report = json.loads(generation_report_path.read_text(encoding="utf-8"))
    else:
        generation_report = {
            "status": "provided_existing_artifact",
            "output_path": str(local_generations),
        }
        write_json(generation_report_path, generation_report)
    run_manifest = {
        "schema_version": 1,
        "experiment_id": config.run.experiment_id,
        "test_fingerprint": test_fingerprint,
        "frozen_manifest": str(frozen_manifest),
        "frozen_subset": subset,
        "selected_examples": len(selected),
        "config_path": str(config_path),
        "config": config.model_dump(mode="json"),
        "adapter_path": str(adapter_path or config.run.output_dir / "adapter"),
        "generations_path": str(local_generations),
        "generation": generation_report,
        "primary_judge_config": str(primary_config) if primary_config else None,
        "shadow_judge_config": str(shadow_config) if shadow_config else None,
        "corpus_index": str(corpus_index_path) if corpus_index_path else None,
        "code": collect_code_provenance(),
    }
    write_json(output_dir / "evaluation_manifest.json", run_manifest)
    if generation_only:
        return run_manifest
    if primary_config is None:
        raise ValueError("primary_config is required unless --generation-only is used")
    summary = score_generation_artifact(
        local_generations,
        primary_config=primary_config,
        shadow_config=shadow_config,
        judge_device=judge_device,
        output_dir=output_dir,
        test_fingerprint=test_fingerprint,
        experiment_id=config.run.experiment_id,
        corpus_index_path=corpus_index_path,
    )
    report = build_generator_report(
        output_dir / "summary.json",
        output_dir / "per_generation.jsonl",
        markdown_path=output_dir / "report.md",
        html_path=output_dir / "report.html",
    )
    result = {"manifest": run_manifest, "summary": summary, "report": report}
    write_json(output_dir / "result.json", result)
    return result
