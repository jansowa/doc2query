"""Fail-closed post-D01 generation, comparison, and probe-input contracts."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any, cast

import torch
import yaml

from doc2query.config import load_config
from doc2query.evaluation.bootstrap import paired_bootstrap
from doc2query.evaluation.datasets import evaluation_fingerprint, load_frozen_records
from doc2query.evaluation.embedder_probe import ProbeRecipe
from doc2query.evaluation.generator import score_generation_artifact
from doc2query.evaluation.statistical_contract import StatisticalContract, build_budget_manifest
from doc2query.generation.batching import generate_text_batch
from doc2query.generation.controlled import generate_query_set
from doc2query.generation.deduplicate import query_key
from doc2query.generation.runner import _control_matrix
from doc2query.models.load_generator import load_generator, load_tokenizer
from doc2query.models.templates import normalize_completion, render_prompt
from doc2query.schemas import AppConfig, QueryControl
from doc2query.utils.records import read_durable_jsonl_prefix, read_records, write_json
from doc2query.utils.reproducibility import set_seed
from doc2query.utils.tracking import collect_code_provenance

D01_GENERATION_CONTRACT = "task05-d01-frozen-dev-generation-v1"
D01_SCORING_CONTRACT = "task05-d01-intrinsic-scoring-v1"
D01_COMPARISON_CONTRACT = "task05-d01-matched-comparison-v1"
D01_PROBE_INPUT_CONTRACT = "task05-d01-probe-input-v1"
ALLOWED_DEVELOPMENT_SUBSETS = frozenset({"dev_intrinsic_rank10"})


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_fingerprint(path: Path | None) -> str | None:
    if path is None:
        return None
    if path.is_file():
        return _file_sha256(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"adapter/model artifact is empty: {path}")
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(str(item.stat().st_size).encode())
        digest.update(b"\0")
        digest.update(_file_sha256(item).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def assert_development_subset(subset: str) -> None:
    """Prevent this post-training pipeline from opening any final-test subset."""
    if subset not in ALLOWED_DEVELOPMENT_SUBSETS:
        raise ValueError(
            "D01 post-training evaluation is restricted to frozen dev_intrinsic_rank10; "
            f"got {subset!r}"
        )


def _positive(record: Mapping[str, Any]) -> Mapping[str, Any]:
    positives = record.get("positives")
    negatives = record.get("hard_negatives")
    if not isinstance(positives, list) or not positives:
        raise ValueError("frozen D01 record requires at least one positive")
    if not isinstance(negatives, list) or len(negatives) < 10:
        raise ValueError("frozen D01 record requires at least 10 hard negatives")
    ordered = sorted(positives, key=lambda item: str(item["doc_id"]))
    return cast(Mapping[str, Any], ordered[0])


def evaluation_group_ids(records: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return the immutable passage-level order used by journals and bootstraps."""
    ids = [f"{record['example_id']}::{_positive(record)['doc_id']}" for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("frozen D01 cohort has duplicate example/positive identities")
    return ids


def _controls(config: AppConfig, passage: str) -> list[QueryControl | None]:
    if not config.generation.controlled:
        return [None] * config.generation.target_query_count
    controls = _control_matrix(config, passage)
    if len(controls) != config.generation.target_query_count:
        raise ValueError(
            "D01 control matrix must yield exactly target_query_count applicable controls; "
            f"got {len(controls)}/{config.generation.target_query_count}"
        )
    return list(controls)


def _generation_identity(
    config: AppConfig,
    records: Sequence[Mapping[str, Any]],
    *,
    config_path: Path,
    frozen_manifest: Path,
    subset: str,
    adapter_path: Path | None,
    model_path: Path | None,
) -> dict[str, Any]:
    group_ids = evaluation_group_ids(records)
    control_payload = [
        [
            item.model_dump(mode="json") if item is not None else None
            for item in _controls(config, str(_positive(row)["text"]))
        ]
        for row in records
    ]
    identity = {
        "schema_version": 1,
        "contract": D01_GENERATION_CONTRACT,
        "experiment_id": config.run.experiment_id,
        "resolved_config_sha256": _canonical_sha256(config.model_dump(mode="json")),
        "config_file_sha256": _file_sha256(config_path),
        "model": {
            "name_or_path": str(model_path)
            if model_path is not None
            else config.model.name_or_path,
            "revision": config.model.revision,
            "artifact_sha256": _artifact_fingerprint(model_path),
        },
        "adapter": {
            "path": str(adapter_path) if adapter_path is not None else None,
            "artifact_sha256": _artifact_fingerprint(adapter_path),
        },
        "cohort": {
            "frozen_manifest": str(frozen_manifest),
            "subset": subset,
            "fingerprint": evaluation_fingerprint(frozen_manifest, subset),
            "group_count": len(records),
            "group_ids_sha256": hashlib.sha256("\n".join(group_ids).encode()).hexdigest(),
        },
        "controls_sha256": _canonical_sha256(control_payload),
        "seed_contract": {
            "base_seed": config.run.seed,
            "group_stride": 1000,
            "attempt_stride": 1,
        },
        "generation": config.generation.model_dump(mode="json"),
        "final_tests_used": [],
    }
    return {**identity, "identity_sha256": _canonical_sha256(identity)}


def _archive_partial_state(output_path: Path, previous: Any, current: Any) -> Path:
    root = output_path.parent / "interrupted-generation"
    stamp = f"{_canonical_sha256(previous)[:12]}-to-{_canonical_sha256(current)[:12]}"
    destination = root / stamp
    suffix = 1
    while destination.exists():
        destination = root / f"{stamp}-{suffix}"
        suffix += 1
    destination.mkdir(parents=True)
    for path in (
        output_path.with_suffix(output_path.suffix + ".journal.jsonl"),
        output_path.with_suffix(output_path.suffix + ".identity.json"),
        output_path.with_suffix(output_path.suffix + ".summary.json"),
    ):
        if path.exists():
            os.replace(path, destination / path.name)
    write_json(
        destination / "archive_manifest.json",
        {
            "reason": "incompatible_generation_resume_identity",
            "previous_identity": previous,
            "requested_identity": current,
        },
    )
    return destination


def _append_durable(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_flatten(journal: Path, output_path: Path) -> int:
    rows = read_durable_jsonl_prefix(journal)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for group in rows:
            for query in group["queries"]:
                handle.write(json.dumps(query, ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output_path)
    return count


def _model_backend(
    config: AppConfig, *, adapter_path: Path | None, model_path: Path | None
) -> tuple[Callable[[str, int], str], str]:
    tokenizer = load_tokenizer(config)
    model, precision = load_generator(
        config,
        for_training=False,
        model_path=str(model_path) if model_path is not None else None,
    )
    if adapter_path is not None:
        try:
            from peft import PeftModel
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("install PEFT to load a D01 adapter") from exc
        loader: Any = getattr(PeftModel, "from_" + "pretrained")
        model = loader(model, adapter_path, is_trainable=False)
    model.eval()

    def backend(prompt: str, seed: int) -> str:
        # Retry/deduplication is stateful. Per-attempt seeding makes batch-size 1 the
        # trajectory-preserving choice and permits an exact passage-level restart.
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

    return backend, precision.label


def generate_frozen_dev(
    config_path: Path,
    *,
    frozen_manifest: Path,
    subset: str,
    output_path: Path,
    adapter_path: Path | None = None,
    model_path: Path | None = None,
    max_examples: int | None = None,
    backend: Callable[[str, int], str] | None = None,
    precision_label: str | None = None,
    archive_incompatible: bool = False,
    progress_every: int = 10,
) -> dict[str, Any]:
    """Generate exactly one durable passage group at a time from frozen dev."""
    assert_development_subset(subset)
    if progress_every < 1:
        raise ValueError("progress_every must be positive")
    config = load_config(config_path)
    records = load_frozen_records(frozen_manifest, subset)
    if max_examples is not None:
        if max_examples < 1:
            raise ValueError("max_examples must be positive")
        records = records[:max_examples]
    identity = _generation_identity(
        config,
        records,
        config_path=config_path,
        frozen_manifest=frozen_manifest,
        subset=subset,
        adapter_path=adapter_path,
        model_path=model_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    journal = output_path.with_suffix(output_path.suffix + ".journal.jsonl")
    identity_path = output_path.with_suffix(output_path.suffix + ".identity.json")
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    existing_identity = (
        json.loads(identity_path.read_text(encoding="utf-8")) if identity_path.exists() else None
    )
    if existing_identity is not None and existing_identity != identity:
        archived = None
        if archive_incompatible:
            archived = _archive_partial_state(output_path, existing_identity, identity)
        message = "D01 generation resume identity mismatch"
        if archived is not None:
            message += f"; incompatible state archived at {archived}"
        raise ValueError(message)
    if existing_identity is None and journal.exists() and journal.stat().st_size:
        archived = None
        if archive_incompatible:
            archived = _archive_partial_state(output_path, {"identity": "missing"}, identity)
        message = "D01 generation journal exists without identity"
        if archived is not None:
            message += f"; orphan state archived at {archived}"
        raise ValueError(message)
    if existing_identity is None:
        temporary = identity_path.with_suffix(identity_path.suffix + ".tmp")
        write_json(temporary, identity)
        os.replace(temporary, identity_path)
    groups = read_durable_jsonl_prefix(journal)
    expected_ids = evaluation_group_ids(records)
    actual_ids = [str(row.get("evaluation_group_id")) for row in groups]
    if actual_ids != expected_ids[: len(actual_ids)]:
        raise ValueError("D01 generation journal is not the exact frozen cohort prefix")
    if len(groups) > len(records):
        raise ValueError("D01 generation journal is longer than its frozen cohort")
    if output_path.exists():
        if len(groups) != len(records):
            raise ValueError("atomic D01 output exists before its journal is complete")
        if summary_path.is_file():
            existing_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if not isinstance(existing_summary, dict):
                raise ValueError("D01 generation summary must be a mapping")
            return existing_summary
    if backend is None and len(groups) < len(records):
        backend, loaded_precision = _model_backend(
            config, adapter_path=adapter_path, model_path=model_path
        )
        precision_label = loaded_precision
    if len(groups) < len(records):
        assert backend is not None
    active_backend = backend
    started = time.perf_counter()
    resumed = len(groups)
    attempts = sum(int(row["stats"]["attempts"]) for row in groups)
    duplicates = sum(int(row["stats"]["duplicate_outputs"]) for row in groups)
    invalid = sum(int(row["stats"]["invalid_outputs"]) for row in groups)
    exhausted = sum(int(bool(row["stats"]["exhausted"])) for row in groups)
    generated = sum(len(row["queries"]) for row in groups)
    for index in range(resumed, len(records)):
        record = records[index]
        positive = dict(_positive(record))
        passage = str(positive["text"]).strip()
        controls = _controls(config, passage)
        seed = config.run.seed + index * 1000
        generated_items: list[tuple[str, QueryControl | None, int, int]]
        if config.generation.controlled:
            assert active_backend is not None
            assert all(item is not None for item in controls)
            result = generate_query_set(
                passage,
                [item for item in controls if item is not None],
                active_backend,
                seed=seed,
                max_attempts_per_query=config.generation.max_attempts_per_query,
            )
            generated_items = [
                (item.text, item.control, item.seed, item.attempt) for item in result.queries
            ]
            group_stats = {
                "attempts": result.attempts,
                "duplicate_outputs": result.duplicate_outputs,
                "invalid_outputs": result.invalid_outputs,
                "exhausted": result.exhausted,
            }
        else:
            generated_items = []
            seen: set[str] = set()
            invalid_count = duplicate_count = attempt_count = 0
            for candidate_index in range(config.generation.target_query_count):
                for attempt in range(1, config.generation.max_attempts_per_query + 1):
                    item_seed = (
                        seed
                        + candidate_index * config.generation.max_attempts_per_query
                        + attempt
                        - 1
                    )
                    attempt_count += 1
                    try:
                        assert active_backend is not None
                        text = normalize_completion(
                            active_backend(render_prompt(passage), item_seed)
                        )
                    except ValueError:
                        invalid_count += 1
                        continue
                    key = query_key(text)
                    if key in seen:
                        duplicate_count += 1
                        continue
                    seen.add(key)
                    generated_items.append((text, None, item_seed, attempt))
                    break
            group_stats = {
                "attempts": attempt_count,
                "duplicate_outputs": duplicate_count,
                "invalid_outputs": invalid_count,
                "exhausted": len(generated_items) != config.generation.target_query_count,
            }
        group_id = expected_ids[index]
        queries = []
        for candidate_index, (text, control, item_seed, attempt) in enumerate(generated_items):
            control_payload = control.model_dump(mode="json") if control is not None else None
            queries.append(
                {
                    "evaluation_id": f"{group_id}::candidate::{candidate_index}",
                    "evaluation_group_id": group_id,
                    "experiment_id": config.run.experiment_id,
                    "example_id": str(record["example_id"]),
                    "doc_id": str(positive["doc_id"]),
                    "positive": positive,
                    "positives": record["positives"],
                    "hard_negatives": record["hard_negatives"],
                    "positive_count": len(record["positives"]),
                    "reference": str(record["query"]),
                    "metadata": record.get("metadata", {}),
                    "generated": text,
                    "mode": "controlled" if control is not None else "matched_uncontrolled",
                    "candidate_index": candidate_index,
                    "control": control_payload,
                    "requested_form": control_payload.get("form") if control_payload else None,
                    "requested_intent": control_payload.get("intent") if control_payload else None,
                    "intent_applicable": (
                        control_payload.get("intent_applicable") if control_payload else None
                    ),
                    "seed": item_seed,
                    "attempt": attempt,
                    "generation_identity_sha256": identity["identity_sha256"],
                    "frozen_subset": subset,
                    "frozen_cohort_fingerprint": identity["cohort"]["fingerprint"],
                    "final_tests_used": [],
                }
            )
        group_row = {
            "evaluation_group_id": group_id,
            "group_index": index,
            "queries": queries,
            "stats": group_stats,
        }
        _append_durable(journal, group_row)
        groups.append(group_row)
        attempts += int(group_stats["attempts"])
        duplicates += int(group_stats["duplicate_outputs"])
        invalid += int(group_stats["invalid_outputs"])
        exhausted += int(bool(group_stats["exhausted"]))
        generated += len(queries)
        completed = index + 1
        if completed == len(records) or completed % progress_every == 0:
            elapsed = time.perf_counter() - started
            rate = (completed - resumed) / elapsed if elapsed > 0 else 0.0
            eta = (len(records) - completed) / rate if rate > 0 else float("inf")
            print(
                f"[D01 generation] {completed}/{len(records)} passages "
                f"queries={generated} attempts={attempts} invalid={invalid} "
                f"duplicates={duplicates} exhausted={exhausted} rate={rate:.3f}/s "
                f"eta={eta / 60:.1f} min",
                file=sys.stderr,
                flush=True,
            )
    output_count = (
        sum(len(row["queries"]) for row in groups)
        if output_path.exists()
        else _atomic_flatten(journal, output_path)
    )
    elapsed = time.perf_counter() - started
    summary = {
        "schema_version": 1,
        "status": "measured",
        "contract": D01_GENERATION_CONTRACT,
        "experiment_id": config.run.experiment_id,
        "identity": identity,
        "source_passage_count": len(records),
        "target_queries_per_passage": config.generation.target_query_count,
        "generation_count": output_count,
        "attempts": attempts,
        "invalid_outputs": invalid,
        "duplicate_outputs": duplicates,
        "exhausted_groups": exhausted,
        "effective_candidate_count_mean": output_count / len(records),
        "resumed_group_count": resumed,
        "trajectory_batch_size": 1,
        "trajectory_batching_reason": "stateful retry/deduplication with per-attempt seeds",
        "elapsed_seconds": elapsed,
        "passages_per_second": (len(records) - resumed) / elapsed if elapsed else None,
        "precision": precision_label or "injected-backend",
        "peak_vram_allocated_bytes": (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        ),
        "peak_vram_reserved_bytes": (
            torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None
        ),
        "journal_path": str(journal),
        "output_path": str(output_path),
        "final_tests_used": [],
        "code": collect_code_provenance(),
    }
    temporary_summary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    write_json(temporary_summary, summary)
    os.replace(temporary_summary, summary_path)
    return summary


def score_d01_artifact(
    generations_path: Path,
    *,
    generation_summary_path: Path,
    output_dir: Path,
    primary_config: Path,
    shadow_config: Path,
    judge_device: str | None = None,
    primary_judge_device: str | None = None,
    shadow_judge_device: str | None = None,
    corpus_index_path: Path | None = None,
    scoring_batch_size: int = 16,
    progress_every: int = 100,
    archive_incompatible: bool = False,
) -> dict[str, Any]:
    """Score a self-contained D01 artifact through the shared resumable Harness."""
    generation_summary = json.loads(generation_summary_path.read_text(encoding="utf-8"))
    if generation_summary.get("contract") != D01_GENERATION_CONTRACT:
        raise ValueError("D01 scoring requires a compatible generation summary")
    if generation_summary.get("final_tests_used") != []:
        raise ValueError("D01 generation provenance must declare final_tests_used=[]")
    identity = generation_summary.get("identity", {})
    cohort = identity.get("cohort", {}) if isinstance(identity, Mapping) else {}
    assert_development_subset(str(cohort.get("subset", "")))
    generation_rows = list(read_records(generations_path))
    expected_count = int(generation_summary.get("generation_count", 0))
    if len(generation_rows) != expected_count:
        raise ValueError("D01 generation artifact count differs from its summary")
    evaluation_ids = [str(row.get("evaluation_id", "")) for row in generation_rows]
    if not all(evaluation_ids) or len(evaluation_ids) != len(set(evaluation_ids)):
        raise ValueError("D01 generation artifact has missing or duplicate evaluation IDs")
    for row in generation_rows:
        if row.get("generation_identity_sha256") != identity.get("identity_sha256"):
            raise ValueError("D01 generation row identity differs from its summary")
        if row.get("frozen_subset") != cohort.get("subset"):
            raise ValueError("D01 generation row is not from the declared frozen subset")
        if row.get("frozen_cohort_fingerprint") != cohort.get("fingerprint"):
            raise ValueError("D01 generation row has a different frozen cohort fingerprint")
        if row.get("final_tests_used") != []:
            raise ValueError("D01 generation row must declare final_tests_used=[]")
    result = score_generation_artifact(
        generations_path,
        primary_config=primary_config,
        shadow_config=shadow_config,
        judge_device=judge_device,
        primary_judge_device=primary_judge_device,
        shadow_judge_device=shadow_judge_device,
        output_dir=output_dir,
        test_fingerprint=str(cohort["fingerprint"]),
        experiment_id=str(generation_summary["experiment_id"]),
        corpus_index_path=corpus_index_path,
        scoring_batch_size=scoring_batch_size,
        progress_every=progress_every,
        archive_incompatible_scoring=archive_incompatible,
    )
    result["contract"] = D01_SCORING_CONTRACT
    result["generation_identity_sha256"] = identity["identity_sha256"]
    result["generation_contract"] = {
        "cohort": identity["cohort"],
        "seed_contract": identity["seed_contract"],
        "max_new_tokens": identity["generation"]["max_new_tokens"],
        "do_sample": identity["generation"]["do_sample"],
        "temperature": identity["generation"]["temperature"],
        "top_p": identity["generation"]["top_p"],
        "target_query_count": identity["generation"]["target_query_count"],
        "max_attempts_per_query": identity["generation"]["max_attempts_per_query"],
    }
    result["generation_stats"] = {
        field: generation_summary[field]
        for field in (
            "attempts",
            "invalid_outputs",
            "duplicate_outputs",
            "exhausted_groups",
            "effective_candidate_count_mean",
        )
    }
    result["source_passage_count"] = generation_summary["source_passage_count"]
    result["target_queries_per_passage"] = generation_summary["target_queries_per_passage"]
    primary_payload = yaml.safe_load(primary_config.read_text(encoding="utf-8"))
    if not isinstance(primary_payload, Mapping):
        raise ValueError("primary judge config must be a mapping")
    result["primary_judge_name"] = primary_payload.get("name_or_path")
    result["primary_judge_revision"] = primary_payload.get("revision")
    result["final_tests_used"] = []
    write_json(output_dir / "summary.json", result)
    return result


def _per_passage_metric(path: Path, metric: str) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    generated: dict[str, list[str]] = defaultdict(list)
    for row in read_records(path):
        identifier = str(row["evaluation_group_id"])
        generated[identifier].append(str(row.get("generated", "")))
        value = row.get(metric)
        if isinstance(value, (int, float)):
            values[identifier].append(float(value))
    if metric == "duplicate_rate":
        return {
            identifier: 1.0 - len({query_key(text) for text in texts}) / len(texts)
            for identifier, texts in generated.items()
            if texts
        }
    return {identifier: fmean(items) for identifier, items in values.items() if items}


def _budget_from_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    passages = int(summary.get("source_passage_count", summary.get("example_count", 0)))
    count = int(summary.get("generation_count", 0))
    if passages < 1 or count < 1 or count % passages:
        raise ValueError("comparison artifacts do not define a uniform passage/K budget")
    generation = summary.get("generation_contract")
    if not isinstance(generation, Mapping) or not isinstance(generation.get("max_new_tokens"), int):
        raise ValueError("comparison artifacts lack a generation token ceiling")
    max_attempts = generation.get("max_attempts_per_query")
    if not isinstance(max_attempts, int):
        raise ValueError("comparison artifacts lack a generation attempt ceiling")
    return {
        "definition_version": "d01-generation-budget-v1",
        "completion_token_ceiling": (count * int(generation["max_new_tokens"]) * max_attempts),
        "query_count": count,
        "unique_passage_count": passages,
        "queries_per_passage": count // passages,
        "max_new_tokens": int(generation["max_new_tokens"]),
        "max_attempts_per_query": max_attempts,
    }


def assemble_matched_report(
    *,
    baseline_summary_path: Path,
    baseline_rows_path: Path,
    variant_summary_path: Path,
    variant_rows_path: Path,
    comparison_contract_path: Path,
    output_json: Path,
    output_markdown: Path,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20_260_721,
) -> dict[str, Any]:
    """Build a matched dev report; never promote an arm without extrinsic evidence."""
    baseline = json.loads(baseline_summary_path.read_text(encoding="utf-8"))
    variant = json.loads(variant_summary_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    for label, summary in (("baseline", baseline), ("variant", variant)):
        if summary.get("status") != "measured":
            errors.append(f"{label} scoring summary is not measured")
        if summary.get("final_tests_used") != []:
            errors.append(f"{label} must declare final_tests_used=[]")
    if baseline.get("test_fingerprint") != variant.get("test_fingerprint"):
        errors.append("frozen cohort fingerprints differ")
    baseline_generation = baseline.get("generation_contract")
    variant_generation = variant.get("generation_contract")
    if not isinstance(baseline_generation, Mapping) or not isinstance(variant_generation, Mapping):
        errors.append("generation budget/provenance contract is missing")
    else:
        for field in (
            "cohort",
            "seed_contract",
            "max_new_tokens",
            "do_sample",
            "temperature",
            "top_p",
            "target_query_count",
            "max_attempts_per_query",
        ):
            if baseline_generation.get(field) != variant_generation.get(field):
                errors.append(f"generation contract differs: {field}")
    baseline_budget: dict[str, Any] | None
    variant_budget: dict[str, Any] | None
    try:
        baseline_budget = _budget_from_summary(baseline)
        variant_budget = _budget_from_summary(variant)
    except ValueError as exc:
        errors.append(str(exc))
        baseline_budget = None
        variant_budget = None
    if baseline_budget != variant_budget:
        errors.append("matched comparison budget differs")
    contract = StatisticalContract.load(comparison_contract_path)
    common_metrics = (
        "pool_recall_at_1",
        "pool_recall_at_5",
        "pool_mrr",
        "pool_ndcg_at_10",
        "pool_margin",
        "content_jaccard",
        "normalized_lcs",
        "copy_density",
        "format_valid",
        "duplicate_rate",
        "judge_rank_disagreement",
        "corpus_round_trip_at_20",
        "sentence_level_source_hit",
    )
    bootstrap: dict[str, Any] = {}
    if not errors:
        for metric in common_metrics:
            left = _per_passage_metric(baseline_rows_path, metric)
            right = _per_passage_metric(variant_rows_path, metric)
            if left and right:
                bootstrap[metric] = paired_bootstrap(
                    left, right, samples=bootstrap_samples, seed=bootstrap_seed
                )
    metric_aliases = {
        "corpus_round_trip_at_20": "corpus_round_trip_at_20",
        "sentence_level_source_hit": "sentence_level_source_hit",
        "format_valid_rate": "format_valid",
    }
    guardrails: dict[str, Any] = {}
    non_inferiority = contract.payload["non_inferiority"]
    for name, rule in non_inferiority.items():
        metric = metric_aliases[str(rule["metric"])]
        measured = bootstrap.get(metric)
        if measured is None:
            guardrails[name] = {
                "status": "not_measured",
                "metric": metric,
                "margin": float(rule["margin"]),
            }
        else:
            passed = float(measured["ci95_low"]) >= -float(rule["margin"])
            guardrails[name] = {
                "status": "passed" if passed else "failed",
                "metric": metric,
                "margin": float(rule["margin"]),
                "bootstrap": measured,
            }
    guardrail_statuses = {str(value["status"]) for value in guardrails.values()}
    intrinsic_guardrail_decision = (
        "stop"
        if "failed" in guardrail_statuses
        else "continue"
        if guardrail_statuses == {"passed"}
        else "not_measured"
    )
    # Intrinsic D01 can establish guardrails, not the main probe-embedder outcome.
    decision = "stop" if intrinsic_guardrail_decision == "stop" else "not_measured"
    reasons = list(errors)
    if not reasons:
        reasons.append("comparable probe embedder result is not measured")
    budgets_matched = baseline_budget is not None and baseline_budget == variant_budget
    report: dict[str, Any] = {
        "schema_version": 1,
        "contract": D01_COMPARISON_CONTRACT,
        "status": "incomplete" if errors else "intrinsic_complete",
        "decision": decision,
        "decision_reasons": reasons,
        "baseline": {
            "experiment_id": baseline.get("experiment_id"),
            "summary": baseline,
            "comparison_budget": baseline_budget,
        },
        "variant": {
            "experiment_id": variant.get("experiment_id"),
            "summary": variant,
            "comparison_budget": variant_budget,
        },
        "statistical_contract": contract.reference(),
        "budget_difference": {
            "completion_token_ceiling": (
                int(variant_budget["completion_token_ceiling"])
                - int(baseline_budget["completion_token_ceiling"])
                if baseline_budget is not None and variant_budget is not None
                else None
            ),
            "actual_generation_attempts": (
                int(variant.get("generation_stats", {}).get("attempts", 0))
                - int(baseline.get("generation_stats", {}).get("attempts", 0))
            ),
            "matched": budgets_matched,
        },
        "paired_passage_bootstrap": bootstrap,
        "intrinsic_guardrails": guardrails,
        "intrinsic_guardrail_decision": intrinsic_guardrail_decision,
        "style_accuracy_comparison_policy": (
            "variant_only; uncontrolled baseline has no requested-style target"
        ),
        "automatic_promotion": False,
        "final_tests_used": [],
    }
    write_json(output_json, report)
    lines = [
        "# Matched post-D01 intrinsic comparison",
        "",
        f"- Status: `{report['status']}`",
        f"- Decision: `{decision}`",
        f"- Intrinsic guardrail decision: `{intrinsic_guardrail_decision}`",
        f"- Budgets matched: `{str(budgets_matched).lower()}`",
        "- Automatic promotion: `false`",
        "- Final tests used: `[]`",
        "",
        "## Decision reasons",
        "",
        *[f"- {reason}" for reason in reasons],
        "",
        "## Paired bootstrap (variant - baseline, passage/source-query unit)",
        "",
        "| Metric | Difference | 95% CI | N |",
        "|---|---:|---:|---:|",
    ]
    for metric, item in bootstrap.items():
        lines.append(
            f"| {metric} | {item['difference']:.6f} | "
            f"[{item['ci95_low']:.6f}, {item['ci95_high']:.6f}] | {item['query_count']} |"
        )
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def materialize_probe_inputs(
    *,
    generations_path: Path,
    generation_summary_path: Path,
    scoring_summary_path: Path,
    scoring_rows_path: Path,
    comparison_report_path: Path,
    probe_recipe_path: Path,
    output_path: Path,
    selection_policy: str = "all_matched_k4",
) -> dict[str, Any]:
    """Materialize later probe inputs only after all D01 dev guardrails exist."""
    generation = json.loads(generation_summary_path.read_text(encoding="utf-8"))
    scoring = json.loads(scoring_summary_path.read_text(encoding="utf-8"))
    comparison = json.loads(comparison_report_path.read_text(encoding="utf-8"))
    recipe_raw = yaml.safe_load(probe_recipe_path.read_text(encoding="utf-8"))
    if not isinstance(recipe_raw, Mapping):
        raise ValueError("probe recipe must be a mapping")
    recipe = ProbeRecipe.from_dict(recipe_raw)
    if recipe.negative_recipe.strategy != "hn0_filter":
        raise ValueError("D01 probe inputs require the frozen HN0+filter recipe")
    if recipe.negative_recipe.false_negative_policy != "drop":
        raise ValueError("D01 probe inputs require possible-false-negative policy=drop")
    calibration = recipe.negative_recipe.load_calibration()
    if calibration is None:
        raise ValueError("D01 probe inputs require a pinned dev-only calibration")
    if (
        generation.get("status") != "measured"
        or generation.get("contract") != D01_GENERATION_CONTRACT
    ):
        raise ValueError("probe materialization requires a measured D01 generation report")
    if scoring.get("status") != "measured" or scoring.get("contract") != D01_SCORING_CONTRACT:
        raise ValueError("probe materialization requires a measured D01 scoring report")
    if (
        comparison.get("contract") != D01_COMPARISON_CONTRACT
        or comparison.get("status") != "intrinsic_complete"
    ):
        raise ValueError("probe materialization requires a complete matched intrinsic report")
    if not bool(comparison.get("budget_difference", {}).get("matched")):
        raise ValueError("probe materialization requires a matched comparison budget")
    if comparison.get("intrinsic_guardrail_decision") != "continue":
        raise ValueError("probe materialization requires all P-04 intrinsic guardrails to pass")
    for payload in (generation, scoring, comparison):
        if payload.get("final_tests_used") != []:
            raise ValueError("probe materialization forbids final-test provenance")
    rows = list(read_records(generations_path))
    scored_rows = {str(row["evaluation_id"]): row for row in read_records(scoring_rows_path)}
    if scoring.get("primary_judge_name") != calibration.primary_judge_name:
        raise ValueError("D01 scoring primary judge differs from the HN filter calibration")
    if scoring.get("primary_judge_revision") != calibration.primary_judge_revision:
        raise ValueError("D01 scoring primary revision differs from the HN filter calibration")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("final_tests_used") != []:
            raise ValueError("generation row used a final-test subset")
        grouped[str(row["evaluation_group_id"])].append(row)
    expected_k = int(generation["target_queries_per_passage"])
    if expected_k != 4 or any(len(items) != expected_k for items in grouped.values()):
        raise ValueError("probe inputs require exact K=4 with no exhausted passage groups")
    if int(generation.get("exhausted_groups", 0)) or int(generation.get("invalid_outputs", 0)):
        raise ValueError("probe inputs require zero exhausted groups and invalid outputs")
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8") as handle:
        for group_id in sorted(grouped):
            for row in sorted(grouped[group_id], key=lambda item: int(item["candidate_index"])):
                scored = scored_rows.get(str(row["evaluation_id"]))
                if scored is None:
                    raise ValueError("probe inputs require a scored row for every generated query")
                negative_scores = scored.get("primary_negative_scores")
                if not isinstance(negative_scores, list) or len(negative_scores) != len(
                    row["hard_negatives"]
                ):
                    raise ValueError(
                        "probe inputs require identity-aligned primary negative scores"
                    )
                retained_negatives = [
                    negative
                    for negative, score in zip(row["hard_negatives"], negative_scores, strict=True)
                    if float(score) < calibration.threshold
                ]
                if not retained_negatives:
                    raise ValueError("HN0+filter/drop removed every negative for a probe pair")
                handle.write(
                    json.dumps(
                        {
                            "pair_id": row["evaluation_id"],
                            "query": row["generated"],
                            "positive": row["positive"],
                            "hard_negatives": retained_negatives,
                            "source_example_id": row["example_id"],
                            "source_passage_id": row["doc_id"],
                            "generator_experiment_id": row["experiment_id"],
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output_path)
    identity = generation["identity"]
    budget = build_budget_manifest(
        token_count=(
            recipe.max_steps
            * recipe.batch_size
            * recipe.max_length
            * (2 + recipe.negatives_per_example)
        ),
        pair_count=len(rows),
        unique_passage_count=len(grouped),
        queries_per_passage=expected_k,
    )
    manifest = {
        "schema_version": 1,
        "contract": D01_PROBE_INPUT_CONTRACT,
        "status": "materialized",
        "generator_experiment_id": generation["experiment_id"],
        "generator_identity_sha256": identity["identity_sha256"],
        "adapter_fingerprint": identity["adapter"]["artifact_sha256"],
        "frozen_cohort_fingerprint": identity["cohort"]["fingerprint"],
        "selection_policy": selection_policy,
        "selection_policy_fingerprint": _canonical_sha256(selection_policy),
        "negative_recipe": "HN0+filter",
        "negative_recipe_manifest": recipe.negative_recipe.manifest(calibration),
        "possible_false_negative_policy": "drop",
        "comparison_budget": budget,
        "output_path": str(output_path),
        "output_sha256": _file_sha256(output_path),
        "training_started": False,
        "final_tests_used": [],
    }
    write_json(output_path.with_suffix(output_path.suffix + ".manifest.json"), manifest)
    return manifest
