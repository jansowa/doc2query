"""One-off, dev-only W05 hard-negative sensitivity contracts for Task 04 P-03."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sqlite3
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from statistics import fmean
from typing import Any, cast

import torch
import torch.nn.functional as functional
import yaml

from doc2query.config import load_config
from doc2query.evaluation.bootstrap import paired_bootstrap
from doc2query.evaluation.corpus import sha256_file
from doc2query.evaluation.embedder_probe import (
    MeanPoolEncoder,
    ProbeRecipe,
    _encode,
    _tokenize,
)
from doc2query.evaluation.retrieval import (
    aggregate_query_metrics,
    candidate_pool_metrics_from_rank,
)
from doc2query.models.load_generator import load_generator, load_tokenizer
from doc2query.models.templates import render_prompt
from doc2query.reranker.base import FrozenRerankerConfig, PairScorer
from doc2query.utils.records import JsonlWriter, read_records, write_json
from doc2query.utils.reproducibility import set_seed
from doc2query.utils.tracking import collect_code_provenance

GENERATOR_NAME = "speakleash/Bielik-1.5B-v3"
GENERATOR_REVISION = "4b25049621bf3952a1fc9314c89773102eda0333"
GENERATOR_CHECKPOINT = "runs/W05-1.5B-50K-8GB/checkpoint-3125"
CALIBRATION_FINGERPRINT = "9ee4280f18e684b0dc3bb7fd885801b5ae8821af758e2845ab349c559613b3f4"
BM25_FINGERPRINT = "e5df243227e8e877550c283e2f7c882fa931ee38d849d39e8f2e2a51dc182119"
SENSITIVITY_CONTRACT_VERSION = "p03-w05-sensitivity-v1"
ARM_NAMES = ("hn0", "hn0_filter", "hn1_bm25")
FINAL_TEST_MARKERS = ("test_native", "test_translated", "test_embedder", "final_test")


def canonical_fingerprint(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def ordered_ids_fingerprint(ids: Sequence[str]) -> str:
    return canonical_fingerprint(list(ids))


def deterministic_ids(records: Iterable[Mapping[str, Any]], limit: int, seed: int) -> list[str]:
    """Select train IDs independently of source ordering."""
    if limit < 1:
        raise ValueError("train ID limit must be positive")
    eligible: set[str] = set()
    for record in records:
        metadata = record.get("metadata")
        split = metadata.get("split") if isinstance(metadata, Mapping) else record.get("split")
        positives = record.get("positives")
        negatives = record.get("hard_negatives")
        if (
            str(split) == "train"
            and isinstance(positives, list)
            and positives
            and isinstance(negatives, list)
            and negatives
        ):
            eligible.add(str(record["example_id"]))
    ranked = sorted(
        eligible,
        key=lambda example_id: (
            hashlib.sha256(f"{seed}:{example_id}".encode()).hexdigest(),
            example_id,
        ),
    )
    if len(ranked) < limit:
        raise ValueError(f"only {len(ranked)} eligible train IDs exist; requested {limit}")
    return ranked[:limit]


def assert_no_test_ids(train_ids: Sequence[str], forbidden_ids: set[str]) -> None:
    overlap = sorted(set(train_ids) & forbidden_ids)
    if overlap:
        raise ValueError(f"P-03 train cohort contains frozen test IDs: {overlap[:3]}")


def frozen_test_ids(manifest_path: Path) -> set[str]:
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    sets = raw.get("sets")
    if not isinstance(sets, Mapping):
        raise ValueError("frozen evaluation manifest has no sets mapping")
    result: set[str] = set()
    for name, entry in sets.items():
        if not str(name).startswith("test"):
            continue
        if not isinstance(entry, Mapping):
            raise ValueError(f"invalid frozen set manifest: {name}")
        id_path = Path(str(entry["id_path"]))
        for row in read_records(id_path):
            value = row.get("example_id", row.get("case_id"))
            if value is not None:
                result.add(str(value))
    return result


def freeze_train_cohort(
    train_path: Path,
    *,
    output_path: Path,
    limit: int,
    seed: int,
    forbidden_ids: set[str],
) -> dict[str, Any]:
    selected_ids = deterministic_ids(read_records(train_path), limit, seed)
    assert_no_test_ids(selected_ids, forbidden_ids)
    fingerprint = ordered_ids_fingerprint(selected_ids)
    artifact = {
        "schema_version": 1,
        "scope": "task04-p03-train-only",
        "source_path": str(train_path),
        "source_sha256": sha256_file(train_path),
        "selection": "sha256(seed:example_id), ascending",
        "seed": seed,
        "count": len(selected_ids),
        "ordered_ids": selected_ids,
        "ordered_ids_fingerprint": fingerprint,
        "forbidden_final_test_overlap": 0,
    }
    if output_path.is_file():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing != artifact:
            raise ValueError("frozen P-03 train cohort drifted; refusing to overwrite it")
    else:
        write_json(output_path, artifact)
    return artifact


def materialize_selected_train(
    train_path: Path,
    ordered_ids: Sequence[str],
) -> list[dict[str, Any]]:
    wanted = set(ordered_ids)
    by_id = {
        str(record["example_id"]): record
        for record in read_records(train_path)
        if str(record["example_id"]) in wanted
    }
    missing = sorted(wanted - by_id.keys())
    if missing:
        raise ValueError(f"frozen train records are missing: {missing[:3]}")
    return [by_id[example_id] for example_id in ordered_ids]


def _generation_contract(raw: Mapping[str, Any], cohort_fingerprint: str) -> dict[str, Any]:
    generator = _mapping(raw, "generator")
    return {
        "contract_version": SENSITIVITY_CONTRACT_VERSION,
        "checkpoint": str(generator["checkpoint"]),
        "model_name_or_path": str(generator["model_name_or_path"]),
        "revision": str(generator["revision"]),
        "config": {
            "max_length": int(generator["max_length"]),
            "max_new_tokens": int(generator["max_new_tokens"]),
            "do_sample": bool(generator["do_sample"]),
            "num_return_sequences": int(generator["num_return_sequences"]),
        },
        "seed": int(generator["seed"]),
        "train_ids_fingerprint": cohort_fingerprint,
    }


def _open_generation_journal(path: Path, contract_fingerprint: str) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS generations ("
        "example_id TEXT PRIMARY KEY, ordinal INTEGER UNIQUE NOT NULL, payload TEXT NOT NULL)"
    )
    existing = connection.execute(
        "SELECT value FROM metadata WHERE key='contract_fingerprint'"
    ).fetchone()
    if existing is not None and str(existing[0]) != contract_fingerprint:
        connection.close()
        raise ValueError("generation resume contract mismatch")
    connection.execute(
        "INSERT OR IGNORE INTO metadata(key, value) VALUES('contract_fingerprint', ?)",
        (contract_fingerprint,),
    )
    connection.commit()
    return connection


def _write_generation_artifact(
    connection: sqlite3.Connection,
    output_path: Path,
    expected_count: int,
) -> None:
    rows = connection.execute("SELECT payload FROM generations ORDER BY ordinal").fetchall()
    if len(rows) != expected_count:
        raise RuntimeError(f"generation journal has {len(rows)} rows; expected {expected_count}")
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with JsonlWriter(temporary) as writer:
        for (payload,) in rows:
            value = json.loads(str(payload))
            if not isinstance(value, dict):
                raise ValueError("generation journal payload must be an object")
            writer.write(value)
    os.replace(temporary, output_path)


def generate_w05_queries(
    records: Sequence[dict[str, Any]],
    *,
    raw_config: Mapping[str, Any],
    cohort_fingerprint: str,
    journal_path: Path,
    output_path: Path,
    mock: bool = False,
    interrupt_after: int | None = None,
) -> dict[str, Any]:
    """Generate exactly one greedy query per frozen train record with SQLite resume."""
    generator = _mapping(raw_config, "generator")
    contract = _generation_contract(raw_config, cohort_fingerprint)
    fingerprint = canonical_fingerprint(contract)
    connection = _open_generation_journal(journal_path, fingerprint)
    completed = {
        str(row[0]) for row in connection.execute("SELECT example_id FROM generations").fetchall()
    }
    elapsed_row = connection.execute(
        "SELECT value FROM metadata WHERE key='elapsed_seconds'"
    ).fetchone()
    previous_elapsed = float(elapsed_row[0]) if elapsed_row is not None else 0.0
    peak_row = connection.execute(
        "SELECT value FROM metadata WHERE key='peak_vram_allocated_bytes'"
    ).fetchone()
    previous_peak = int(peak_row[0]) if peak_row is not None else 0
    model: Any = None
    tokenizer: Any = None
    config: Any = None
    started = time.perf_counter()
    generated_now = 0
    if not mock and len(completed) < len(records):
        config = load_config(Path(str(generator["config"])))
        if (
            config.model.name_or_path != GENERATOR_NAME
            or config.model.revision != GENERATOR_REVISION
        ):
            raise ValueError("W05 config does not pin the required generator name and revision")
        tokenizer = load_tokenizer(config)
        model, _precision = load_generator(config, for_training=False)
        from peft import PeftModel

        adapter_loader: Any = getattr(PeftModel, "from_" + "pretrained")
        model = adapter_loader(
            model,
            Path(str(generator["checkpoint"])),
            is_trainable=False,
            local_files_only=True,
        )
        model.eval()
    session_elapsed = 0.0
    session_peak = 0
    try:
        for ordinal, record in enumerate(records):
            example_id = str(record["example_id"])
            if example_id in completed:
                continue
            positives = sorted(
                cast(list[dict[str, Any]], record["positives"]),
                key=lambda value: str(value["doc_id"]),
            )
            passage = str(positives[0]["text"])
            if mock:
                generated = f"mock query {example_id}"
            else:
                prompt = render_prompt(passage, config.training.baseline)
                prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
                max_length = int(generator["max_length"])
                if len(prompt_ids) > max_length:
                    prefix = min(config.training.min_prompt_tokens, max_length)
                    suffix = max_length - prefix
                    prompt_ids = prompt_ids[:prefix] + (prompt_ids[-suffix:] if suffix else [])
                input_ids = torch.tensor([prompt_ids], dtype=torch.long)
                device = next(model.parameters()).device
                input_ids = input_ids.to(device)
                attention_mask = torch.ones_like(input_ids)
                with torch.inference_mode():
                    output = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=int(generator["max_new_tokens"]),
                        do_sample=False,
                        num_return_sequences=1,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                generated = tokenizer.decode(
                    output[0, input_ids.shape[1] :], skip_special_tokens=True
                ).strip()
            if not generated:
                raise ValueError(f"empty deterministic W05 generation: {example_id}")
            row = {
                "example_id": example_id,
                "generated": generated,
                "mode": "deterministic",
                "candidate_index": 0,
                "checkpoint": str(generator["checkpoint"]),
                "revision": str(generator["revision"]),
                "config": contract["config"],
                "seed": int(generator["seed"]),
                "fingerprint": fingerprint,
            }
            connection.execute(
                "INSERT INTO generations(example_id, ordinal, payload) VALUES(?, ?, ?)",
                (
                    example_id,
                    ordinal,
                    json.dumps(row, ensure_ascii=False, sort_keys=True),
                ),
            )
            connection.commit()
            generated_now += 1
            if interrupt_after is not None and generated_now >= interrupt_after:
                raise InterruptedError("deliberate generation interruption")
        _write_generation_artifact(connection, output_path, len(records))
    finally:
        session_elapsed = time.perf_counter() - started
        session_peak = (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() and not mock else 0
        )
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES('elapsed_seconds', ?)",
            (str(previous_elapsed + session_elapsed),),
        )
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES('peak_vram_allocated_bytes', ?)",
            (str(max(previous_peak, session_peak)),),
        )
        connection.commit()
        connection.close()
    elapsed = previous_elapsed + session_elapsed
    return {
        "status": "measured",
        "count": len(records),
        "resumed_records": len(completed),
        "generated_now": generated_now,
        "fingerprint": fingerprint,
        "elapsed_seconds": elapsed,
        "throughput_examples_per_second": len(records) / elapsed if elapsed else None,
        "peak_vram_allocated_bytes": (
            max(previous_peak, session_peak) if max(previous_peak, session_peak) else None
        ),
    }


def negative_recipe_for_arm(base: ProbeRecipe, arm: str) -> ProbeRecipe:
    if arm not in ARM_NAMES:
        raise ValueError(f"unsupported P-03 sensitivity arm: {arm}")
    negative = replace(
        base.negative_recipe,
        strategy=cast(Any, arm),
        bm25_index_fingerprint=BM25_FINGERPRINT if arm == "hn1_bm25" else None,
    )
    return replace(base, negative_recipe=negative)


def common_cohort(
    arm_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    frozen_order: Sequence[str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if set(arm_rows) != set(ARM_NAMES):
        raise ValueError("P-03 requires exactly HN0, HN0+filter and HN1 BM25")
    maps = {
        arm: {str(row["example_id"]): dict(row) for row in rows} for arm, rows in arm_rows.items()
    }
    shared = set.intersection(*(set(values) for values in maps.values()))
    ordered = [example_id for example_id in frozen_order if example_id in shared]
    if not ordered:
        raise ValueError("P-03 common legal-negative cohort is empty")
    result = {arm: [maps[arm][example_id] for example_id in ordered] for arm in ARM_NAMES}
    reference = result["hn0"]
    for arm in ARM_NAMES[1:]:
        for left, right in zip(reference, result[arm], strict=True):
            for field in ("example_id", "query", "positive"):
                if left[field] != right[field]:
                    raise ValueError(f"P-03 common cohort drift in {field} for arm {arm}")
    fingerprint = ordered_ids_fingerprint(ordered)
    return result, {
        "count": len(ordered),
        "ordered_example_ids": ordered,
        "ordered_example_ids_fingerprint": fingerprint,
        "dropped_from_frozen_count": len(frozen_order) - len(ordered),
        "drop_rate": (len(frozen_order) - len(ordered)) / len(frozen_order),
    }


def token_budget(
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    max_length: int,
    max_steps: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    """Report actual unpadded and the enforced fixed padded training budget."""
    unpadded = {
        field: sum(
            min(len(tokenizer.encode(str(row[field]), add_special_tokens=True)), max_length)
            for row in rows
        )
        for field in ("query", "positive", "negative")
    }
    return {
        "cohort_examples": len(rows),
        "seed": seed,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "max_length": max_length,
        "sequences_per_example": 3,
        "padding": "max_length",
        "tokens_per_step": batch_size * max_length * 3,
        "total_padded_tokens": max_steps * batch_size * max_length * 3,
        "cohort_unpadded_tokens": unpadded,
    }


def assert_equal_budget(contracts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    compared = (
        "seed",
        "max_steps",
        "batch_size",
        "max_length",
        "sequences_per_example",
        "padding",
        "tokens_per_step",
        "total_padded_tokens",
        "cohort_examples",
    )
    first = contracts[ARM_NAMES[0]]
    mismatches = [
        f"{arm}.{field}"
        for arm in ARM_NAMES[1:]
        for field in compared
        if contracts[arm].get(field) != first.get(field)
    ]
    if mismatches:
        raise ValueError("P-03 arm budget drift: " + ", ".join(mismatches))
    return {field: first.get(field) for field in compared}


def sensitivity_contract(
    *,
    recipe: ProbeRecipe,
    arm: str,
    cohort: Mapping[str, Any],
    generation_fingerprint: str,
    dev_fingerprint: str,
    budget: Mapping[str, Any],
) -> dict[str, Any]:
    negative = recipe.negative_recipe
    calibration = negative.load_calibration()
    if calibration is None:
        calibration = replace(negative, strategy="hn0_filter").load_calibration()
    if calibration is None:  # pragma: no cover - replacement is a filtered recipe
        raise RuntimeError("P-03 sensitivity requires pinned calibration provenance")
    return {
        "contract_version": SENSITIVITY_CONTRACT_VERSION,
        "diagnostic_scope": "W05 hard-negative recipe sensitivity; not generator comparison",
        "arm": arm,
        "hard_negative_strategy": arm,
        "probe_model_name_or_path": recipe.model_name_or_path,
        "probe_model_revision": recipe.revision,
        "probe_recipe_version": recipe.recipe_version,
        "tokenizer_name_or_path": recipe.model_name_or_path,
        "tokenizer_revision": recipe.revision,
        "max_length": recipe.max_length,
        "batch_size": recipe.batch_size,
        "max_steps": recipe.max_steps,
        "learning_rate": recipe.learning_rate,
        "warmup_ratio": recipe.warmup_ratio,
        "seed": recipe.seed,
        "loss": recipe.loss,
        "false_negative_policy": negative.false_negative_policy,
        "calibration_artifact_fingerprint": calibration.artifact_fingerprint,
        "primary_judge_name": calibration.primary_judge_name,
        "primary_judge_revision": calibration.primary_judge_revision,
        "possible_false_negative_threshold": calibration.threshold,
        "bm25_index_fingerprint": negative.bm25_index_fingerprint,
        "cohort_fingerprint": cohort["ordered_example_ids_fingerprint"],
        "cohort_count": cohort["count"],
        "generation_fingerprint": generation_fingerprint,
        "dev_fingerprint": dev_fingerprint,
        "budget": dict(budget),
        "final_tests_used": [],
    }


SENSITIVITY_ALLOWED_DRIFT = frozenset(
    {
        "arm",
        "hard_negative_strategy",
        "bm25_index_fingerprint",
        "budget.cohort_unpadded_tokens",
    }
)


def assert_sensitivity_compatible(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> dict[str, Any]:
    """Allow only negative-strategy provenance to differ between sensitivity arms."""
    keys = sorted(set(left) | set(right))
    mismatches = []
    for key in keys:
        if key in SENSITIVITY_ALLOWED_DRIFT:
            continue
        if key == "budget":
            left_budget = cast(Mapping[str, Any], left.get(key, {}))
            right_budget = cast(Mapping[str, Any], right.get(key, {}))
            for budget_key in sorted(set(left_budget) | set(right_budget)):
                dotted = f"budget.{budget_key}"
                if dotted not in SENSITIVITY_ALLOWED_DRIFT and (
                    left_budget.get(budget_key) != right_budget.get(budget_key)
                ):
                    mismatches.append(dotted)
        elif left.get(key) != right.get(key):
            mismatches.append(key)
    if mismatches:
        raise ValueError("P-03 sensitivity comparison contract drift: " + ", ".join(mismatches))
    return {
        "contract_version": left["contract_version"],
        "cohort_fingerprint": left["cohort_fingerprint"],
        "dev_fingerprint": left["dev_fingerprint"],
    }


def _checkpoint_identity(
    recipe: ProbeRecipe, contract: Mapping[str, Any], rows_fingerprint: str
) -> str:
    return canonical_fingerprint(
        {
            "recipe": asdict(recipe),
            "contract": dict(contract),
            "rows_fingerprint": rows_fingerprint,
        }
    )


def _rows_fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    return canonical_fingerprint([dict(row) for row in rows])


def _batch_indices(size: int, batch_size: int, seed: int, step: int) -> list[int]:
    batches_per_epoch = math.ceil(size / batch_size)
    epoch, offset = divmod(step, batches_per_epoch)
    generator = torch.Generator().manual_seed(seed + epoch)
    permutation = torch.randperm(size, generator=generator).tolist()
    start = offset * batch_size
    indexes = permutation[start : start + batch_size]
    if len(indexes) < batch_size:
        indexes += permutation[: batch_size - len(indexes)]
    return [int(value) for value in indexes]


def _latest_probe_checkpoint(output_dir: Path, identity: str) -> tuple[Path | None, int]:
    candidates = []
    for path in output_dir.glob("checkpoint-*"):
        try:
            step = int(path.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        state_path = path / "training_state.pt"
        identity_path = path / "identity.json"
        if state_path.is_file() and identity_path.is_file() and (path / "model").is_dir():
            stored = json.loads(identity_path.read_text(encoding="utf-8"))
            if stored.get("identity") != identity:
                raise ValueError("probe resume identity mismatch")
            candidates.append((step, path))
    if not candidates:
        return None, 0
    step, path = max(candidates)
    return path, step


def _save_probe_checkpoint(
    output_dir: Path,
    *,
    step: int,
    identity: str,
    model: MeanPoolEncoder,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    elapsed_seconds: float,
) -> None:
    target = output_dir / f"checkpoint-{step}"
    temporary = output_dir / f".checkpoint-{step}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    model.backbone.save_pretrained(temporary / "model", safe_serialization=True)
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "elapsed_seconds": elapsed_seconds,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        },
        temporary / "training_state.pt",
    )
    write_json(temporary / "identity.json", {"identity": identity})
    os.replace(temporary, target)
    checkpoints = sorted(
        (
            (int(path.name.removeprefix("checkpoint-")), path)
            for path in output_dir.glob("checkpoint-*")
            if path.is_dir()
        ),
        reverse=True,
    )
    for _old_step, old_path in checkpoints[2:]:
        shutil.rmtree(old_path)


def train_sensitivity_probe(
    rows: list[dict[str, Any]],
    *,
    recipe: ProbeRecipe,
    output_dir: Path,
    contract: Mapping[str, Any],
    checkpoint_steps: int,
) -> dict[str, Any]:
    """Train a fixed-budget probe and resume from trajectory-compatible checkpoints."""
    summary_path = output_dir / "train_summary.json"
    if summary_path.is_file() and (output_dir / "model").is_dir():
        return cast(dict[str, Any], json.loads(summary_path.read_text(encoding="utf-8")))
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_fingerprint = _rows_fingerprint(rows)
    identity = _checkpoint_identity(recipe, contract, rows_fingerprint)
    checkpoint, start_step = _latest_probe_checkpoint(output_dir, identity)
    set_seed(recipe.seed)
    from transformers import AutoTokenizer

    tokenizer_loader: Any = getattr(AutoTokenizer, "from_" + "pretrained")
    tokenizer = tokenizer_loader(
        recipe.model_name_or_path,
        revision=recipe.revision,
        trust_remote_code=False,
        local_files_only=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if checkpoint is None:
        model = MeanPoolEncoder(recipe.model_name_or_path, recipe.revision)
    else:
        model = MeanPoolEncoder(str(checkpoint / "model"), "main")
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe.learning_rate)
    warmup_steps = int(recipe.max_steps * recipe.warmup_ratio)

    def learning_rate_scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, recipe.max_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_scale)
    previous_elapsed = 0.0
    if checkpoint is not None:
        state: dict[str, Any] = torch.load(
            checkpoint / "training_state.pt", map_location=device, weights_only=False
        )
        if int(state["step"]) != start_step:
            raise ValueError("probe checkpoint step mismatch")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        previous_elapsed = float(state.get("elapsed_seconds", 0.0))
        torch.set_rng_state(state["torch_rng_state"])
        cuda_rng_states = state.get("cuda_rng_states")
        if torch.cuda.is_available() and cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)
    losses: list[float] = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    for step in range(start_step, recipe.max_steps):
        indexes = _batch_indices(len(rows), recipe.batch_size, recipe.seed, step)
        batch = [rows[index] for index in indexes]
        queries = model(
            _tokenize(
                tokenizer,
                [str(row["query"]) for row in batch],
                recipe.max_length,
                device,
                padding="max_length",
            )
        )
        positives = model(
            _tokenize(
                tokenizer,
                [str(row["positive"]) for row in batch],
                recipe.max_length,
                device,
                padding="max_length",
            )
        )
        negatives = model(
            _tokenize(
                tokenizer,
                [str(row["negative"]) for row in batch],
                recipe.max_length,
                device,
                padding="max_length",
            )
        )
        documents = torch.cat((positives, negatives), dim=0)
        logits = queries @ documents.T / 0.05
        targets = torch.arange(queries.shape[0], device=device)
        loss = functional.cross_entropy(logits, targets)
        torch.autograd.backward(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
        completed_step = step + 1
        if completed_step % checkpoint_steps == 0 and completed_step < recipe.max_steps:
            _save_probe_checkpoint(
                output_dir,
                step=completed_step,
                identity=identity,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                elapsed_seconds=previous_elapsed + time.perf_counter() - started,
            )
    model.backbone.save_pretrained(output_dir / "model", safe_serialization=True)
    tokenizer.save_pretrained(output_dir / "model")
    elapsed = time.perf_counter() - started
    total_elapsed = previous_elapsed + elapsed
    summary = {
        "schema_version": 1,
        "status": "measured",
        "sensitivity_contract": dict(contract),
        "rows_fingerprint": rows_fingerprint,
        "train_examples": len(rows),
        "steps": recipe.max_steps,
        "resumed_from_step": start_step,
        "first_session_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "elapsed_seconds_this_session": elapsed,
        "elapsed_seconds_total": total_elapsed,
        "throughput_steps_per_second": (
            recipe.max_steps / total_elapsed if total_elapsed else None
        ),
        "peak_vram_allocated_bytes": (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        ),
        "code": collect_code_provenance(),
    }
    write_json(summary_path, summary)
    for path in output_dir.glob("checkpoint-*"):
        if path.is_dir():
            shutil.rmtree(path)
    return summary


def evaluate_probe_on_dev(
    model_path: Path,
    records: Sequence[dict[str, Any]],
    *,
    recipe: ProbeRecipe,
    output_dir: Path,
    dev_fingerprint: str,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate natural queries only against their frozen dev candidate pools."""
    from transformers import AutoTokenizer

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_loader: Any = getattr(AutoTokenizer, "from_" + "pretrained")
    tokenizer = tokenizer_loader(model_path, trust_remote_code=False, local_files_only=True)
    model = MeanPoolEncoder(str(model_path), "main")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    metric_rows: list[dict[str, float | int]] = []
    started = time.perf_counter()
    latencies: list[float] = []
    with JsonlWriter(output_dir / "dev_per_query.jsonl") as writer:
        for record in records:
            positives = sorted(
                cast(list[dict[str, Any]], record["positives"]),
                key=lambda value: str(value["doc_id"]),
            )
            negatives = sorted(
                cast(list[dict[str, Any]], record["hard_negatives"]),
                key=lambda value: str(value["doc_id"]),
            )
            if not positives or len(negatives) < 10:
                raise ValueError("P-03 dev evaluation requires positive + 10 hard negatives")
            query = str(record["query"])
            documents = positives + negatives
            query_started = time.perf_counter()
            query_embedding = _encode(
                model, tokenizer, [query], max_length=recipe.max_length, device=device
            )
            document_embeddings = _encode(
                model,
                tokenizer,
                [str(document["text"]) for document in documents],
                max_length=recipe.max_length,
                device=device,
            )
            latencies.append(time.perf_counter() - query_started)
            scores = (query_embedding @ document_embeddings.T).squeeze(0)
            positive_score = max(float(scores[index]) for index in range(len(positives)))
            rank = 1 + sum(float(score) >= positive_score for score in scores[len(positives) :])
            metrics = candidate_pool_metrics_from_rank(
                rank,
                candidate_count=len(documents),
            )
            row = {"example_id": str(record["example_id"]), **metrics}
            writer.write(row)
            metric_rows.append(metrics)
    elapsed = time.perf_counter() - started
    summary = {
        "schema_version": 1,
        "status": "measured",
        "scope": "task04-p03-dev-only-sensitivity",
        "dataset_name": "dev_intrinsic_rank10",
        "test_fingerprint": dev_fingerprint,
        "query_count": len(metric_rows),
        "metrics": aggregate_query_metrics(metric_rows),
        "sensitivity_contract": dict(contract),
        "elapsed_seconds": elapsed,
        "throughput_queries_per_second": len(metric_rows) / elapsed if elapsed else None,
        "latency_seconds_per_query": fmean(latencies) if latencies else None,
        "peak_vram_allocated_bytes": (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        ),
        "final_tests_used": [],
    }
    write_json(output_dir / "dev_summary.json", summary)
    return summary


def _metric_map(path: Path, metric: str) -> dict[str, float]:
    return {
        str(row["example_id"]): float(row[metric])
        for row in read_records(path)
        if isinstance(row.get(metric), (int, float))
    }


def compare_sensitivity_arms(
    arm_dirs: Mapping[str, Path],
    *,
    output_path: Path,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    summaries = {
        arm: json.loads((path / "dev_summary.json").read_text(encoding="utf-8"))
        for arm, path in arm_dirs.items()
    }
    reference_contract = cast(Mapping[str, Any], summaries["hn0"]["sensitivity_contract"])
    compatibility = {}
    for arm in ARM_NAMES[1:]:
        compatibility[arm] = assert_sensitivity_compatible(
            reference_contract,
            cast(Mapping[str, Any], summaries[arm]["sensitivity_contract"]),
        )
    fingerprints = {summary.get("test_fingerprint") for summary in summaries.values()}
    if len(fingerprints) != 1 or not next(iter(fingerprints)):
        raise ValueError("P-03 sensitivity arms must use the same frozen dev fingerprint")
    metrics = ("pool_mrr", "pool_ndcg_at_10", "pool_hard_negative_win_rate")
    comparisons: dict[str, Any] = {}
    decisions = []
    for left, right in (("hn0", "hn0_filter"), ("hn0", "hn1_bm25"), ("hn0_filter", "hn1_bm25")):
        name = f"{right}_minus_{left}"
        comparisons[name] = {}
        for metric in metrics:
            result = paired_bootstrap(
                _metric_map(arm_dirs[left] / "dev_per_query.jsonl", metric),
                _metric_map(arm_dirs[right] / "dev_per_query.jsonl", metric),
                samples=samples,
                seed=seed,
            )
            comparisons[name][metric] = result
            if float(result["ci95_low"]) > 0 or float(result["ci95_high"]) < 0:
                decisions.append("statistically_separated")
            else:
                decisions.append("inconclusive_without_equivalence_margin")
    outcome = (
        "statistically_separated"
        if "statistically_separated" in decisions
        else "inconclusive_without_p04_equivalence_contract"
    )
    report = {
        "schema_version": 1,
        "status": "measured",
        "scope": "W05 hard-negative recipe sensitivity; not a generator comparison or final result",
        "difference": "right_minus_left",
        "frozen_dev_fingerprint": str(next(iter(fingerprints))),
        "compatibility": compatibility,
        "arms": {
            arm: {
                "metrics": summaries[arm]["metrics"],
                "throughput_queries_per_second": summaries[arm]["throughput_queries_per_second"],
                "peak_vram_allocated_bytes": summaries[arm]["peak_vram_allocated_bytes"],
            }
            for arm in ARM_NAMES
        },
        "paired_query_bootstrap": comparisons,
        "outcome": outcome,
        "recipe_selected": None,
        "requires_adr": True,
        "final_tests_used": [],
    }
    write_json(output_path, report)
    return report


def write_sensitivity_adr(path: Path, report: Mapping[str, Any]) -> None:
    outcome = str(report["outcome"])
    content = f"""# ADR: P-03 W05 hard-negative sensitivity

Status: diagnostic measurement recorded; no recipe selected

The one-off W05 comparison used only the frozen development set. Its outcome is
`{outcome}`. This is a diagnosis of the hard-negative recipe, not a generator
comparison and not a final result.

No HN recipe is selected by this ADR. P-04 remains required before an
equivalence/non-inferiority claim or any generator comparison. No native,
translated, embedder, or other final test was opened.

Machine-readable evidence: `{report.get("artifact_path", "sensitivity_report.json")}`.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"P-03 sensitivity config requires mapping: {key}")
    return value


def load_sensitivity_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("P-03 sensitivity config must be a mapping")
    generator = _mapping(raw, "generator")
    pins = _mapping(raw, "pins")
    inputs = _mapping(raw, "inputs")
    if raw.get("scope") != "task04-p03-only":
        raise ValueError("runner scope must remain task04-p03-only")
    required = {
        "model_name_or_path": GENERATOR_NAME,
        "revision": GENERATOR_REVISION,
        "checkpoint": GENERATOR_CHECKPOINT,
    }
    for field, expected in required.items():
        if str(generator.get(field)) != expected:
            raise ValueError(f"generator.{field} must be pinned to {expected}")
    if pins.get("calibration_fingerprint") != CALIBRATION_FINGERPRINT:
        raise ValueError("P-03 calibration fingerprint drift")
    if pins.get("bm25_fingerprint") != BM25_FINGERPRINT:
        raise ValueError("P-03 BM25 fingerprint drift")
    dev_subset = str(inputs.get("dev_subset", ""))
    if dev_subset != "dev_intrinsic_rank10" or any(
        marker in dev_subset.lower() for marker in FINAL_TEST_MARKERS
    ):
        raise ValueError("P-03 sensitivity evaluation must be frozen dev_intrinsic_rank10")
    return raw


def verify_project_paths(raw: Mapping[str, Any], root: Path) -> None:
    paths: list[str] = []
    generator = _mapping(raw, "generator")
    inputs = _mapping(raw, "inputs")
    probe = _mapping(raw, "probe")
    paths.extend(str(generator[key]) for key in ("checkpoint", "config"))
    paths.extend(
        str(inputs[key])
        for key in (
            "train",
            "frozen_evaluation_manifest",
            "train_corpus",
            "bm25_index",
            "calibration",
        )
    )
    paths.extend(str(probe[key]) for key in ("recipe", "primary_judge"))
    paths.extend((str(raw["output_dir"]), str(raw["report_dir"])))
    resolved_root = root.resolve()
    for value in paths:
        path = (root / value).resolve()
        if not path.is_relative_to(resolved_root):
            raise ValueError(f"P-03 path escapes the project partition: {value}")


def preflight(raw: Mapping[str, Any], root: Path, *, require_model_cache: bool) -> dict[str, Any]:
    verify_project_paths(raw, root)
    generator = _mapping(raw, "generator")
    inputs = _mapping(raw, "inputs")
    probe = _mapping(raw, "probe")
    required_paths = [
        root / str(generator["checkpoint"]),
        root / str(generator["config"]),
        root / "runs/W05-1.5B-50K-8GB/run_manifest.json",
        root / str(inputs["train"]),
        root / str(inputs["frozen_evaluation_manifest"]),
        root / str(inputs["train_corpus"]),
        root / str(inputs["bm25_index"]) / "manifest.json",
        root / str(inputs["calibration"]),
        root / str(probe["recipe"]),
        root / str(probe["primary_judge"]),
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "P-03 required project artifacts are missing: " + ", ".join(missing)
        )
    checkpoint_config = json.loads(
        (root / str(generator["checkpoint"]) / "adapter_config.json").read_text(encoding="utf-8")
    )
    if checkpoint_config.get("base_model_name_or_path") != GENERATOR_NAME:
        raise ValueError("W05 checkpoint base model does not match the pinned generator")
    run_manifest = json.loads(
        (root / "runs/W05-1.5B-50K-8GB/run_manifest.json").read_text(encoding="utf-8")
    )
    run_model = run_manifest.get("config", {}).get("model", {})
    if (
        run_model.get("name_or_path") != GENERATOR_NAME
        or run_model.get("revision") != GENERATOR_REVISION
    ):
        raise ValueError("W05 run manifest does not pin the required base revision")
    calibration_raw = json.loads((root / str(inputs["calibration"])).read_text(encoding="utf-8"))
    if calibration_raw.get("artifact_fingerprint") != CALIBRATION_FINGERPRINT:
        raise ValueError("calibration artifact fingerprint mismatch")
    bm25_raw = json.loads(
        (root / str(inputs["bm25_index"]) / "manifest.json").read_text(encoding="utf-8")
    )
    if bm25_raw.get("index_fingerprint") != BM25_FINGERPRINT:
        raise ValueError("BM25 artifact fingerprint mismatch")
    cache: dict[str, Any] = {"required": require_model_cache, "snapshots": {}}
    if require_model_cache:
        from huggingface_hub import snapshot_download

        app_config = load_config(root / str(generator["config"]))
        recipe_raw = yaml.safe_load((root / str(probe["recipe"])).read_text(encoding="utf-8"))
        if not isinstance(recipe_raw, dict):
            raise ValueError("probe recipe must be a mapping")
        recipe = ProbeRecipe.from_dict(recipe_raw)
        judge_raw = yaml.safe_load((root / str(probe["primary_judge"])).read_text(encoding="utf-8"))
        if not isinstance(judge_raw, dict):
            raise ValueError("primary judge config must be a mapping")
        judge = FrozenRerankerConfig(**judge_raw)
        models = (
            (app_config.model.name_or_path, app_config.model.revision),
            (recipe.model_name_or_path, recipe.revision),
            (judge.name_or_path, judge.revision),
        )
        for name, revision in models:
            try:
                snapshot = snapshot_download(
                    repo_id=name,
                    revision=revision,
                    local_files_only=True,
                )
            except Exception as exc:
                raise RuntimeError(
                    "P-03 BLOCKED: pinned model snapshot is not legally available in the "
                    f"project-local Hugging Face cache: {name}@{revision}. Accept any gated "
                    "terms in Hugging Face and populate HF_HOME; access checks are not bypassed."
                ) from exc
            cast(dict[str, str], cache["snapshots"])[f"{name}@{revision}"] = snapshot
    return {
        "status": "ready",
        "scope": "task04-p03-only",
        "required_artifacts": len(required_paths),
        "cache": cache,
        "final_tests_used": [],
    }


class MockPrimary(PairScorer):
    name = "sdadas/polish-reranker-roberta-v3"

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        return [0.0 for _query, _passage in pairs]


def mock_smoke(output_dir: Path) -> dict[str, Any]:
    """Exercise resume, common-cohort, budget and comparator contracts without models/GPU."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    records: list[dict[str, Any]] = [
        {
            "example_id": f"train-{index}",
            "query": f"natural {index}",
            "metadata": {"split": "train"},
            "positives": [{"doc_id": f"p-{index}", "text": f"positive {index}"}],
            "hard_negatives": [{"doc_id": f"n-{index}", "text": f"negative {index}"}],
        }
        for index in range(4)
    ]
    raw = {
        "generator": {
            "checkpoint": GENERATOR_CHECKPOINT,
            "model_name_or_path": GENERATOR_NAME,
            "revision": GENERATOR_REVISION,
            "max_length": 32,
            "max_new_tokens": 8,
            "do_sample": False,
            "num_return_sequences": 1,
            "seed": 42,
        }
    }
    journal = output_dir / "generation.sqlite"
    generations = output_dir / "generations.jsonl"
    try:
        generate_w05_queries(
            records,
            raw_config=raw,
            cohort_fingerprint="f" * 64,
            journal_path=journal,
            output_path=generations,
            mock=True,
            interrupt_after=2,
        )
    except InterruptedError:
        pass
    generation = generate_w05_queries(
        records,
        raw_config=raw,
        cohort_fingerprint="f" * 64,
        journal_path=journal,
        output_path=generations,
        mock=True,
    )
    rows: list[dict[str, Any]] = [
        {
            "example_id": record["example_id"],
            "query": f"mock query {record['example_id']}",
            "positive": record["positives"][0]["text"],
            "negative": record["hard_negatives"][0]["text"],
        }
        for record in records
    ]
    arms, cohort = common_cohort(
        {arm: rows for arm in ARM_NAMES},
        [r["example_id"] for r in records],
    )
    budgets = {
        arm: {
            "cohort_examples": len(arms[arm]),
            "seed": 42,
            "max_steps": 2,
            "batch_size": 2,
            "max_length": 8,
            "sequences_per_example": 3,
            "padding": "max_length",
            "tokens_per_step": 48,
            "total_padded_tokens": 96,
            "cohort_unpadded_tokens": {},
        }
        for arm in ARM_NAMES
    }
    budget = assert_equal_budget(budgets)
    report = {
        "status": "passed",
        "mock_only": True,
        "generation_resume": generation,
        "cohort": cohort,
        "budget": budget,
        "final_tests_used": [],
    }
    write_json(output_dir / "smoke_report.json", report)
    return report
