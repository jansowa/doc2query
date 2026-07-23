"""Prospective, development-only S00 prompting baseline."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import sqlite3
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import torch
import yaml

from doc2query.config import load_config
from doc2query.data.style_labels import label_query
from doc2query.evaluation.datasets import load_frozen_records
from doc2query.models.load_generator import load_generator, load_tokenizer
from doc2query.utils.records import JsonlWriter, write_json
from doc2query.utils.reproducibility import set_seed
from doc2query.utils.tracking import collect_code_provenance

CONTRACT_VERSION = "task03-s00-prompting-v1"
PROMPT_STRATEGIES = ("zero_shot", "few_shot")
GENERATION_TRAJECTORY_VERSION = "s00-batched-generation-v2"
# Journal created by e6ecfb3 before batching was introduced. Its completed
# rows use the same frozen cohort, prompts and decoding contract and are safe
# to retain. New journals use the trajectory version above instead of a git
# commit so execution-only optimizations do not invalidate resume state.
LEGACY_RESUME_IDENTITIES = frozenset(
    {"feefd6d189cddb0ec9f059579eaf280fc8ce83246940024fc98fe150ba1d2280"}
)


def canonical_fingerprint(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _records_fingerprint(records: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda row: str(row["example_id"])):
        payload = json.dumps(
            record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        digest.update(payload)
        digest.update(b"\n")
    return digest.hexdigest()


def _hash_order(seed: int, identifier: str) -> tuple[str, str]:
    return hashlib.sha256(f"{seed}:{identifier}".encode()).hexdigest(), identifier


def load_contract(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("S00 contract must be a mapping")
    if raw.get("contract_version") != CONTRACT_VERSION:
        raise ValueError(f"S00 contract_version must be {CONTRACT_VERSION}")
    access = raw.get("data_access")
    if not isinstance(access, dict) or access.get("development_only") is not True:
        raise ValueError("S00 contract must be development_only")
    if access.get("final_tests_used") != []:
        raise ValueError("S00 final_tests_used must remain empty")
    forbidden = tuple(str(value).casefold() for value in access.get("forbidden_markers", []))
    protected_values = (
        raw.get("frozen_manifest"),
        raw.get("source_subset"),
        raw.get("target_subset"),
        raw.get("output_dir"),
    )
    for value in protected_values:
        normalized = str(value).casefold()
        if any(marker in normalized for marker in forbidden):
            raise ValueError(f"S00 contract references forbidden final-test marker: {value}")
    if raw.get("source_subset") != "dev_intrinsic_rank10":
        raise ValueError("S00 source_subset must be frozen dev_intrinsic_rank10")
    if raw.get("target_subset") != "dev_s00_5000" or int(raw.get("target_size", 0)) != 5000:
        raise ValueError("S00 target cohort must be the prospective dev_s00_5000")
    harness = raw.get("harness")
    if not isinstance(harness, dict) or harness.get("version") != "1.1":
        raise ValueError("S00 requires Harness v1.1")
    if harness.get("comparison_contract_version") != "task04-p04-v1":
        raise ValueError("S00 must cite the frozen task04-p04-v1 contract")
    return raw


def _positive_doc_ids(record: Mapping[str, Any]) -> set[str]:
    positives = record.get("positives")
    if not isinstance(positives, list) or not positives:
        raise ValueError("S00 records require at least one positive")
    return {str(value["doc_id"]) for value in positives}


def _select_exemplars(
    records: Sequence[dict[str, Any]],
    *,
    target_ids: set[str],
    target_doc_ids: set[str],
    spec: Mapping[str, Any],
) -> list[dict[str, Any]]:
    wanted = {str(key): int(value) for key, value in spec["form_counts"].items()}
    buckets: dict[str, list[dict[str, Any]]] = {key: [] for key in wanted}
    for record in records:
        example_id = str(record["example_id"])
        if example_id in target_ids or _positive_doc_ids(record) & target_doc_ids:
            continue
        labels = label_query(str(record["query"]))
        form = labels.form.value
        if form in buckets and labels.form_confidence >= 0.9:
            buckets[form].append(record)
    selected: list[dict[str, Any]] = []
    seed = int(spec["seed"])
    for form, count in sorted(wanted.items()):
        ordered = sorted(buckets[form], key=lambda row: _hash_order(seed, str(row["example_id"])))
        if len(ordered) < count:
            raise ValueError(f"not enough disjoint S00 exemplars for form {form}")
        for record in ordered[:count]:
            selected.append({**record, "s00_form": form})
    selected.sort(key=lambda row: (str(row["s00_form"]), str(row["example_id"])))
    if len(selected) != int(spec["count"]):
        raise ValueError("S00 exemplar count does not match form_counts")
    return selected


def prepare_s00(
    contract_path: Path,
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Verify frozen dev and materialize an immutable derived 5k cohort."""
    contract = load_contract(contract_path)
    destination = output_dir or Path(str(contract["output_dir"]))
    cohort_dir = destination / "cohort"
    manifest_path = cohort_dir / "manifest.json"
    preparation_path = destination / "preparation.json"
    if manifest_path.exists() or preparation_path.exists():
        existing = json.loads(preparation_path.read_text(encoding="utf-8"))
        if existing.get("contract_fingerprint") != canonical_fingerprint(contract):
            raise ValueError("existing S00 preparation has a different contract fingerprint")
        load_frozen_records(manifest_path, str(contract["target_subset"]))
        return cast(dict[str, Any], existing)

    frozen_manifest = Path(str(contract["frozen_manifest"]))
    source = load_frozen_records(frozen_manifest, str(contract["source_subset"]))
    manifest = json.loads(frozen_manifest.read_text(encoding="utf-8"))
    source_spec = manifest["sets"][str(contract["source_subset"])]
    if source_spec["records_sha256"] != contract["source_subset_fingerprint"]:
        raise ValueError("frozen S00 dev fingerprint does not match the prospective contract")
    target = sorted(
        source,
        key=lambda row: _hash_order(int(contract["target_seed"]), str(row["example_id"])),
    )[: int(contract["target_size"])]
    target_ids = {str(row["example_id"]) for row in target}
    target_doc_ids = set().union(*(_positive_doc_ids(row) for row in target))
    exemplar_source = load_frozen_records(
        frozen_manifest, str(contract["exemplars"]["source_subset"])
    )
    exemplars = _select_exemplars(
        exemplar_source,
        target_ids=target_ids,
        target_doc_ids=target_doc_ids,
        spec=contract["exemplars"],
    )
    cohort_dir.mkdir(parents=True, exist_ok=False)
    ids_path = cohort_dir / f"{contract['target_subset']}.ids.jsonl"
    with JsonlWriter(ids_path) as writer:
        for identifier in sorted(target_ids):
            writer.write({"id": identifier})
    exemplar_path = cohort_dir / "few_shot_exemplars.jsonl"
    with JsonlWriter(exemplar_path) as writer:
        for row in exemplars:
            positive = sorted(row["positives"], key=lambda value: str(value["doc_id"]))[0]
            writer.write(
                {
                    "example_id": str(row["example_id"]),
                    "doc_id": str(positive["doc_id"]),
                    "passage": str(positive["text"]),
                    "query": str(row["query"]),
                    "form": str(row["s00_form"]),
                }
            )
    id_payload = "".join(f"{value}\n" for value in sorted(target_ids)).encode()
    derived_manifest = {
        "schema_version": 1,
        "version": CONTRACT_VERSION,
        "parent_manifest": str(frozen_manifest),
        "parent_subset": str(contract["source_subset"]),
        "parent_records_sha256": str(source_spec["records_sha256"]),
        "selection_policy": "sha256(seed:example_id), ascending",
        "seed": int(contract["target_seed"]),
        "sets": {
            str(contract["target_subset"]): {
                "name": str(contract["target_subset"]),
                "source_path": str(source_spec["source_path"]),
                "source_sha256": str(source_spec["source_sha256"]),
                "id_path": str(ids_path),
                "id_field": "example_id",
                "id_count": len(target),
                "id_list_sha256": hashlib.sha256(id_payload).hexdigest(),
                "records_sha256": _records_fingerprint(target),
                "population_count": len(source),
                "excluded_count": len(source) - len(target),
                "exclusion_reason": "prospective_hash_sample_for_s00_dev_only",
            }
        },
    }
    write_json(manifest_path, derived_manifest)
    preparation = {
        "status": "ready_not_run",
        "contract_version": CONTRACT_VERSION,
        "contract_path": str(contract_path),
        "contract_fingerprint": canonical_fingerprint(contract),
        "parent_manifest_sha256": _sha256_file(frozen_manifest),
        "target_manifest": str(manifest_path),
        "target_subset": str(contract["target_subset"]),
        "target_count": len(target),
        "target_records_sha256": _records_fingerprint(target),
        "exemplar_count": len(exemplars),
        "exemplar_form_counts": dict(Counter(str(row["s00_form"]) for row in exemplars)),
        "exemplar_ids": [str(row["example_id"]) for row in exemplars],
        "example_id_overlap": sorted(target_ids & {str(row["example_id"]) for row in exemplars}),
        "positive_doc_id_overlap": sorted(
            target_doc_ids & set().union(*(_positive_doc_ids(row) for row in exemplars))
        ),
        "final_tests_used": [],
        "code": collect_code_provenance(),
    }
    write_json(preparation_path, preparation)
    return preparation


def _prompt_parts(
    strategy: str,
    exemplars: Sequence[Mapping[str, Any]],
    *,
    max_passage_characters: int,
) -> tuple[str, str]:
    instruction = (
        "Wygeneruj jedno polskie zapytanie wyszukiwawcze, na które można odpowiedzieć "
        "wyłącznie na podstawie podanego pasażu. Nie kopiuj długich fragmentów "
        "pasażu. Zachowaj konieczne nazwy własne, liczby i terminy. Zwróć wyłącznie "
        "zapytanie, bez komentarza, numeracji i prefiksu.\n\n"
    )
    if strategy == "few_shot":
        examples = []
        for index, row in enumerate(exemplars, start=1):
            passage = str(row["passage"]).strip()[:max_passage_characters]
            examples.append(
                f"Przykład {index} (forma: {row['form']}):\nPasaż:\n{passage}\n"
                f"Zapytanie:\n{str(row['query']).strip()}\n"
            )
        instruction += "\n".join(examples) + "\n"
    elif strategy != "zero_shot":
        raise ValueError(f"unknown S00 prompt strategy: {strategy}")
    return instruction + "Pasaż:\n", "\n\nZapytanie:\n"


def encode_prompt(
    tokenizer: Any,
    passage: str,
    *,
    strategy: str,
    exemplars: Sequence[Mapping[str, Any]],
    max_prompt_tokens: int,
    min_target_passage_tokens: int,
    max_exemplar_characters: int,
) -> tuple[list[int], str]:
    prefix, suffix = _prompt_parts(
        strategy, exemplars, max_passage_characters=max_exemplar_characters
    )
    prefix_ids = list(tokenizer.encode(prefix, add_special_tokens=False))
    suffix_ids = list(tokenizer.encode(suffix, add_special_tokens=False))
    available = max_prompt_tokens - len(prefix_ids) - len(suffix_ids)
    if available < min_target_passage_tokens:
        raise ValueError("few-shot prompt leaves too few tokens for the target passage")
    passage_ids = list(tokenizer.encode(passage.strip(), add_special_tokens=False))
    if len(passage_ids) > available:
        head = (available + 1) // 2
        passage_ids = passage_ids[:head] + passage_ids[-(available - head) :]
    ids = prefix_ids + passage_ids + suffix_ids
    return ids, tokenizer.decode(ids, skip_special_tokens=False)


def _open_journal(
    path: Path,
    identity: str,
    *,
    compatible_identities: frozenset[str] = frozenset(),
) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS generations("
        "strategy TEXT NOT NULL, mode TEXT NOT NULL, example_id TEXT NOT NULL, "
        "candidate_index INTEGER NOT NULL, ordinal INTEGER NOT NULL, payload TEXT NOT NULL, "
        "PRIMARY KEY(strategy, mode, example_id, candidate_index))"
    )
    row = connection.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
    if row is None:
        connection.execute("INSERT INTO metadata(key, value) VALUES('identity', ?)", (identity,))
        connection.commit()
    elif row[0] != identity and row[0] not in compatible_identities:
        connection.close()
        raise ValueError("S00 generation resume identity mismatch")
    return connection


def _batch_metadata_key(strategy: str, mode: str) -> str:
    return f"prompt_batch_size:{strategy}:{mode}"


def _effective_batch_size(
    connection: sqlite3.Connection,
    *,
    strategy: str,
    mode: str,
    requested: int,
    minimum: int,
) -> int:
    key = _batch_metadata_key(strategy, mode)
    row = connection.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
    effective = min(requested, int(row[0])) if row is not None else requested
    if effective < minimum:
        raise ValueError("saved S00 batch size is below the configured minimum")
    return effective


def _save_batch_size(
    connection: sqlite3.Connection, *, strategy: str, mode: str, value: int
) -> None:
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES(?, ?)",
        (_batch_metadata_key(strategy, mode), str(value)),
    )
    connection.commit()


def _left_pad_batch(
    sequences: Sequence[Sequence[int]], *, pad_token_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    if not sequences:
        raise ValueError("cannot pad an empty S00 generation batch")
    width = max(len(sequence) for sequence in sequences)
    input_ids = torch.full((len(sequences), width), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    for row, sequence in enumerate(sequences):
        length = len(sequence)
        input_ids[row, width - length :] = torch.tensor(sequence, dtype=torch.long, device=device)
        attention_mask[row, width - length :] = 1
    return input_ids, attention_mask


def _is_cuda_oom(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        "out of memory" in str(error).casefold() and "cuda" in str(error).casefold()
    )


def _generate_model_batch(
    model: Any,
    tokenizer: Any,
    prompt_ids: Sequence[Sequence[int]],
    *,
    mode: Mapping[str, Any],
    max_new_tokens: int,
) -> list[str]:
    """Keep CUDA batch tensors in a short-lived frame so OOM retry can release them."""
    device = next(model.parameters()).device
    tensor, attention_mask = _left_pad_batch(
        prompt_ids,
        pad_token_id=int(tokenizer.pad_token_id),
        device=device,
    )
    kwargs: dict[str, Any] = {
        "input_ids": tensor,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
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
    with torch.inference_mode():
        sequences = model.generate(**kwargs)
    return [
        tokenizer.decode(sequence[tensor.shape[1] :], skip_special_tokens=True).strip()
        for sequence in sequences
    ]


def _batch_seed(target_seed: int, strategy: str, mode: str, example_ids: Sequence[str]) -> int:
    joined = ",".join(example_ids)
    return int(
        hashlib.sha256(f"{target_seed}:{strategy}:{mode}:{joined}".encode()).hexdigest()[:8],
        16,
    )


def _duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _progress(done: int, total: int, elapsed: float) -> None:
    rate = done / elapsed if done and elapsed else 0.0
    eta = (total - done) / rate if rate else 0.0
    print(
        f"[S00 generation] {done}/{total} ({100 * done / total:5.1f}%) "
        f"elapsed={_duration(elapsed)} rate={rate:.3f}/s eta={_duration(eta)}",
        file=sys.stderr,
        flush=True,
    )


def _write_strategy_artifact(
    connection: sqlite3.Connection,
    path: Path,
    *,
    strategy: str,
    expected: int,
) -> None:
    count = int(
        connection.execute(
            "SELECT COUNT(*) FROM generations WHERE strategy=?", (strategy,)
        ).fetchone()[0]
    )
    if count != expected:
        raise RuntimeError(f"incomplete S00 {strategy} journal: {count}/{expected}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with JsonlWriter(temporary) as writer:
        rows = connection.execute(
            "SELECT payload FROM generations WHERE strategy=? "
            "ORDER BY ordinal, mode, candidate_index",
            (strategy,),
        )
        for (payload,) in rows:
            value = json.loads(payload)
            if not isinstance(value, dict):
                raise ValueError("S00 journal payload must be an object")
            writer.write(value)
    os.replace(temporary, path)


def generate_s00(
    contract_path: Path,
    *,
    output_dir: Path | None = None,
    batch_size: int = 8,
    min_batch_size: int = 1,
    mock: bool = False,
    interrupt_after: int | None = None,
    mock_oom_above: int | None = None,
) -> dict[str, Any]:
    """Generate zero/few-shot greedy and sampled queries with exact SQLite resume."""
    if batch_size < 1 or min_batch_size < 1 or min_batch_size > batch_size:
        raise ValueError("S00 requires 1 <= min_batch_size <= batch_size")
    contract = load_contract(contract_path)
    destination = output_dir or Path(str(contract["output_dir"]))
    preparation = prepare_s00(contract_path, output_dir=destination)
    target = load_frozen_records(
        Path(str(preparation["target_manifest"])), str(contract["target_subset"])
    )
    exemplars = []
    exemplar_path = destination / "cohort" / "few_shot_exemplars.jsonl"
    with exemplar_path.open(encoding="utf-8") as handle:
        exemplars = [json.loads(line) for line in handle if line.strip()]
    identity_payload = {
        "contract": contract,
        "target_records_sha256": preparation["target_records_sha256"],
        "exemplars_sha256": _sha256_file(exemplar_path),
        "generation_trajectory_version": GENERATION_TRAJECTORY_VERSION,
    }
    identity = canonical_fingerprint(identity_payload)
    journal_path = destination / "generation.sqlite"
    connection = _open_journal(
        journal_path, identity, compatible_identities=LEGACY_RESUME_IDENTITIES
    )
    completed = {
        (str(row[0]), str(row[1]), str(row[2]), int(row[3]))
        for row in connection.execute(
            "SELECT strategy, mode, example_id, candidate_index FROM generations"
        )
    }
    legacy_completed = int(
        connection.execute(
            "SELECT COUNT(*) FROM generations "
            "WHERE instr(payload, '\"generation_trajectory_version\"') = 0"
        ).fetchone()[0]
    )
    modes = list(contract["generation"]["modes"])
    expected_per_strategy = len(target) * sum(int(mode["num_return_sequences"]) for mode in modes)
    total = len(PROMPT_STRATEGIES) * expected_per_strategy
    model = tokenizer = app_config = None
    if not mock and len(completed) < total:
        app_config = load_config(Path(str(contract["model_config"])))
        if (
            app_config.model.name_or_path != contract["model_name_or_path"]
            or app_config.model.revision != contract["model_revision"]
        ):
            raise ValueError("S00 generator config does not match pinned model identity")
        tokenizer = load_tokenizer(app_config)
        model, _precision = load_generator(app_config, for_training=False)
        model.eval()
    started = time.perf_counter()
    generated_now = 0
    oom_retries = 0
    effective_batch_sizes: dict[str, int] = {}
    _progress(len(completed), total, 0.0)
    try:
        for strategy in PROMPT_STRATEGIES:
            for mode in modes:
                name = str(mode["name"])
                candidate_count = int(mode["num_return_sequences"])
                pending: list[tuple[int, dict[str, Any]]] = []
                for ordinal, record in enumerate(target):
                    example_id = str(record["example_id"])
                    keys = [(strategy, name, example_id, index) for index in range(candidate_count)]
                    if all(key in completed for key in keys):
                        continue
                    if any(key in completed for key in keys):
                        raise ValueError("partial multi-candidate S00 generation group in journal")
                    pending.append((ordinal, record))
                effective_batch = _effective_batch_size(
                    connection,
                    strategy=strategy,
                    mode=name,
                    requested=batch_size,
                    minimum=min_batch_size,
                )
                position = 0
                while position < len(pending):
                    chunk = pending[position : position + effective_batch]
                    example_ids = [str(record["example_id"]) for _ordinal, record in chunk]
                    seed = _batch_seed(int(contract["target_seed"]), strategy, name, example_ids)
                    prompts: list[str] = []
                    prompt_ids: list[list[int]] = []
                    try:
                        if mock and mock_oom_above is not None and len(chunk) > mock_oom_above:
                            raise torch.cuda.OutOfMemoryError("mock CUDA out of memory")
                        if mock:
                            generated = [
                                f"mock {strategy} {name} {example_id} {candidate_index}"
                                for example_id in example_ids
                                for candidate_index in range(candidate_count)
                            ]
                            prompts = [
                                f"mock prompt {strategy} {example_id}" for example_id in example_ids
                            ]
                        else:
                            assert tokenizer is not None and model is not None
                            for _ordinal, record in chunk:
                                positive = sorted(
                                    record["positives"], key=lambda row: str(row["doc_id"])
                                )[0]
                                encoded, prompt = encode_prompt(
                                    tokenizer,
                                    str(positive["text"]),
                                    strategy=strategy,
                                    exemplars=exemplars,
                                    max_prompt_tokens=int(
                                        contract["generation"]["max_prompt_tokens"]
                                    ),
                                    min_target_passage_tokens=int(
                                        contract["prompt"]["min_target_passage_tokens"]
                                    ),
                                    max_exemplar_characters=int(
                                        contract["exemplars"]["max_passage_characters"]
                                    ),
                                )
                                prompt_ids.append(encoded)
                                prompts.append(prompt)
                            set_seed(seed)
                            generated = _generate_model_batch(
                                model,
                                tokenizer,
                                prompt_ids,
                                mode=mode,
                                max_new_tokens=int(contract["generation"]["max_new_tokens"]),
                            )
                        expected = len(chunk) * candidate_count
                        if len(generated) != expected:
                            raise RuntimeError(
                                f"S00 generation returned {len(generated)}/{expected} sequences"
                            )
                    except BaseException as error:
                        if not _is_cuda_oom(error):
                            raise
                        connection.rollback()
                        oom_retries += 1
                        if effective_batch <= min_batch_size:
                            raise RuntimeError(
                                "S00 CUDA OOM at minimum prompt batch size "
                                f"{min_batch_size}; lower S00_MIN_BATCH_SIZE or prompt length"
                            ) from error
                        previous_batch = effective_batch
                        effective_batch = max(min_batch_size, effective_batch // 2)
                        _save_batch_size(
                            connection,
                            strategy=strategy,
                            mode=name,
                            value=effective_batch,
                        )
                        error.__traceback__ = None
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        print(
                            f"[S00 OOM] {strategy}/{name}: prompt batch "
                            f"{previous_batch} -> {effective_batch}; retrying",
                            file=sys.stderr,
                            flush=True,
                        )
                        continue
                    _save_batch_size(
                        connection,
                        strategy=strategy,
                        mode=name,
                        value=effective_batch,
                    )
                    for chunk_index, (ordinal, record) in enumerate(chunk):
                        example_id = str(record["example_id"])
                        positive = sorted(record["positives"], key=lambda row: str(row["doc_id"]))[
                            0
                        ]
                        for candidate_index in range(candidate_count):
                            text_index = chunk_index * candidate_count + candidate_index
                            payload = {
                                "example_id": example_id,
                                "positive": positive,
                                "hard_negatives": record["hard_negatives"],
                                "positive_count": len(record["positives"]),
                                "reference": str(record["query"]),
                                "metadata": record.get("metadata", {}),
                                "experiment_id": f"S00-{strategy}",
                                "generation_run_id": f"S00-{strategy}-{name}",
                                "evaluation_id": (f"{example_id}::{name}::{candidate_index}"),
                                "mode": name,
                                "candidate_index": candidate_index,
                                "generation_config": mode,
                                "prompt_version": str(contract["prompt"]["version"]),
                                "prompt_strategy": strategy,
                                "prompt_sha256": hashlib.sha256(
                                    prompts[chunk_index].encode()
                                ).hexdigest(),
                                "seed": seed,
                                "prompt_batch_size": len(chunk),
                                "generation_trajectory_version": (GENERATION_TRAJECTORY_VERSION),
                                "generated": generated[text_index],
                            }
                            connection.execute(
                                "INSERT INTO generations("
                                "strategy, mode, example_id, candidate_index, "
                                "ordinal, payload) VALUES(?, ?, ?, ?, ?, ?)",
                                (
                                    strategy,
                                    name,
                                    example_id,
                                    candidate_index,
                                    ordinal,
                                    json.dumps(payload, ensure_ascii=False, sort_keys=True),
                                ),
                            )
                    connection.commit()
                    position += len(chunk)
                    generated_now += len(chunk) * candidate_count
                    done = len(completed) + generated_now
                    interval = int(contract["generation"]["progress_every"])
                    if (
                        generated_now == len(chunk) * candidate_count
                        or done % interval < len(chunk) * candidate_count
                        or done == total
                    ):
                        _progress(done, total, time.perf_counter() - started)
                    if interrupt_after is not None and generated_now >= interrupt_after:
                        raise InterruptedError("deliberate S00 generation interruption")
                effective_batch_sizes[f"{strategy}/{name}"] = effective_batch
        artifacts = {}
        for strategy in PROMPT_STRATEGIES:
            path = destination / f"{strategy}.generations.jsonl"
            _write_strategy_artifact(
                connection, path, strategy=strategy, expected=expected_per_strategy
            )
            artifacts[strategy] = str(path)
    finally:
        connection.close()
    elapsed = time.perf_counter() - started
    summary = {
        "status": "complete",
        "contract_version": CONTRACT_VERSION,
        "generation_identity": identity,
        "target_count": len(target),
        "expected_generation_count": total,
        "resumed_generation_count": len(completed),
        "legacy_resumed_generation_count": legacy_completed,
        "generated_now": generated_now,
        "requested_prompt_batch_size": batch_size,
        "effective_prompt_batch_sizes": effective_batch_sizes,
        "oom_retries": oom_retries,
        "elapsed_seconds_this_invocation": elapsed,
        "final_tests_used": [],
        "artifacts": artifacts,
        "code": collect_code_provenance(),
    }
    write_json(destination / "generation_summary.json", summary)
    return summary
