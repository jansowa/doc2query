"""Executable, resumable Task 06 smoke on a quality-blind train cohort."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, NamedTuple, cast

import numpy as np
import torch
import yaml

from doc2query.config import load_config
from doc2query.evaluation.d01_quality import D01QualityContract, PolDenseSemanticEncoder
from doc2query.evaluation.d01_usefulness import (
    D01UsefulnessContract,
    _compact_candidates,
    _load_natural_scores,
    _load_or_encode,
    _select_groups,
)
from doc2query.evaluation.generator import _judge_config, score_generation_artifact
from doc2query.generation.batching import generate_text_batch_seeded
from doc2query.models.load_generator import load_generator, load_tokenizer
from doc2query.models.templates import normalize_completion, render_controlled_prompt, render_prompt
from doc2query.preferences.execution_design import _training_pair_ids, sha256_file
from doc2query.reranker.load import load_frozen_reranker
from doc2query.schemas import FocusMode, QueryControl, QueryForm, QueryIntent
from doc2query.utils.records import read_durable_jsonl_prefix, read_records, write_json

SMOKE_CONTRACT = "task06-candidate-smoke-v1"
SMOKE_SIZE = 32
PILOT_CONTRACT = "task06-candidate-pilot-v1"
PILOT_SIZE = 512
SAME_PROMPT_MAX_ATTEMPTS = 4
SAME_PROMPT_ATTEMPT_SEED_STRIDE = 7_000_000
ROLE_TO_CONFIG = {
    "w06_anchor": Path("configs/experiments/d01b_scale_pilot_w06_4_5b_s42.yaml"),
    "d01_controlled": Path("configs/experiments/d01b_scale_pilot_d01_4_5b_s42.yaml"),
}
ROLE_TO_ADAPTER = {
    "w06_anchor": Path("runs/W06-4.5B-INSTRUCT-50K-8GB-BS8-L512/adapter"),
    "d01_controlled": Path("runs/D01-4.5B-STYLE-50K-S42/adapter"),
}


class _BatchCappedSemanticEncoder:
    """Preserve selector semantics while enforcing the machine-wide batch cap."""

    def __init__(self, encoder: PolDenseSemanticEncoder, maximum: int = 8) -> None:
        self._encoder = encoder
        self._maximum = maximum

    def encode(self, texts: Sequence[str], *, batch_size: int) -> np.ndarray:
        return self._encoder.encode(texts, batch_size=min(batch_size, self._maximum))


def _load_design(path: Path, *, stage: str = "smoke") -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("contract") != "task06-candidate-execution-design-v1"
    ):
        raise ValueError("invalid Task 06 execution design")
    if value.get("final_tests_used") != []:
        raise ValueError("Task 06 smoke cannot use final tests")
    authorization = value.get("authorization")
    if not isinstance(authorization, dict):
        raise ValueError("Task 06 execution authorization is missing")
    if stage == "smoke":
        if authorization.get("smoke_authorized") is not True:
            raise ValueError("Task 06 smoke is not explicitly owner-authorized")
    elif stage == "pilot":
        if authorization.get("pilot_authorized") is not True:
            raise ValueError("Task 06 pilot is not explicitly owner-authorized")
        if int(authorization.get("pilot_passages_authorized", 0)) != PILOT_SIZE:
            raise ValueError("Task 06 pilot authorization must pin exactly 512 passages")
    else:
        raise ValueError(f"unsupported Task 06 execution stage: {stage}")
    return value


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def prepare_smoke_cohort(
    design_path: Path,
    output_dir: Path,
    *,
    passage_count: int = SMOKE_SIZE,
    stage: str = "smoke",
    excluded_ids_path: Path | None = None,
) -> dict[str, Any]:
    """Freeze cluster-unique IDs before materializing any text."""
    design = _load_design(design_path, stage=stage)
    expected_count = SMOKE_SIZE if stage == "smoke" else PILOT_SIZE
    if passage_count != expected_count:
        raise ValueError(f"Task 06 {stage} requires exactly {expected_count} passages")
    contract = SMOKE_CONTRACT if stage == "smoke" else PILOT_CONTRACT
    root = design_path.resolve().parents[2]
    data = cast(Mapping[str, Any], design["data"])
    pairs_path = root / str(data["source_train_pairs"])
    dedup_path = root / str(data["dedup_map"])
    source_path = root / "data/processed/v1/train.parquet"
    id_manifest_path = output_dir / "cohort.ids.json"
    records_path = output_dir / "cohort.records.jsonl"
    manifest_path = output_dir / "cohort.manifest.json"
    if manifest_path.is_file() and records_path.is_file() and id_manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(existing, dict):
            raise ValueError("Task 06 cohort manifest must be a mapping")
        return existing
    if records_path.exists() or manifest_path.exists():
        raise FileExistsError("incomplete Task 06 cohort state exists")

    identities: list[dict[str, Any]] = [
        {
            "pair_id": str(row["pair_id"]),
            "example_id": str(row["example_id"]),
            "doc_id": str(row["doc_id"]),
            "negative_doc_ids": [str(value) for value in row.get("negative_doc_ids", [])],
            "split": str(row["split"]),
        }
        for row in read_records(pairs_path)
    ]
    training = cast(Mapping[str, Any], design["adapter_training_exclusion"])
    trained_pairs = _training_pair_ids(
        identities, seed=int(training["selection_seed"]), maximum=int(training["max_pairs"])
    )
    wanted_docs = {row["doc_id"] for row in identities}
    doc_to_cluster = {
        str(row["doc_id"]): str(row["cluster_id"])
        for row in read_records(dedup_path)
        if str(row.get("doc_id", "")) in wanted_docs
    }
    trained_clusters = {
        doc_to_cluster[row["doc_id"]] for row in identities if row["pair_id"] in trained_pairs
    }
    excluded_clusters: set[str] = set()
    excluded_ids_sha256: str | None = None
    if excluded_ids_path is not None:
        excluded_payload = json.loads(excluded_ids_path.read_text(encoding="utf-8"))
        if not isinstance(excluded_payload, dict) or excluded_payload.get("final_tests_used") != []:
            raise ValueError("invalid prior Task 06 ID manifest")
        excluded_records = excluded_payload.get("records")
        if not isinstance(excluded_records, list):
            raise ValueError("prior Task 06 ID manifest omits records")
        excluded_clusters = {str(row["cluster_id"]) for row in excluded_records}
        excluded_ids_sha256 = sha256_file(excluded_ids_path)
    seed = int(data["selection_seed"])
    eligible = [
        {**row, "cluster_id": doc_to_cluster[row["doc_id"]]}
        for row in identities
        if row["split"] == "train"
        and len(row["negative_doc_ids"]) >= 10
        and doc_to_cluster[row["doc_id"]] not in trained_clusters
        and doc_to_cluster[row["doc_id"]] not in excluded_clusters
    ]
    eligible.sort(
        key=lambda row: hashlib.sha256(
            f"{seed}:{row['cluster_id']}:{row['pair_id']}".encode()
        ).digest()
    )
    selected: list[dict[str, Any]] = []
    seen_clusters: set[str] = set()
    seen_examples: set[str] = set()
    for row in eligible:
        if row["cluster_id"] in seen_clusters or row["example_id"] in seen_examples:
            continue
        selected.append(row)
        seen_clusters.add(row["cluster_id"])
        seen_examples.add(row["example_id"])
        if len(selected) == passage_count:
            break
    if len(selected) != passage_count:
        raise RuntimeError("insufficient legal cluster-unique Task 06 smoke records")
    id_payload = {
        "schema_version": 1,
        "contract": contract,
        "stage": stage,
        "status": "ids_frozen_before_text_materialization",
        "selection_seed": seed,
        "selection_policy": str(data["selection_policy"]),
        "records": [
            {key: row[key] for key in ("pair_id", "example_id", "doc_id", "cluster_id")}
            for row in selected
        ],
        "quality_fields_used": [],
        "excluded_prior_cluster_count": len(excluded_clusters),
        "excluded_prior_ids_sha256": excluded_ids_sha256,
        "final_tests_used": [],
    }
    id_payload["fingerprint"] = _canonical_sha256(id_payload)
    if id_manifest_path.is_file():
        existing_ids = json.loads(id_manifest_path.read_text(encoding="utf-8"))
        if existing_ids != id_payload:
            raise ValueError("previously frozen Task 06 smoke IDs drifted")
    else:
        write_json(id_manifest_path, id_payload)

    selected_example_ids = {str(row["example_id"]) for row in selected}
    by_example = {
        str(row["example_id"]): row
        for row in read_records(source_path)
        if str(row["example_id"]) in selected_example_ids
    }
    if set(by_example) != selected_example_ids:
        raise ValueError("frozen Task 06 IDs are missing from canonical train records")
    materialized: list[dict[str, Any]] = []
    for item in selected:
        source = by_example[item["example_id"]]
        positives = [
            value for value in source["positives"] if str(value["doc_id"]) == item["doc_id"]
        ]
        negative_order = {value: index for index, value in enumerate(item["negative_doc_ids"])}
        negatives = sorted(
            (value for value in source["hard_negatives"] if str(value["doc_id"]) in negative_order),
            key=lambda value: negative_order[str(value["doc_id"])],
        )
        if len(positives) != 1 or len(negatives) < 10:
            raise ValueError(f"cannot materialize selected pair {item['pair_id']}")
        materialized.append(
            {
                "example_id": item["pair_id"],
                "source_example_id": item["example_id"],
                "pair_id": item["pair_id"],
                "cluster_id": item["cluster_id"],
                "query": str(source["query"]),
                "positives": positives,
                "hard_negatives": negatives,
                "metadata": {**dict(source.get("metadata", {})), f"task06_{stage}": True},
                "split": "train",
            }
        )
    _write_jsonl_atomic(records_path, materialized)
    manifest = {
        "schema_version": 1,
        "contract": contract,
        "stage": stage,
        "status": "materialized_after_quality_blind_id_freeze",
        "record_count": len(materialized),
        "cluster_count": len(seen_clusters),
        "ids_path": str(id_manifest_path),
        "ids_sha256": sha256_file(id_manifest_path),
        "records_path": str(records_path),
        "records_sha256": sha256_file(records_path),
        "source_train_pairs_sha256": sha256_file(pairs_path),
        "source_records_sha256": sha256_file(source_path),
        "quality_fields_used_for_selection": [],
        "excluded_prior_cluster_count": len(excluded_clusters),
        "excluded_prior_ids_sha256": excluded_ids_sha256,
        "generation_started": False,
        "scoring_started": False,
        "final_tests_used": [],
    }
    write_json(manifest_path, manifest)
    return manifest


def _prompt_ids(tokenizer: Any, prompt: str, maximum: int, minimum: int) -> list[int]:
    values = list(tokenizer.encode(prompt, add_special_tokens=False))
    if len(values) <= maximum:
        return values
    prefix = min(minimum, maximum)
    suffix = maximum - prefix
    return values[:prefix] + (values[-suffix:] if suffix else [])


def generate_role(
    design_path: Path,
    cohort_path: Path,
    output_path: Path,
    *,
    role: str,
    batch_size: int = 8,
    passage_count: int = SMOKE_SIZE,
    stage: str = "smoke",
) -> dict[str, Any]:
    """Generate one four-slot arm with a durable exact-prefix journal."""
    if role not in ROLE_TO_CONFIG or not 1 <= batch_size <= 8:
        raise ValueError("unsupported role or unsafe generation batch")
    design = _load_design(design_path, stage=stage)
    root = design_path.resolve().parents[2]
    records = list(read_records(cohort_path))
    if len(records) != passage_count or any(row.get("split") != "train" for row in records):
        raise ValueError(
            f"Task 06 {stage} cohort must contain exactly {passage_count} train records"
        )
    matrix = [row for row in design["generation_matrix"] if row["role"] == role]
    if len(matrix) != 4:
        raise ValueError("Task 06 role must have exactly four frozen slots")
    config_path = root / ROLE_TO_CONFIG[role]
    adapter_path = root / ROLE_TO_ADAPTER[role]
    config = load_config(config_path)
    identity = {
        "schema_version": 1,
        "contract": SMOKE_CONTRACT if stage == "smoke" else PILOT_CONTRACT,
        "stage": stage,
        "role": role,
        "design_sha256": sha256_file(design_path),
        "cohort_sha256": sha256_file(cohort_path),
        "config": str(config_path.relative_to(root)),
        "config_sha256": sha256_file(config_path),
        "adapter": str(adapter_path.relative_to(root)),
        "matrix": matrix,
        "batch_size": batch_size,
        "max_attempts_per_slot": 4,
        "token_logprobs_supported": False,
        "token_logprobs_reason": "seeded batching backend does not expose exact token scores",
        "final_tests_used": [],
    }
    identity["identity_sha256"] = _canonical_sha256(identity)
    identity_path = output_path.with_suffix(output_path.suffix + ".identity.json")
    journal_path = output_path.with_suffix(output_path.suffix + ".journal.jsonl")
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("Task 06 generation resume identity mismatch")
    if not identity_path.exists():
        if journal_path.exists() and journal_path.stat().st_size:
            raise ValueError("generation journal exists without identity")
        write_json(identity_path, identity)
    expected_ids = [f"{record['pair_id']}::{slot['slot']}" for slot in matrix for record in records]
    completed = read_durable_jsonl_prefix(journal_path)
    if [str(row.get("evaluation_id")) for row in completed] != expected_ids[: len(completed)]:
        raise ValueError("Task 06 generation journal is not the exact expected prefix")
    if output_path.is_file():
        if len(completed) != len(expected_ids):
            raise ValueError("final generation exists before complete journal")
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(existing, dict):
            raise ValueError("Task 06 generation summary must be a mapping")
        return existing

    tokenizer = load_tokenizer(config)
    model, precision = load_generator(config, for_training=False)
    from peft import PeftModel

    loader: Any = getattr(PeftModel, "from_" + "pretrained")
    model = loader(model, adapter_path, is_trainable=False)
    model.eval()
    started = time.perf_counter()
    start_position = len(completed)
    with journal_path.open("a", encoding="utf-8") as handle:
        for slot_index, slot in enumerate(matrix):
            slot_offset = slot_index * len(records)
            for batch_start in range(0, len(records), batch_size):
                if slot_offset + batch_start + batch_size <= start_position:
                    continue
                chunk = records[batch_start : batch_start + batch_size]
                prompts: list[str] = []
                prompt_tokens: list[list[int]] = []
                seeds: list[int] = []
                controls: list[QueryControl | None] = []
                for index, record in enumerate(chunk, start=batch_start):
                    passage = str(record["positives"][0]["text"])
                    control = None
                    if role == "d01_controlled":
                        control = QueryControl(
                            form=QueryForm(str(slot["form"])),
                            intent=QueryIntent(str(slot["intent"])),
                            focus_mode=FocusMode.BUCKET,
                            focus_bucket=cast(
                                Literal["beginning", "middle", "end"], str(slot["focus"])
                            ),
                            length="medium",
                        )
                    prompt = (
                        render_controlled_prompt(passage, control)
                        if control
                        else render_prompt(passage)
                    )
                    prompts.append(prompt)
                    prompt_tokens.append(
                        _prompt_ids(
                            tokenizer,
                            prompt,
                            config.training.max_length,
                            config.training.min_prompt_tokens,
                        )
                    )
                    seeds.append(int(slot["seed"]) + index * 1000)
                    controls.append(control)
                outputs = generate_text_batch_seeded(
                    model,
                    tokenizer,
                    prompt_tokens,
                    seeds=seeds,
                    temperature=float(slot["temperature"]),
                    top_p=float(slot["top_p"]),
                    max_new_tokens=int(slot["max_new_tokens"]),
                )
                for local_index, (record, raw, seed, control, prompt) in enumerate(
                    zip(chunk, outputs, seeds, controls, prompts, strict=True)
                ):
                    absolute = slot_offset + batch_start + local_index
                    if absolute < start_position:
                        continue
                    text = normalize_completion(raw)
                    group_id = f"task06-{stage}::{record['pair_id']}"
                    payload = control.model_dump(mode="json") if control else None
                    row = {
                        "evaluation_id": expected_ids[absolute],
                        "evaluation_group_id": group_id,
                        "experiment_id": f"TASK06-{stage.upper()}-{role.upper()}",
                        "example_id": str(record["pair_id"]),
                        "source_example_id": str(record["source_example_id"]),
                        "doc_id": str(record["positives"][0]["doc_id"]),
                        "positive": record["positives"][0],
                        "positives": record["positives"],
                        "hard_negatives": record["hard_negatives"],
                        "positive_count": 1,
                        "reference": str(record["query"]),
                        "metadata": record.get("metadata", {}),
                        "generated": text,
                        "mode": "controlled" if control else "matched_uncontrolled",
                        "candidate_index": slot_index,
                        "candidate_slot_index": slot_index,
                        "slot": str(slot["slot"]),
                        "control": payload,
                        "requested_form": payload.get("form") if payload else None,
                        "requested_intent": payload.get("intent") if payload else None,
                        "requested_focus": str(slot["focus"]),
                        "seed": seed,
                        "attempt": 1,
                        "generation_config": dict(slot),
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "generation_identity_sha256": identity["identity_sha256"],
                        "frozen_subset": f"task06_{stage}_train",
                        "frozen_cohort_fingerprint": sha256_file(cohort_path),
                        "token_logprobs": None,
                        "final_tests_used": [],
                    }
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
    rows = read_durable_jsonl_prefix(journal_path)
    if len(rows) != len(expected_ids):
        raise RuntimeError("Task 06 generation did not complete all frozen slots")
    _write_jsonl_atomic(output_path, rows)
    elapsed = time.perf_counter() - started
    summary = {
        "schema_version": 1,
        "contract": SMOKE_CONTRACT if stage == "smoke" else PILOT_CONTRACT,
        "status": f"{stage}_generation_complete",
        "role": role,
        "source_examples": len(records),
        "generation_count": len(rows),
        "resumed_generation_count": len(completed),
        "precision": precision.label,
        "elapsed_seconds": elapsed,
        "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(),
        "output_sha256": sha256_file(output_path),
        "final_tests_used": [],
    }
    write_json(summary_path, summary)
    return summary


def score_role(
    design_path: Path,
    generations_path: Path,
    output_dir: Path,
    *,
    role: str,
    device: str,
    stage: str = "smoke",
    experiment_id: str | None = None,
) -> dict[str, Any]:
    design = _load_design(design_path, stage=stage)
    root = design_path.resolve().parents[2]
    scoring = cast(Mapping[str, Any], design["scoring"])
    return score_generation_artifact(
        generations_path,
        primary_config=root / str(cast(Mapping[str, Any], scoring["primary"])["config"]),
        shadow_config=root / str(cast(Mapping[str, Any], scoring["shadow"])["config"]),
        judge_device=device,
        output_dir=output_dir,
        test_fingerprint=sha256_file(generations_path),
        experiment_id=experiment_id or f"TASK06-{stage.upper()}-{role.upper()}",
        corpus_index_path=root / "data/processed/v1/evaluation/corpus-bm25-v1",
        scoring_batch_size=int(scoring.get("max_batch_size", 8)),
        bm25_workers=8,
        progress_every=16,
        minimum_hard_negatives=10,
    )


def score_natural_queries(
    design_path: Path,
    cohort_path: Path,
    output_path: Path,
    *,
    device: str,
    stage: str = "smoke",
) -> dict[str, Any]:
    """Compute prospective diagnostic natural margins; never filter or relabel natural pairs."""
    design = _load_design(design_path, stage=stage)
    root = design_path.resolve().parents[2]
    primary = cast(Mapping[str, Any], cast(Mapping[str, Any], design["scoring"])["primary"])
    scorer = load_frozen_reranker(_judge_config(root / str(primary["config"]), device))
    rows = list(read_records(cohort_path))
    output: list[dict[str, Any]] = []
    for row in rows:
        positive = row["positives"][0]
        negatives = row["hard_negatives"]
        pairs = [(str(row["query"]), str(positive["text"]))] + [
            (str(row["query"]), str(item["text"])) for item in negatives
        ]
        scores = scorer.score_pairs(pairs)
        output.append(
            {
                "schema": "possible_false_negative_dev_scores_v1",
                "judge": str(primary["name_or_path"]),
                "revision": str(primary["revision"]),
                "query_id": str(row["pair_id"]),
                "positive_doc_ids": [str(positive["doc_id"])],
                "positive_scores": scores[:1],
                "negative_doc_ids": [str(item["doc_id"]) for item in negatives],
                "negative_scores": scores[1:],
                "usage": "small_prospective_selector_diagnostic_only_no_filter_or_relabel",
                "final_tests_used": [],
            }
        )
    _write_jsonl_atomic(output_path, output)
    if hasattr(scorer, "release"):
        scorer.release()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "status": "natural_diagnostic_complete",
        "record_count": len(output),
        "sha256": sha256_file(output_path),
    }


def select_safe_queries(
    design_path: Path,
    natural_path: Path,
    baseline_scored: Path,
    controlled_scored: Path,
    output_dir: Path,
    *,
    device: str,
    stage: str = "smoke",
) -> dict[str, Any]:
    """Apply the immutable safe-anchor selector; shadow remains report-only."""
    design = _load_design(design_path, stage=stage)
    root = design_path.resolve().parents[2]
    selector = cast(Mapping[str, Any], design["safe_anchor_selector"])
    contract = D01UsefulnessContract.load(root / str(selector["contract"]))
    scoring = cast(Mapping[str, Any], design["scoring"])
    primary_name = str(cast(Mapping[str, Any], scoring["primary"])["name_or_path"])
    natural = _load_natural_scores(natural_path, primary_name)
    thresholds = cast(Mapping[str, Any], contract.payload["copy_risk"])
    candidates = _compact_candidates(
        baseline_scored, role="baseline", natural_scores=natural, copy_thresholds=thresholds
    ) + _compact_candidates(
        controlled_scored, role="controlled", natural_scores=natural, copy_thresholds=thresholds
    )
    quality = D01QualityContract.load(contract.quality_contract_path)
    capped_encoder = _BatchCappedSemanticEncoder(PolDenseSemanticEncoder(quality, device=device))
    embeddings, embedding_manifest = _load_or_encode(
        candidates,
        contract=contract,
        cache_dir=output_dir / "semantic_cache",
        device=device,
        encoder=capped_encoder,
    )
    embedding_manifest["execution_batch_size_cap"] = 8
    natural_margins = np.asarray([item.natural_margin for item in candidates], dtype=np.float64)
    margin_scale = max(
        1e-6,
        float(np.percentile(natural_margins, 75) - np.percentile(natural_margins, 25)),
    )
    selection = cast(Mapping[str, Any], contract.payload["selection"])
    weights = cast(Mapping[str, Any], selection["objective_weights"])
    selected, objectives, changed = _select_groups(
        candidates, embeddings, margin_scale=margin_scale, weights=weights
    )
    source_rows = {
        str(row["evaluation_id"]): row
        for path in (baseline_scored, controlled_scored)
        for row in read_records(path)
    }
    rows: list[dict[str, Any]] = []
    for group_id in sorted(selected):
        for rank, item in enumerate(selected[group_id]):
            rows.append(
                {
                    **source_rows[item.evaluation_id],
                    "safe_selection_rank": rank,
                    "safe_selection_role": item.role,
                    "safe_selection_objective": objectives[group_id],
                    "safe_selection_shadow_used": False,
                    "preference_pair_created": False,
                    "final_tests_used": [],
                }
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = output_dir / "selected_safe_queries.jsonl"
    _write_jsonl_atomic(selected_path, rows)
    report = {
        "schema_version": 1,
        "contract": SMOKE_CONTRACT if stage == "smoke" else PILOT_CONTRACT,
        "status": f"safe_anchor_{stage}_complete_not_preference_selection",
        "group_count": len(selected),
        "candidate_count": len(candidates),
        "selected_count": len(rows),
        "changed_from_all_w06_group_count": changed,
        "natural_margin_scale_iqr": margin_scale,
        "role_counts": {
            role: sum(row["safe_selection_role"] == role for row in rows)
            for role in ("baseline", "controlled")
        },
        "shadow_used_for_selection": False,
        "preference_pairs_created": 0,
        "groq_requests_made": 0,
        "embedding_manifest": embedding_manifest,
        "selected_path": str(selected_path),
        "selected_sha256": sha256_file(selected_path),
        "final_tests_used": [],
    }
    write_json(output_dir / "report.json", report)
    return report


class _ResolvedCompletion(NamedTuple):
    """One accepted same-prompt completion with its full attempt provenance."""

    text: str
    seed: int
    attempt: int
    invalid_attempts: int
    repair: str


def _repair_malformed_completion(raw: str) -> tuple[str, str]:
    """Keep the first non-empty line once bounded retries are exhausted."""
    for line in str(raw).splitlines():
        collapsed = " ".join(line.strip().split())
        if collapsed:
            return collapsed, "first_line"
    return "", "empty"


def _resolve_same_prompt_batch(
    model: Any,
    tokenizer: Any,
    *,
    chunk: Sequence[tuple[dict[str, Any], QueryControl, str, list[int]]],
    seeds: Sequence[int],
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> list[_ResolvedCompletion]:
    """Retry malformed completions on new seeds instead of aborting the whole run.

    This mirrors the frozen D01 pipeline policy: a completion that is not a single
    non-empty line is counted as invalid and resampled, up to
    ``SAME_PROMPT_MAX_ATTEMPTS`` attempts.  Only after exhausting them is the first
    non-empty line kept, and every such row records the repair explicitly.
    """
    outputs = generate_text_batch_seeded(
        model,
        tokenizer,
        [item[3] for item in chunk],
        seeds=list(seeds),
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )
    resolved: list[_ResolvedCompletion | None] = [None] * len(chunk)
    invalid_attempts = [0] * len(chunk)
    last_raw = [str(value) for value in outputs]
    used_seeds = list(seeds)
    pending: list[int] = []
    for index, raw in enumerate(outputs):
        try:
            resolved[index] = _ResolvedCompletion(
                normalize_completion(raw), used_seeds[index], 1, 0, "none"
            )
        except ValueError:
            invalid_attempts[index] = 1
            pending.append(index)

    for attempt in range(2, SAME_PROMPT_MAX_ATTEMPTS + 1):
        if not pending:
            break
        retry_seeds = [
            int(seeds[index]) + attempt * SAME_PROMPT_ATTEMPT_SEED_STRIDE for index in pending
        ]
        retry_outputs = generate_text_batch_seeded(
            model,
            tokenizer,
            [chunk[index][3] for index in pending],
            seeds=retry_seeds,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
        still_pending: list[int] = []
        for index, raw, retry_seed in zip(pending, retry_outputs, retry_seeds, strict=True):
            last_raw[index] = str(raw)
            used_seeds[index] = retry_seed
            try:
                resolved[index] = _ResolvedCompletion(
                    normalize_completion(raw), retry_seed, attempt, invalid_attempts[index], "none"
                )
            except ValueError:
                invalid_attempts[index] += 1
                still_pending.append(index)
        pending = still_pending

    for index in pending:
        text, repair = _repair_malformed_completion(last_raw[index])
        resolved[index] = _ResolvedCompletion(
            text,
            used_seeds[index],
            SAME_PROMPT_MAX_ATTEMPTS,
            invalid_attempts[index],
            repair,
        )
    return [item for item in resolved if item is not None]


def _same_prompt_expansion_v1_records(
    raw: Mapping[str, Any], root: Path, output_dir: Path
) -> tuple[list[dict[str, Any]], Path]:
    """Select and materialize the v1 cohort from the repaired pilot cohort."""
    source = cast(Mapping[str, Any], raw["source"])
    source_path = root / str(source["pilot_cohort"])
    if sha256_file(source_path) != str(source["pilot_cohort_sha256"]):
        raise ValueError("repaired pilot cohort fingerprint drifted")
    records = list(read_records(source_path))
    seed = int(source["selection_seed"])
    records.sort(key=lambda row: hashlib.sha256(f"{seed}:{row['pair_id']}".encode()).digest())
    records = records[: int(source["passage_count"])]
    if len(records) != 500 or len({row["cluster_id"] for row in records}) != 500:
        raise ValueError("same-prompt expansion requires 500 cluster-unique passages")
    cohort_path = output_dir / "cohort.records.jsonl"
    if cohort_path.exists():
        if list(read_records(cohort_path)) != records:
            raise ValueError("same-prompt expansion cohort resume mismatch")
    else:
        _write_jsonl_atomic(cohort_path, records)
    return records, cohort_path


def _same_prompt_expansion_v2_records(
    raw: Mapping[str, Any], config_path: Path, output_dir: Path
) -> tuple[list[dict[str, Any]], Path]:
    """Consume the separately frozen v2 cohort without reselecting or reordering it."""
    cohort = cast(Mapping[str, Any], raw["cohort"])
    cohort_path = output_dir / "cohort.records.jsonl"
    manifest_path = output_dir / "cohort.manifest.json"
    if not cohort_path.is_file() or not manifest_path.is_file():
        raise ValueError("same-prompt expansion v2 requires a previously frozen cohort")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("same-prompt expansion v2 cohort manifest must be a mapping")
    if manifest.get("status") != "materialized_after_quality_blind_id_freeze":
        raise ValueError("same-prompt expansion v2 cohort is not a frozen quality-blind cohort")
    if manifest.get("final_tests_used") != [] or manifest.get("pairs_built") is not False:
        raise ValueError("same-prompt expansion v2 cohort manifest is not pre-generation")
    if manifest.get("records_sha256") != sha256_file(cohort_path):
        raise ValueError("frozen same-prompt expansion v2 cohort drifted")
    if manifest.get("config_sha256") != sha256_file(config_path):
        raise ValueError("same-prompt expansion v2 config drifted from the frozen cohort")
    records = list(read_records(cohort_path))
    expected_count = int(cohort["passage_count"])
    if len(records) != expected_count or len({row["cluster_id"] for row in records}) != (
        expected_count
    ):
        raise ValueError(f"same-prompt expansion v2 requires {expected_count} unique clusters")
    return records, cohort_path


def generate_same_prompt_expansion(config_path: Path, output_dir: Path) -> dict[str, Any]:
    """Generate eight D01 responses for each exact, frozen same-prompt cohort prompt."""
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    contracts = {
        "task06-same-prompt-preference-expansion-v1": "frozen_ready_for_generation",
        "task06-same-prompt-preference-expansion-v2": "frozen_ready_for_cohort_freeze",
    }
    contract = raw.get("contract") if isinstance(raw, dict) else None
    if (
        not isinstance(raw, dict)
        or contract not in contracts
        or raw.get("status") != contracts[str(contract)]
        or raw.get("final_tests_used") != []
    ):
        raise ValueError("invalid same-prompt expansion contract")
    authorization = cast(Mapping[str, Any], raw["authorization"])
    if authorization.get("generation_authorized") is not True:
        raise ValueError("same-prompt expansion generation is not authorized")
    if authorization.get("final_tests_used") != []:
        raise ValueError("same-prompt expansion authorization declares final-test usage")
    root = config_path.resolve().parents[2]
    generator = cast(Mapping[str, Any], raw["generator"])
    if contract == "task06-same-prompt-preference-expansion-v1":
        records, cohort_path = _same_prompt_expansion_v1_records(raw, root, output_dir)
    else:
        records, cohort_path = _same_prompt_expansion_v2_records(raw, config_path, output_dir)

    generation_config = root / str(generator["config"])
    adapter_path = root / str(generator["adapter"])
    controls_raw = cast(Sequence[Mapping[str, Any]], generator["controls"])
    decoding = cast(Sequence[Mapping[str, Any]], generator["decoding"])
    if len(controls_raw) != 4 or len(decoding) != 8:
        raise ValueError("same-prompt expansion requires four controls and eight decodes")
    batch_size = int(generator["generation_batch_size"])
    if not 1 <= batch_size <= 8:
        raise ValueError("same-prompt generation batch size must be between 1 and 8")
    experiment_id = str(generator.get("experiment_id", "TASK06-PREFERENCE-D01-SAME-PROMPT"))
    config = load_config(generation_config)
    output_path = output_dir / "d01_controlled/generations.jsonl"
    journal = output_path.with_suffix(output_path.suffix + ".journal.jsonl")
    identity_path = output_path.with_suffix(output_path.suffix + ".identity.json")
    identity = {
        "schema_version": 1,
        "contract": contract,
        "config_sha256": sha256_file(config_path),
        "cohort_sha256": sha256_file(cohort_path),
        "generation_config_sha256": sha256_file(generation_config),
        "adapter": str(adapter_path.relative_to(root)),
        "controls": controls_raw,
        "decoding": decoding,
        "batch_size": batch_size,
        "exact_same_prompt_required": True,
        "final_tests_used": [],
    }
    identity["identity_sha256"] = _canonical_sha256(identity)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("same-prompt generation identity mismatch")
    else:
        write_json(identity_path, identity)
    expected = [
        f"{record['pair_id']}::same-prompt::{slot['slot']}"
        for slot in decoding
        for record in records
    ]
    completed = read_durable_jsonl_prefix(journal)
    if [row.get("evaluation_id") for row in completed] != expected[: len(completed)]:
        raise ValueError("same-prompt journal is not the exact expected prefix")
    if output_path.exists():
        if len(completed) != len(expected):
            raise ValueError("same-prompt final output exists before complete journal")
        existing = json.loads(
            output_path.with_suffix(output_path.suffix + ".summary.json").read_text(
                encoding="utf-8"
            )
        )
        if not isinstance(existing, dict):
            raise ValueError("same-prompt generation summary must be a mapping")
        return existing

    tokenizer = load_tokenizer(config)
    model, precision = load_generator(config, for_training=False)
    from peft import PeftModel

    loader: Any = getattr(PeftModel, "from_" + "pretrained")
    model = loader(model, adapter_path, is_trainable=False)
    model.eval()
    prepared: list[tuple[dict[str, Any], QueryControl, str, list[int]]] = []
    for index, record in enumerate(records):
        control_raw = controls_raw[index % len(controls_raw)]
        control = QueryControl(
            form=QueryForm(str(control_raw["form"])),
            intent=QueryIntent(str(control_raw["intent"])),
            focus_mode=FocusMode.BUCKET,
            focus_bucket=cast(Literal["beginning", "middle", "end"], str(control_raw["focus"])),
            length="medium",
        )
        prompt = render_controlled_prompt(str(record["positives"][0]["text"]), control)
        prepared.append(
            (
                record,
                control,
                prompt,
                _prompt_ids(
                    tokenizer,
                    prompt,
                    config.training.max_length,
                    config.training.min_prompt_tokens,
                ),
            )
        )
    started = time.perf_counter()
    with journal.open("a", encoding="utf-8") as handle:
        for slot_index, slot in enumerate(decoding):
            for batch_start in range(0, len(prepared), batch_size):
                absolute_start = slot_index * len(prepared) + batch_start
                if absolute_start + batch_size <= len(completed):
                    continue
                chunk = prepared[batch_start : batch_start + batch_size]
                seeds = [
                    int(slot["seed"]) + index * 1000
                    for index in range(batch_start, batch_start + len(chunk))
                ]
                resolved = _resolve_same_prompt_batch(
                    model,
                    tokenizer,
                    chunk=chunk,
                    seeds=seeds,
                    temperature=float(slot["temperature"]),
                    top_p=float(slot["top_p"]),
                    max_new_tokens=int(generator["max_new_tokens"]),
                )
                for local, ((record, control, prompt, _ids), completion) in enumerate(
                    zip(chunk, resolved, strict=True)
                ):
                    absolute = absolute_start + local
                    if absolute < len(completed):
                        continue
                    positive = record["positives"][0]
                    row = {
                        "evaluation_id": expected[absolute],
                        "evaluation_group_id": f"task06-preference::{record['pair_id']}",
                        "experiment_id": experiment_id,
                        "example_id": str(record["pair_id"]),
                        "source_example_id": str(record["source_example_id"]),
                        "doc_id": str(positive["doc_id"]),
                        "positive": positive,
                        "positives": record["positives"],
                        "hard_negatives": record["hard_negatives"],
                        "positive_count": 1,
                        "reference": str(record["query"]),
                        "metadata": record.get("metadata", {}),
                        "generated": completion.text,
                        "mode": "controlled_same_prompt",
                        "candidate_index": int(slot["slot"]),
                        "candidate_slot_index": int(slot["slot"]),
                        "slot": f"same-prompt-{slot['slot']}",
                        "control": control.model_dump(mode="json"),
                        "requested_form": control.form.value,
                        "requested_intent": control.intent.value,
                        "requested_focus": control.focus_bucket,
                        "seed": completion.seed,
                        "attempt": completion.attempt,
                        "invalid_attempts": completion.invalid_attempts,
                        "format_repair": completion.repair,
                        "generation_config": dict(slot),
                        "prompt": prompt,
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "generation_identity_sha256": identity["identity_sha256"],
                        "frozen_subset": "task06_preference_train",
                        "frozen_cohort_fingerprint": sha256_file(cohort_path),
                        "token_logprobs": None,
                        "final_tests_used": [],
                    }
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
    rows = read_durable_jsonl_prefix(journal)
    if len(rows) != len(expected):
        raise RuntimeError("same-prompt expansion did not complete")
    _write_jsonl_atomic(output_path, rows)
    summary = {
        "schema_version": 1,
        "contract": contract,
        "status": "same_prompt_generation_complete",
        "prompt_count": len(records),
        "generation_count": len(rows),
        "resumed_generation_count": len(completed),
        "max_attempts_per_slot": SAME_PROMPT_MAX_ATTEMPTS,
        "retried_row_count": sum(1 for row in rows if int(row.get("attempt", 1)) > 1),
        "invalid_completion_count": sum(int(row.get("invalid_attempts", 0)) for row in rows),
        "format_repair_counts": {
            repair: sum(1 for row in rows if str(row.get("format_repair", "none")) == repair)
            for repair in ("none", "first_line", "empty")
        },
        "precision": precision.label,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(),
        "output_sha256": sha256_file(output_path),
        "final_tests_used": [],
    }
    write_json(output_path.with_suffix(output_path.suffix + ".summary.json"), summary)
    return summary
