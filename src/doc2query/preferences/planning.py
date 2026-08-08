"""Quality-blind planning of controlled candidate-generation requests."""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from doc2query.data.focus_labels import split_sentences
from doc2query.data.style_labels import intent_applicable
from doc2query.models.templates import render_controlled_prompt
from doc2query.preferences.schemas import CandidateGenerationRequest, CandidatePlanningConfig
from doc2query.schemas import FocusMode, QueryControl
from doc2query.utils.records import read_records, write_json


@dataclass(frozen=True)
class PlanningPassage:
    passage_id: str
    passage_cluster_id: str
    passage: str
    split: Literal["train", "dev"]
    source_pair_ids: tuple[str, ...]


@dataclass(frozen=True)
class _RequestChoice:
    control: QueryControl
    temperature: float
    seed: int

    def axes(self) -> tuple[str, str, str, str, str]:
        focus = (
            f"{self.control.focus_mode.value}:{self.control.focus_bucket}"
            if self.control.focus_bucket is not None
            else f"{self.control.focus_mode.value}:{self.control.focus_sentence_id}"
        )
        return (
            self.control.form.value,
            self.control.intent.value,
            focus,
            format(self.temperature, ".12g"),
            str(self.seed),
        )


def load_planning_config(path: Path) -> CandidatePlanningConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("candidate planning config must be a mapping")
    return CandidatePlanningConfig.model_validate(payload)


def planning_fingerprint(config: CandidatePlanningConfig) -> str:
    payload = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _focus_controls(mode: FocusMode, sentence_count: int) -> list[dict[str, Any]]:
    if mode == FocusMode.NONE:
        return [{}]
    if mode == FocusMode.BUCKET:
        return [{"focus_bucket": bucket} for bucket in ("beginning", "middle", "end")]
    if sentence_count == 0:
        return []
    sentence_ids = sorted({0, sentence_count // 2, sentence_count - 1})
    return [{"focus_sentence_id": sentence_id} for sentence_id in sentence_ids]


def _choice_tie_break(choice: _RequestChoice, passage_id: str, plan_seed: int) -> str:
    payload = {
        "axes": choice.axes(),
        "passage_id": passage_id,
        "plan_seed": plan_seed,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _candidate_choices(
    passage: PlanningPassage, config: CandidatePlanningConfig
) -> list[_RequestChoice]:
    sentence_count = len(split_sentences(passage.passage))
    pool: list[_RequestChoice] = []
    for form in config.forms:
        for intent in config.intents:
            applicable = intent_applicable(intent, passage.passage)
            if applicable is False:
                continue
            for mode in config.focus_modes:
                for focus_fields in _focus_controls(mode, sentence_count):
                    control = QueryControl(
                        form=form,
                        intent=intent,
                        intent_applicable=applicable,
                        focus_mode=mode,
                        **focus_fields,
                    )
                    for temperature in config.temperatures:
                        for seed in config.seeds:
                            pool.append(_RequestChoice(control, temperature, seed))
    if len(pool) < config.target_candidates_per_passage:
        raise ValueError(
            f"passage {passage.passage_id} has only {len(pool)} eligible generation choices"
        )
    return pool


def _coverage_first_choices(
    passage: PlanningPassage, config: CandidatePlanningConfig
) -> list[_RequestChoice]:
    remaining = _candidate_choices(passage, config)
    seen: list[set[str]] = [set() for _ in range(5)]
    selected: list[_RequestChoice] = []
    while len(selected) < config.target_candidates_per_passage:
        ranked = sorted(
            remaining,
            key=lambda choice: (
                -sum(value not in seen[index] for index, value in enumerate(choice.axes())),
                _choice_tie_break(choice, passage.passage_id, config.plan_seed),
            ),
        )
        choice = ranked[0]
        remaining.remove(choice)
        selected.append(choice)
        for index, value in enumerate(choice.axes()):
            seen[index].add(value)
    return selected


def prepare_planning_passages(
    source_records: Iterable[Mapping[str, Any]],
    dedup_records: Iterable[Mapping[str, Any]],
    config: CandidatePlanningConfig,
) -> list[PlanningPassage]:
    """Consolidate multi-positive rows and inherit leakage-safe cluster identity."""
    documents: dict[str, dict[str, Any]] = {}
    source_pairs: dict[str, set[str]] = defaultdict(set)
    allowed = set(config.allowed_splits)
    for row in source_records:
        split = str(row.get("split", ""))
        if split == "test":
            raise ValueError("candidate planning must never consume test rows")
        if split not in allowed:
            continue
        passage_id = str(row.get("doc_id", "")).strip()
        passage = str(row.get("passage", "")).strip()
        pair_id = str(row.get("pair_id", "")).strip()
        if not passage_id or not passage or not pair_id:
            raise ValueError("planning input requires doc_id, passage and pair_id")
        previous = documents.setdefault(passage_id, {"passage": passage, "split": split})
        if previous != {"passage": passage, "split": split}:
            raise ValueError(f"document identity drift for {passage_id}")
        source_pairs[passage_id].add(pair_id)
    wanted = set(documents)
    clusters: dict[str, str] = {}
    for row in dedup_records:
        doc_id = str(row.get("doc_id", ""))
        if doc_id in wanted:
            cluster_id = str(row.get("cluster_id", "")).strip()
            if not cluster_id:
                raise ValueError(f"missing cluster_id for {doc_id}")
            previous_cluster = clusters.setdefault(doc_id, cluster_id)
            if previous_cluster != cluster_id:
                raise ValueError(f"multiple clusters for document {doc_id}")
    missing = sorted(wanted - set(clusters))
    if missing:
        raise ValueError(f"dedup map is missing {len(missing)} planning documents")
    cluster_splits: dict[str, str] = {}
    passages: list[PlanningPassage] = []
    for passage_id, document in sorted(documents.items()):
        cluster_id = clusters[passage_id]
        split = str(document["split"])
        prior_split = cluster_splits.setdefault(cluster_id, split)
        if prior_split != split:
            raise ValueError(f"passage cluster {cluster_id} crosses planning splits")
        if split not in {"train", "dev"}:  # proven above, narrows the runtime contract
            raise AssertionError("unsupported planning split")
        passages.append(
            PlanningPassage(
                passage_id=passage_id,
                passage_cluster_id=cluster_id,
                passage=str(document["passage"]),
                split=cast(Literal["train", "dev"], split),
                source_pair_ids=tuple(sorted(source_pairs[passage_id])),
            )
        )
    return passages


def iter_generation_requests(
    passages: Sequence[PlanningPassage], config: CandidatePlanningConfig
) -> Iterator[CandidateGenerationRequest]:
    fingerprint = planning_fingerprint(config)
    for passage in passages:
        for candidate_index, choice in enumerate(_coverage_first_choices(passage, config)):
            identity = {
                "candidate_index": candidate_index,
                "control": choice.control.model_dump(mode="json"),
                "passage_id": passage.passage_id,
                "plan_fingerprint": fingerprint,
                "seed": choice.seed,
                "temperature": choice.temperature,
            }
            request_id = hashlib.sha256(
                json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()[:32]
            yield CandidateGenerationRequest(
                request_id=request_id,
                plan_id=config.plan_id,
                plan_fingerprint=fingerprint,
                candidate_index=candidate_index,
                passage_id=passage.passage_id,
                passage_cluster_id=passage.passage_cluster_id,
                passage=passage.passage,
                source_pair_ids=list(passage.source_pair_ids),
                split=passage.split,
                prompt=render_controlled_prompt(passage.passage, choice.control),
                control=choice.control,
                temperature=choice.temperature,
                top_p=config.top_p,
                max_new_tokens=config.max_new_tokens,
                seed=choice.seed,
            )


def _source_fingerprint(passages: Sequence[PlanningPassage]) -> str:
    digest = hashlib.sha256()
    for passage in passages:
        row = {
            "passage_cluster_id": passage.passage_cluster_id,
            "passage_id": passage.passage_id,
            "passage_sha256": hashlib.sha256(passage.passage.encode()).hexdigest(),
            "source_pair_ids": passage.source_pair_ids,
            "split": passage.split,
        }
        digest.update(json.dumps(row, sort_keys=True).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def write_generation_plan(
    source_path: Path,
    dedup_map_path: Path,
    config_path: Path,
    output_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Materialize an atomic request plan without loading a generator."""
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError("candidate planning outputs already exist")
    config = load_planning_config(config_path)
    passages = prepare_planning_passages(
        read_records(source_path), read_records(dedup_map_path), config
    )
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    request_digest = hashlib.sha256()
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for request in iter_generation_requests(passages, config):
                payload = json.dumps(
                    request.model_dump(mode="json"), ensure_ascii=False, sort_keys=True
                )
                handle.write(payload + "\n")
                request_digest.update(payload.encode())
                request_digest.update(b"\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "contract": "task06-candidate-generation-plan-v1",
        "status": "planned_not_generated",
        "plan_id": config.plan_id,
        "plan_fingerprint": planning_fingerprint(config),
        "source_fingerprint": _source_fingerprint(passages),
        "passage_count": len(passages),
        "request_count": count,
        "requests_per_passage": config.target_candidates_per_passage,
        "requests_sha256": request_digest.hexdigest(),
        "source_path": str(source_path),
        "dedup_map_path": str(dedup_map_path),
        "config_path": str(config_path),
        "output_path": str(output_path),
        "generation_started": False,
        "scoring_started": False,
        "final_tests_used": [],
    }
    write_json(manifest_path, manifest)
    return manifest
