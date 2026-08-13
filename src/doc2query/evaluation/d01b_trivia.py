"""Prospective external TriviaQA development-cohort preparation for Task 05."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from doc2query.utils.records import JsonlWriter, JsonParquetWriter, read_records, write_json

CONTRACT = "task05-d01b-trivia-external-dev-cohort-v1"
SOURCE_SHA256 = "2ed9f62ae99b3c8e66274e70e9af975e10feaf31b1f154c7976ab24dccda10ac"
README_SHA256 = "bb88a54c3e3b24f377d06f8665a75e0d4e4ea5bd120eec419204b3e6a50a5f3b"
SOURCE_QUERY_COUNT = 60413
THRESHOLD = 23.5
SELECTION_SEED = 20260810
SELECTED_QUERY_COUNT = 8000
NEAR_DUPLICATE_JACCARD = 0.85


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _ids_sha256(ids: Sequence[str], *, sort_ids: bool = False) -> str:
    values = sorted(ids) if sort_ids else list(ids)
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def _records_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda row: str(row["example_id"])):
        payload = json.dumps(
            record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        digest.update(payload)
        digest.update(b"\n")
    return digest.hexdigest()


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _text_sha256(text: str) -> str:
    return hashlib.sha256(_normalize(text).encode()).hexdigest()


def _shingles(text: str, size: int = 5) -> frozenset[tuple[str, ...]]:
    words = _normalize(text).split()
    if len(words) < size:
        return frozenset({tuple(words)}) if words else frozenset()
    return frozenset(tuple(words[index : index + size]) for index in range(len(words) - size + 1))


def _aligned_list(row: Mapping[str, Any], field: str, expected: int | None = None) -> list[Any]:
    value = row.get(field)
    if not isinstance(value, list) or (expected is not None and len(value) != expected):
        raise ValueError(f"TriviaQA field {field} has invalid shape")
    return value


def _validated_metadata(row: Mapping[str, Any]) -> tuple[str, list[int]]:
    query_id = str(row.get("query_id", "")).strip()
    if not query_id or not str(row.get("query", "")).strip():
        raise ValueError("TriviaQA query ID/text is empty")
    if bool(row.get("translation_missing")):
        raise ValueError("TriviaQA Polish translation is marked missing")
    positives = _aligned_list(row, "pos")
    count = len(positives)
    pos_ids = [str(value).strip() for value in _aligned_list(row, "pos_id", count)]
    scores = _aligned_list(row, "pos_scores_stronger_reranker", count)
    _aligned_list(row, "pos_is_synthetic", count)
    negatives = _aligned_list(row, "neg", 10)
    neg_ids = [str(value).strip() for value in _aligned_list(row, "neg_id", 10)]
    _aligned_list(row, "neg_selection_tier", 10)
    if len(set(neg_ids)) != 10 or any(not value for value in neg_ids):
        raise ValueError("TriviaQA negatives require ten unique non-empty IDs")
    retained = [index for index, score in enumerate(scores) if float(score) > THRESHOLD]
    if any(not pos_ids[index] for index in retained):
        raise ValueError("retained TriviaQA positive ID is empty")
    if set(pos_ids[index] for index in retained) & set(neg_ids):
        raise ValueError("retained TriviaQA positive overlaps a negative ID")
    if len(negatives) != 10:
        raise AssertionError("negative shape validation drifted")
    return query_id, retained


def _pilot_training_passages(paths: Sequence[Path]) -> list[str]:
    by_source_id: dict[str, str] = {}
    for path in paths:
        for row in read_records(path):
            source_id = str(row["source_passage_id"])
            positive = cast(Mapping[str, Any], row["positive"])
            text = str(positive["text"])
            previous = by_source_id.setdefault(source_id, text)
            if previous != text:
                raise ValueError("pilot source passage ID maps to multiple texts")
    if not by_source_id:
        raise ValueError("pilot training passages are empty")
    return list(by_source_id.values())


def _near_duplicate_count(texts: Sequence[str], pilot_texts: Sequence[str]) -> int:
    pilot_sets = [_shingles(text) for text in pilot_texts]
    postings: dict[tuple[str, ...], set[int]] = defaultdict(set)
    for index, shingles in enumerate(pilot_sets):
        for shingle in shingles:
            postings[shingle].add(index)
    matches = 0
    for text in texts:
        shingles = _shingles(text)
        candidates: set[int] = set()
        for shingle in shingles:
            candidates.update(postings.get(shingle, ()))
        if any(
            len(shingles & pilot_sets[index]) / max(1, len(shingles | pilot_sets[index]))
            >= NEAR_DUPLICATE_JACCARD
            for index in candidates
        ):
            matches += 1
    return matches


def _document(doc_id: str, text: str, *, role: str, metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "doc_id": f"trivia::{doc_id}",
        "text": text,
        "metadata": {"source": "mining-negatives/trivia-mined-negatives", "role": role}
        | dict(metadata),
    }


def prepare_trivia_external_dev(
    *,
    source_path: Path,
    readme_path: Path,
    policy_path: Path,
    pilot_inputs: Sequence[Path],
    output_dir: Path,
) -> dict[str, Any]:
    """Audit IDs/metadata and atomically materialize the frozen external cohort."""
    if sha256_file(source_path) != SOURCE_SHA256 or sha256_file(readme_path) != README_SHA256:
        raise ValueError("TriviaQA source or dataset card fingerprint drifted")
    if not policy_path.is_file():
        raise ValueError("prospective TriviaQA cohort policy is missing")
    if output_dir.exists():
        raise FileExistsError(f"external cohort already exists: {output_dir}")
    pilot_texts = _pilot_training_passages(pilot_inputs)
    pilot_hashes = {_text_sha256(text) for text in pilot_texts}

    source_count = 0
    eligible: list[tuple[str, str]] = []
    seen_query_ids: set[str] = set()
    retained_positive_count = 0
    exact_overlap_excluded = 0
    ineligible_no_strong_positive = 0
    translation_missing_excluded = 0
    for row in read_records(source_path):
        source_count += 1
        if bool(row.get("translation_missing")):
            translation_missing_excluded += 1
            continue
        query_id, retained = _validated_metadata(row)
        if query_id in seen_query_ids:
            raise ValueError("duplicate TriviaQA query_id")
        seen_query_ids.add(query_id)
        if not retained:
            ineligible_no_strong_positive += 1
            continue
        positives = cast(list[str], row["pos"])
        if any(_text_sha256(positives[index]) in pilot_hashes for index in retained):
            exact_overlap_excluded += 1
            continue
        retained_positive_count += len(retained)
        order = hashlib.sha256(f"{SELECTION_SEED}:{query_id}".encode()).hexdigest()
        eligible.append((order, query_id))
    eligible.sort()
    selected_ids = [query_id for _order, query_id in eligible[:SELECTED_QUERY_COUNT]]
    if source_count != SOURCE_QUERY_COUNT or len(selected_ids) != SELECTED_QUERY_COUNT:
        raise ValueError("TriviaQA population cannot satisfy the frozen cohort size")
    selected_set = set(selected_ids)

    stage = output_dir.with_name(output_dir.name + ".staging")
    if stage.exists():
        raise FileExistsError(f"stale TriviaQA staging directory: {stage}")
    stage.mkdir(parents=True)
    records_path = stage / "dev.jsonl"
    documents_path = stage / "documents.parquet"
    ids_path = stage / "dev.ids.jsonl"
    manifest_path = stage / "manifest.json"
    documents: dict[str, dict[str, Any]] = {}
    selected_records: list[dict[str, Any]] = []
    selected_positive_texts: list[str] = []
    try:
        for row in read_records(source_path):
            if bool(row.get("translation_missing")):
                continue
            query_id, retained = _validated_metadata(row)
            if query_id not in selected_set:
                continue
            positives = cast(list[str], row["pos"])
            pos_ids = cast(list[str], row["pos_id"])
            scores = cast(list[float], row["pos_scores_stronger_reranker"])
            synthetic = cast(list[bool], row["pos_is_synthetic"])
            negatives = cast(list[str], row["neg"])
            neg_ids = cast(list[str], row["neg_id"])
            tiers = cast(list[str], row["neg_selection_tier"])
            positive_docs = [
                _document(
                    pos_ids[index],
                    positives[index],
                    role="positive",
                    metadata={
                        "pos_scores_stronger_reranker": float(scores[index]),
                        "pos_is_synthetic": bool(synthetic[index]),
                    },
                )
                for index in retained
            ]
            negative_docs = [
                _document(
                    doc_id,
                    text,
                    role="hard_negative",
                    metadata={"neg_selection_tier": str(tier)},
                )
                for doc_id, text, tier in zip(neg_ids, negatives, tiers, strict=True)
            ]
            for document in positive_docs + negative_docs:
                doc_id = str(document["doc_id"])
                previous = documents.setdefault(doc_id, document)
                if previous["text"] != document["text"]:
                    raise ValueError("TriviaQA document ID maps to multiple texts")
            selected_positive_texts.extend(str(document["text"]) for document in positive_docs)
            selected_records.append(
                {
                    "example_id": query_id,
                    "query": str(row["query"]),
                    "positives": positive_docs,
                    "hard_negatives": negative_docs,
                    "metadata": {
                        "source": "mining-negatives/trivia-mined-negatives",
                        "source_file_sha256": SOURCE_SHA256,
                        "language": "pl",
                        "split": "external_dev",
                    },
                }
            )
        selected_rank = {query_id: index for index, query_id in enumerate(selected_ids)}
        selected_records.sort(key=lambda item: selected_rank[str(item["example_id"])])
        if len(selected_records) != SELECTED_QUERY_COUNT:
            raise ValueError("selected TriviaQA records are incomplete")
        near_duplicates = _near_duplicate_count(selected_positive_texts, pilot_texts)
        if near_duplicates:
            raise ValueError(
                "selected TriviaQA positives near-duplicate pilot training passages: "
                f"{near_duplicates}"
            )
        with JsonlWriter(records_path) as writer:
            for record in selected_records:
                writer.write(record)
        with JsonlWriter(ids_path) as writer:
            for query_id in sorted(selected_ids):
                writer.write({"id": query_id})
        with JsonParquetWriter(documents_path) as writer:
            for doc_id in sorted(documents):
                writer.write(documents[doc_id])
        records_sha256 = sha256_file(records_path)
        ids_sha256 = _ids_sha256(selected_ids)
        ids_sorted_sha256 = _ids_sha256(selected_ids, sort_ids=True)
        manifest = {
            "schema_version": 1,
            "version": CONTRACT,
            "status": "materialized_before_model_evaluation",
            "seed": SELECTION_SEED,
            "selection_policy": {
                "unit": "query_id",
                "order": 'sha256("20260810:<query_id>"), query_id',
                "positive_filter": "pos_scores_stronger_reranker > 23.50",
                "selected_count": SELECTED_QUERY_COUNT,
                "quality_outcomes_used": [],
                "split_mutation": False,
            },
            "audit": {
                "source_count": source_count,
                "translation_missing_excluded": translation_missing_excluded,
                "ineligible_no_strong_positive": ineligible_no_strong_positive,
                "eligible_after_exact_training_overlap_exclusion": len(eligible),
                "retained_positive_count_in_eligible_population": retained_positive_count,
                "exact_positive_training_overlap_excluded_queries": exact_overlap_excluded,
                "selected_positive_near_duplicate_training_count": 0,
                "pilot_training_passage_count": len(pilot_texts),
                "raw_ids_reported": False,
                "text_manually_inspected": False,
            },
            "source": {
                "path": str(source_path),
                "sha256": SOURCE_SHA256,
                "dataset_card": str(readme_path),
                "dataset_card_sha256": README_SHA256,
                "policy": str(policy_path),
                "policy_sha256": sha256_file(policy_path),
            },
            "sets": {
                "dev_d01b_trivia_external_v1": {
                    "name": "dev_d01b_trivia_external_v1",
                    "source_path": str(output_dir / records_path.name),
                    "source_sha256": records_sha256,
                    "id_path": str(output_dir / ids_path.name),
                    "id_field": "example_id",
                    "id_count": SELECTED_QUERY_COUNT,
                    "id_list_sha256": ids_sorted_sha256,
                    "records_sha256": _records_sha256(selected_records),
                    "population_count": len(eligible),
                    "excluded_count": len(eligible) - SELECTED_QUERY_COUNT,
                    "exclusion_reason": "prospective deterministic external-dev cohort",
                }
            },
            "selected_id_list_sha256_selection_order": ids_sha256,
            "documents": {
                "path": str(output_dir / documents_path.name),
                "sha256": sha256_file(documents_path),
                "count": len(documents),
            },
            "authorization": {
                "probe_training": False,
                "model_evaluation": False,
                "final_tests": False,
            },
            "selection_claim": None,
            "retained_for_finalist_freeze": False,
            "four_point_five_b_full_authorized": False,
            "final_tests_used": [],
        }
        write_json(manifest_path, manifest)
        os.replace(stage, output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
