"""Deterministic, development-only P-05 common-cohort materialization."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from doc2query.evaluation.corpus import sha256_file
from doc2query.evaluation.statistical_contract import (
    BUDGET_DEFINITION_VERSION,
    BUDGET_FIELDS,
)
from doc2query.utils.records import JsonlWriter, JsonParquetWriter, read_records, write_json

MATERIALIZATION_SCHEMA_VERSION = 1
W05_GENERATOR_ID = "W05-1.5B-50K-8GB"
_ALLOWED_SPLITS = {"train", "dev"}
_FORBIDDEN_DATASET_NAMES = {
    "test_native_pl",
    "test_translated_msmarco_pl",
    "test_embedder",
    "test_intrinsic",
    "test_adversarial",
    "final_test",
    "final_tests",
}


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _load_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _assert_development_path(path: Path, label: str) -> None:
    lowered = path.name.lower()
    if any(name in lowered for name in _FORBIDDEN_DATASET_NAMES):
        raise ValueError(f"{label} points to a forbidden final-test artifact: {path}")


def _contains_forbidden_reference(value: Any) -> bool:
    if isinstance(value, str):
        lowered = value.lower()
        return any(name in lowered for name in _FORBIDDEN_DATASET_NAMES)
    if isinstance(value, Mapping):
        return any(_contains_forbidden_reference(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_forbidden_reference(item) for item in value)
    return False


def _validate_fingerprint_manifest(
    path: Path,
    *,
    artifact_path: Path,
    label: str,
) -> dict[str, Any]:
    _assert_development_path(path, label)
    raw = _load_object(path, label)
    if raw.get("final_tests_used") != []:
        raise ValueError(f"{label} must declare final_tests_used=[]")
    if _contains_forbidden_reference(raw):
        raise ValueError(f"{label} references a forbidden final-test dataset")
    splits = raw.get("splits")
    if not isinstance(splits, list) or not splits or not set(map(str, splits)) <= _ALLOWED_SPLITS:
        raise ValueError(f"{label} must be restricted to train/dev splits")
    declared_path = raw.get("artifact_path")
    if (
        not isinstance(declared_path, str)
        or Path(declared_path).resolve() != artifact_path.resolve()
    ):
        raise ValueError(f"{label} artifact_path does not match the explicit input")
    expected = raw.get("sha256")
    if not isinstance(expected, str) or expected != sha256_file(artifact_path):
        raise ValueError(f"{label} fingerprint drift")
    return raw


def _validate_budget(raw: Mapping[str, Any], expected_k: int) -> dict[str, Any]:
    if raw.get("definition_version") != BUDGET_DEFINITION_VERSION:
        raise ValueError(f"P-05 requires budget definition {BUDGET_DEFINITION_VERSION}")
    for field in BUDGET_FIELDS:
        value = raw.get(field)
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"P-05 budget requires positive integer {field}")
    pair_count = int(raw["pair_count"])
    passage_count = int(raw["unique_passage_count"])
    if int(raw["queries_per_passage"]) != expected_k:
        raise ValueError("P-05 budget queries_per_passage differs from explicit uniform K")
    if pair_count != passage_count * expected_k:
        raise ValueError("P-05 budget pair_count is not divisible into uniform K")
    if pair_count % 8 or passage_count % 4 or int(raw["token_count"]) % 4:
        raise ValueError(
            "P-05 budget must be divisible for an exact 25% prefix and exact 50/50 mix"
        )
    return {
        "definition_version": BUDGET_DEFINITION_VERSION,
        **{field: int(raw[field]) for field in BUDGET_FIELDS},
    }


def _natural_identity(record: Mapping[str, Any]) -> tuple[str, str, str, str]:
    pair_id = str(record.get("pair_id") or record.get("example_id") or "").strip()
    doc_id = str(record.get("doc_id") or "").strip()
    passage = str(record.get("passage") or "").strip()
    positives = record.get("positives")
    if isinstance(positives, list) and positives:
        first = positives[0]
        if isinstance(first, Mapping):
            doc_id = doc_id or str(first.get("doc_id") or "").strip()
            passage = passage or str(first.get("text") or "").strip()
    query = str(record.get("query") or "").strip()
    if not pair_id or not doc_id or not passage or not query:
        raise ValueError("natural pairs require pair_id/example_id, doc_id, passage and query")
    split = record.get("split")
    if split not in _ALLOWED_SPLITS:
        raise ValueError("P-05 natural pairs must explicitly use only train/dev")
    return pair_id, doc_id, passage, query


def _generation_identity(record: Mapping[str, Any]) -> tuple[str, str]:
    pair_id = str(record.get("pair_id") or record.get("example_id") or "").strip()
    query = str(record.get("generated") or record.get("query") or "").strip()
    if not pair_id or not query:
        raise ValueError("W05 generations require pair_id/example_id and a non-empty query")
    if "split" in record and record.get("split") not in _ALLOWED_SPLITS:
        raise ValueError("P-05 W05 generations may use only train/dev")
    if "generator_id" in record and record.get("generator_id") != W05_GENERATOR_ID:
        raise ValueError("W05 generation has an incompatible generator_id")
    return pair_id, query


def _write_records(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if path.suffix == ".jsonl":
        writer_type: type[JsonlWriter] | type[JsonParquetWriter] = JsonlWriter
    elif path.suffix == ".parquet":
        writer_type = JsonParquetWriter
    else:
        raise ValueError("P-05 outputs must have .jsonl or .parquet suffix")
    with writer_type(path) as writer:
        for row in rows:
            writer.write(row)


def _stage_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    screen = len(rows) // 4
    return {
        "dev_screen": dict(Counter(str(row["query_source"]) for row in rows[:screen])),
        "dev_confirm": dict(Counter(str(row["query_source"]) for row in rows)),
    }


def materialize_p05_cohort(
    *,
    natural_pairs_path: Path,
    w05_generations_path: Path,
    natural_fingerprint_path: Path,
    w05_fingerprint_path: Path,
    budget_path: Path,
    gold_output_path: Path,
    synthetic_output_path: Path,
    mixed_output_path: Path,
    manifest_output_path: Path,
    seed: int,
    queries_per_passage: int | None = None,
) -> dict[str, Any]:
    """Materialize three comparison arms without reading tests or executing a plan."""
    explicit_paths = (
        natural_pairs_path,
        w05_generations_path,
        natural_fingerprint_path,
        w05_fingerprint_path,
        budget_path,
        gold_output_path,
        synthetic_output_path,
        mixed_output_path,
        manifest_output_path,
    )
    for path in explicit_paths:
        _assert_development_path(path, "P-05 path")
    output_paths = {
        gold_output_path.resolve(),
        synthetic_output_path.resolve(),
        mixed_output_path.resolve(),
        manifest_output_path.resolve(),
    }
    if len(output_paths) != 4:
        raise ValueError("P-05 output paths must be distinct")
    input_paths = {path.resolve() for path in explicit_paths[:5]}
    if output_paths & input_paths:
        raise ValueError("P-05 outputs must not overwrite any explicit input")
    if queries_per_passage is not None and queries_per_passage < 1:
        raise ValueError("queries_per_passage must be positive")
    expected_k = queries_per_passage or 1

    _validate_fingerprint_manifest(
        natural_fingerprint_path, artifact_path=natural_pairs_path, label="natural fingerprint"
    )
    w05_manifest = _validate_fingerprint_manifest(
        w05_fingerprint_path, artifact_path=w05_generations_path, label="W05 fingerprint"
    )
    natural_sha = sha256_file(natural_pairs_path)
    if w05_manifest.get("generator_id") != W05_GENERATOR_ID:
        raise ValueError("W05 fingerprint manifest has an incompatible generator_id")
    if w05_manifest.get("source_data_sha256") != natural_sha:
        raise ValueError("W05 provenance does not match the canonical natural pairs")
    budget = _validate_budget(_load_object(budget_path, "P-04 budget"), expected_k)

    natural_by_pair: dict[str, dict[str, Any]] = {}
    identities: dict[str, tuple[str, str, str]] = {}
    example_ids: set[str] = set()
    passages_by_doc: dict[str, str] = {}
    pairs_by_doc: dict[str, list[str]] = defaultdict(list)
    for record in read_records(natural_pairs_path):
        pair_id, doc_id, passage, query = _natural_identity(record)
        if pair_id in natural_by_pair:
            raise ValueError(f"duplicate natural pair_id: {pair_id}")
        example_id = str(record.get("example_id") or pair_id)
        if example_id in example_ids:
            raise ValueError(f"duplicate natural example_id: {example_id}")
        example_ids.add(example_id)
        if doc_id in passages_by_doc and passages_by_doc[doc_id] != passage:
            raise ValueError(f"different passage text for doc_id {doc_id}")
        passages_by_doc[doc_id] = passage
        natural_by_pair[pair_id] = deepcopy(record)
        identities[pair_id] = (doc_id, passage, query)
        pairs_by_doc[doc_id].append(pair_id)

    w05_by_pair: dict[str, dict[str, Any]] = {}
    for record in read_records(w05_generations_path):
        pair_id, _query = _generation_identity(record)
        if pair_id in w05_by_pair:
            raise ValueError(f"duplicate W05 pair_id: {pair_id}")
        if pair_id not in natural_by_pair:
            raise ValueError(f"W05 pair_id is absent from natural pairs: {pair_id}")
        doc_id, passage, _natural_query = identities[pair_id]
        generated_doc = str(record.get("doc_id") or doc_id)
        generated_passage = str(record.get("passage") or passage)
        if generated_doc != doc_id or generated_passage != passage:
            raise ValueError(f"different passage provenance for pair_id {pair_id}")
        w05_by_pair[pair_id] = deepcopy(record)
    if set(w05_by_pair) != set(natural_by_pair):
        missing = sorted(set(natural_by_pair) - set(w05_by_pair))
        raise ValueError(f"missing W05 query for natural pair_id: {missing[0]}")

    counts = {len(pair_ids) for pair_ids in pairs_by_doc.values()}
    if counts != {expected_k}:
        raise ValueError("inputs do not provide the explicit uniform K queries per passage")
    passage_budget = int(budget["unique_passage_count"])
    if len(pairs_by_doc) < passage_budget:
        raise ValueError("P-04 budget exceeds the available common cohort")
    ordered_docs = sorted(
        pairs_by_doc,
        key=lambda doc_id: (hashlib.sha256(f"{seed}:{doc_id}".encode()).hexdigest(), doc_id),
    )[:passage_budget]
    ordered_pairs: list[str] = []
    for doc_id in ordered_docs:
        ordered_pairs.extend(
            sorted(
                pairs_by_doc[doc_id],
                key=lambda pair_id: (
                    hashlib.sha256(f"{seed}:{doc_id}:{pair_id}".encode()).hexdigest(),
                    pair_id,
                ),
            )
        )
    if len(ordered_pairs) != int(budget["pair_count"]):
        raise ValueError("materialized cohort does not match the P-04 pair budget")

    gold_rows: list[dict[str, Any]] = []
    synthetic_rows: list[dict[str, Any]] = []
    mixed_rows: list[dict[str, Any]] = []
    cohort_items: list[dict[str, str]] = []
    for index, pair_id in enumerate(ordered_pairs):
        doc_id, passage, natural_query = identities[pair_id]
        synthetic_query = str(
            w05_by_pair[pair_id].get("generated") or w05_by_pair[pair_id].get("query")
        ).strip()
        natural_row = deepcopy(natural_by_pair[pair_id])
        natural_row["p05_pair_id"] = pair_id
        natural_row["p05_doc_id"] = doc_id
        natural_row["query_source"] = "natural"
        gold_rows.append(natural_row)
        common_generation = {
            "example_id": str(natural_row.get("example_id") or pair_id),
            "pair_id": pair_id,
            "doc_id": doc_id,
            "passage": passage,
            "mode": "deterministic",
            "candidate_index": 0,
        }
        synthetic_rows.append(
            common_generation
            | {
                "generated": synthetic_query,
                "query_source": "synthetic_w05",
                "generator_id": W05_GENERATOR_ID,
            }
        )
        use_natural = index % 2 == 0
        mixed_rows.append(
            common_generation
            | {
                "generated": natural_query if use_natural else synthetic_query,
                "query_source": "natural" if use_natural else "synthetic_w05",
                "generator_id": (
                    "natural-gold" if use_natural else W05_GENERATOR_ID
                ),
            }
        )
        cohort_items.append(
            {
                "doc_id": doc_id,
                "pair_id": pair_id,
                "passage_sha256": hashlib.sha256(passage.encode()).hexdigest(),
            }
        )

    expected_half = len(mixed_rows) // 2
    mixed_stages = _stage_counts(mixed_rows)
    if mixed_stages != {
        "dev_screen": {"natural": expected_half // 4, "synthetic_w05": expected_half // 4},
        "dev_confirm": {"natural": expected_half, "synthetic_w05": expected_half},
    }:
        raise AssertionError("internal P-05 ordering failed exact 50/50 allocation")

    _write_records(gold_output_path, gold_rows)
    _write_records(synthetic_output_path, synthetic_rows)
    _write_records(mixed_output_path, mixed_rows)
    outputs = {
        "gold_natural": gold_output_path,
        "w05_synthetic": synthetic_output_path,
        "mixed_50_50": mixed_output_path,
    }
    manifest: dict[str, Any] = {
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "materialization_id": "task03-p05-common-cohort-v1",
        "seed": seed,
        "inputs": {
            "natural_pairs": {
                "path": str(natural_pairs_path.resolve()),
                "sha256": natural_sha,
                "fingerprint_manifest_path": str(natural_fingerprint_path.resolve()),
                "fingerprint_manifest_sha256": sha256_file(natural_fingerprint_path),
            },
            "w05_generations": {
                "path": str(w05_generations_path.resolve()),
                "sha256": sha256_file(w05_generations_path),
                "fingerprint_manifest_path": str(w05_fingerprint_path.resolve()),
                "fingerprint_manifest_sha256": sha256_file(w05_fingerprint_path),
                "generator_id": W05_GENERATOR_ID,
            },
            "budget": {"path": str(budget_path.resolve()), "sha256": sha256_file(budget_path)},
        },
        "outputs": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "pair_count": len(gold_rows),
                "token_count": int(budget["token_count"]),
                "unique_passage_count": len(ordered_docs),
                "queries_per_passage": expected_k,
                "source_counts": (
                    {"natural": len(gold_rows)}
                    if name == "gold_natural"
                    else {"synthetic_w05": len(synthetic_rows)}
                    if name == "w05_synthetic"
                    else dict(Counter(str(row["query_source"]) for row in mixed_rows))
                ),
            }
            for name, path in outputs.items()
        },
        "stage_pair_counts": mixed_stages,
        "cohort_fingerprint": _canonical_hash(cohort_items),
        "cohort_order": "sha256(seed:doc_id), then sha256(seed:doc_id:pair_id)",
        "comparison_budget": budget,
        "negative_recipe": {"strategy": "HN0+filter", "false_negative_policy": "drop"},
        "final_tests_used": [],
    }
    write_json(manifest_output_path, manifest)
    return manifest
