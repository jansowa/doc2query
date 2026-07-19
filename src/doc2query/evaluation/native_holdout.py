"""Safe import and immutable manifests for native/translated Polish holdouts."""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from doc2query.evaluation.datasets import load_frozen_records
from doc2query.evaluation.translationese import aggregate_translationese
from doc2query.utils.records import JsonlWriter, read_records, write_json

HOLDOUT_SCHEMA_VERSION = 1
HOLDOUT_VERSION = "task04-native-pl-v1"
HoldoutProfile = Literal["quick", "medium", "full"]
PROFILE_LIMITS: dict[HoldoutProfile, int | None] = {
    "quick": 100,
    "medium": 500,
    "full": None,
}
DIAGNOSTIC_PROFILES: tuple[HoldoutProfile, HoldoutProfile] = ("quick", "medium")


def _canonical(record: dict[str, Any]) -> bytes:
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ids_hash(ids: Iterable[str]) -> str:
    return hashlib.sha256("".join(f"{value}\n" for value in sorted(ids)).encode()).hexdigest()


def _records_hash(records: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda row: str(row["example_id"])):
        digest.update(_canonical(record))
        digest.update(b"\n")
    return digest.hexdigest()


def _selection_key(set_name: str, identifier: str) -> tuple[str, str]:
    payload = f"{HOLDOUT_VERSION}:{set_name}:{identifier}".encode()
    return hashlib.sha256(payload).hexdigest(), identifier


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().casefold()
    if normalized in {"1", "true", "t", "yes"}:
        return True
    if normalized in {"0", "false", "f", "no", ""}:
        return False
    raise ValueError(f"unsupported PolQA relevance value: {value!r}")


def _polqa_rows(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix.casefold() == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            yield from csv.DictReader(handle)
        return
    yield from read_records(path)


@dataclass(frozen=True)
class PolQAImport:
    records: list[dict[str, Any]]
    judged_documents: list[dict[str, Any]]
    audit: dict[str, Any]


def import_polqa_test(path: Path) -> PolQAImport:
    """Convert the official PolQA test rows into the canonical retrieval contract."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_rows = 0
    for row in _polqa_rows(path):
        source_rows += 1
        split = str(row.get("split", "test")).strip().casefold()
        if split not in {"test", ""}:
            raise ValueError(f"PolQA holdout accepts only test rows, got split={split!r}")
        question_id = str(row.get("question_id", "")).strip()
        question = str(row.get("question", "")).strip()
        if not question_id or not question:
            raise ValueError("PolQA row is missing question_id or question")
        grouped[question_id].append(dict(row))

    records: list[dict[str, Any]] = []
    documents: dict[str, dict[str, Any]] = {}
    excluded_no_corpus_positive = 0
    for question_id, rows in sorted(grouped.items()):
        questions = {str(row["question"]).strip() for row in rows}
        if len(questions) != 1:
            raise ValueError(f"PolQA question text conflict for question_id={question_id}")
        positives: dict[str, dict[str, Any]] = {}
        negatives: dict[str, dict[str, Any]] = {}
        for row in rows:
            raw_passage_id = str(row.get("passage_id", "")).strip()
            passage_text = str(row.get("passage_wiki") or row.get("passage_text") or "").strip()
            if not raw_passage_id or not passage_text:
                continue
            doc_id = f"polqa:{raw_passage_id}"
            document = {
                "doc_id": doc_id,
                "text": passage_text,
                "metadata": {
                    "source": "ipipan/polqa",
                    "source_passage_id": raw_passage_id,
                    "title": str(row.get("passage_title", "")).strip(),
                    "language": "pl",
                },
            }
            previous = documents.get(doc_id)
            if previous is not None and previous["text"] != passage_text:
                raise ValueError(f"PolQA passage text conflict for passage_id={raw_passage_id}")
            documents[doc_id] = document
            target = positives if _bool(row.get("relevant")) else negatives
            target[doc_id] = document
        for doc_id in positives:
            negatives.pop(doc_id, None)
        if not positives:
            excluded_no_corpus_positive += 1
            continue
        records.append(
            {
                "example_id": f"polqa:{question_id}",
                "query": questions.pop(),
                "positives": [positives[key] for key in sorted(positives)],
                "hard_negatives": [negatives[key] for key in sorted(negatives)],
                "metadata": {
                    "source": "ipipan/polqa",
                    "source_question_id": question_id,
                    "split": "test",
                    "language": "pl",
                    "language_origin": "native_polish",
                    "usage_policy": "evaluation_only_no_tuning",
                },
            }
        )
    if not records:
        raise ValueError("PolQA import produced no test query with a corpus-backed positive")
    audit = {
        "source_rows": source_rows,
        "source_questions": len(grouped),
        "accepted_questions": len(records),
        "excluded_no_corpus_positive": excluded_no_corpus_positive,
        "judged_document_count": len(documents),
        "translationese": aggregate_translationese(str(row["query"]) for row in records),
    }
    return PolQAImport(records, [documents[key] for key in sorted(documents)], audit)


def _write_records(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with JsonlWriter(path) as writer:
        for record in records:
            writer.write(record)


def _profile_records(
    records: list[dict[str, Any]], set_name: str, profile: HoldoutProfile
) -> list[dict[str, Any]]:
    ordered = sorted(
        records,
        key=lambda row: _selection_key(set_name, str(row["example_id"])),
    )
    limit = PROFILE_LIMITS[profile]
    return ordered if limit is None else ordered[:limit]


def _diagnostic_documents(
    records: Iterable[dict[str, Any]],
    *,
    output_path: Path,
    role: str,
) -> dict[str, Any]:
    documents: dict[str, dict[str, Any]] = {}
    for record in records:
        for group in ("positives", "hard_negatives"):
            for raw_document in record.get(group, []):
                document = dict(raw_document)
                doc_id = str(document["doc_id"])
                previous = documents.get(doc_id)
                if previous is not None and str(previous["text"]) != str(document["text"]):
                    raise ValueError(f"conflicting diagnostic corpus document: {doc_id}")
                documents[doc_id] = document
    _write_records(output_path, (documents[key] for key in sorted(documents)))
    return {
        "status": "materialized",
        "path": str(output_path),
        "sha256": _sha256_file(output_path),
        "document_count": len(documents),
        "role": role,
    }


def adapt_polqa_corpus(source_path: Path, output_path: Path) -> dict[str, Any]:
    """Stream the official PolQA corpus into canonical documents with duplicate checks."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    index_path = output_path.with_suffix(output_path.suffix + ".ids.sqlite")
    if output_path.exists() or partial_path.exists() or index_path.exists():
        raise FileExistsError(f"PolQA corpus output/staging path already exists: {output_path}")
    connection = sqlite3.connect(index_path)
    connection.execute(
        "CREATE TABLE documents (doc_id TEXT PRIMARY KEY, text_sha256 TEXT NOT NULL)"
    )
    count = 0
    try:
        with JsonlWriter(partial_path) as writer:
            for row in read_records(source_path):
                raw_id = str(row.get("id", "")).strip()
                text = str(row.get("text", "")).strip()
                if not raw_id or not text:
                    raise ValueError("PolQA corpus row is missing id or text")
                doc_id = f"polqa:{raw_id}"
                text_hash = hashlib.sha256(text.encode()).hexdigest()
                try:
                    connection.execute(
                        "INSERT INTO documents(doc_id, text_sha256) VALUES (?, ?)",
                        (doc_id, text_hash),
                    )
                except sqlite3.IntegrityError as exc:
                    previous = connection.execute(
                        "SELECT text_sha256 FROM documents WHERE doc_id = ?", (doc_id,)
                    ).fetchone()
                    if previous is None or str(previous[0]) != text_hash:
                        raise ValueError(
                            f"conflicting duplicate PolQA corpus ID: {raw_id}"
                        ) from exc
                    continue
                writer.write(
                    {
                        "doc_id": doc_id,
                        "text": text,
                        "metadata": {
                            "source": "ipipan/polqa",
                            "source_passage_id": raw_id,
                            "title": str(row.get("title", "")).strip(),
                            "language": "pl",
                        },
                    }
                )
                count += 1
                if count % 100_000 == 0:
                    connection.commit()
        connection.commit()
        partial_path.replace(output_path)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise
    finally:
        connection.close()
        index_path.unlink(missing_ok=True)
    return {
        "status": "materialized",
        "path": str(output_path),
        "sha256": _sha256_file(output_path),
        "source_path": str(source_path),
        "source_sha256": _sha256_file(source_path),
        "document_count": count,
        "role": "full_profile_corpus",
    }


def _materialized_set(
    *,
    name: str,
    records_path: Path,
    records: list[dict[str, Any]],
    language_origin: str,
    output_dir: Path,
) -> dict[str, Any]:
    ids = sorted(str(record["example_id"]) for record in records)
    ids_path = output_dir / f"{name}.ids.jsonl"
    _write_records(ids_path, ({"id": identifier} for identifier in ids))
    profiles: dict[str, Any] = {}
    for profile, limit in PROFILE_LIMITS.items():
        selected = sorted(ids, key=lambda value: _selection_key(name, value))
        if limit is not None:
            selected = selected[:limit]
        profile_path = output_dir / f"{name}.{profile}.ids.jsonl"
        _write_records(profile_path, ({"id": identifier} for identifier in sorted(selected)))
        profiles[profile] = {
            "id_path": str(profile_path),
            "id_count": len(selected),
            "id_list_sha256": _ids_hash(selected),
            "selection": f"sha256({HOLDOUT_VERSION}:{name}:example_id), ascending",
            "query_limit": limit,
            "comparison_eligible": profile == "full",
        }
    return {
        "name": name,
        "status": "materialized",
        "records_path": str(records_path),
        "records_sha256": _records_hash(records),
        "source_sha256": _sha256_file(records_path),
        "id_path": str(ids_path),
        "id_count": len(ids),
        "id_list_sha256": _ids_hash(ids),
        "language_origin": language_origin,
        "usage_policy": "evaluation_only_no_tuning",
        "profiles": profiles,
        "translationese": aggregate_translationese(str(row["query"]) for row in records),
    }


def freeze_native_pl_holdout(
    *,
    translated_manifest: Path,
    output_dir: Path,
    polqa_test_path: Path | None = None,
    polqa_passages_path: Path | None = None,
    polqa_revision: str = "d78d036ef08ab3b9f4d85a2893f4d3a0c95a6f37",
) -> dict[str, Any]:
    """Freeze translated data now and native data only when the real source exists."""
    if len(polqa_revision) != 40:
        raise ValueError("PolQA revision must be a full 40-character commit")
    if polqa_test_path is None and polqa_passages_path is not None:
        raise ValueError("PolQA passages require the matching pinned test artifact")
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"frozen holdout manifest already exists: {manifest_path}")
    output_dir.mkdir(parents=True, exist_ok=False)

    translated = load_frozen_records(translated_manifest, "test_embedder")
    translated_path = output_dir / "test_translated_msmarco_pl.jsonl"
    _write_records(translated_path, translated)
    sets: dict[str, Any] = {
        "test_translated_msmarco_pl": _materialized_set(
            name="test_translated_msmarco_pl",
            records_path=translated_path,
            records=translated,
            language_origin="machine_translated_from_english",
            output_dir=output_dir,
        )
    }
    artifacts: dict[str, Any] = {}
    for profile in DIAGNOSTIC_PROFILES:
        artifacts[f"translated_{profile}_corpus"] = _diagnostic_documents(
            _profile_records(
                translated,
                "test_translated_msmarco_pl",
                profile,
            ),
            output_path=output_dir / f"test_translated_msmarco_pl.{profile}_documents.jsonl",
            role=f"{profile}_diagnostic_corpus_not_full_retrieval",
        )
    blockers: list[dict[str, str]] = []
    if polqa_test_path is None:
        sets["test_native_pl"] = {
            "name": "test_native_pl",
            "status": "missing_source_artifact",
            "source_repo": "ipipan/polqa",
            "source_revision": polqa_revision,
            "required_file": "data/test.csv or the pinned converted test Parquet",
            "records_sha256": None,
            "id_list_sha256": None,
            "usage_policy": "evaluation_only_no_tuning",
        }
        blockers.append(
            {
                "artifact": "test_native_pl",
                "reason": "official PolQA test file was not available locally",
            }
        )
    else:
        imported = import_polqa_test(polqa_test_path)
        native_path = output_dir / "test_native_pl.jsonl"
        _write_records(native_path, imported.records)
        sets["test_native_pl"] = {
            **_materialized_set(
                name="test_native_pl",
                records_path=native_path,
                records=imported.records,
                language_origin="native_polish",
                output_dir=output_dir,
            ),
            "source_repo": "ipipan/polqa",
            "source_revision": polqa_revision,
            "source_artifact_path": str(polqa_test_path),
            "source_artifact_sha256": _sha256_file(polqa_test_path),
            "import_audit": imported.audit,
        }
        for profile in DIAGNOSTIC_PROFILES:
            artifacts[f"native_{profile}_corpus"] = _diagnostic_documents(
                _profile_records(imported.records, "test_native_pl", profile),
                output_path=output_dir / f"test_native_pl.{profile}_documents.jsonl",
                role=f"{profile}_diagnostic_corpus_not_full_retrieval",
            )
        overlap_path = output_dir / "native_translated_exact_overlap.json"
        write_json(
            overlap_path,
            audit_exact_overlap(imported.records, translated),
        )
        artifacts["native_translated_exact_overlap"] = {
            "status": "materialized",
            "path": str(overlap_path),
            "sha256": _sha256_file(overlap_path),
            "near_duplicate_status": "not_measured",
        }
        if polqa_passages_path is None:
            blockers.append(
                {
                    "artifact": "native_full_corpus",
                    "reason": (
                        "official PolQA passages.jsonl was not imported; required for full profile"
                    ),
                }
            )
        else:
            artifacts["native_full_corpus"] = adapt_polqa_corpus(
                polqa_passages_path,
                output_dir / "test_native_pl.full_documents.jsonl",
            )
    manifest = {
        "schema_version": HOLDOUT_SCHEMA_VERSION,
        "version": HOLDOUT_VERSION,
        "status": "complete" if not blockers else "incomplete",
        "immutability": "existing manifest is never overwritten",
        "native_usage_policy": "final_test_only_never_for_tuning_or_threshold_selection",
        "profile_contract": {
            "quick": "100 deterministic queries; judged corpus; target 5-10 minute diagnostic",
            "medium": "500 deterministic queries; judged corpus; broader diagnostic",
            "full": "all frozen queries; full source corpus; comparison-eligible",
            "cross_profile_comparison": "forbidden",
        },
        "translated_source_manifest": str(translated_manifest),
        "translated_source_manifest_sha256": _sha256_file(translated_manifest),
        "sets": sets,
        "artifacts": artifacts,
        "blockers": blockers,
    }
    write_json(manifest_path, manifest)
    return manifest


def _load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != HOLDOUT_SCHEMA_VERSION:
        raise ValueError(f"unsupported native holdout manifest: {path}")
    return value


def holdout_set_status(manifest_path: Path, set_name: str) -> str:
    manifest = _load_manifest(manifest_path)
    spec = manifest.get("sets", {}).get(set_name)
    if not isinstance(spec, dict):
        raise KeyError(set_name)
    return str(spec.get("status", "unknown"))


def holdout_artifact_path(manifest_path: Path, artifact_name: str) -> Path | None:
    manifest = _load_manifest(manifest_path)
    artifact = manifest.get("artifacts", {}).get(artifact_name)
    if not isinstance(artifact, dict) or artifact.get("status") != "materialized":
        return None
    path = Path(str(artifact["path"]))
    if not path.is_file() or _sha256_file(path) != artifact.get("sha256"):
        raise RuntimeError(f"holdout artifact fingerprint mismatch: {artifact_name}")
    return path


def load_holdout_records(
    manifest_path: Path,
    set_name: str,
    *,
    profile: HoldoutProfile,
) -> list[dict[str, Any]]:
    """Verify base records and the selected profile ID hash before loading."""
    manifest = _load_manifest(manifest_path)
    spec = manifest.get("sets", {}).get(set_name)
    if not isinstance(spec, dict):
        raise KeyError(set_name)
    if spec.get("status") != "materialized":
        raise RuntimeError(f"holdout set {set_name} is not materialized: {spec.get('status')}")
    records_path = Path(str(spec["records_path"]))
    if _sha256_file(records_path) != spec["source_sha256"]:
        raise RuntimeError(f"frozen source fingerprint mismatch for {set_name}")
    records = list(read_records(records_path))
    if _records_hash(records) != spec["records_sha256"]:
        raise RuntimeError(f"frozen record fingerprint mismatch for {set_name}")
    profile_spec = spec.get("profiles", {}).get(profile)
    if not isinstance(profile_spec, dict):
        raise KeyError(f"unknown profile for {set_name}: {profile}")
    ids_path = Path(str(profile_spec["id_path"]))
    ids = [str(row["id"]) for row in read_records(ids_path)]
    if len(ids) != profile_spec["id_count"] or _ids_hash(ids) != profile_spec["id_list_sha256"]:
        raise RuntimeError(f"frozen profile ID-list fingerprint mismatch for {set_name}/{profile}")
    wanted = set(ids)
    selected = [row for row in records if str(row["example_id"]) in wanted]
    if len(selected) != len(wanted):
        raise RuntimeError(f"frozen profile {set_name}/{profile} is missing records")
    return sorted(selected, key=lambda row: str(row["example_id"]))


def holdout_fingerprint(manifest_path: Path, set_name: str, profile: HoldoutProfile) -> str:
    manifest = _load_manifest(manifest_path)
    spec = manifest.get("sets", {}).get(set_name)
    if not isinstance(spec, dict) or spec.get("status") != "materialized":
        raise RuntimeError(f"holdout set {set_name} is not materialized")
    profile_spec = spec.get("profiles", {}).get(profile)
    if not isinstance(profile_spec, dict):
        raise KeyError(profile)
    payload = {
        "records_sha256": spec["records_sha256"],
        "profile_id_list_sha256": profile_spec["id_list_sha256"],
        "profile": profile,
    }
    return hashlib.sha256(_canonical(payload)).hexdigest()


def verify_native_holdout_manifest(manifest_path: Path) -> dict[str, Any]:
    """Verify every materialized set/profile and preserve explicit missing artifacts."""
    manifest = _load_manifest(manifest_path)
    verified: dict[str, Any] = {}
    verified_artifacts: dict[str, dict[str, Any]] = {}
    missing: dict[str, str] = {}
    for set_name, raw_spec in manifest.get("sets", {}).items():
        if not isinstance(raw_spec, dict):
            raise ValueError(f"invalid holdout set specification: {set_name}")
        if raw_spec.get("status") != "materialized":
            missing[str(set_name)] = str(raw_spec.get("status", "unknown"))
            continue
        profiles = {}
        for profile in PROFILE_LIMITS:
            records = load_holdout_records(
                manifest_path,
                str(set_name),
                profile=profile,
            )
            profiles[profile] = {
                "count": len(records),
                "fingerprint": holdout_fingerprint(manifest_path, str(set_name), profile),
            }
        verified[str(set_name)] = profiles
    for artifact_name, raw_artifact in manifest.get("artifacts", {}).items():
        if not isinstance(raw_artifact, dict) or raw_artifact.get("status") != "materialized":
            continue
        path = Path(str(raw_artifact["path"]))
        if not path.is_file() or _sha256_file(path) != raw_artifact.get("sha256"):
            raise RuntimeError(f"holdout artifact fingerprint mismatch: {artifact_name}")
        verified_artifacts[str(artifact_name)] = {
            "path": str(path),
            "sha256": str(raw_artifact["sha256"]),
        }
    return {
        "version": manifest["version"],
        "status": "complete" if not missing else "incomplete",
        "verified": verified,
        "verified_artifacts": verified_artifacts,
        "missing": missing,
    }


def audit_exact_overlap(
    native_records: Iterable[dict[str, Any]],
    translated_records: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Report exact normalized query/document overlap without claiming near-dedup coverage."""

    def normalized(value: str) -> str:
        return " ".join(value.casefold().split())

    def inventory(records: Iterable[dict[str, Any]]) -> tuple[set[str], set[str]]:
        queries: set[str] = set()
        documents: set[str] = set()
        for record in records:
            queries.add(normalized(str(record["query"])))
            for group in ("positives", "hard_negatives"):
                documents.update(
                    hashlib.sha256(normalized(str(doc["text"])).encode()).hexdigest()
                    for doc in record.get(group, [])
                )
        return queries, documents

    native_queries, native_documents = inventory(native_records)
    translated_queries, translated_documents = inventory(translated_records)
    return {
        "method": "exact_casefold_whitespace_query_and_sha256_normalized_document",
        "native_query_count": len(native_queries),
        "translated_query_count": len(translated_queries),
        "exact_query_overlap_count": len(native_queries & translated_queries),
        "native_document_count": len(native_documents),
        "translated_document_count": len(translated_documents),
        "exact_document_overlap_count": len(native_documents & translated_documents),
        "near_duplicate_overlap": None,
        "near_duplicate_status": "not_measured",
    }
