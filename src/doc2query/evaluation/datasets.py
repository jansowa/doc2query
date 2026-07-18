"""Freeze evaluation IDs and fingerprints without modifying Task 01 split files."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from doc2query.utils.records import JsonlWriter, read_records, write_json

SCHEMA_VERSION = 1


def _canonical(record: dict[str, Any]) -> bytes:
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_id(record: dict[str, Any]) -> str:
    for field in ("example_id", "case_id", "pair_id"):
        if field in record:
            return str(record[field])
    raise ValueError("evaluation record has no stable example_id/case_id/pair_id")


def _records_fingerprint(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=_record_id):
        digest.update(_canonical(record))
        digest.update(b"\n")
    return digest.hexdigest()


def _ids_hash(ids: list[str]) -> str:
    payload = "".join(f"{value}\n" for value in sorted(ids)).encode()
    return hashlib.sha256(payload).hexdigest()


def _deterministic_panel(
    records: list[dict[str, Any]], *, size: int, seed: int
) -> list[dict[str, Any]]:
    def order(record: dict[str, Any]) -> tuple[str, str]:
        identifier = _record_id(record)
        key = hashlib.sha256(f"{seed}:{identifier}".encode()).hexdigest()
        return key, identifier

    return sorted(records, key=order)[:size]


@dataclass(frozen=True)
class FrozenSetSpec:
    name: str
    source_path: Path
    records: list[dict[str, Any]]
    population_count: int
    exclusion_reason: str | None = None


def _write_set(output_dir: Path, spec: FrozenSetSpec) -> dict[str, Any]:
    ids = sorted(_record_id(record) for record in spec.records)
    ids_path = output_dir / f"{spec.name}.ids.jsonl"
    with JsonlWriter(ids_path) as writer:
        for identifier in ids:
            writer.write({"id": identifier})
    return {
        "name": spec.name,
        "source_path": str(spec.source_path),
        "source_sha256": _sha256_file(spec.source_path),
        "id_path": str(ids_path),
        "id_field": ("case_id" if spec.records and "case_id" in spec.records[0] else "example_id"),
        "id_count": len(ids),
        "id_list_sha256": _ids_hash(ids),
        "records_sha256": _records_fingerprint(spec.records),
        "population_count": spec.population_count,
        "excluded_count": spec.population_count - len(ids),
        "exclusion_reason": spec.exclusion_reason,
    }


def freeze_evaluation_sets(
    *,
    dev_path: Path,
    test_path: Path,
    adversarial_path: Path,
    output_dir: Path,
    human_panel_size: int = 300,
    generation_panel_size: int = 100,
    seed: int = 42,
) -> dict[str, Any]:
    """Create an immutable manifest and ID lists; existing manifests are never replaced."""
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"frozen evaluation manifest already exists: {manifest_path}")
    dev = list(read_records(dev_path))
    test = list(read_records(test_path))
    adversarial = list(read_records(adversarial_path))
    dev_rank10 = [record for record in dev if len(record.get("hard_negatives", [])) >= 10]
    test_rank10 = [record for record in test if len(record.get("hard_negatives", [])) >= 10]
    human_panel = _deterministic_panel(test_rank10, size=human_panel_size, seed=seed)
    generation_panel = _deterministic_panel(test_rank10, size=generation_panel_size, seed=seed + 1)
    output_dir.mkdir(parents=True, exist_ok=False)
    reason = "fewer_than_10_hard_negatives_after_v1_cross_split_cleanup"
    specs = [
        FrozenSetSpec("dev_intrinsic", dev_path, dev, len(dev)),
        FrozenSetSpec("dev_intrinsic_rank10", dev_path, dev_rank10, len(dev), reason),
        FrozenSetSpec("test_intrinsic", test_path, test, len(test)),
        FrozenSetSpec("test_intrinsic_rank10", test_path, test_rank10, len(test), reason),
        FrozenSetSpec("test_adversarial", adversarial_path, adversarial, len(adversarial)),
        FrozenSetSpec("test_human_panel", test_path, human_panel, len(test_rank10)),
        FrozenSetSpec("test_embedder", test_path, test, len(test)),
        FrozenSetSpec("test_embedder_rank10", test_path, test_rank10, len(test), reason),
        FrozenSetSpec("test_generator_panel_rank10", test_path, generation_panel, len(test_rank10)),
    ]
    sets = {spec.name: _write_set(output_dir, spec) for spec in specs}
    split_manifest = test_path.parent / "split_manifest.json"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "version": "task04-v1",
        "seed": seed,
        "selection_policy": {
            "rank10": "retain existing v1 records with at least 10 hard_negatives",
            "panels": "sha256(seed:example_id), ascending; selected before model evaluation",
            "split_mutation": False,
        },
        "source_split_manifest": str(split_manifest),
        "source_split_manifest_sha256": _sha256_file(split_manifest),
        "sets": sets,
    }
    write_json(manifest_path, manifest)
    return manifest


def _load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported frozen evaluation manifest: {path}")
    return value


def load_frozen_records(manifest_path: Path, set_name: str) -> list[dict[str, Any]]:
    """Load a set only after verifying source, ID-list, count and record fingerprints."""
    manifest = _load_manifest(manifest_path)
    raw_spec = manifest.get("sets", {}).get(set_name)
    if not isinstance(raw_spec, dict):
        raise KeyError(f"unknown frozen evaluation set: {set_name}")
    source = Path(str(raw_spec["source_path"]))
    ids_path = Path(str(raw_spec["id_path"]))
    if _sha256_file(source) != raw_spec["source_sha256"]:
        raise RuntimeError(f"frozen source fingerprint mismatch for {set_name}")
    ids = [str(row["id"]) for row in read_records(ids_path)]
    if len(ids) != raw_spec["id_count"] or _ids_hash(ids) != raw_spec["id_list_sha256"]:
        raise RuntimeError(f"frozen ID-list fingerprint mismatch for {set_name}")
    wanted = set(ids)
    records = [record for record in read_records(source) if _record_id(record) in wanted]
    if len(records) != len(wanted):
        raise RuntimeError(f"frozen set {set_name} is missing records in its source")
    if _records_fingerprint(records) != raw_spec["records_sha256"]:
        raise RuntimeError(f"frozen record fingerprint mismatch for {set_name}")
    return sorted(records, key=_record_id)


def verify_frozen_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = _load_manifest(manifest_path)
    verified = {}
    for name in manifest["sets"]:
        records = load_frozen_records(manifest_path, name)
        verified[name] = len(records)
    return {"version": manifest["version"], "verified": verified}


def evaluation_fingerprint(manifest_path: Path, set_name: str) -> str:
    manifest = _load_manifest(manifest_path)
    spec = manifest.get("sets", {}).get(set_name)
    if not isinstance(spec, dict):
        raise KeyError(set_name)
    return str(spec["records_sha256"])
