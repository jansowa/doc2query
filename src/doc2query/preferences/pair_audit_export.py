"""Deterministic, blind export of tentative Task 06 pairs for the dual-LLM audit.

The export consumes finished ``tentative_pairs`` artifacts, draws a deterministic
stratified sample under the frozen ``audit_sample`` block of the pair policy, and
publishes three separate files:

* ``blind_pairs.jsonl`` — exactly the five fields the blind reviewer may see;
* ``machine_key.jsonl`` — the unblinding key, never sent to any reviewer;
* ``sample.jsonl`` — the full pair records of the sample, for later analysis.

The A/B orientation is committed **before** any rating exists: every blind row carries
``orientation_commitment = sha256(salt || pair_id || orientation)`` and the manifest
publishes the salt, so a later reader can prove the orientation was not retro-fitted.

Nothing here calls an API, loads a model or authorizes DPO training.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from doc2query.preferences.diversity_gate import _reject_final_test_path
from doc2query.preferences.pair_policy import (
    TentativePairManifest,
    TentativePairPolicy,
    load_pair_policy,
)
from doc2query.schemas import StrictModel
from doc2query.training.dpo import (
    SHA256_PATTERN,
    canonical_fingerprint,
    file_sha256,
    ordered_ids_fingerprint,
)
from doc2query.utils.records import JsonlWriter, read_records, write_json

EXPORT_CONTRACT = "task06-preference-audit-blind-export-v1"
EXPORT_STATUS = "blind_export_frozen_not_reviewed"
BLIND_FIELDS = ("audit_id", "passage", "query_a", "query_b", "orientation_commitment")


class StratumAllocation(StrictModel):
    cohort_id: str = Field(min_length=1)
    requested_form: str = Field(min_length=1)
    primary_margin_gap_band: str = Field(min_length=1)
    population: int = Field(ge=1)
    allocated: int = Field(ge=0)


class BlindExportManifest(StrictModel):
    schema_version: Literal[1]
    contract: Literal["task06-preference-audit-blind-export-v1"]
    status: Literal["blind_export_frozen_not_reviewed"]
    policy_id: str = Field(min_length=1)
    policy_sha256: str = Field(pattern=SHA256_PATTERN)
    source_cohorts: list[str] = Field(min_length=1)
    source_pair_manifest_sha256: dict[str, str]
    population_pair_count: int = Field(ge=1)
    target_pair_count: int = Field(ge=1)
    sampled_pair_count: int = Field(ge=1)
    shortfall_pair_count: int = Field(ge=0)
    development_gate_met: bool
    seed: int = Field(ge=0)
    strata: list[StratumAllocation] = Field(min_length=1)
    orientation_commitment_salt: str = Field(min_length=1)
    orientation_balance: dict[str, int]
    audit_ids_fingerprint: str = Field(pattern=SHA256_PATTERN)
    blind_pairs: dict[str, Any]
    machine_key: dict[str, Any]
    sample: dict[str, Any]
    report: dict[str, Any]
    blind_fields: list[str] = Field(min_length=5, max_length=5)
    ratings_collected: Literal[False]
    human_evidence_claimed: Literal[False]
    task07_training_authorized: Literal[False]
    final_tests_used: list[str] = Field(max_length=0)
    manifest_fingerprint: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def counts_and_fingerprint_are_valid(self) -> BlindExportManifest:
        if self.sampled_pair_count + self.shortfall_pair_count != self.target_pair_count:
            raise ValueError("sampled and shortfall counts must cover the target")
        if self.development_gate_met != (self.shortfall_pair_count == 0):
            raise ValueError("the development gate is met only without a shortfall")
        if sum(row.allocated for row in self.strata) != self.sampled_pair_count:
            raise ValueError("stratum allocations must sum to the sampled count")
        if list(self.blind_fields) != list(BLIND_FIELDS):
            raise ValueError("blind rows must expose exactly the frozen field set")
        payload = self.model_dump(mode="json")
        fingerprint = payload.pop("manifest_fingerprint")
        if fingerprint != canonical_fingerprint(payload):
            raise ValueError("blind export manifest fingerprint mismatch")
        return self


def gap_band_label(gap: float, policy: TentativePairPolicy) -> str:
    """Assign a pair to its frozen, contiguous primary-margin-gap band."""
    bands = policy.audit_sample.primary_margin_gap_bands
    for low, high in bands:
        if low <= gap < high:
            return f"[{low},{high})"
    raise ValueError(f"primary margin gap {gap} falls outside the frozen bands")


def _stratum_rng_seed(seed: int, key: tuple[str, str, str]) -> int:
    """Derive a per-stratum seed so a stratum's draw never depends on the others."""
    digest = hashlib.sha256("\0".join((str(seed), *key)).encode())
    return int(digest.hexdigest()[:16], 16)


def _largest_remainder(populations: Sequence[int], target: int) -> list[int]:
    """Allocate a target across strata proportionally, breaking ties deterministically."""
    total = sum(populations)
    if target >= total:
        return list(populations)
    exact = [population * target / total for population in populations]
    allocated = [min(int(value), populations[index]) for index, value in enumerate(exact)]
    remainders = sorted(
        range(len(populations)),
        key=lambda index: (-(exact[index] - allocated[index]), index),
    )
    position = 0
    while sum(allocated) < target:
        index = remainders[position % len(remainders)]
        if allocated[index] < populations[index]:
            allocated[index] += 1
        position += 1
        if position > len(remainders) * target + len(remainders):  # pragma: no cover - safety
            raise ValueError("largest-remainder allocation failed to converge")
    return allocated


def _load_pairs(
    pair_dirs: Sequence[Path], policy_path: Path, policy: TentativePairPolicy
) -> tuple[list[dict[str, Any]], dict[str, str], list[str]]:
    policy_sha256 = file_sha256(policy_path)
    pairs: list[dict[str, Any]] = []
    manifest_hashes: dict[str, str] = {}
    cohorts: list[str] = []
    for pair_dir in pair_dirs:
        manifest_path = pair_dir / "manifest.json"
        pairs_path = pair_dir / "pairs.jsonl"
        for path in (manifest_path, pairs_path):
            _reject_final_test_path(path)
            if not path.is_file():
                raise ValueError(f"missing tentative pair artifact: {path}")
        manifest = TentativePairManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        if manifest.policy_id != policy.policy_id or manifest.policy_sha256 != policy_sha256:
            raise ValueError(f"{pair_dir}: pairs were built by a different frozen policy")
        if manifest.cohort_id not in policy.authorized_cohorts:
            raise ValueError(f"{pair_dir}: cohort is not authorized for pair building")
        if file_sha256(pairs_path) != manifest.pairs.sha256:
            raise ValueError(f"{pair_dir}: pairs.jsonl drifted from its manifest")
        rows = list(read_records(pairs_path))
        if len(rows) != manifest.pair_count:
            raise ValueError(f"{pair_dir}: pair count drifted from its manifest")
        for row in rows:
            if row.get("final_tests_used") != []:
                raise ValueError("a tentative pair declares final-test usage")
            if str(row["policy_id"]) != policy.policy_id:
                raise ValueError("a tentative pair pins a different policy")
        manifest_hashes[manifest.cohort_id] = file_sha256(manifest_path)
        cohorts.append(manifest.cohort_id)
        pairs.extend(rows)
    if len(cohorts) != len(set(cohorts)):
        raise ValueError("the same cohort was supplied twice")
    ids = [str(row["pair_id"]) for row in pairs]
    if len(ids) != len(set(ids)):
        raise ValueError("tentative pair IDs must be unique across cohorts")
    clusters = [str(row["passage_cluster_id"]) for row in pairs]
    if len(clusters) != len(set(clusters)):
        raise ValueError("the audit population must not repeat a near-duplicate cluster")
    return sorted(pairs, key=lambda row: str(row["pair_id"])), manifest_hashes, sorted(cohorts)


def _sample(
    pairs: Sequence[Mapping[str, Any]], policy: TentativePairPolicy
) -> tuple[list[dict[str, Any]], list[StratumAllocation]]:
    """Draw the frozen stratified sample; population below target is taken whole."""
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in pairs:
        key = (
            str(row["cohort_id"]),
            str(row["requested_form"]),
            gap_band_label(float(row["primary_margin_gap"]), policy),
        )
        grouped.setdefault(key, []).append(dict(row))
    keys = sorted(grouped)
    populations = [len(grouped[key]) for key in keys]
    allocations = _largest_remainder(populations, policy.audit_sample.target_pair_count)
    sampled: list[dict[str, Any]] = []
    strata: list[StratumAllocation] = []
    for key, population, allocated in zip(keys, populations, allocations, strict=True):
        candidates = sorted(grouped[key], key=lambda row: str(row["pair_id"]))
        if allocated >= population:
            drawn = candidates
        else:
            rng = random.Random(_stratum_rng_seed(policy.audit_sample.seed, key))
            drawn = sorted(rng.sample(candidates, allocated), key=lambda row: str(row["pair_id"]))
        sampled.extend(drawn)
        strata.append(
            StratumAllocation(
                cohort_id=key[0],
                requested_form=key[1],
                primary_margin_gap_band=key[2],
                population=population,
                allocated=len(drawn),
            )
        )
    sampled.sort(key=lambda row: str(row["pair_id"]))
    return sampled, strata


def export_blind_audit_sample(
    *,
    pair_dirs: Sequence[Path],
    policy_path: Path,
    output_dir: Path,
) -> BlindExportManifest:
    """Freeze a blind, deterministic audit export from finished tentative pairs."""
    policy = load_pair_policy(policy_path)
    if output_dir.exists():
        raise FileExistsError(f"Task 06 blind audit export already exists: {output_dir}")
    pairs, manifest_hashes, cohorts = _load_pairs(pair_dirs, policy_path, policy)
    sampled, strata = _sample(pairs, policy)
    if not sampled:
        raise ValueError("the audit export refuses an empty sample")

    salt = f"{EXPORT_CONTRACT}:{policy.audit_sample.seed}"
    blind_rows: list[dict[str, str]] = []
    key_rows: list[dict[str, Any]] = []
    orientation: Counter[str] = Counter()
    for index, row in enumerate(sampled):
        pair_id = str(row["pair_id"])
        chosen_is_a = index % 2 == 0
        option = "A" if chosen_is_a else "B"
        orientation[option] += 1
        audit_id = hashlib.sha256(
            f"{policy.audit_sample.seed}\0{pair_id}".encode()
        ).hexdigest()[:24]
        commitment = hashlib.sha256(f"{salt}\0{pair_id}\0{option}".encode()).hexdigest()
        blind_rows.append(
            {
                "audit_id": audit_id,
                "passage": str(row["passage"]),
                "query_a": str(row["chosen"] if chosen_is_a else row["rejected"]),
                "query_b": str(row["rejected"] if chosen_is_a else row["chosen"]),
                "orientation_commitment": commitment,
            }
        )
        key_rows.append(
            {
                "audit_id": audit_id,
                "pair_id": pair_id,
                "cohort_id": str(row["cohort_id"]),
                "group_id": str(row["group_id"]),
                "automatic_chosen_option": option,
                "orientation_commitment": commitment,
                "primary_margin_gap": float(row["primary_margin_gap"]),
                "primary_margin_gap_band": gap_band_label(
                    float(row["primary_margin_gap"]), policy
                ),
                "requested_form": str(row["requested_form"]),
                "requested_intent": str(row["requested_intent"]),
                "rejected_failure_types": list(row["rejected_failure_types"]),
                "split": str(row["split"]),
            }
        )
    for blind in blind_rows:
        if set(blind) != set(BLIND_FIELDS):
            raise ValueError("a blind row does not expose exactly the frozen field set")
        if blind["query_a"] == blind["query_b"]:
            raise ValueError("a blind row contains two identical queries")

    target = policy.audit_sample.target_pair_count
    shortfall = max(0, target - len(sampled))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"
    try:
        staging.mkdir()
        blind_path = staging / "blind_pairs.jsonl"
        with JsonlWriter(blind_path) as writer:
            for blind in blind_rows:
                writer.write(dict(blind))
        key_path = staging / "machine_key.jsonl"
        with JsonlWriter(key_path) as writer:
            for key_row in key_rows:
                writer.write(key_row)
        sample_path = staging / "sample.jsonl"
        with JsonlWriter(sample_path) as writer:
            for row in sampled:
                writer.write(row)
        report_payload: dict[str, Any] = {
            "schema_version": 1,
            "contract": EXPORT_CONTRACT,
            "status": EXPORT_STATUS,
            "policy_id": policy.policy_id,
            "source_cohorts": cohorts,
            "population_pair_count": len(pairs),
            "target_pair_count": target,
            "sampled_pair_count": len(sampled),
            "shortfall_pair_count": shortfall,
            "development_gate_met": shortfall == 0,
            "strata": [row.model_dump(mode="json") for row in strata],
            "orientation_balance": dict(sorted(orientation.items())),
            "rejected_failure_type_counts": dict(
                sorted(
                    Counter(
                        label for row in sampled for label in row["rejected_failure_types"]
                    ).items()
                )
            ),
            "requested_form_counts": dict(
                sorted(Counter(str(row["requested_form"]) for row in sampled).items())
            ),
            "requested_intent_counts": dict(
                sorted(Counter(str(row["requested_intent"]) for row in sampled).items())
            ),
            "ratings_collected": False,
            "human_evidence_claimed": False,
            "task07_training_authorized": False,
            "final_tests_used": [],
        }
        report_path = staging / "report.json"
        write_json(report_path, report_payload)
        payload: dict[str, Any] = {
            "schema_version": 1,
            "contract": EXPORT_CONTRACT,
            "status": EXPORT_STATUS,
            "policy_id": policy.policy_id,
            "policy_sha256": file_sha256(policy_path),
            "source_cohorts": cohorts,
            "source_pair_manifest_sha256": dict(sorted(manifest_hashes.items())),
            "population_pair_count": len(pairs),
            "target_pair_count": target,
            "sampled_pair_count": len(sampled),
            "shortfall_pair_count": shortfall,
            "development_gate_met": shortfall == 0,
            "seed": policy.audit_sample.seed,
            "strata": [row.model_dump(mode="json") for row in strata],
            "orientation_commitment_salt": salt,
            "orientation_balance": dict(sorted(orientation.items())),
            "audit_ids_fingerprint": ordered_ids_fingerprint(
                [row["audit_id"] for row in blind_rows]
            ),
            "blind_pairs": {
                "path": blind_path.name,
                "sha256": file_sha256(blind_path),
                "record_count": len(blind_rows),
            },
            "machine_key": {
                "path": key_path.name,
                "sha256": file_sha256(key_path),
                "record_count": len(key_rows),
            },
            "sample": {
                "path": sample_path.name,
                "sha256": file_sha256(sample_path),
                "record_count": len(sampled),
            },
            "report": {
                "path": report_path.name,
                "sha256": file_sha256(report_path),
                "record_count": 1,
            },
            "blind_fields": list(BLIND_FIELDS),
            "ratings_collected": False,
            "human_evidence_claimed": False,
            "task07_training_authorized": False,
            "final_tests_used": [],
        }
        payload["manifest_fingerprint"] = canonical_fingerprint(payload)
        manifest = BlindExportManifest.model_validate(payload)
        write_json(staging / "manifest.json", manifest.model_dump(mode="json"))
        if output_dir.exists():
            raise FileExistsError(f"Task 06 blind audit export already exists: {output_dir}")
        os.replace(staging, output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_orientation_commitments(export_dir: Path) -> int:
    """Re-derive every commitment from the published salt and the unblinding key."""
    manifest = BlindExportManifest.model_validate_json(
        (export_dir / "manifest.json").read_text(encoding="utf-8")
    )
    blind = {str(row["audit_id"]): row for row in read_records(export_dir / "blind_pairs.jsonl")}
    checked = 0
    for row in read_records(export_dir / "machine_key.jsonl"):
        audit_id = str(row["audit_id"])
        expected = hashlib.sha256(
            f"{manifest.orientation_commitment_salt}\0{row['pair_id']}\0"
            f"{row['automatic_chosen_option']}".encode()
        ).hexdigest()
        if expected != row["orientation_commitment"]:
            raise ValueError(f"machine key commitment mismatch for {audit_id}")
        if expected != blind[audit_id]["orientation_commitment"]:
            raise ValueError(f"blind row commitment mismatch for {audit_id}")
        checked += 1
    if checked != manifest.sampled_pair_count:
        raise ValueError("commitment verification did not cover the whole sample")
    return checked


def load_blind_export_manifest(path: Path) -> BlindExportManifest:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return BlindExportManifest.model_validate(raw)
