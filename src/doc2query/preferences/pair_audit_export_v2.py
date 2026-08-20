"""Deterministic, blind export of defect-anchored Task 06 pairs (policy v2).

Mechanically this is the v1 blind export with one dimension swapped: the sample is
stratified and quota-balanced by **axis** instead of by primary-margin band, because
policy v2 does not order pairs by margin and must not smuggle it back in as a design
dimension.  Everything else is deliberately identical, so the frozen audit rubric sees
the same shape of input:

* ``blind_pairs.jsonl`` — exactly the same five fields a blind reviewer may see;
* ``machine_key.jsonl`` — the unblinding key, never sent to any reviewer;
* ``sample.jsonl`` — the full pair records of the sample, for later analysis.

The A/B orientation is committed **before** any rating exists:
``orientation_commitment = sha256(salt || pair_id || orientation)``, with the salt
published in the manifest.

The axis quota (250/250) reallocates unused quota to the other axis when one axis is
short, and reports the shortfall — it never relaxes a threshold to fill the sample.

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
from doc2query.preferences.pair_audit_export import (
    BLIND_FIELDS,
    _largest_remainder,
    _stratum_rng_seed,
)
from doc2query.preferences.pair_policy_v2 import (
    RELEASED_AXES,
    DefectPairManifest,
    DefectPairPolicy,
    load_defect_pair_policy,
)
from doc2query.schemas import StrictModel
from doc2query.training.dpo import (
    SHA256_PATTERN,
    canonical_fingerprint,
    file_sha256,
    ordered_ids_fingerprint,
)
from doc2query.utils.records import JsonlWriter, read_records, write_json

EXPORT_CONTRACT = "task06-defect-pair-audit-blind-export-v2"
EXPORT_STATUS = "blind_export_frozen_not_reviewed"


class AxisStratumAllocation(StrictModel):
    cohort_id: str = Field(min_length=1)
    axis: Literal["A", "B"]
    requested_form: str = Field(min_length=1)
    population: int = Field(ge=1)
    allocated: int = Field(ge=0)


class AxisQuotaOutcome(StrictModel):
    axis: Literal["A", "B"]
    quota: int = Field(ge=0)
    effective_quota: int = Field(ge=0)
    population: int = Field(ge=0)
    allocated: int = Field(ge=0)
    shortfall: int = Field(ge=0)

    @model_validator(mode="after")
    def outcome_is_consistent(self) -> AxisQuotaOutcome:
        if self.allocated > self.population:
            raise ValueError("an axis cannot allocate more pairs than it holds")
        if self.allocated + self.shortfall != self.effective_quota:
            raise ValueError("allocated and shortfall must cover the effective quota")
        return self


class DefectBlindExportManifest(StrictModel):
    schema_version: Literal[1]
    contract: Literal["task06-defect-pair-audit-blind-export-v2"]
    status: Literal["blind_export_frozen_not_reviewed"]
    policy_id: str = Field(min_length=1)
    policy_sha256: str = Field(pattern=SHA256_PATTERN)
    source_cohorts: list[str] = Field(min_length=1)
    source_pair_manifest_sha256: dict[str, str]
    population_pair_count: int = Field(ge=1)
    population_axis_counts: dict[str, int]
    target_pair_count: int = Field(ge=1)
    sampled_pair_count: int = Field(ge=1)
    shortfall_pair_count: int = Field(ge=0)
    development_gate_met: bool
    axis_quotas: list[AxisQuotaOutcome] = Field(min_length=2, max_length=2)
    axis_quota_shortfall: dict[str, int]
    seed: int = Field(ge=0)
    strata: list[AxisStratumAllocation] = Field(min_length=1)
    orientation_commitment_salt: str = Field(min_length=1)
    orientation_balance: dict[str, int]
    audit_ids_fingerprint: str = Field(pattern=SHA256_PATTERN)
    blind_pairs: dict[str, Any]
    machine_key: dict[str, Any]
    sample: dict[str, Any]
    report: dict[str, Any]
    blind_fields: list[str] = Field(min_length=5, max_length=5)
    margin_used_for_stratification: Literal[False]
    ratings_collected: Literal[False]
    human_evidence_claimed: Literal[False]
    task07_training_authorized: Literal[False]
    final_tests_used: list[str] = Field(max_length=0)
    manifest_fingerprint: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def counts_and_fingerprint_are_valid(self) -> DefectBlindExportManifest:
        if self.sampled_pair_count + self.shortfall_pair_count != self.target_pair_count:
            raise ValueError("sampled and shortfall counts must cover the target")
        if self.development_gate_met != (self.shortfall_pair_count == 0):
            raise ValueError("the development gate is met only without a shortfall")
        if sum(row.allocated for row in self.strata) != self.sampled_pair_count:
            raise ValueError("stratum allocations must sum to the sampled count")
        if sum(row.allocated for row in self.axis_quotas) != self.sampled_pair_count:
            raise ValueError("axis allocations must sum to the sampled count")
        if list(self.blind_fields) != list(BLIND_FIELDS):
            raise ValueError("blind rows must expose exactly the frozen field set")
        payload = self.model_dump(mode="json")
        fingerprint = payload.pop("manifest_fingerprint")
        if fingerprint != canonical_fingerprint(payload):
            raise ValueError("defect blind export manifest fingerprint mismatch")
        return self


def _load_pairs(
    pair_dirs: Sequence[Path], policy_path: Path, policy: DefectPairPolicy
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
                raise ValueError(f"missing defect pair artifact: {path}")
        manifest = DefectPairManifest.model_validate_json(
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
                raise ValueError("a defect pair declares final-test usage")
            if str(row["policy_id"]) != policy.policy_id:
                raise ValueError("a defect pair pins a different policy")
            if row.get("margin_used_for_ordering") is not False:
                raise ValueError("a defect pair claims margin ordering")
        manifest_hashes[manifest.cohort_id] = file_sha256(manifest_path)
        cohorts.append(manifest.cohort_id)
        pairs.extend(rows)
    if len(cohorts) != len(set(cohorts)):
        raise ValueError("the same cohort was supplied twice")
    ids = [str(row["pair_id"]) for row in pairs]
    if len(ids) != len(set(ids)):
        raise ValueError("defect pair IDs must be unique across cohorts")
    clusters = [str(row["passage_cluster_id"]) for row in pairs]
    if len(clusters) != len(set(clusters)):
        raise ValueError("the audit population must not repeat a near-duplicate cluster")
    return sorted(pairs, key=lambda row: str(row["pair_id"])), manifest_hashes, sorted(cohorts)


def _effective_quotas(
    populations: Mapping[str, int], policy: DefectPairPolicy
) -> dict[str, int]:
    """Give each axis its quota, then pass unused quota to the axis that can absorb it."""
    quotas = {axis: int(policy.audit_sample.axis_quotas[axis]) for axis in RELEASED_AXES}
    effective = {axis: min(quotas[axis], populations.get(axis, 0)) for axis in RELEASED_AXES}
    unused = sum(quotas[axis] - effective[axis] for axis in RELEASED_AXES)
    for axis in sorted(RELEASED_AXES):
        if unused <= 0:
            break
        headroom = populations.get(axis, 0) - effective[axis]
        taken = min(unused, max(0, headroom))
        effective[axis] += taken
        unused -= taken
    return effective


def _sample(
    pairs: Sequence[Mapping[str, Any]], policy: DefectPairPolicy
) -> tuple[list[dict[str, Any]], list[AxisStratumAllocation], list[AxisQuotaOutcome]]:
    """Draw the frozen axis-balanced stratified sample; no threshold moves to fill it."""
    populations = Counter(str(row["axis"]) for row in pairs)
    effective = _effective_quotas(populations, policy)
    sampled: list[dict[str, Any]] = []
    strata: list[AxisStratumAllocation] = []
    quota_outcomes: list[AxisQuotaOutcome] = []
    for axis in sorted(RELEASED_AXES):
        grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for row in pairs:
            if str(row["axis"]) != axis:
                continue
            key = (str(row["cohort_id"]), axis, str(row["requested_form"]))
            grouped.setdefault(key, []).append(dict(row))
        keys = sorted(grouped)
        axis_allocated = 0
        if keys:
            counts = [len(grouped[key]) for key in keys]
            allocations = _largest_remainder(counts, effective[axis])
            for key, population, allocated in zip(keys, counts, allocations, strict=True):
                candidates = sorted(grouped[key], key=lambda row: str(row["pair_id"]))
                if allocated >= population:
                    drawn = candidates
                else:
                    rng = random.Random(_stratum_rng_seed(policy.audit_sample.seed, key))
                    drawn = sorted(
                        rng.sample(candidates, allocated), key=lambda row: str(row["pair_id"])
                    )
                sampled.extend(drawn)
                axis_allocated += len(drawn)
                strata.append(
                    AxisStratumAllocation(
                        cohort_id=key[0],
                        axis=axis,  # type: ignore[arg-type]
                        requested_form=key[2],
                        population=population,
                        allocated=len(drawn),
                    )
                )
        quota_outcomes.append(
            AxisQuotaOutcome(
                axis=axis,  # type: ignore[arg-type]
                quota=int(policy.audit_sample.axis_quotas[axis]),
                effective_quota=effective[axis],
                population=populations.get(axis, 0),
                allocated=axis_allocated,
                shortfall=effective[axis] - axis_allocated,
            )
        )
    sampled.sort(key=lambda row: str(row["pair_id"]))
    return sampled, strata, quota_outcomes


def export_blind_defect_audit_sample(
    *,
    pair_dirs: Sequence[Path],
    policy_path: Path,
    output_dir: Path,
) -> DefectBlindExportManifest:
    """Freeze a blind, deterministic, axis-balanced audit export from finished v2 pairs."""
    policy = load_defect_pair_policy(policy_path)
    if output_dir.exists():
        raise FileExistsError(f"Task 06 blind defect audit export already exists: {output_dir}")
    pairs, manifest_hashes, cohorts = _load_pairs(pair_dirs, policy_path, policy)
    sampled, strata, quota_outcomes = _sample(pairs, policy)
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
                "axis": str(row["axis"]),
                "automatic_chosen_option": option,
                "orientation_commitment": commitment,
                "chosen_verdict": str(row["chosen_verdict"]),
                "rejected_verdict": str(row["rejected_verdict"]),
                "rejected_defect_labels": list(row["rejected_defect_labels"]),
                "requested_form": str(row["requested_form"]),
                "requested_intent": str(row["requested_intent"]),
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
    axis_shortfall = {row.axis: row.effective_quota - row.allocated for row in quota_outcomes}
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
            "population_axis_counts": dict(
                sorted(Counter(str(row["axis"]) for row in pairs).items())
            ),
            "target_pair_count": target,
            "sampled_pair_count": len(sampled),
            "shortfall_pair_count": shortfall,
            "development_gate_met": shortfall == 0,
            "axis_quotas": [row.model_dump(mode="json") for row in quota_outcomes],
            "sampled_axis_counts": dict(
                sorted(Counter(str(row["axis"]) for row in sampled).items())
            ),
            "strata": [row.model_dump(mode="json") for row in strata],
            "orientation_balance": dict(sorted(orientation.items())),
            "rejected_defect_label_counts": dict(
                sorted(
                    Counter(
                        label for row in sampled for label in row["rejected_defect_labels"]
                    ).items()
                )
            ),
            "rejected_verdict_counts": dict(
                sorted(Counter(str(row["rejected_verdict"]) for row in sampled).items())
            ),
            "requested_form_counts": dict(
                sorted(Counter(str(row["requested_form"]) for row in sampled).items())
            ),
            "requested_intent_counts": dict(
                sorted(Counter(str(row["requested_intent"]) for row in sampled).items())
            ),
            "margin_used_for_stratification": False,
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
            "population_axis_counts": dict(
                sorted(Counter(str(row["axis"]) for row in pairs).items())
            ),
            "target_pair_count": target,
            "sampled_pair_count": len(sampled),
            "shortfall_pair_count": shortfall,
            "development_gate_met": shortfall == 0,
            "axis_quotas": [row.model_dump(mode="json") for row in quota_outcomes],
            "axis_quota_shortfall": dict(sorted(axis_shortfall.items())),
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
            "margin_used_for_stratification": False,
            "ratings_collected": False,
            "human_evidence_claimed": False,
            "task07_training_authorized": False,
            "final_tests_used": [],
        }
        payload["manifest_fingerprint"] = canonical_fingerprint(payload)
        manifest = DefectBlindExportManifest.model_validate(payload)
        write_json(staging / "manifest.json", manifest.model_dump(mode="json"))
        if output_dir.exists():
            raise FileExistsError(
                f"Task 06 blind defect audit export already exists: {output_dir}"
            )
        os.replace(staging, output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_orientation_commitments(export_dir: Path) -> int:
    """Re-derive every commitment from the published salt and the unblinding key."""
    manifest = load_defect_blind_export_manifest(export_dir / "manifest.json")
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


def load_defect_blind_export_manifest(path: Path) -> DefectBlindExportManifest:
    return DefectBlindExportManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))


__all__ = [
    "EXPORT_CONTRACT",
    "AxisQuotaOutcome",
    "AxisStratumAllocation",
    "DefectBlindExportManifest",
    "export_blind_defect_audit_sample",
    "load_defect_blind_export_manifest",
    "verify_orientation_commitments",
]
