"""Axis-A supply after answerability certification: how many groups can actually yield a pair?

The V2-00 inventory measured defect supply *without* an answerability signal, because none
existed yet — axis A had to lean on the corpus round trip, which the dual-LLM audit showed
does not differentiate answerability.  The judge accepted by K1-K3 now supplies that
signal, so this module recomputes the same question with it:

* a `chosen` candidate is admissible when it is clean under the frozen policy checks
  (format + lead-in guard, round trip @20, no entity hallucination, ``pool_margin > 0``,
  no copy risk) **and** the judge says ``yes``;
* an axis-A `rejected` candidate is one that is format-admissible and carries a named
  defect: the judge says ``no``, or the candidate fails the round trip @100;
* a group is pairable when it holds both, on two different candidates that also satisfy
  the frozen diversity constraint on normalized query overlap.

``uncertain`` blocks the `chosen` role and is **not** a defect, exactly as the judge ADR
requires — so it can never manufacture a rejected side.

This is a measurement and a design input for the V2-03 ADR: it reads quality fields
openly, builds no pair, freezes no threshold and orders nothing.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from doc2query.preferences.answerability_judge import judge_item_id
from doc2query.preferences.build import normalized_query_jaccard
from doc2query.preferences.defect_inventory import CandidateProfile, classify_candidate
from doc2query.preferences.pair_policy import (
    TentativePairPolicy,
    _load_gate,
    _load_scoring,
    load_pair_policy,
)
from doc2query.training.dpo import file_sha256
from doc2query.utils.records import read_durable_jsonl_prefix, write_json

SUPPLY_CONTRACT = "task06-axis-a-supply-after-certification-v1"
SUPPLY_STATUS = "design_input_measured_no_pairs_built"


@dataclass(frozen=True)
class CertifiedCandidate:
    """One gate representative with its frozen profile and its answerability verdict."""

    profile: CandidateProfile
    verdict: str | None

    @property
    def admissible_chosen(self) -> bool:
        return self.profile.clean_chosen and self.verdict == "yes"

    @property
    def axis_a_rejected(self) -> bool:
        if not self.profile.format_admissible:
            return False
        return self.verdict == "no" or self.profile.round_trip_100_fail


def load_verdicts(paths: Iterable[Path]) -> dict[str, str]:
    """Read judge journals into ``item_id -> verdict``; only ``verdict`` events count."""
    verdicts: dict[str, str] = {}
    for path in paths:
        for event in read_durable_jsonl_prefix(path):
            if event.get("event") != "verdict":
                continue
            item_id = str(event["item_id"])
            verdict = str(event["verdict"])
            previous = verdicts.get(item_id)
            if previous is not None and previous != verdict:
                raise ValueError(f"journals disagree on {item_id}: {previous} vs {verdict}")
            verdicts[item_id] = verdict
    if not verdicts:
        raise ValueError("no verdicts found in the given journals")
    return verdicts


def load_certified_cohort(
    cohort_dir: Path, policy: TentativePairPolicy, verdicts: Mapping[str, str]
) -> tuple[dict[str, list[CertifiedCandidate]], dict[str, Any]]:
    """Load one frozen cohort's gate representatives joined with judge verdicts."""
    gate, gate_verdicts = _load_gate(cohort_dir / "diversity_gate")
    scoring_dir = cohort_dir / "d01_controlled" / "scoring"
    rows = _load_scoring(scoring_dir / "per_generation.jsonl", scoring_dir / "summary.json", gate)
    representatives = {
        group_id: {str(value) for value in verdict["representative_candidate_ids"]}
        for group_id, verdict in gate_verdicts.items()
        if bool(verdict["eligible"])
    }
    groups: dict[str, list[CertifiedCandidate]] = {group_id: [] for group_id in representatives}
    missing = 0
    for row in rows:
        group_id = str(row["evaluation_group_id"])
        allowed = representatives.get(group_id)
        if allowed is None or str(row["evaluation_id"]) not in allowed:
            continue
        passage = str(cast(Mapping[str, Any], row["positive"])["text"])
        item_id = judge_item_id(str(row["generated"]), passage)
        verdict = verdicts.get(item_id)
        if verdict is None:
            missing += 1
        groups[group_id].append(CertifiedCandidate(classify_candidate(row, policy), verdict))
    pins = {
        "scoring_sha256": file_sha256(scoring_dir / "per_generation.jsonl"),
        "gate_manifest_sha256": file_sha256(cohort_dir / "diversity_gate" / "manifest.json"),
        "candidates_without_verdict": missing,
    }
    return groups, pins


def _pairable(
    chosen: Sequence[CertifiedCandidate],
    rejected: Sequence[CertifiedCandidate],
    *,
    max_jaccard: float,
) -> bool:
    return any(
        left.profile.candidate_id != right.profile.candidate_id
        and normalized_query_jaccard(left.profile.query, right.profile.query) <= max_jaccard
        for left in chosen
        for right in rejected
    )


def summarize_cohort(
    groups: Mapping[str, Sequence[CertifiedCandidate]], policy: TentativePairPolicy
) -> dict[str, Any]:
    """Per-cohort supply: how many groups hold each side, and how many can pair."""
    counters: Counter[str] = Counter()
    verdicts: Counter[str] = Counter()
    for members in groups.values():
        counters["groups"] += 1
        for member in members:
            verdicts[str(member.verdict)] += 1
        chosen = [member for member in members if member.admissible_chosen]
        rejected = [member for member in members if member.axis_a_rejected]
        counters["groups_with_certified_chosen"] += 1 if chosen else 0
        counters["groups_with_axis_a_rejected"] += 1 if rejected else 0
        if chosen and rejected:
            counters["groups_with_both_sides"] += 1
            if _pairable(chosen, rejected, max_jaccard=policy.pairing.max_normalized_query_jaccard):
                counters["pairable_groups"] += 1
        # Kontrola kontrfaktyczna: ile grup mialoby czysty `chosen` BEZ filtra sedziego.
        if any(member.profile.clean_chosen for member in members):
            counters["groups_with_clean_chosen_before_judge"] += 1
    total = counters["groups"] or 1
    return {
        "counts": dict(sorted(counters.items())),
        "candidate_verdicts": dict(sorted(verdicts.items())),
        "pairable_share": counters["pairable_groups"] / total,
        "chosen_supply_kept_by_judge": (
            counters["groups_with_certified_chosen"]
            / (counters["groups_with_clean_chosen_before_judge"] or 1)
        ),
    }


def run_axis_a_supply(
    *,
    cohort_dirs: Iterable[Path],
    journal_paths: Iterable[Path],
    policy_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Measure axis-A supply per cohort and pooled, then publish one JSON report."""
    policy = load_pair_policy(policy_path)
    verdicts = load_verdicts(list(journal_paths))
    per_cohort: dict[str, Any] = {}
    inputs: dict[str, Any] = {}
    pooled: Counter[str] = Counter()
    pooled_verdicts: Counter[str] = Counter()
    for cohort_dir in cohort_dirs:
        groups, pins = load_certified_cohort(cohort_dir, policy, verdicts)
        summary = summarize_cohort(groups, policy)
        per_cohort[cohort_dir.name] = summary
        inputs[cohort_dir.name] = pins
        pooled.update(summary["counts"])
        pooled_verdicts.update(summary["candidate_verdicts"])
    if not per_cohort:
        raise ValueError("axis-A supply needs at least one cohort")
    total = pooled["groups"] or 1
    report = {
        "schema": "task06-axis-a-supply-result-v1",
        "contract": SUPPLY_CONTRACT,
        "status": SUPPLY_STATUS,
        "role": "design_input_for_v2_03_adr",
        "answerability_signal": "task06-answerability-judge-v1 (accepted by K1-K3)",
        "verdicts_loaded": len(verdicts),
        "policy_sha256": file_sha256(policy_path),
        "inputs": inputs,
        "per_cohort": per_cohort,
        "pooled": {
            "counts": dict(sorted(pooled.items())),
            "candidate_verdicts": dict(sorted(pooled_verdicts.items())),
            "pairable_share": pooled["pairable_groups"] / total,
            "chosen_supply_kept_by_judge": (
                pooled["groups_with_certified_chosen"]
                / (pooled["groups_with_clean_chosen_before_judge"] or 1)
            ),
        },
        "pairs_built": False,
        "thresholds_frozen_here": False,
        "task07_training_authorized": False,
        "final_tests_used": [],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, report)
    return report


def load_supply_report(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("contract") != SUPPLY_CONTRACT:
        raise ValueError(f"unsupported axis-A supply contract: {path}")
    return value


__all__ = [
    "CertifiedCandidate",
    "load_certified_cohort",
    "load_supply_report",
    "load_verdicts",
    "run_axis_a_supply",
    "summarize_cohort",
]
