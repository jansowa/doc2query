"""V2-00: defect-supply inventory over frozen, scored same-prompt cohorts.

The inventory answers one design question for the future defect-anchored pair policy
(`reports/plans/task06_defect_anchored_pairs_v2_spec_2026-08-17.md`): **how many
same-prompt groups can actually yield a pair on each defect axis?**  It reads quality
fields openly — it is a declared design input, not a quality-blind gate — but it builds
no pair, ranks nothing for training, and freezes no threshold.  Overlap cut points are
reported at several candidate quantiles precisely so the V2-03 ADR can freeze one of
them prospectively instead of tuning it against pair outcomes.

Axis definitions mirror the specification:

* **A — answerability/grounding**: rejected has a hallucinated entity
  (``entity_preservation < 1``) or fails the corpus round trip at 100.
* **B — lexical easiness**: rejected is answerable-by-proxy yet sits in a high
  ``content_jaccard`` band; chosen sits in a low band.
* **C — focus compliance** (preliminary): rejected has ``focus_accuracy == 0``; the
  labels are known-broken (`split_sentences` defect) and are reported with that caveat.

Everything is CPU-only and reuses the frozen loaders of the v1 pair builder, so every
input keeps its SHA-256 pinning and identity checks.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from doc2query.evaluation.d01_usefulness import _copy_risk as copy_risk_flag
from doc2query.evaluation.retrieval import percentile
from doc2query.preferences.build import normalized_query_jaccard
from doc2query.preferences.pair_policy import (
    TentativePairPolicy,
    _load_gate,
    _load_scoring,
    load_pair_policy,
    violates_lead_in_guard,
)
from doc2query.training.dpo import file_sha256
from doc2query.utils.records import write_json

INVENTORY_CONTRACT = "task06-defect-supply-inventory-v1"
INVENTORY_STATUS = "design_input_measured_no_pairs_built"
# Kandydaci punktów cięcia dla osi B; jeden z nich zamrozi dopiero ADR V2-03.
HIGH_OVERLAP_QUANTILES = (0.75, 0.9)
LOW_OVERLAP_QUANTILE = 0.5


@dataclass(frozen=True)
class CandidateProfile:
    """The handful of frozen fields the inventory needs per candidate."""

    candidate_id: str
    query: str
    format_admissible: bool
    answerable_proxy: bool
    clean_chosen: bool
    axis_a_defect: bool
    entity_hallucination: bool
    round_trip_100_fail: bool
    focus_correct: bool
    focus_wrong: bool
    content_jaccard: float


def classify_candidate(row: Mapping[str, Any], policy: TentativePairPolicy) -> CandidateProfile:
    """Classify one scored candidate against the frozen component contracts."""
    query = str(row["generated"])
    format_admissible = (
        row.get("format_valid") is True
        and row.get("has_prefix") is False
        and row.get("has_metacomment") is False
        and row.get("multiple_query") is False
        and row.get("empty") is False
        and not violates_lead_in_guard(query)
    )
    entity_preservation = float(row["entity_preservation"])
    round_trip_20 = float(row["corpus_round_trip_at_20"]) >= 1.0
    round_trip_100 = float(row["corpus_round_trip_at_100"]) >= 1.0
    entity_hallucination = entity_preservation < 1.0
    answerable_proxy = format_admissible and not entity_hallucination and round_trip_100
    clean_chosen = (
        format_admissible
        and round_trip_20
        and not entity_hallucination
        and float(row["pool_margin"]) > 0.0
        and not copy_risk_flag(row, policy.copy_risk.thresholds())
    )
    focus_accuracy = row.get("focus_accuracy")
    return CandidateProfile(
        candidate_id=str(row["evaluation_id"]),
        query=query,
        format_admissible=format_admissible,
        answerable_proxy=answerable_proxy,
        clean_chosen=clean_chosen,
        axis_a_defect=format_admissible and (entity_hallucination or not round_trip_100),
        entity_hallucination=format_admissible and entity_hallucination,
        round_trip_100_fail=format_admissible and not round_trip_100,
        focus_correct=clean_chosen and focus_accuracy is not None and float(focus_accuracy) == 1.0,
        focus_wrong=(
            answerable_proxy and focus_accuracy is not None and float(focus_accuracy) == 0.0
        ),
        content_jaccard=float(row["content_jaccard"]),
    )


def _pairable(
    chosen_pool: Sequence[CandidateProfile],
    rejected_pool: Sequence[CandidateProfile],
    *,
    max_jaccard: float,
) -> bool:
    """True when at least one admissible pair satisfies the frozen pairing contract."""
    return any(
        chosen.candidate_id != rejected.candidate_id
        and normalized_query_jaccard(chosen.query, rejected.query) <= max_jaccard
        for chosen in chosen_pool
        for rejected in rejected_pool
    )


def load_cohort_profiles(
    cohort_dir: Path, policy: TentativePairPolicy
) -> tuple[dict[str, list[CandidateProfile]], dict[str, str]]:
    """Load one frozen cohort into per-eligible-group candidate profiles.

    Only diversity-gate representatives of eligible groups enter the inventory —
    exactly the population the pair policy is allowed to touch.
    """
    gate, verdicts = _load_gate(cohort_dir / "diversity_gate")
    scoring_dir = cohort_dir / "d01_controlled" / "scoring"
    rows = _load_scoring(scoring_dir / "per_generation.jsonl", scoring_dir / "summary.json", gate)
    representatives: dict[str, set[str]] = {
        group_id: {str(value) for value in verdict["representative_candidate_ids"]}
        for group_id, verdict in verdicts.items()
        if bool(verdict["eligible"])
    }
    groups: dict[str, list[CandidateProfile]] = {group_id: [] for group_id in representatives}
    for row in rows:
        group_id = str(row["evaluation_group_id"])
        allowed = representatives.get(group_id)
        if allowed is None or str(row["evaluation_id"]) not in allowed:
            continue
        groups[group_id].append(classify_candidate(row, policy))
    empty = [group_id for group_id, members in groups.items() if not members]
    if empty:
        raise ValueError(f"eligible groups without representatives in scoring: {empty[:3]}")
    inputs = {
        "scoring_sha256": file_sha256(scoring_dir / "per_generation.jsonl"),
        "gate_manifest_sha256": file_sha256(cohort_dir / "diversity_gate" / "manifest.json"),
        "generation_identity_sha256": gate.generation_identity_sha256,
    }
    return groups, inputs


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "p10": percentile(ordered, 0.10),
        "p25": percentile(ordered, 0.25),
        "p50": percentile(ordered, 0.50),
        "p75": percentile(ordered, 0.75),
        "p90": percentile(ordered, 0.90),
    }


def summarize_supply(
    cohorts: Mapping[str, Mapping[str, Sequence[CandidateProfile]]],
    policy: TentativePairPolicy,
) -> dict[str, Any]:
    """Count, per cohort and pooled, the groups that can yield a pair on each axis."""
    max_jaccard = policy.pairing.max_normalized_query_jaccard
    pooled_overlap = [
        candidate.content_jaccard
        for groups in cohorts.values()
        for members in groups.values()
        for candidate in members
        if candidate.answerable_proxy
    ]
    if not pooled_overlap:
        raise ValueError("the inventory found no answerable-by-proxy candidate at all")
    cuts = {
        f"p{int(quantile * 100)}": float(percentile(sorted(pooled_overlap), quantile) or 0.0)
        for quantile in (LOW_OVERLAP_QUANTILE, *HIGH_OVERLAP_QUANTILES)
    }

    per_cohort: dict[str, Any] = {}
    pooled: Counter[str] = Counter()
    for cohort_id, groups in sorted(cohorts.items()):
        counts: Counter[str] = Counter()
        for members in groups.values():
            counts["eligible_groups"] += 1
            clean = [candidate for candidate in members if candidate.clean_chosen]
            if clean:
                counts["groups_with_clean_chosen"] += 1
            axis_a = [candidate for candidate in members if candidate.axis_a_defect]
            if any(candidate.entity_hallucination for candidate in members):
                counts["groups_with_entity_hallucination"] += 1
            if any(candidate.round_trip_100_fail for candidate in members):
                counts["groups_with_round_trip_100_fail"] += 1
            if _pairable(clean, axis_a, max_jaccard=max_jaccard):
                counts["axis_a_pairable_groups"] += 1
            for label, cut in cuts.items():
                if label == f"p{int(LOW_OVERLAP_QUANTILE * 100)}":
                    continue
                low_cut = cuts[f"p{int(LOW_OVERLAP_QUANTILE * 100)}"]
                high = [
                    candidate
                    for candidate in members
                    if candidate.answerable_proxy and candidate.content_jaccard >= cut
                ]
                low = [
                    candidate
                    for candidate in clean
                    if candidate.content_jaccard <= low_cut
                ]
                if _pairable(low, high, max_jaccard=max_jaccard):
                    counts[f"axis_b_pairable_groups_high_{label}"] += 1
            focus_ok = [candidate for candidate in members if candidate.focus_correct]
            focus_bad = [candidate for candidate in members if candidate.focus_wrong]
            if _pairable(focus_ok, focus_bad, max_jaccard=max_jaccard):
                counts["axis_c_preliminary_pairable_groups"] += 1
        per_cohort[cohort_id] = dict(sorted(counts.items()))
        pooled.update(counts)

    return {
        "schema_version": 1,
        "contract": INVENTORY_CONTRACT,
        "status": INVENTORY_STATUS,
        "policy_id": policy.policy_id,
        "pairing_max_normalized_query_jaccard": max_jaccard,
        "overlap_signal": "content_jaccard",
        "overlap_cut_candidates": cuts,
        "overlap_distribution_answerable_proxy": _distribution(pooled_overlap),
        "per_cohort": per_cohort,
        "pooled": dict(sorted(pooled.items())),
        "axis_c_labels_caveat": (
            "wstępne etykiety focus pochodzą ze starego splittera zdań o zmierzonych "
            "wadach; wiążąca podaż osi C wymaga focus_v2 (zadanie V2-02)"
        ),
        "answerability_judge_pending": True,
        "pairs_built": False,
        "thresholds_frozen_here": False,
        "final_tests_used": [],
    }


def run_inventory(
    *,
    cohort_dirs: Iterable[Path],
    policy_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Measure the defect supply across cohorts and publish one JSON report."""
    policy = load_pair_policy(policy_path)
    cohorts: dict[str, dict[str, list[CandidateProfile]]] = {}
    inputs: dict[str, dict[str, str]] = {}
    for cohort_dir in cohort_dirs:
        groups, pins = load_cohort_profiles(cohort_dir, policy)
        cohorts[cohort_dir.name] = groups
        inputs[cohort_dir.name] = pins
    if not cohorts:
        raise ValueError("the inventory needs at least one cohort")
    report = summarize_supply(cohorts, policy)
    report["inputs"] = {name: dict(sorted(pins.items())) for name, pins in sorted(inputs.items())}
    report["policy_sha256"] = file_sha256(policy_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, report)
    return report


def load_inventory(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("contract") != INVENTORY_CONTRACT:
        raise ValueError(f"unsupported defect inventory contract: {path}")
    return value
