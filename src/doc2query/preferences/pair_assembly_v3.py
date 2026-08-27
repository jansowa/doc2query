"""Etap 3 polityki v3: złożenie par z werdyktów turnieju i pełnych rekordów kohort.

Turniej u operatora zna tylko pasaż i teksty kandydatów. Tutaj dokładamy wszystko, co
czyni z tego artefakt badawczy: proweniencję kohorty, komponenty obu stron, klaster
pasażu, fingerprinty wejść i **ponowną weryfikację guardów** — flagom z pakietu nie
wierzymy na słowo, liczymy je jeszcze raz z pełnych rekordów.

Do zbioru trafiają wyłącznie pary **jednomyślne 6/6** (amendment
`task06_v3_selector_aggregation_amendment_2026-08-27.md`). Oba prerejestrowane
warianty strony `rejected` (`bottom` i `near_miss`) są zapisywane osobno; żaden nie
jest promowany bez pomiaru.

Nic tutaj nie woła API i nie autoryzuje treningu.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import Field, model_validator

from doc2query.preferences.build import normalized_query_jaccard
from doc2query.preferences.pair_policy import (
    _Candidate,
    _cluster_ids,
    _components,
    _format_admissible,
    _load_gate,
    _load_scoring,
)
from doc2query.preferences.pair_policy_v2_1 import (
    _clean_chosen,
    load_defect_pair_policy_v2_1,
)
from doc2query.preferences.pair_policy_v3 import REQUIRED_UNANIMOUS_VOTES
from doc2query.schemas import StrictModel
from doc2query.training.dpo import (
    SHA256_PATTERN,
    canonical_fingerprint,
    file_sha256,
    normalize_task06_query,
    ordered_ids_fingerprint,
)
from doc2query.utils.records import JsonlWriter, read_records, write_json

PAIRS_CONTRACT = "task06-judge-selected-pairs-v3"
PAIRS_STATUS = "pairs_built_not_audited"
MAX_NORMALIZED_QUERY_JACCARD = 0.85
Variant = Literal["bottom", "near_miss"]


class JudgeSelectedPair(StrictModel):
    """Jedna para v3; obie strony w pełnej proweniencji, głosy sędziego zapisane."""

    pair_id: str = Field(min_length=1)
    cohort_id: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    example_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    passage_cluster_id: str = Field(min_length=1)
    split: Literal["train"]
    prompt: str = Field(min_length=1)
    prompt_sha256: str = Field(pattern=SHA256_PATTERN)
    passage: str = Field(min_length=1)
    rejected_variant: Literal["bottom", "near_miss"]
    chosen_candidate_id: str = Field(min_length=1)
    rejected_candidate_id: str = Field(min_length=1)
    chosen: str = Field(min_length=1)
    rejected: str = Field(min_length=1)
    chosen_components: dict[str, Any]
    rejected_components: dict[str, Any]
    votes_for_chosen: int = Field(ge=0, le=6)
    position_flips: int = Field(ge=0, le=3)
    tournament_comparisons: int = Field(ge=1)
    chosen_pool_size: int = Field(ge=1)
    rejected_pool_size: int = Field(ge=2)
    normalized_query_jaccard: float = Field(ge=0.0, le=1.0)
    requested_form: str = Field(min_length=1)
    requested_intent: str = Field(min_length=1)
    selector: Literal["task06-judge-selected-pair-policy-v3"]
    validated_defect_scope: list[str] = Field(min_length=3, max_length=3)
    form_or_focus_compliance_claimed: Literal[False]
    final_tests_used: list[str] = Field(max_length=0)

    @model_validator(mode="after")
    def pair_is_distinct_and_unanimous(self) -> JudgeSelectedPair:
        if self.chosen_candidate_id == self.rejected_candidate_id:
            raise ValueError("chosen and rejected candidate IDs must differ")
        if normalize_task06_query(self.chosen) == normalize_task06_query(self.rejected):
            raise ValueError("chosen i rejected są identyczne po normalizacji Task 06")
        if self.votes_for_chosen < REQUIRED_UNANIMOUS_VOTES:
            raise ValueError(
                f"para wymaga {REQUIRED_UNANIMOUS_VOTES} zgodnych głosów, ma "
                f"{self.votes_for_chosen}"
            )
        if self.normalized_query_jaccard > MAX_NORMALIZED_QUERY_JACCARD:
            raise ValueError("para narusza ograniczenie różnorodności zapytań")
        return self


class JudgeSelectedPairManifest(StrictModel):
    schema_version: Literal[1]
    contract: Literal["task06-judge-selected-pairs-v3"]
    status: Literal["pairs_built_not_audited"]
    rejected_variant: Literal["bottom", "near_miss"]
    selector_adr: str = Field(min_length=1)
    aggregation_amendment: str = Field(min_length=1)
    required_unanimous_votes: Literal[6]
    guard_policy_id: str = Field(min_length=1)
    guard_policy_sha256: str = Field(pattern=SHA256_PATTERN)
    bundle_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    tournament_outcomes_sha256: str = Field(pattern=SHA256_PATTERN)
    cohorts: list[str] = Field(min_length=1)
    tournament_groups: int = Field(ge=1)
    paired_groups: int = Field(ge=0)
    unanimous_groups: int = Field(ge=0)
    pair_count: int = Field(ge=0)
    rejection_counts: dict[str, int]
    pair_ids_fingerprint: str = Field(pattern=SHA256_PATTERN)
    pairs: dict[str, Any]
    report: dict[str, Any]
    validated_defect_scope: list[str] = Field(min_length=3, max_length=3)
    form_or_focus_compliance_claimed: Literal[False]
    audit_completed: Literal[False]
    task07_training_authorized: Literal[False]
    final_tests_used: list[str] = Field(max_length=0)
    manifest_fingerprint: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def counts_and_fingerprint_are_valid(self) -> JudgeSelectedPairManifest:
        if self.pair_count > self.unanimous_groups:
            raise ValueError("nie może powstać więcej par niż grup jednomyślnych")
        payload = self.model_dump(mode="json")
        fingerprint = payload.pop("manifest_fingerprint")
        if fingerprint != canonical_fingerprint(payload):
            raise ValueError("manifest par v3 ma niezgodny fingerprint")
        return self


def _load_cohort_candidates(
    cohort_dirs: Sequence[Path],
) -> tuple[dict[str, dict[str, _Candidate]], dict[str, str], dict[str, str]]:
    """Odczytaj kandydatów, przypisanie grupy do kohorty i klastry pasaży."""
    by_group: dict[str, dict[str, _Candidate]] = {}
    cohort_of_group: dict[str, str] = {}
    clusters: dict[str, str] = {}
    for cohort_dir in cohort_dirs:
        scoring_dir = cohort_dir / "d01_controlled" / "scoring"
        gate, _verdicts = _load_gate(cohort_dir / "diversity_gate")
        rows = _load_scoring(
            scoring_dir / "per_generation.jsonl", scoring_dir / "summary.json", gate
        )
        clusters.update(_cluster_ids(cohort_dir / "cohort.records.jsonl"))
        for row in rows:
            group_id = str(row["evaluation_group_id"])
            cohort_of_group[group_id] = cohort_dir.name
            by_group.setdefault(group_id, {})[str(row["evaluation_id"])] = _Candidate(
                candidate_id=str(row["evaluation_id"]),
                candidate_index=int(row["candidate_index"]),
                query=str(row["generated"]),
                row=row,
            )
    return by_group, cohort_of_group, clusters


def assemble_pairs(
    *,
    cohort_dirs: Sequence[Path],
    tournament_dir: Path,
    bundle_dir: Path,
    policy_path: Path,
    output_dir: Path,
    variant: Variant,
) -> JudgeSelectedPairManifest:
    """Złóż pary jednego wariantu z jednomyślnych rozstrzygnięć turnieju."""
    if output_dir.exists():
        raise FileExistsError(f"wyjście już istnieje: {output_dir}")
    policy = load_defect_pair_policy_v2_1(policy_path)
    outcomes_path = tournament_dir / "tournament_outcomes.jsonl"
    outcomes = list(read_records(outcomes_path))
    by_group, cohort_of_group, clusters = _load_cohort_candidates(cohort_dirs)

    pairs: list[JudgeSelectedPair] = []
    rejected_counts: Counter[str] = Counter()
    unanimous_groups = 0
    paired_groups = 0
    seen_clusters: set[str] = set()

    for outcome in sorted(outcomes, key=lambda row: str(row["group_id"])):
        group_id = str(outcome["group_id"])
        if not outcome.get("paired"):
            rejected_counts[str(outcome.get("reason", "unpaired"))] += 1
            continue
        paired_groups += 1
        confirmation = (outcome.get("confirmations") or {}).get(variant)
        if confirmation is None:
            rejected_counts["variant_missing"] += 1
            continue
        if not confirmation.get("complete"):
            rejected_counts["confirmation_incomplete"] += 1
            continue
        if not confirmation.get("unanimous"):
            rejected_counts["not_unanimous"] += 1
            continue
        unanimous_groups += 1

        members = by_group.get(group_id)
        if members is None:
            raise ValueError(f"grupa {group_id} nie występuje w kohortach")
        chosen = members.get(str(outcome["best"]))
        rejected = members.get(str(confirmation["rejected_candidate_id"]))
        if chosen is None or rejected is None:
            raise ValueError(f"grupa {group_id}: brak kandydata wskazanego przez turniej")
        # Guard liczony PONOWNIE z pełnych rekordów: flagom z pakietu nie wierzymy.
        if not _clean_chosen(chosen, policy):
            rejected_counts["chosen_failed_reverification"] += 1
            continue
        if not _format_admissible(rejected):
            rejected_counts["rejected_failed_reverification"] += 1
            continue
        jaccard = normalized_query_jaccard(chosen.query, rejected.query)
        if jaccard > MAX_NORMALIZED_QUERY_JACCARD:
            rejected_counts["near_duplicate_query_pair"] += 1
            continue
        example_id = str(chosen.row["example_id"])
        cluster = clusters.get(example_id)
        if cluster is None:
            raise ValueError(f"grupa {group_id}: brak klastra dla {example_id}")
        if cluster in seen_clusters:
            rejected_counts["duplicate_passage_cluster"] += 1
            continue
        seen_clusters.add(cluster)
        positive = cast(Mapping[str, Any], chosen.row["positive"])
        pair_id = canonical_fingerprint(
            {
                "contract": PAIRS_CONTRACT,
                "variant": variant,
                "chosen": chosen.candidate_id,
                "rejected": rejected.candidate_id,
            }
        )[:32]
        pairs.append(
            JudgeSelectedPair(
                pair_id=pair_id,
                cohort_id=cohort_of_group[group_id],
                group_id=group_id,
                example_id=example_id,
                doc_id=str(chosen.row["doc_id"]),
                passage_cluster_id=cluster,
                split="train",
                prompt=str(chosen.row["prompt"]),
                prompt_sha256=str(chosen.row["prompt_sha256"]),
                passage=str(positive["text"]),
                rejected_variant=variant,
                chosen_candidate_id=chosen.candidate_id,
                rejected_candidate_id=rejected.candidate_id,
                chosen=chosen.query,
                rejected=rejected.query,
                chosen_components=_components(chosen),
                rejected_components=_components(rejected),
                votes_for_chosen=int(confirmation["votes_for_first"]),
                position_flips=int(confirmation.get("position_flips", 0)),
                tournament_comparisons=int(outcome["comparisons"]),
                chosen_pool_size=int(outcome["chosen_pool_size"]),
                rejected_pool_size=int(outcome["rejected_pool_size"]),
                normalized_query_jaccard=jaccard,
                requested_form=str(chosen.row["requested_form"]),
                requested_intent=str(chosen.row["requested_intent"]),
                selector="task06-judge-selected-pair-policy-v3",
                validated_defect_scope=["ungrounded", "copy_verbatim", "too_general"],
                form_or_focus_compliance_claimed=False,
                final_tests_used=[],
            )
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"
    try:
        staging.mkdir()
        pairs_path = staging / "pairs.jsonl"
        with JsonlWriter(pairs_path) as writer:
            for pair in pairs:
                writer.write(pair.model_dump(mode="json"))
        report = {
            "schema_version": 1,
            "contract": PAIRS_CONTRACT,
            "rejected_variant": variant,
            "pair_count": len(pairs),
            "rejection_counts": dict(sorted(rejected_counts.items())),
            "cohort_counts": dict(sorted(Counter(p.cohort_id for p in pairs).items())),
            "requested_form_counts": dict(sorted(Counter(p.requested_form for p in pairs).items())),
            "requested_intent_counts": dict(
                sorted(Counter(p.requested_intent for p in pairs).items())
            ),
            "normalized_query_jaccard_mean": (
                sum(p.normalized_query_jaccard for p in pairs) / len(pairs) if pairs else None
            ),
            "chosen_pool_size_mean": (
                sum(p.chosen_pool_size for p in pairs) / len(pairs) if pairs else None
            ),
            "validated_defect_scope": ["ungrounded", "copy_verbatim", "too_general"],
            "form_or_focus_compliance_claimed": False,
            "final_tests_used": [],
        }
        report_path = staging / "report.json"
        write_json(report_path, report)
        payload: dict[str, Any] = {
            "schema_version": 1,
            "contract": PAIRS_CONTRACT,
            "status": PAIRS_STATUS,
            "rejected_variant": variant,
            "selector_adr": "reports/decisions/task06_judge_selected_pair_policy_v3.md",
            "aggregation_amendment": (
                "reports/decisions/task06_v3_selector_aggregation_amendment_2026-08-27.md"
            ),
            "required_unanimous_votes": REQUIRED_UNANIMOUS_VOTES,
            "guard_policy_id": policy.policy_id,
            "guard_policy_sha256": file_sha256(policy_path),
            "bundle_manifest_sha256": file_sha256(bundle_dir / "manifest.json"),
            "tournament_outcomes_sha256": file_sha256(outcomes_path),
            "cohorts": sorted({p.cohort_id for p in pairs}) or sorted(cohort_of_group.values()),
            "tournament_groups": len(outcomes),
            "paired_groups": paired_groups,
            "unanimous_groups": unanimous_groups,
            "pair_count": len(pairs),
            "rejection_counts": dict(sorted(rejected_counts.items())),
            "pair_ids_fingerprint": ordered_ids_fingerprint([p.pair_id for p in pairs]),
            "pairs": {
                "path": pairs_path.name,
                "sha256": file_sha256(pairs_path),
                "record_count": len(pairs),
            },
            "report": {
                "path": report_path.name,
                "sha256": file_sha256(report_path),
                "record_count": 1,
            },
            "validated_defect_scope": ["ungrounded", "copy_verbatim", "too_general"],
            "form_or_focus_compliance_claimed": False,
            "audit_completed": False,
            "task07_training_authorized": False,
            "final_tests_used": [],
        }
        payload["manifest_fingerprint"] = canonical_fingerprint(payload)
        manifest = JudgeSelectedPairManifest.model_validate(payload)
        write_json(staging / "manifest.json", manifest.model_dump(mode="json"))
        if output_dir.exists():
            raise FileExistsError(f"wyjście już istnieje: {output_dir}")
        os.replace(staging, output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_pair_manifest(path: Path) -> JudgeSelectedPairManifest:
    return JudgeSelectedPairManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))


__all__ = [
    "MAX_NORMALIZED_QUERY_JACCARD",
    "PAIRS_CONTRACT",
    "JudgeSelectedPair",
    "JudgeSelectedPairManifest",
    "assemble_pairs",
    "load_pair_manifest",
]
