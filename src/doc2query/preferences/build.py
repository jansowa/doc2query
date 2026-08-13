"""Deterministic candidate selection and TRL-compatible preference export."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from doc2query.preferences.schemas import CandidateSet, ScoredCandidate, SelectionPolicy
from doc2query.utils.records import JsonlWriter, JsonParquetWriter, write_json


def _normalized_tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.casefold(), flags=re.UNICODE))


def normalized_query_jaccard(left: str, right: str) -> float:
    """Pairing-contract query similarity used by SelectionPolicy thresholds."""
    left_tokens = _normalized_tokens(left)
    right_tokens = _normalized_tokens(right)
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 1.0


def _candidate_is_chosen_eligible(row: ScoredCandidate, policy: SelectionPolicy) -> bool:
    score = row.scores
    if score.format_score < policy.min_chosen_format_score:
        return False
    if policy.require_chosen_answerable and not score.answerability_flag:
        return False
    return not (policy.reject_possible_false_negative_chosen and score.possible_false_negative)


def _candidate_is_rejected_eligible(row: ScoredCandidate, policy: SelectionPolicy) -> bool:
    if row.scores.format_score < policy.min_rejected_format_score:
        return False
    return not (
        policy.min_rejected_ground_score is not None
        and row.scores.ground_score < policy.min_rejected_ground_score
    )


def _validate_candidate_groups(candidates: Sequence[ScoredCandidate]) -> None:
    ids: set[str] = set()
    passage_splits: dict[str, str] = {}
    cluster_splits: dict[str, str] = {}
    for row in candidates:
        if row.candidate_id in ids:
            raise ValueError(f"duplicate candidate_id: {row.candidate_id}")
        ids.add(row.candidate_id)
        for key, split, label in (
            (row.passage_id, row.split, "passage"),
            (row.passage_cluster_id, row.split, "passage cluster"),
        ):
            target = passage_splits if label == "passage" else cluster_splits
            previous = target.setdefault(key, split)
            if previous != split:
                raise ValueError(f"{label} {key} crosses splits: {previous} vs {split}")


def select_candidate_sets(
    candidates: Sequence[ScoredCandidate], policy: SelectionPolicy
) -> tuple[list[CandidateSet], dict[str, Any]]:
    """Select deterministic, non-trivial chosen/rejected sets without model calls."""
    _validate_candidate_groups(candidates)
    groups: dict[tuple[str, str], list[ScoredCandidate]] = defaultdict(list)
    for row in candidates:
        groups[(row.split, row.passage_id)].append(row)

    selected: list[CandidateSet] = []
    skipped: Counter[str] = Counter()
    rejected_types: Counter[str] = Counter()
    for (_split, passage_id), rows in sorted(groups.items()):
        prompts = {row.prompt for row in rows}
        clusters = {row.passage_cluster_id for row in rows}
        if len(prompts) != 1 or len(clusters) != 1:
            raise ValueError(f"passage {passage_id} has inconsistent prompt or cluster identity")
        ordered = sorted(rows, key=lambda row: (-row.scores.total_score, row.candidate_id))
        chosen = next((row for row in ordered if _candidate_is_chosen_eligible(row, policy)), None)
        if chosen is None:
            skipped["no_eligible_chosen"] += 1
            continue
        rejected = [
            row
            for row in ordered
            if row.candidate_id != chosen.candidate_id
            and _candidate_is_rejected_eligible(row, policy)
            and chosen.scores.total_score - row.scores.total_score >= policy.min_score_margin
            and normalized_query_jaccard(chosen.query, row.query)
            <= policy.max_normalized_query_jaccard
        ]
        if not rejected:
            skipped["no_eligible_rejected"] += 1
            continue
        if policy.strategy == "top_vs_near_miss":
            rejected.sort(key=lambda row: (-row.scores.total_score, row.candidate_id))
        else:
            rejected.sort(key=lambda row: (row.scores.total_score, row.candidate_id))
        rejected = rejected[: policy.max_pairs_per_passage]
        margins = [chosen.scores.total_score - row.scores.total_score for row in rejected]
        set_id = hashlib.sha256(
            (chosen.candidate_id + "\0" + "\0".join(row.candidate_id for row in rejected)).encode()
        ).hexdigest()[:24]
        selected.append(
            CandidateSet(
                set_id=set_id,
                passage_id=passage_id,
                passage_cluster_id=next(iter(clusters)),
                split=chosen.split,
                prompt=next(iter(prompts)),
                chosen_candidate_id=chosen.candidate_id,
                rejected_candidate_ids=[row.candidate_id for row in rejected],
                score_margins=margins,
                strategy=policy.strategy,
            )
        )
        for row in rejected:
            rejected_types.update(row.failure_types or ["unclassified"])

    report: dict[str, Any] = {
        "candidate_count": len(candidates),
        "group_count": len(groups),
        "selected_set_count": len(selected),
        "pair_count": sum(len(item.rejected_candidate_ids) for item in selected),
        "skipped_groups": dict(sorted(skipped.items())),
        "rejected_failure_types": dict(sorted(rejected_types.items())),
        "policy": policy.model_dump(mode="json"),
    }
    return selected, report


def _preference_row(
    candidate_set: CandidateSet,
    chosen: ScoredCandidate,
    rejected: ScoredCandidate,
    margin: float,
) -> dict[str, Any]:
    return {
        "preference_id": f"{candidate_set.set_id}::{rejected.candidate_id}",
        "prompt": candidate_set.prompt,
        "chosen": chosen.query,
        "rejected": rejected.query,
        "split": candidate_set.split,
        "passage_id": candidate_set.passage_id,
        "passage_cluster_id": candidate_set.passage_cluster_id,
        "chosen_candidate_id": chosen.candidate_id,
        "rejected_candidate_id": rejected.candidate_id,
        "score_margin": margin,
        "rejected_failure_types": rejected.failure_types,
        "chosen_scores": chosen.scores.model_dump(mode="json"),
        "rejected_scores": rejected.scores.model_dump(mode="json"),
        "chosen_provenance": chosen.provenance.model_dump(mode="json"),
        "rejected_provenance": rejected.provenance.model_dump(mode="json"),
        "chosen_controls": chosen.controls,
        "rejected_controls": rejected.controls,
    }


def _write_rows(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    writer_class = JsonParquetWriter if path.suffix == ".parquet" else JsonlWriter
    if path.suffix not in {".jsonl", ".parquet"}:
        raise ValueError("output rows must use .jsonl or .parquet")
    with writer_class(path) as writer:
        for row in rows:
            writer.write(dict(row))


def build_preference_dataset(
    candidates: Sequence[ScoredCandidate],
    candidate_sets: Sequence[CandidateSet],
    output_dir: Path,
    *,
    output_format: str = "parquet",
) -> dict[str, Any]:
    """Write preference and mandatory continued-SFT controls by inherited split."""
    _validate_candidate_groups(candidates)
    candidate_by_id = {row.candidate_id: row for row in candidates}
    preference_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    continued_sft: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    role_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for item in candidate_sets:
        chosen = candidate_by_id.get(item.chosen_candidate_id)
        if chosen is None:
            raise ValueError(f"missing chosen candidate: {item.chosen_candidate_id}")
        if chosen.split != item.split or chosen.passage_id != item.passage_id:
            raise ValueError(f"chosen identity drift in set {item.set_id}")
        for rejected_id, margin in zip(
            item.rejected_candidate_ids, item.score_margins, strict=True
        ):
            rejected = candidate_by_id.get(rejected_id)
            if rejected is None:
                raise ValueError(f"missing rejected candidate: {rejected_id}")
            if rejected.split != item.split or rejected.passage_id != item.passage_id:
                raise ValueError(f"rejected identity drift in set {item.set_id}")
            if chosen.query.casefold() == rejected.query.casefold():
                raise ValueError(f"normalized-identical pair in set {item.set_id}")
            preference_rows[item.split].append(_preference_row(item, chosen, rejected, margin))
            role_counts[chosen.query.casefold()]["chosen"] += 1
            role_counts[rejected.query.casefold()]["rejected"] += 1
        continued_sft[item.split][chosen.candidate_id] = {
            "prompt": item.prompt,
            "completion": chosen.query,
            "candidate_id": chosen.candidate_id,
            "passage_id": item.passage_id,
            "passage_cluster_id": item.passage_cluster_id,
            "split": item.split,
            "raw_total_score": chosen.scores.total_score,
            "scores": chosen.scores.model_dump(mode="json"),
            "provenance": chosen.provenance.model_dump(mode="json"),
        }

    extension = ".parquet" if output_format == "parquet" else ".jsonl"
    if output_format not in {"parquet", "jsonl"}:
        raise ValueError("output_format must be parquet or jsonl")
    for split in ("train", "dev", "test"):
        _write_rows(output_dir / f"{split}{extension}", preference_rows[split])
        _write_rows(
            output_dir / f"continued_sft_{split}{extension}",
            continued_sft[split].values(),
        )
    one_sided = {
        query: dict(counts)
        for query, counts in role_counts.items()
        if not (counts["chosen"] and counts["rejected"])
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "contract": "task06-preference-data-v1",
        "candidate_count": len(candidates),
        "candidate_set_count": len(candidate_sets),
        "pair_counts": {split: len(preference_rows[split]) for split in ("train", "dev", "test")},
        "continued_sft_counts": {
            split: len(continued_sft[split]) for split in ("train", "dev", "test")
        },
        "one_sided_normalized_query_count": len(one_sided),
        "one_sided_normalized_query_examples": dict(sorted(one_sided.items())[:50]),
        "final_tests_used": [],
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def serialize_candidate_sets(path: Path, sets: Sequence[CandidateSet]) -> None:
    _write_rows(path, (item.model_dump(mode="json") for item in sets))


def candidate_fingerprint(candidates: Sequence[ScoredCandidate]) -> str:
    digest = hashlib.sha256()
    for row in sorted(candidates, key=lambda item: item.candidate_id):
        digest.update(json.dumps(row.model_dump(mode="json"), sort_keys=True).encode())
        digest.update(b"\n")
    return digest.hexdigest()
