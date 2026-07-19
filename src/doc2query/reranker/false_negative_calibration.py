"""Data-driven calibration of the possible-false-negative operating point."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from doc2query.evaluation.probe_negatives import (
    CALIBRATION_ARTIFACT_TYPE,
    CALIBRATION_OPERATOR,
    CALIBRATION_SCHEMA_VERSION,
    CALIBRATION_SCORE_KIND,
    calibration_artifact_fingerprint,
)
from doc2query.reranker.base import FrozenRerankerConfig
from doc2query.reranker.load import load_frozen_reranker
from doc2query.utils.records import JsonlWriter, read_records

SELECTION_METHOD = "query_macro_youden_j_maximum_with_highest_threshold_tie_break_on_frozen_dev"


@dataclass(frozen=True)
class QueryScoreGroup:
    """Known-positive and inherited-negative raw logits for one development query."""

    query_id: str
    positive_scores: tuple[float, ...]
    negative_scores: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.query_id or not self.positive_scores or not self.negative_scores:
            raise ValueError("each calibration query requires positive and negative scores")
        scores = (*self.positive_scores, *self.negative_scores)
        if not all(math.isfinite(value) for value in scores):
            raise ValueError("calibration scores must be finite")


@dataclass(frozen=True)
class YoudenOperatingPoint:
    threshold: float
    query_macro_true_positive_rate: float
    query_macro_false_positive_rate: float
    youden_j: float


def select_query_macro_youden_threshold(
    groups: Sequence[QueryScoreGroup],
) -> YoudenOperatingPoint:
    """Maximize query-macro Youden J with a conservative deterministic tie break."""
    if not groups:
        raise ValueError("threshold calibration requires at least one query")
    query_weight = 1.0 / len(groups)
    events: dict[float, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for group in groups:
        positive_weight = query_weight / len(group.positive_scores)
        negative_weight = query_weight / len(group.negative_scores)
        for score in group.positive_scores:
            events[score][0] += positive_weight
        for score in group.negative_scores:
            events[score][1] += negative_weight
    highest = max(events)
    best = YoudenOperatingPoint(
        threshold=math.nextafter(highest, math.inf),
        query_macro_true_positive_rate=0.0,
        query_macro_false_positive_rate=0.0,
        youden_j=0.0,
    )
    true_positive_rate = 0.0
    false_positive_rate = 0.0
    for threshold in sorted(events, reverse=True):
        positive_delta, negative_delta = events[threshold]
        true_positive_rate += positive_delta
        false_positive_rate += negative_delta
        youden_j = true_positive_rate - false_positive_rate
        # Descending traversal makes the first exact optimum the highest threshold.
        if youden_j > best.youden_j + 1e-12:
            best = YoudenOperatingPoint(
                threshold=threshold,
                query_macro_true_positive_rate=true_positive_rate,
                query_macro_false_positive_rate=false_positive_rate,
                youden_j=youden_j,
            )
    return best


def _query_youden_at_threshold(group: QueryScoreGroup, threshold: float) -> float:
    sensitivity = sum(score >= threshold for score in group.positive_scores) / len(
        group.positive_scores
    )
    false_positive_rate = sum(score >= threshold for score in group.negative_scores) / len(
        group.negative_scores
    )
    return sensitivity - false_positive_rate


def bootstrap_fixed_threshold_youden_ci(
    groups: Sequence[QueryScoreGroup],
    *,
    threshold: float,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    """Query bootstrap CI for J at the selected operating point."""
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    values = [_query_youden_at_threshold(group, threshold) for group in groups]
    generator = random.Random(seed)
    estimates = sorted(
        sum(values[generator.randrange(len(values))] for _ in values) / len(values)
        for _ in range(samples)
    )
    low_index = max(0, math.ceil(0.025 * samples) - 1)
    high_index = min(samples - 1, math.ceil(0.975 * samples) - 1)
    return estimates[low_index], estimates[high_index]


def load_query_score_groups(
    scores_path: Path,
    *,
    expected_judge: str,
) -> list[QueryScoreGroup]:
    """Load Task 02 group scores without over-weighting multi-positive queries."""
    positives: dict[str, dict[str, float]] = defaultdict(dict)
    negatives: dict[str, dict[str, float]] = defaultdict(dict)
    for row in read_records(scores_path):
        judge = str(row.get("judge", ""))
        if judge != expected_judge:
            raise ValueError(f"score judge {judge!r} does not match {expected_judge!r}")
        query_id = str(row.get("query_id", ""))
        if row.get("schema") == "possible_false_negative_dev_scores_v1":
            positive_doc_ids = row.get("positive_doc_ids")
            negative_doc_ids = row.get("negative_doc_ids")
            positive_values = row.get("positive_scores")
            negative_values = row.get("negative_scores")
            if (
                not query_id
                or not isinstance(positive_doc_ids, list)
                or not isinstance(negative_doc_ids, list)
                or not isinstance(positive_values, list)
                or not isinstance(negative_values, list)
                or len(positive_doc_ids) != len(positive_values)
                or len(negative_doc_ids) != len(negative_values)
            ):
                raise ValueError("calibration score group has invalid query/document provenance")
            for doc_id, raw_score in zip(positive_doc_ids, positive_values, strict=True):
                positives[query_id][str(doc_id)] = float(raw_score)
            for doc_id, raw_score in zip(negative_doc_ids, negative_values, strict=True):
                negatives[query_id][str(doc_id)] = float(raw_score)
            if positives[query_id].keys() & negatives[query_id].keys():
                raise ValueError("one query/document pair is both positive and negative")
            continue
        positive_doc_id = str(row.get("positive_doc_id", ""))
        negative_doc_ids = row.get("negative_doc_ids")
        document_scores = row.get("document_scores")
        if (
            not query_id
            or not positive_doc_id
            or not isinstance(negative_doc_ids, list)
            or not isinstance(document_scores, list)
            or len(document_scores) != len(negative_doc_ids) + 1
        ):
            raise ValueError("scores lack complete query/document provenance")
        positive_score = float(document_scores[0])
        previous_positive = positives[query_id].setdefault(positive_doc_id, positive_score)
        if not math.isclose(previous_positive, positive_score, abs_tol=1e-7):
            raise ValueError("one query/document pair has inconsistent positive scores")
        for doc_id, raw_score in zip(negative_doc_ids, document_scores[1:], strict=True):
            negative_id = str(doc_id)
            negative_score = float(raw_score)
            if negative_id in positives[query_id]:
                raise ValueError("one query/document pair is both positive and negative")
            previous_negative = negatives[query_id].setdefault(negative_id, negative_score)
            if not math.isclose(previous_negative, negative_score, abs_tol=1e-7):
                raise ValueError("one query/document pair has inconsistent negative scores")
    if positives.keys() != negatives.keys():
        raise ValueError("every calibration query must have both score classes")
    return [
        QueryScoreGroup(
            query_id=query_id,
            positive_scores=tuple(positives[query_id].values()),
            negative_scores=tuple(negatives[query_id].values()),
        )
        for query_id in sorted(positives)
    ]


def score_frozen_dev_for_false_negative_calibration(
    *,
    input_path: Path,
    output_path: Path,
    judge: FrozenRerankerConfig,
) -> dict[str, Any]:
    """Score every known positive and inherited negative in an immutable dev file."""
    scorer = load_frozen_reranker(judge)
    query_count = 0
    positive_count = 0
    negative_count = 0
    minimum_negatives: int | None = None
    buffered: list[tuple[str, str, list[dict[str, Any]], list[dict[str, Any]]]] = []

    def flush(writer: JsonlWriter) -> None:
        pairs = [
            (query, str(document["text"]))
            for _query_id, query, positive_docs, negative_docs in buffered
            for document in [*positive_docs, *negative_docs]
        ]
        ordered = sorted(
            enumerate(pairs),
            key=lambda item: len(item[1][0].split()) + len(item[1][1].split()),
        )
        ordered_scores = scorer.score_pairs([pair for _index, pair in ordered])
        scores = [0.0] * len(pairs)
        for (index, _pair), score in zip(ordered, ordered_scores, strict=True):
            scores[index] = score
        offset = 0
        for query_id, _query, positive_docs, negative_docs in buffered:
            group_size = len(positive_docs) + len(negative_docs)
            group_scores = scores[offset : offset + group_size]
            offset += group_size
            writer.write(
                {
                    "schema": "possible_false_negative_dev_scores_v1",
                    "judge": scorer.name,
                    "query_id": query_id,
                    "positive_doc_ids": [str(document["doc_id"]) for document in positive_docs],
                    "negative_doc_ids": [str(document["doc_id"]) for document in negative_docs],
                    "positive_scores": group_scores[: len(positive_docs)],
                    "negative_scores": group_scores[len(positive_docs) :],
                }
            )
        if offset != len(scores):
            raise RuntimeError("batched calibration scorer lost pair alignment")
        buffered.clear()

    with JsonlWriter(output_path) as writer:
        for record in read_records(input_path):
            metadata = record.get("metadata")
            split = str(metadata.get("split", "")) if isinstance(metadata, dict) else ""
            if not split.lower().startswith("dev") or "test" in split.lower():
                raise ValueError("calibration scorer accepts frozen dev records only")
            query_id = str(record.get("example_id", ""))
            query = str(record.get("query", ""))
            positives = record.get("positives")
            negatives = record.get("hard_negatives")
            if (
                not query_id
                or not query
                or not isinstance(positives, list)
                or not positives
                or not isinstance(negatives, list)
                or not negatives
            ):
                raise ValueError("each dev record requires a query and both document classes")
            positive_docs = sorted(positives, key=lambda value: str(value["doc_id"]))
            negative_docs = sorted(negatives, key=lambda value: str(value["doc_id"]))
            buffered.append((query_id, query, positive_docs, negative_docs))
            query_count += 1
            positive_count += len(positive_docs)
            negative_count += len(negative_docs)
            minimum_negatives = (
                len(negative_docs)
                if minimum_negatives is None
                else min(minimum_negatives, len(negative_docs))
            )
            if len(buffered) >= 256:
                flush(writer)
        if buffered:
            flush(writer)
    return {
        "fit_split": "dev",
        "queries": query_count,
        "positive_pairs": positive_count,
        "inherited_negative_pairs": negative_count,
        "minimum_inherited_negatives_per_query": minimum_negatives,
        "judge": judge.name_or_path,
        "judge_revision": judge.revision,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_calibration_artifact(
    *,
    scores_path: Path,
    fit_dataset_path: Path,
    fit_split: str,
    judge: FrozenRerankerConfig,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 42,
) -> dict[str, Any]:
    """Build the pinned Task 02 artifact consumed by P-03."""
    normalized_split = fit_split.strip().lower()
    if not normalized_split.startswith("dev") or "test" in normalized_split:
        raise ValueError("possible-false-negative calibration is restricted to a dev split")
    groups = load_query_score_groups(scores_path, expected_judge=judge.name_or_path)
    operating_point = select_query_macro_youden_threshold(groups)
    ci_low, ci_high = bootstrap_fixed_threshold_youden_ci(
        groups,
        threshold=operating_point.threshold,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    dataset_fingerprint = sha256_file(fit_dataset_path)
    scores_sha256 = sha256_file(scores_path)
    counts = {
        "queries": len(groups),
        "positive_pairs": sum(len(group.positive_scores) for group in groups),
        "inherited_negative_pairs": sum(len(group.negative_scores) for group in groups),
    }
    identity = hashlib.sha256(
        json.dumps(
            {
                "dataset": dataset_fingerprint,
                "judge_revision": judge.revision,
                "method": SELECTION_METHOD,
                "scores": scores_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()[:16]
    payload: dict[str, Any] = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "artifact_type": CALIBRATION_ARTIFACT_TYPE,
        "artifact_id": f"pfn-dev-v1-{identity}",
        "fit_split": normalized_split,
        "fit_dataset_fingerprint": dataset_fingerprint,
        "primary_judge": {
            "name_or_path": judge.name_or_path,
            "revision": judge.revision,
            "license": judge.license,
            "trust_remote_code": judge.trust_remote_code,
            "max_length": judge.max_length,
        },
        "score_kind": CALIBRATION_SCORE_KIND,
        "comparison_operator": CALIBRATION_OPERATOR,
        "threshold": operating_point.threshold,
        "selection_method": SELECTION_METHOD,
        "selection_objective": "maximize query-macro sensitivity minus false-positive rate",
        "tie_break": "highest threshold among equal maxima (conservative specificity)",
        "source_scores_sha256": scores_sha256,
        "counts": counts,
        "operating_point": {
            "query_macro_true_positive_rate": (operating_point.query_macro_true_positive_rate),
            "query_macro_false_positive_rate": (operating_point.query_macro_false_positive_rate),
            "youden_j": operating_point.youden_j,
            "youden_j_query_bootstrap_95_ci": [ci_low, ci_high],
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
        },
        "label_contract": {
            "positive": "frozen dev known positive document",
            "negative": "frozen dev inherited hard negative document",
            "warning": (
                "Inherited hard-negative labels may be noisy; the threshold flags candidates "
                "whose frozen-primary score lies in the empirically positive-like region and "
                "does not prove false negativity."
            ),
        },
        "tests_used_for_threshold_tuning": [],
    }
    payload["artifact_fingerprint"] = calibration_artifact_fingerprint(payload)
    return payload
