"""Fingerprint-safe run comparison with probe-first ranking."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

from doc2query.evaluation.bootstrap import assert_same_test_fingerprint, paired_bootstrap
from doc2query.evaluation.probe_negatives import assert_same_negative_contract
from doc2query.evaluation.retrieval import CORPUS_RETRIEVAL
from doc2query.utils.records import read_records, write_json


def _per_query(path: Path, metric: str) -> dict[str, float]:
    result = {}
    for row in read_records(path):
        value = row.get(metric)
        if isinstance(value, (int, float)):
            result[str(row["example_id"])] = float(value)
    return result


def compare_retrieval_runs(
    left_summary_path: Path,
    right_summary_path: Path,
    *,
    left_per_query_path: Path,
    right_per_query_path: Path,
    output_path: Path,
    samples: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    left = json.loads(left_summary_path.read_text(encoding="utf-8"))
    right = json.loads(right_summary_path.read_text(encoding="utf-8"))
    fingerprint = assert_same_test_fingerprint(left, right)
    if left.get("protocol") != CORPUS_RETRIEVAL or right.get("protocol") != CORPUS_RETRIEVAL:
        raise ValueError("probe comparison requires corpus_retrieval summaries")
    if left.get("corpus_sha256") != right.get("corpus_sha256"):
        raise ValueError("probe comparison requires the same frozen corpus fingerprint")
    negative_contract = assert_same_negative_contract(left, right)
    metrics = (
        "corpus_recall_at_1",
        "corpus_recall_at_5",
        "corpus_recall_at_10",
        "corpus_recall_at_100",
        "corpus_mrr_at_10",
        "corpus_ndcg_at_10",
        "corpus_map",
    )
    report = {
        "test_fingerprint": fingerprint,
        "left": str(left_summary_path),
        "right": str(right_summary_path),
        "negative_contract": negative_contract,
        "bootstrap": {
            metric: paired_bootstrap(
                _per_query(left_per_query_path, metric),
                _per_query(right_per_query_path, metric),
                samples=samples,
                seed=seed,
            )
            for metric in metrics
        },
    }
    write_json(output_path, report)
    return report


def _generator_query_means(path: Path, *, mode: str, metric: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in read_records(path):
        value = row.get(metric)
        if str(row.get("mode")) == mode and isinstance(value, (int, float)):
            grouped[str(row["example_id"])].append(float(value))
    return {key: fmean(values) for key, values in grouped.items()}


def compare_generator_runs(
    left_summary_path: Path,
    right_summary_path: Path,
    *,
    left_per_generation_path: Path,
    right_per_generation_path: Path,
    output_path: Path,
    samples: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Bootstrap query-level means separately for deterministic and diverse decoding."""
    left = json.loads(left_summary_path.read_text(encoding="utf-8"))
    right = json.loads(right_summary_path.read_text(encoding="utf-8"))
    fingerprint = assert_same_test_fingerprint(left, right)
    left_corpus = left.get("protocols", {}).get(CORPUS_RETRIEVAL, {})
    right_corpus = right.get("protocols", {}).get(CORPUS_RETRIEVAL, {})
    if left_corpus.get("status") != "measured" or right_corpus.get("status") != "measured":
        raise ValueError("generator comparison requires measured corpus_retrieval round-trip")
    left_index = (left_corpus.get("index") or {}).get("index_fingerprint")
    right_index = (right_corpus.get("index") or {}).get("index_fingerprint")
    if not left_index or left_index != right_index:
        raise ValueError("generator comparison requires the same frozen corpus index fingerprint")
    metrics = (
        "pool_recall_at_1",
        "pool_recall_at_5",
        "pool_mrr",
        "pool_ndcg_at_10",
        "pool_margin",
        "corpus_round_trip_at_1",
        "corpus_round_trip_at_5",
        "corpus_round_trip_at_20",
        "corpus_round_trip_at_100",
        "content_jaccard",
        "copy_density",
        "sentence_level_source_hit",
    )
    report = {
        "test_fingerprint": fingerprint,
        "corpus_index_fingerprint": left_index,
        "unit": "frozen natural-query record; diverse candidates averaged within query",
        "difference": "right_minus_left",
        "modes": {
            mode: {
                metric: paired_bootstrap(
                    _generator_query_means(left_per_generation_path, mode=mode, metric=metric),
                    _generator_query_means(right_per_generation_path, mode=mode, metric=metric),
                    samples=samples,
                    seed=seed,
                )
                for metric in metrics
            }
            for mode in ("deterministic", "diverse")
        },
    }
    write_json(output_path, report)
    return report


def rank_variants(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank only measured probe results; intrinsic reward never substitutes for them."""
    measured = [
        summary
        for summary in summaries
        if isinstance(summary.get("probe_embedder", {}).get("corpus_ndcg_at_10"), (int, float))
    ]
    return sorted(
        measured,
        key=lambda value: (
            -float(value["probe_embedder"]["corpus_ndcg_at_10"]),
            -float(value["probe_embedder"].get("corpus_mrr_at_10", 0.0)),
            str(value.get("experiment_id", "")),
        ),
    )
