"""Implementations behind the Task 02 thin command wrappers."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, cast

import yaml

from doc2query.reranker.base import FrozenRerankerConfig, PairScorer
from doc2query.reranker.benchmark import aggregate, disagreement
from doc2query.reranker.calibrate import PercentileCalibrator, RobustZCalibrator
from doc2query.reranker.focus import assign_focus
from doc2query.reranker.infer import GroupScore, score_group
from doc2query.reranker.load import load_frozen_reranker
from doc2query.rewards.calibration import (
    dominance_warnings,
    overlap_band_reward,
    pearson_matrix,
    quantile,
)
from doc2query.rewards.lexical import lexical_metrics
from doc2query.text.cache import AnalysisCache
from doc2query.text.normalization import SimplePolishNormalizer, SpacyPolishNormalizer
from doc2query.utils.records import JsonlWriter, read_records, write_json


def _model_config(path: Path) -> FrozenRerankerConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("model config must be a mapping")
    return FrozenRerankerConfig(**raw)


def _document_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("text"), str):
        return str(value["text"])
    raise ValueError("document must be a string or object with text")


def _groups(record: dict[str, Any]) -> list[tuple[str, str, str, list[str]]]:
    positives = record.get("positives")
    negatives = record.get("hard_negatives")
    if not isinstance(positives, list) or not positives:
        raise ValueError("record requires at least one positive")
    if not isinstance(negatives, list) or len(negatives) < 10:
        raise ValueError("benchmark requires at least 10 hard negatives")
    base_id = str(record.get("example_id", ""))
    query = str(record["query"])
    negative_texts = [_document_text(value) for value in negatives]
    groups = []
    for index, positive in enumerate(positives):
        doc_id = positive.get("doc_id") if isinstance(positive, dict) else None
        suffix = str(doc_id) if doc_id is not None else str(index)
        groups.append((f"{base_id}::{suffix}", query, _document_text(positive), negative_texts))
    return groups


def _slice_values(record: dict[str, Any], positive: str) -> dict[str, str]:
    raw_metadata = record.get("metadata")
    metadata = cast(dict[str, Any], raw_metadata) if isinstance(raw_metadata, dict) else {}
    length = len(positive.split())
    return {
        "domain": str(metadata.get("domain", "unknown")),
        "passage_length": "short" if length < 128 else "medium" if length < 512 else "long",
        "query_type": str(record.get("query_style", metadata.get("query_type", "unknown"))),
        "difficulty": str(metadata.get("baseline_difficulty", "unknown")),
    }


def _load_judges(configs: list[Path]) -> list[PairScorer]:
    if len(configs) < 1:
        raise ValueError("at least one judge config is required")
    return [load_frozen_reranker(_model_config(path)) for path in configs]


def benchmark_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark frozen rerankers on 1 positive + 10 hard negatives."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--judge-config", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    judges = _load_judges(args.judge_config)
    if len(judges) < 2:
        raise ValueError("benchmark requires primary and independent shadow judge")
    by_judge: dict[str, list[GroupScore]] = defaultdict(list)
    by_slice: dict[str, dict[str, dict[str, list[GroupScore]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with JsonlWriter(args.output_dir / "scores.jsonl") as writer:
        for record in read_records(args.input):
            for example_id, query, positive, negatives in _groups(record):
                slices = _slice_values(record, positive)
                for judge in judges:
                    result = score_group(
                        judge,
                        example_id=example_id,
                        query=query,
                        positive=positive,
                        negatives=negatives,
                    )
                    by_judge[judge.name].append(result)
                    for dimension, value in slices.items():
                        by_slice[judge.name][dimension][value].append(result)
                    writer.write(result.to_dict() | {"slices": slices})
    names = [judge.name for judge in judges]
    report = {
        "status": "measured",
        "input": str(args.input),
        "elapsed_seconds": time.perf_counter() - started,
        "judges": {name: aggregate(by_judge[name]) for name in names},
        "slices": {
            name: {
                dimension: {value: aggregate(rows) for value, rows in values.items()}
                for dimension, values in by_slice[name].items()
            }
            for name in names
        },
        "disagreement": disagreement(by_judge[names[0]], by_judge[names[1]]),
        "note": "raw logits are never averaged across judges",
    }
    write_json(args.output_dir / "benchmark.json", report)
    return 0


def score_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score canonical query/positive/negative groups.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--judge-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    judge = _load_judges([args.judge_config])[0]
    with JsonlWriter(args.output) as writer:
        for record in read_records(args.input):
            for example_id, query, positive, negatives in _groups(record):
                writer.write(
                    score_group(
                        judge,
                        example_id=example_id,
                        query=query,
                        positive=positive,
                        negatives=negatives,
                    ).to_dict()
                )
    return 0


def calibrate_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fit score and margin calibrators without changing a judge."
    )
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("robust_z", "percentile"), default="robust_z")
    args = parser.parse_args(argv)
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"score": [], "margin": []})
    for row in read_records(args.scores):
        grouped[str(row["judge"])]["score"].append(float(row["positive_score"]))
        grouped[str(row["judge"])]["margin"].append(float(row["margin"]))
    fitter = RobustZCalibrator.fit if args.method == "robust_z" else PercentileCalibrator.fit
    write_json(
        args.output,
        {
            "fit_split": "calibration_only_not_test",
            "judges": {
                judge: {name: fitter(values).to_dict() for name, values in components.items()}
                for judge, components in grouped.items()
            },
        },
    )
    return 0


def focus_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Assign soft sentence focus labels using a frozen judge."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--judge-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ambiguity-margin", type=float, default=0.1)
    args = parser.parse_args(argv)
    judge = _load_judges([args.judge_config])[0]
    with JsonlWriter(args.output) as writer:
        for row in read_records(args.input):
            result = dict(row)
            result.update(
                assign_focus(
                    judge,
                    str(row["query"]),
                    str(row["passage"]),
                    ambiguity_margin=args.ambiguity_margin,
                ).to_dict()
            )
            writer.write(result)
    return 0


def analysis_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Precompute CPU text analyses into an SQLite cache."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--text-field", action="append", default=[])
    parser.add_argument("--backend", choices=("simple", "spacy_pl"), default="simple")
    parser.add_argument("--spacy-model", default="pl_core_news_lg")
    args = parser.parse_args(argv)
    fields = args.text_field or ["query", "passage"]
    normalizer = (
        SimplePolishNormalizer()
        if args.backend == "simple"
        else SpacyPolishNormalizer(args.spacy_model)
    )
    analyzed = 0
    with AnalysisCache(args.cache, normalizer) as cache:
        for row in read_records(args.input):
            for field in fields:
                value = row.get(field)
                if isinstance(value, str):
                    cache.analyze(value)
                    analyzed += 1
    print(json.dumps({"analyzed": analyzed, "namespace": normalizer.cache_namespace}))
    return 0


def rewards_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate lexical reward bands on natural dev queries."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--low-percentile", type=float, default=0.1)
    parser.add_argument("--high-percentile", type=float, default=0.9)
    args = parser.parse_args(argv)
    normalizer = SimplePolishNormalizer()
    components: list[dict[str, float]] = []
    human: list[float] = []
    for row in read_records(args.input):
        query = normalizer.analyze(str(row["query"]))
        passage = normalizer.analyze(str(row["passage"]))
        metric = lexical_metrics(query, passage)
        current = {
            "content_jaccard": metric.content_jaccard,
            "copy_density": metric.copy_density,
            "normalized_lcs": metric.normalized_lcs,
        }
        for optional in ("primary_margin", "shadow_margin"):
            if optional in row:
                current[optional] = float(row[optional])
        components.append(current)
        if "human_answerability" in row:
            human.append(float(row["human_answerability"]))
    overlaps = [row["content_jaccard"] for row in components]
    epsilon = 1e-6
    low = min(max(quantile(overlaps, args.low_percentile), epsilon), 1 - 2 * epsilon)
    high = min(max(quantile(overlaps, args.high_percentile), low + epsilon), 1 - epsilon)
    for row in components:
        row["overlap_reward"] = overlap_band_reward(row["content_jaccard"], low=low, high=high)
        if "primary_margin" in row:
            row["total"] = row["overlap_reward"] + row["primary_margin"]
    correlations = pearson_matrix(components)
    report: dict[str, Any] = {
        "status": "measured_from_input",
        "count": len(components),
        "overlap_band": {"low": low, "high": high},
        "component_means": {key: fmean(row[key] for row in components) for key in components[0]},
        "correlations": correlations,
        "dominance_warnings": dominance_warnings(correlations) if "total" in correlations else [],
        "judges_reported_separately": True,
    }
    if "total" not in correlations:
        report["composite_note"] = "not computed: primary_margin is absent"
    if human and len(human) == len(components):
        labeled = [
            dict(row, human_answerability=label)
            for row, label in zip(components, human, strict=True)
        ]
        report["human_label_correlations"] = pearson_matrix(labeled)["human_answerability"]
    else:
        report["human_label_correlations"] = None
        report["human_label_note"] = "input did not contain a complete human_answerability column"
    write_json(args.output, report)
    return 0
