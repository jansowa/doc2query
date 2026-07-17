"""Argparse entry points used by thin Task 01 scripts."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import yaml

from doc2query.data.deduplicate import deduplicate_documents
from doc2query.data.index import build_document_index
from doc2query.data.invert import invert_doc2query_pairs
from doc2query.data.report import build_data_report
from doc2query.data.split import SplitConfig, build_splits
from doc2query.data.validate import ValidationPolicy, validate_dataset


def _policy(path: Path | None) -> ValidationPolicy:
    if path is None:
        return ValidationPolicy.defaults()
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("validation policy must be a mapping")
    modes = value.pop("modes", {})
    if not isinstance(modes, dict):
        raise ValueError("validation policy modes must be a mapping")
    return replace(ValidationPolicy.defaults(modes), **value)


def validate_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate canonical retrieval JSONL/Parquet.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--rejected", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--policy", type=Path)
    args = parser.parse_args(argv)
    report = validate_dataset(
        args.input,
        accepted_path=args.accepted,
        rejected_path=args.rejected,
        report_path=args.report,
        policy=_policy(args.policy),
    )
    return 2 if report["contains_error_policy_violations"] else 0


def index_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a disk-backed canonical document index.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_document_index(
        args.input,
        sqlite_path=args.sqlite,
        documents_path=args.documents,
        report_path=args.report,
    )
    return 2 if report["conflicting_document_ids"] else 0


def deduplicate_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cluster exact and near-duplicate documents.")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-hamming-distance", type=int, default=3)
    parser.add_argument("--bands", type=int, default=4)
    parser.add_argument("--candidate-cap", type=int, default=500)
    parser.add_argument(
        "--resume-if-available",
        action="store_true",
        help="resume a compatible SQLite checkpoint, or start from zero when none exists",
    )
    args = parser.parse_args(argv)
    deduplicate_documents(
        args.index,
        output_path=args.output,
        report_path=args.report,
        max_hamming_distance=args.max_hamming_distance,
        bands=args.bands,
        candidate_cap=args.candidate_cap,
        resume_if_available=args.resume_if_available,
    )
    return 0


def split_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build deterministic component-level splits.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dedup-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.90)
    parser.add_argument("--dev-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--keep-cross-split-negatives", action="store_true")
    args = parser.parse_args(argv)
    build_splits(
        args.input,
        args.dedup_map,
        output_dir=args.output_dir,
        config=SplitConfig(
            train_ratio=args.train_ratio,
            dev_ratio=args.dev_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            version=args.version,
            remove_cross_split_negatives=not args.keep_cross_split_negatives,
        ),
    )
    return 0


def invert_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Invert retrieval records to passage-query pairs.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--split")
    parser.add_argument("--max-positives-per-query", type=int)
    args = parser.parse_args(argv)
    invert_doc2query_pairs(
        args.input,
        output_path=args.output,
        report_path=args.report,
        split=args.split,
        max_positives_per_query=args.max_positives_per_query,
    )
    return 0


def report_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build JSON and HTML data audit reports.")
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path)
    parser.add_argument("--dedup-report", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--tokenizer-config", type=Path)
    args = parser.parse_args(argv)
    build_data_report(
        args.input,
        json_path=args.json,
        html_path=args.html,
        validation_report=args.validation_report,
        dedup_report=args.dedup_report,
        split_manifest=args.split_manifest,
        tokenizer_config=args.tokenizer_config,
    )
    return 0
