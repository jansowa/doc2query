from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from doc2query.evaluation.datasets import freeze_evaluation_sets
from doc2query.evaluation.native_holdout import (
    adapt_polqa_corpus,
    audit_exact_overlap,
    freeze_native_pl_holdout,
    holdout_fingerprint,
    load_holdout_records,
    verify_native_holdout_manifest,
)
from doc2query.evaluation.report import build_embedder_report
from doc2query.evaluation.translationese import (
    aggregate_translationese,
    translationese_indicators,
)
from doc2query.utils.records import JsonlWriter


def _canonical(identifier: str) -> dict[str, Any]:
    return {
        "example_id": identifier,
        "query": f"Jakie jest testowe pytanie numer {identifier}?",
        "positives": [{"doc_id": f"p-{identifier}", "text": "Pozytywny dokument."}],
        "hard_negatives": [
            {"doc_id": f"n-{identifier}-{index}", "text": f"Negatyw {index}."}
            for index in range(10)
        ],
        "metadata": {"split": "test"},
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with JsonlWriter(path) as writer:
        for row in rows:
            writer.write(row)


def _translated_manifest(tmp_path: Path) -> Path:
    source = tmp_path / "test.jsonl"
    adversarial = tmp_path / "adversarial.jsonl"
    _write_jsonl(source, [_canonical(str(index)) for index in range(8)])
    _write_jsonl(adversarial, [{"case_id": "a", "passage": "A", "query": "Q"}])
    (tmp_path / "split_manifest.json").write_text("{}\n", encoding="utf-8")
    frozen = tmp_path / "task04-v1"
    freeze_evaluation_sets(
        dev_path=source,
        test_path=source,
        adversarial_path=adversarial,
        output_dir=frozen,
        human_panel_size=1,
        generation_panel_size=1,
    )
    return frozen / "manifest.json"


def _write_polqa(path: Path, *, split: str = "test") -> None:
    fields = (
        "question_id",
        "question",
        "passage_id",
        "passage_title",
        "passage_wiki",
        "passage_text",
        "relevant",
        "split",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for question_id in ("10", "20", "30"):
            writer.writerow(
                {
                    "question_id": question_id,
                    "question": f"Gdzie znajduje się obiekt {question_id}?",
                    "passage_id": f"{question_id}-0",
                    "passage_title": "Obiekt",
                    "passage_wiki": f"Obiekt {question_id} znajduje się w Polsce.",
                    "passage_text": "",
                    "relevant": "True",
                    "split": split,
                }
            )
            writer.writerow(
                {
                    "question_id": question_id,
                    "question": f"Gdzie znajduje się obiekt {question_id}?",
                    "passage_id": f"{question_id}-1",
                    "passage_title": "Inny",
                    "passage_wiki": "Inny obiekt znajduje się za granicą.",
                    "passage_text": "",
                    "relevant": "False",
                    "split": split,
                }
            )


def test_missing_native_is_explicit_and_has_no_fabricated_hash(tmp_path: Path) -> None:
    manifest = freeze_native_pl_holdout(
        translated_manifest=_translated_manifest(tmp_path),
        output_dir=tmp_path / "holdout",
    )
    native = manifest["sets"]["test_native_pl"]
    translated = manifest["sets"]["test_translated_msmarco_pl"]
    assert manifest["status"] == "incomplete"
    assert native["status"] == "missing_source_artifact"
    assert native["records_sha256"] is None
    assert native["id_list_sha256"] is None
    assert translated["status"] == "materialized"
    assert len(translated["records_sha256"]) == 64
    assert len(translated["profiles"]["quick"]["id_list_sha256"]) == 64
    assert manifest["artifacts"]["translated_quick_corpus"]["document_count"] == 88
    verified = verify_native_holdout_manifest(tmp_path / "holdout" / "manifest.json")
    assert verified["status"] == "incomplete"
    assert verified["missing"] == {"test_native_pl": "missing_source_artifact"}
    assert "translated_quick_corpus" in verified["verified_artifacts"]


def test_polqa_import_freezes_profiles_and_detects_tampering(tmp_path: Path) -> None:
    source = tmp_path / "polqa-test.csv"
    _write_polqa(source)
    output = tmp_path / "holdout"
    manifest = freeze_native_pl_holdout(
        translated_manifest=_translated_manifest(tmp_path),
        output_dir=output,
        polqa_test_path=source,
        polqa_revision="a" * 40,
    )
    native = manifest["sets"]["test_native_pl"]
    assert native["status"] == "materialized"
    assert native["id_count"] == 3
    assert native["source_artifact_sha256"]
    records = load_holdout_records(output / "manifest.json", "test_native_pl", profile="quick")
    assert [row["example_id"] for row in records] == ["polqa:10", "polqa:20", "polqa:30"]
    assert records[0]["metadata"]["usage_policy"] == "evaluation_only_no_tuning"
    assert records[0]["positives"][0]["doc_id"] == "polqa:10-0"
    assert len(holdout_fingerprint(output / "manifest.json", "test_native_pl", "quick")) == 64
    assert manifest["artifacts"]["native_quick_corpus"]["document_count"] == 6
    overlap = manifest["artifacts"]["native_translated_exact_overlap"]
    assert overlap["status"] == "materialized"
    assert overlap["near_duplicate_status"] == "not_measured"
    profile_ids = output / "test_native_pl.quick.ids.jsonl"
    profile_ids.write_text('{"id":"polqa:changed"}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="profile ID-list"):
        load_holdout_records(output / "manifest.json", "test_native_pl", profile="quick")


def test_polqa_import_rejects_non_test_rows(tmp_path: Path) -> None:
    source = tmp_path / "polqa-validation.csv"
    _write_polqa(source, split="validation")
    with pytest.raises(ValueError, match="only test rows"):
        freeze_native_pl_holdout(
            translated_manifest=_translated_manifest(tmp_path),
            output_dir=tmp_path / "holdout",
            polqa_test_path=source,
        )


def test_full_polqa_corpus_adapter_rejects_conflicting_ids(tmp_path: Path) -> None:
    source = tmp_path / "passages.jsonl"
    _write_jsonl(
        source,
        [
            {"id": "1-0", "title": "Pierwszy", "text": "Pierwszy dokument."},
            {"id": "2-0", "title": "Drugi", "text": "Drugi dokument."},
        ],
    )
    output = tmp_path / "documents.jsonl"
    result = adapt_polqa_corpus(source, output)
    assert result["document_count"] == 2
    assert len(result["sha256"]) == 64
    first = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert first["doc_id"] == "polqa:1-0"

    conflicting = tmp_path / "conflicting.jsonl"
    _write_jsonl(
        conflicting,
        [
            {"id": "same", "text": "A"},
            {"id": "same", "text": "B"},
        ],
    )
    with pytest.raises(ValueError, match="conflicting duplicate"):
        adapt_polqa_corpus(conflicting, tmp_path / "conflicting-output.jsonl")


def test_exact_overlap_does_not_claim_near_duplicate_measurement() -> None:
    native = [_canonical("same")]
    translated = [_canonical("same"), _canonical("other")]
    result = audit_exact_overlap(native, translated)
    assert result["exact_query_overlap_count"] == 1
    assert result["exact_document_overlap_count"] > 0
    assert result["near_duplicate_overlap"] is None
    assert result["near_duplicate_status"] == "not_measured"


def test_translationese_signal_is_transparent_and_weak_ascii_flag_is_not_dominant() -> None:
    polish = translationese_indicators("Gdzie znajduje się stolica województwa?")
    residue = translationese_indicators("What jest nazwą tego miasta ?")
    ascii_only = translationese_indicators(
        "Jak nazywa sie bardzo znany budynek stojacy w centrum miasta"
    )
    assert polish["risk_score"] == 0.0
    assert residue["flags"]["english_residue"]
    assert residue["flags"]["suspicious_punctuation_spacing"]
    assert residue["risk_score"] > ascii_only["risk_score"]
    aggregate = aggregate_translationese(["Gdzie leży Łódź?", "What to jest ?"])
    assert aggregate["query_count"] == 2
    assert aggregate["interpretation"].startswith("diagnostic_only")


def test_embedder_report_without_native_is_incomplete_and_keeps_missing_values(
    tmp_path: Path,
) -> None:
    result = {
        "report_status": "incomplete",
        "comparison_eligible": False,
        "incomplete_reasons": ["native holdout manifest was not supplied"],
        "evaluation_sets": {
            "test_native_pl": {
                "status": "not_measured",
                "profile": "quick",
                "test_fingerprint": None,
                "metrics": None,
            },
            "test_translated_msmarco_pl": {
                "status": "measured",
                "profile": "full",
                "test_fingerprint": "f" * 64,
                "corpus_candidate_count": 100,
                "metrics": {"corpus_ndcg_at_10": 0.5},
                "translationese": {"mean_risk_score": 0.1},
            },
        },
    }
    report = build_embedder_report(
        result,
        markdown_path=tmp_path / "report.md",
        json_path=tmp_path / "report.json",
    )
    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    persisted = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert not report["complete"]
    assert not report["comparison_eligible"]
    assert persisted["native"]["metrics"] is None
    assert "NOT MEASURED" in markdown
    assert "test_native_pl" in markdown
    assert "test_translated_msmarco_pl" in markdown
