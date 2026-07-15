from pathlib import Path
from typing import Any

import pytest

from doc2query.data.deduplicate import deduplicate_documents, load_dedup_map
from doc2query.data.index import build_document_index
from doc2query.data.invert import invert_doc2query_pairs
from doc2query.data.report import build_data_report
from doc2query.data.split import SplitConfig, build_splits
from doc2query.data.token_lengths import load_tokenizer_specs
from doc2query.data.validate import ValidationPolicy, validate_dataset, validate_record
from doc2query.utils.records import JsonlWriter, read_records


def _positive_text(index: int) -> str:
    if index == 0 or index == 1:
        return "Pompa ciepła pobiera energię z otoczenia i ogrzewa budynek mieszkalny."
    if index == 2:
        return "Panel słoneczny zamienia światło na energię elektryczną dla domu przez cały dzień."
    if index == 3:
        return (
            "Panel słoneczny zamienia światło na energię elektryczną dla budynku przez cały dzień."
        )
    return f"Dokument pozytywny numer {index} opisuje unikalny fakt naukowy i jego zastosowanie."


def _records(count: int = 18) -> list[dict[str, Any]]:
    result = []
    for index in range(count):
        next_index = (index + 1) % count
        negatives = [
            {"doc_id": f"p-{next_index}", "text": _positive_text(next_index), "metadata": {}}
        ]
        negatives.extend(
            {
                "doc_id": f"n-{index}-{negative}",
                "text": (
                    f"Trudny negatywny dokument {index} wariant {negative} opisuje podobny temat, "
                    "ale inny fakt szczegółowy."
                ),
                "metadata": {},
            }
            for negative in range(9)
        )
        result.append(
            {
                "example_id": f"q-{index}",
                "query": f"Jak działa rozwiązanie numer {index}?",
                "positives": [
                    {"doc_id": f"p-{index}", "text": _positive_text(index), "metadata": {}}
                ],
                "hard_negatives": negatives,
                "metadata": {"language": "pl", "domain": "test" if index % 2 else "science"},
            }
        )
    return result


def _write(path: Path, records: list[dict[str, Any]]) -> None:
    with JsonlWriter(path) as writer:
        for record in records:
            writer.write(record)


def test_validation_reports_malformed_and_duplicate_documents() -> None:
    malformed = {"example_id": "", "query": "", "positives": [], "hard_negatives": []}
    rules = {issue.rule for issue in validate_record(malformed)}
    assert {"missing_id", "empty_query", "too_few_positives", "too_few_negatives"} <= rules

    empty_passage = _records(2)[0]
    empty_passage["positives"][0]["text"] = ""
    assert "empty_document" in {issue.rule for issue in validate_record(empty_passage)}

    duplicate = _records(1)[0]
    duplicate["hard_negatives"][1]["doc_id"] = duplicate["hard_negatives"][0]["doc_id"]
    duplicate_rules = {issue.rule for issue in validate_record(duplicate)}
    assert "duplicate_doc_id" in duplicate_rules


def test_malformed_json_is_reported_without_losing_following_rows(tmp_path: Path) -> None:
    input_path = tmp_path / "malformed.jsonl"
    valid_path = tmp_path / "valid.jsonl"
    _write(valid_path, [_records(2)[0]])
    input_path.write_text("{not-json}\n" + valid_path.read_text(encoding="utf-8"), encoding="utf-8")
    report = validate_dataset(
        input_path,
        accepted_path=tmp_path / "accepted.jsonl",
        rejected_path=tmp_path / "rejected.jsonl",
        report_path=tmp_path / "report.json",
    )
    assert report["counts"]["rule:malformed_json"] == 1
    assert report["counts"]["accepted"] == 1
    assert report["contains_error_policy_violations"] is True


def test_data_change_changes_validation_fingerprint(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    records = _records(2)
    _write(input_path, records)
    first = validate_dataset(
        input_path,
        accepted_path=tmp_path / "accepted-1.jsonl",
        rejected_path=tmp_path / "rejected-1.jsonl",
        report_path=tmp_path / "report-1.json",
        policy=ValidationPolicy.defaults(),
    )
    records[0]["query"] += " zmiana"
    _write(input_path, records)
    second = validate_dataset(
        input_path,
        accepted_path=tmp_path / "accepted-2.jsonl",
        rejected_path=tmp_path / "rejected-2.jsonl",
        report_path=tmp_path / "report-2.json",
        policy=ValidationPolicy.defaults(),
    )
    assert first["accepted_fingerprint"] != second["accepted_fingerprint"]


def test_duplicate_query_id_is_rejected_globally(tmp_path: Path) -> None:
    record = _records(2)[0]
    input_path = tmp_path / "duplicates.jsonl"
    _write(input_path, [record, record])
    report = validate_dataset(
        input_path,
        accepted_path=tmp_path / "accepted.jsonl",
        rejected_path=tmp_path / "rejected.jsonl",
        report_path=tmp_path / "report.json",
    )
    assert report["counts"]["accepted"] == 1
    assert report["counts"]["rejected"] == 1
    assert report["counts"]["rule:duplicate_example_id"] == 1


def test_tokenizer_audit_config_has_three_pinned_bielik_variants() -> None:
    specs = load_tokenizer_specs(Path("configs/data/bielik_tokenizers.yaml"))
    assert {spec.label for spec in specs} == {"bielik_1_5b", "bielik_4_5b", "bielik_7b"}
    assert all(len(spec.revision) == 40 for spec in specs)


def test_document_index_reports_conflicting_text_for_same_id(tmp_path: Path) -> None:
    records = _records(2)
    records[1]["positives"][0]["doc_id"] = "p-0"
    records[1]["positives"][0]["text"] = "Sprzeczny tekst przypisany do tego samego dokumentu."
    input_path = tmp_path / "conflicts.jsonl"
    _write(input_path, records)
    report = build_document_index(
        input_path,
        sqlite_path=tmp_path / "documents.sqlite",
        documents_path=tmp_path / "documents.parquet",
        report_path=tmp_path / "index.json",
    )
    assert report["conflicting_document_ids"] >= 1
    assert report["conflict_examples"][0]["doc_id"] == "p-0"
    with pytest.raises(ValueError, match="conflicting doc_id"):
        deduplicate_documents(
            tmp_path / "documents.sqlite",
            output_path=tmp_path / "dedup.parquet",
            report_path=tmp_path / "dedup.json",
        )


def test_full_data_pipeline_is_deterministic_and_leakage_safe(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    _write(input_path, _records())
    validation = validate_dataset(
        input_path,
        accepted_path=tmp_path / "accepted.jsonl",
        rejected_path=tmp_path / "rejected.jsonl",
        report_path=tmp_path / "validation.json",
    )
    assert validation["counts"]["accepted"] == 18

    index_report = build_document_index(
        tmp_path / "accepted.jsonl",
        sqlite_path=tmp_path / "documents.sqlite",
        documents_path=tmp_path / "documents.parquet",
        report_path=tmp_path / "index.json",
    )
    assert index_report["conflicting_document_ids"] == 0
    deduplicate_documents(
        tmp_path / "documents.sqlite",
        output_path=tmp_path / "dedup_map.parquet",
        report_path=tmp_path / "dedup.json",
        max_hamming_distance=12,
        bands=16,
    )
    dedup = load_dedup_map(tmp_path / "dedup_map.parquet")
    assert dedup["p-0"] == dedup["p-1"]
    assert dedup["p-2"] == dedup["p-3"]

    config = SplitConfig(train_ratio=0.6, dev_ratio=0.2, test_ratio=0.2, seed=7)
    first = build_splits(
        tmp_path / "accepted.jsonl",
        tmp_path / "dedup_map.parquet",
        output_dir=tmp_path / "split-a",
        config=config,
    )
    second = build_splits(
        tmp_path / "accepted.jsonl",
        tmp_path / "dedup_map.parquet",
        output_dir=tmp_path / "split-b",
        config=config,
    )
    assert first["input_fingerprint"] == second["input_fingerprint"]
    assert first["dedup_fingerprint"] == second["dedup_fingerprint"]
    assignments_a = {
        row["query_id"]: row["split"]
        for row in read_records(tmp_path / "split-a" / "split_assignments.parquet")
    }
    assignments_b = {
        row["query_id"]: row["split"]
        for row in read_records(tmp_path / "split-b" / "split_assignments.parquet")
    }
    assert assignments_a == assignments_b
    assert assignments_a["q-0"] == assignments_a["q-1"]
    assert assignments_a["q-2"] == assignments_a["q-3"]
    assert set(assignments_a.values()) == {"train", "dev", "test"}

    positive_split: dict[str, str] = {}
    split_records: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "dev", "test"):
        rows = list(read_records(tmp_path / "split-a" / f"{split}.parquet"))
        split_records[split] = rows
        for record in rows:
            for positive in record["positives"]:
                canonical = dedup[str(positive["doc_id"])]
                assert canonical not in positive_split or positive_split[canonical] == split
                positive_split[canonical] = split
    assert sum(len(rows) for rows in split_records.values()) == 18
    for split, rows in split_records.items():
        for record in rows:
            for negative in record["hard_negatives"]:
                canonical = dedup[str(negative["doc_id"])]
                assert canonical not in positive_split or positive_split[canonical] == split

    with pytest.raises(FileExistsError, match="frozen split"):
        build_splits(
            tmp_path / "accepted.jsonl",
            tmp_path / "dedup_map.parquet",
            output_dir=tmp_path / "split-a",
            config=config,
        )

    invert_report = invert_doc2query_pairs(
        tmp_path / "split-a" / "train.parquet",
        output_path=tmp_path / "train_doc2query.parquet",
        report_path=tmp_path / "invert.json",
        split="train",
    )
    assert invert_report["output_pairs"] == len(split_records["train"])
    audit = build_data_report(
        [tmp_path / "split-a" / f"{split}.parquet" for split in ("train", "dev", "test")],
        json_path=tmp_path / "data_audit.json",
        html_path=tmp_path / "data_audit.html",
        validation_report=tmp_path / "validation.json",
        dedup_report=tmp_path / "dedup.json",
        split_manifest=tmp_path / "split-a" / "split_manifest.json",
    )
    assert audit["counts"]["records"] == 18
    assert (tmp_path / "data_audit.html").is_file()


def test_inversion_preserves_all_multi_positive_pairs(tmp_path: Path) -> None:
    record = _records(1)[0]
    record["positives"].append(
        {
            "doc_id": "p-extra",
            "text": "Drugi poprawny dokument zawiera inny istotny fakt.",
            "metadata": {},
        }
    )
    input_path = tmp_path / "multi.jsonl"
    _write(input_path, [record])
    report = invert_doc2query_pairs(
        input_path,
        output_path=tmp_path / "pairs.parquet",
        report_path=tmp_path / "pairs.json",
        split="train",
    )
    rows = list(read_records(tmp_path / "pairs.parquet"))
    assert report["output_pairs"] == 2
    assert {row["doc_id"] for row in rows} == {"p-0", "p-extra"}
    assert all(row["positive_count"] == 2 for row in rows)
