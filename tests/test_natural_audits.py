from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from doc2query.data.style_labels import QueryLabels
from doc2query.evaluation import natural_audits
from doc2query.schemas import QueryForm, QueryIntent
from doc2query.utils.records import read_records


def _record(index: int, *, domain: str = "large", query: str | None = None) -> dict[str, Any]:
    return {
        "example_id": f"q-{index:03d}",
        "query": query or f"jak działa urządzenie {index}?",
        "positives": [
            {
                "doc_id": f"d-{index:03d}",
                "text": f"Urządzenie {index} ma 20 kg i działa następująco.",
                "metadata": {},
            }
        ],
        "hard_negatives": [],
        "metadata": {"source": "fixture", "domain": domain, "split": "dev"},
    }


def _contract(path: Path, manifest: Path) -> Path:
    value = json.loads(
        Path("configs/evaluation/task05_natural_audits_v1.json").read_text(encoding="utf-8")
    )
    value["frozen_manifest"] = str(manifest)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _patch_frozen(monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, Any]]) -> None:
    monkeypatch.setattr(natural_audits, "load_frozen_records", lambda *_args: rows)
    monkeypatch.setattr(natural_audits, "evaluation_fingerprint", lambda *_args: "frozen-fp")


def _materialize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, Any]]
) -> tuple[Path, dict[str, Any]]:
    manifest = tmp_path / "frozen.json"
    manifest.write_text("{}", encoding="utf-8")
    contract = _contract(tmp_path / "contract.json", manifest)
    _patch_frozen(monkeypatch, rows)
    output = tmp_path / "output"
    result = natural_audits.materialize_natural_audits(
        contract, output_dir=output, max_records=len(rows)
    )
    return output, result


def test_contract_rejects_final_subset_and_nonempty_final_tests(tmp_path: Path) -> None:
    source = json.loads(
        Path("configs/evaluation/task05_natural_audits_v1.json").read_text(encoding="utf-8")
    )
    source["frozen_subset"] = "test_intrinsic_rank10"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="only frozen dev"):
        natural_audits.load_contract(path)
    source["frozen_subset"] = "dev_intrinsic_rank10"
    source["final_tests_used"] = ["test_intrinsic_rank10"]
    path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match=r"final_tests_used=\[\]"):
        natural_audits.load_contract(path)


def test_stratified_selection_is_deterministic_and_keeps_small_domain() -> None:
    rows = [
        {
            "example_id": f"q-{index}",
            "doc_id": f"d-{index}",
            "domain": "small" if index < 2 else "large",
            "predicted_form": "unknown" if index == 0 else "full_question",
        }
        for index in range(10)
    ]
    first = natural_audits.stratified_sample(
        rows,
        size=5,
        axes=["domain", "predicted_form"],
        seed=7,
        small_domain_minimum=3,
    )
    second = natural_audits.stratified_sample(
        list(reversed(rows)),
        size=5,
        axes=["domain", "predicted_form"],
        seed=7,
        small_domain_minimum=3,
    )
    assert [row["example_id"] for row in first] == [row["example_id"] for row in second]
    assert any(row["allocation_domain"] == "__small_domains__" for row in first)
    assert any(row["predicted_form"] == "unknown" for row in first)


def test_materialization_separates_blind_forms_and_machine_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [_record(index, domain="tiny" if index == 0 else "large") for index in range(12)]
    output, result = _materialize(tmp_path, monkeypatch, rows)
    assert result["review_status"] == {
        "label_audit": "NOT MEASURED",
        "concept_audit": "NOT MEASURED",
    }
    assert result["final_tests_used"] == []
    with (output / "label_audit_blind.csv").open(encoding="utf-8", newline="") as handle:
        blind = list(csv.DictReader(handle))
    key = list(read_records(output / "label_audit_machine_key.jsonl"))
    assert [row["audit_id"] for row in blind] == [row["audit_id"] for row in key]
    assert "predicted_form" not in blind[0]
    assert "confidence" not in blind[0]
    assert "predicted_form" in key[0]
    with (output / "concept_audit_blind.csv").open(encoding="utf-8", newline="") as handle:
        concept_blind = next(csv.DictReader(handle))
    assert "c01=" in concept_blind["candidate_concepts"]
    assert json.loads((output / "identity.json").read_text())["final_tests_used"] == []


def test_resume_repairs_truncated_tail_and_does_not_relabel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [_record(index) for index in range(6)]
    manifest = tmp_path / "frozen.json"
    manifest.write_text("{}", encoding="utf-8")
    contract = _contract(tmp_path / "contract.json", manifest)
    _patch_frozen(monkeypatch, rows)
    calls = 0

    def labeler(_query: str) -> QueryLabels:
        nonlocal calls
        calls += 1
        return QueryLabels(QueryForm.UNKNOWN, QueryIntent.UNKNOWN, 0.0, 0.0)

    output = tmp_path / "resume"
    natural_audits.materialize_natural_audits(
        contract, output_dir=output, max_records=6, labeler=labeler
    )
    assert calls == 6
    with (output / "calibration.journal.jsonl").open("ab") as handle:
        handle.write(b'{"crash":')
    natural_audits.materialize_natural_audits(
        contract, output_dir=output, max_records=6, labeler=labeler
    )
    assert calls == 6
    assert len(list(read_records(output / "calibration.journal.jsonl"))) == 6
    assert all(
        row["predicted_form"] == "unknown"
        for row in read_records(output / "calibration_rows.jsonl")
    )


def test_changed_identity_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [_record(index) for index in range(5)]
    output, _ = _materialize(tmp_path, monkeypatch, rows)
    contract = tmp_path / "contract.json"
    value = json.loads(contract.read_text(encoding="utf-8"))
    value["seed"] += 1
    contract.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch"):
        natural_audits.materialize_natural_audits(contract, output_dir=output, max_records=5)


def test_incompatible_identity_can_be_archived_recoverably(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [_record(index) for index in range(4)]
    output, _ = _materialize(tmp_path, monkeypatch, rows)
    contract = tmp_path / "contract.json"
    value = json.loads(contract.read_text(encoding="utf-8"))
    value["seed"] += 1
    contract.write_text(json.dumps(value), encoding="utf-8")
    result = natural_audits.materialize_natural_audits(
        contract,
        output_dir=output,
        max_records=4,
        archive_incompatible=True,
    )
    archives = list((output / "interrupted").glob("*/archive_manifest.json"))
    assert len(archives) == 1
    assert result["identity"]["seed"] == value["seed"]


def _write_ratings(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_label_aggregator_is_fail_closed_then_computes_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, _ = _materialize(tmp_path, monkeypatch, [_record(0), _record(1, query="hasło")])
    key = list(read_records(output / "label_audit_machine_key.jsonl"))
    incomplete_ratings = tmp_path / "one.csv"
    _write_ratings(
        incomplete_ratings,
        [
            {
                "audit_id": key[0]["audit_id"],
                "reviewer_id": "r1",
                "gold_form": key[0]["predicted_form"],
                "gold_intent": key[0]["predicted_intent"],
                "intent_adequate": "yes",
            }
        ],
    )
    report = natural_audits.aggregate_label_audit(
        output / "label_audit_machine_key.jsonl",
        [incomplete_ratings],
        adjudication=None,
        output_dir=tmp_path / "agg-incomplete",
        required_reviewers=2,
    )
    assert report["status"] == "incomplete"
    ratings = tmp_path / "two.csv"
    complete_rows: list[dict[str, str]] = []
    for item in key:
        for reviewer in ("r1", "r2"):
            complete_rows.append(
                {
                    "audit_id": item["audit_id"],
                    "reviewer_id": reviewer,
                    "gold_form": item["predicted_form"],
                    "gold_intent": item["predicted_intent"],
                    "intent_adequate": "yes",
                }
            )
    _write_ratings(ratings, complete_rows)
    report = natural_audits.aggregate_label_audit(
        output / "label_audit_machine_key.jsonl",
        [ratings],
        adjudication=None,
        output_dir=tmp_path / "agg-complete",
    )
    assert report["status"] == "complete"
    assert report["agreement"]["form"]["method"] == "cohen_kappa"
    form = key[0]["predicted_form"]
    assert report["form"]["confusion_matrix"][form][form] >= 1


def test_single_reviewer_completes_without_claiming_agreement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, _ = _materialize(tmp_path, monkeypatch, [_record(0)])
    item = next(read_records(output / "label_audit_machine_key.jsonl"))
    ratings = tmp_path / "single.csv"
    _write_ratings(
        ratings,
        [
            {
                "audit_id": item["audit_id"],
                "reviewer_id": "owner",
                "gold_form": item["predicted_form"],
                "gold_intent": item["predicted_intent"],
                "intent_adequate": "yes",
            }
        ],
    )
    report = natural_audits.aggregate_label_audit(
        output / "label_audit_machine_key.jsonl",
        [ratings],
        adjudication=None,
        output_dir=tmp_path / "single-agg",
    )
    assert report["status"] == "complete"
    assert report["agreement"]["form"] == {
        "method": "not_measured_fewer_than_two_reviewers",
        "value": None,
        "reviewer_count": 1,
    }


def test_concept_aggregator_counts_audit_error_types(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, _ = _materialize(tmp_path, monkeypatch, [_record(0)])
    proposal = next(read_records(output / "concept_audit_machine_proposals.jsonl"))
    ratings = tmp_path / "concept-ratings.csv"
    rows = []
    for reviewer in ("r1", "r2"):
        rows.append(
            {
                "audit_id": proposal["audit_id"],
                "reviewer_id": reviewer,
                "correct_concept_ids": "c01",
                "spurious_concept_ids": "c02",
                "missing_important_concepts": "mechanizm działania",
                "numbers_units_correct": "yes",
                "over_fragmented": "no",
                "duplicate_concepts": "yes",
                "useful_for_coverage": "yes",
            }
        )
    _write_ratings(ratings, rows)
    report = natural_audits.aggregate_concept_audit(
        output / "concept_audit_machine_proposals.jsonl",
        [ratings],
        adjudication=None,
        output_dir=tmp_path / "concept-agg",
    )
    assert report["status"] == "complete"
    assert report["concept_error_totals"] == {
        "correct": 2,
        "spurious": 2,
        "missing": 2,
        "duplicates": 2,
    }
    assert report["ratings"]["numbers_units_correct"] == {"yes": 2}
