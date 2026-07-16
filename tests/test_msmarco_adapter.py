import pytest

from doc2query.data.msmarco_pl import (
    DATASET_REVISION,
    NoPositiveAfterSourceScoreFilterError,
    adapt_msmarco_pl_record,
)
from doc2query.data.validate import validate_record
from doc2query.reranker.commands import build_scoring_groups


def _raw_record() -> dict[str, object]:
    return {
        "query_id": 334621,
        "query": "jak obliczyć godziny chłodzenia?",
        "pos": ["Pierwszy pasaż.", "Temperatura wynosi 45ËF."],
        "pos_id": ["p1", "p2"],
        "pos_scores": [27, 24.5],
        "pos_is_synthetic": [False, True],
        "neg": [f"Negatywny pasaż {index}." for index in range(10)],
        "neg_id": [f"n{index}" for index in range(10)],
        "neg_scores": [17.5 - index / 10 for index in range(10)],
        "difference_between_max_scores": 9.5,
    }


def test_adapter_preserves_parallel_provenance() -> None:
    adapted = adapt_msmarco_pl_record(_raw_record())
    assert adapted["example_id"] == "334621"
    assert len(adapted["positives"]) == 2
    assert len(adapted["hard_negatives"]) == 10
    assert adapted["positives"][1]["metadata"] == {
        "source_en_score": 24.5,
        "source_score_language": "en",
        "is_synthetic_positive": True,
        "text_quality_flags": ["possible_mojibake"],
    }
    assert adapted["hard_negatives"][0]["doc_id"] == "n0"
    assert adapted["metadata"]["source_revision"] == DATASET_REVISION
    assert adapted["metadata"]["synthetic_positive_count"] == 1

    groups = build_scoring_groups(adapted)
    assert groups[1].positive_doc_id == "p2"
    assert groups[1].positive_is_synthetic is True
    assert groups[1].source_en_positive_score == 24.5
    assert groups[1].negative_doc_ids == tuple(f"n{index}" for index in range(10))
    assert groups[1].source_en_negative_scores[0] == 17.5


def test_adapter_rejects_unaligned_parallel_fields() -> None:
    record = _raw_record()
    record["pos_scores"] = [27]
    with pytest.raises(ValueError, match="unaligned parallel fields"):
        adapt_msmarco_pl_record(record)


def test_adapter_requires_ten_negatives() -> None:
    record = _raw_record()
    for field in ("neg", "neg_id", "neg_scores"):
        value = record[field]
        assert isinstance(value, list)
        record[field] = value[:9]
    with pytest.raises(ValueError, match="ten hard negatives"):
        adapt_msmarco_pl_record(record)


def test_adapter_preserves_document_id_conflicts_for_validation() -> None:
    record = _raw_record()
    positive_ids = record["pos_id"]
    negative_ids = record["neg_id"]
    assert isinstance(positive_ids, list)
    assert isinstance(negative_ids, list)
    positive_ids[1] = "p1"
    negative_ids[0] = "p1"

    adapted = adapt_msmarco_pl_record(record)

    assert adapted["metadata"]["duplicate_positive_doc_ids"] == ["p1"]
    assert adapted["metadata"]["positive_negative_overlap_doc_ids"] == ["p1"]
    rules = {issue.rule for issue in validate_record(adapted)}
    assert {"duplicate_doc_id", "positive_negative_overlap"} <= rules


def test_adapter_filters_low_source_score_positives() -> None:
    record = _raw_record()
    record["pos_scores"] = [23.49, 23.50]

    adapted = adapt_msmarco_pl_record(record, min_positive_score=23.50)

    assert [item["doc_id"] for item in adapted["positives"]] == ["p2"]
    assert adapted["metadata"]["source_en_positive_score_filter"] == {
        "minimum_inclusive": 23.50,
        "removed_count": 1,
        "removed_doc_ids": ["p1"],
    }


def test_adapter_reports_records_emptied_by_source_score_filter() -> None:
    record = _raw_record()
    record["pos_scores"] = [23.49, 20.0]

    with pytest.raises(NoPositiveAfterSourceScoreFilterError) as error:
        adapt_msmarco_pl_record(record, min_positive_score=23.50)

    assert error.value.removed_count == 2
