from __future__ import annotations

import pytest

from doc2query.evaluation.hn_gate import (
    ARMS,
    assert_dev_only_contract,
    compare_to_reference,
    deduplicate_scoring_pairs,
    positive_aware_keep,
    select_dev_records,
    stable_union,
)


def test_dev_contract_fails_closed() -> None:
    assert_dev_only_contract(
        {"evaluation_subset": "dev_intrinsic_rank10", "final_tests_used": [], "training_runs": []}
    )
    with pytest.raises(ValueError, match="dev"):
        assert_dev_only_contract(
            {"evaluation_subset": "test_embedder", "final_tests_used": [], "training_runs": []}
        )
    with pytest.raises(ValueError, match="training"):
        assert_dev_only_contract(
            {"evaluation_subset": "dev_intrinsic", "final_tests_used": [], "training_runs": ["x"]}
        )


def test_deterministic_dev_selection_and_rank10_guard() -> None:
    records = [{"example_id": str(index), "hard_negatives": list(range(10))} for index in range(5)]
    assert select_dev_records(records, limit=3, seed=42) == select_dev_records(
        list(reversed(records)), limit=3, seed=42
    )


def test_positive_aware_filter_and_union_provenance() -> None:
    assert positive_aware_keep(negative_score=1.0, positive_score=2.0, absolute_threshold=3.0)
    assert not positive_aware_keep(negative_score=2.0, positive_score=2.0, absolute_threshold=3.0)
    union = stable_union(
        [{"doc_id": "a", "rank": 1, "score": 4.0}],
        [
            {"doc_id": "a", "rank": 2, "score": 0.8},
            {"doc_id": "b", "rank": 1, "score": 0.9},
        ],
    )
    assert union[0]["doc_id"] == "a"
    assert set(union[0]["miners"]) == {"bm25", "biencoder"}


def test_comparator_requires_all_arms_and_same_ids() -> None:
    row = {
        "example_id": "q1",
        "pool_mrr": 1.0,
        "pool_ndcg_at_10": 1.0,
    }
    arms = {arm: [row] for arm in ARMS}
    result = compare_to_reference(arms, samples=10, seed=1)
    assert result["hn3_union_positive_filter_minus_hn0_filter"]["pool_mrr"]["difference"] == 0
    with pytest.raises(ValueError, match="exactly"):
        compare_to_reference({"hn0": [row]}, samples=10, seed=1)


def test_scoring_pair_deduplication_is_reversible_and_length_bucketed() -> None:
    pairs = [("q", "long passage"), ("q", "x"), ("q", "long passage")]
    unique, inverse = deduplicate_scoring_pairs(pairs)
    assert len(unique) == 2
    assert [unique[index] for index in inverse] == pairs
    assert len(unique[0][0]) + len(unique[0][1]) <= len(unique[1][0]) + len(unique[1][1])
