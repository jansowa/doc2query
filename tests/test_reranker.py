from collections.abc import Sequence

import pytest

from doc2query.reranker.base import FrozenRerankerConfig
from doc2query.reranker.benchmark import aggregate, aggregate_query_macro, disagreement
from doc2query.reranker.focus import assign_focus
from doc2query.reranker.infer import score_group


class TokenScorer:
    name = "token-test-judge"

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        return [
            float(len(set(query.lower().split()) & set(text.lower().split())))
            for query, text in pairs
        ]


def test_config_requires_full_sha_and_prohibits_remote_code() -> None:
    with pytest.raises(ValueError, match="40-character"):
        FrozenRerankerConfig("model", "main", "license")
    with pytest.raises(ValueError, match="remote"):
        FrozenRerankerConfig("model", "a" * 40, "license", trust_remote_code=True)


def test_group_retrieval_metrics() -> None:
    result = score_group(
        TokenScorer(),
        example_id="x",
        query="pompa ciepła",
        positive="pompa ciepła ogrzewa dom",
        negatives=["samochód elektryczny", "pompa wodna"],
    )
    assert result.positive_rank == 1
    assert result.recall_at_1 == 1
    assert result.ndcg_at_10 == 1
    assert aggregate([result])["mrr"] == 1


def test_query_macro_does_not_overweight_multi_positive_queries() -> None:
    successful = score_group(
        TokenScorer(),
        example_id="q1::p1",
        query_id="q1",
        query="trafne",
        positive="trafne",
        negatives=["inne"],
    )
    second_positive = score_group(
        TokenScorer(),
        example_id="q1::p2",
        query_id="q1",
        query="trafne",
        positive="trafne także",
        negatives=["inne"],
    )
    failed = score_group(
        TokenScorer(),
        example_id="q2::p3",
        query_id="q2",
        query="trafne",
        positive="inne",
        negatives=["trafne"],
    )
    rows = [successful, second_positive, failed]
    assert aggregate(rows)["recall_at_1"] == pytest.approx(2 / 3)
    macro = aggregate_query_macro(rows)
    assert macro["recall_at_1"] == pytest.approx(0.5)
    assert macro["query_count"] == 2
    assert macro["pair_count"] == 3


def test_score_artifact_keeps_document_provenance() -> None:
    result = score_group(
        TokenScorer(),
        example_id="q::p",
        query_id="q",
        query="pompa",
        positive="pompa ciepła",
        negatives=["samochód"],
        positive_doc_id="p",
        negative_doc_ids=("n",),
        positive_index=1,
        positive_is_synthetic=True,
        source_en_positive_score=25.0,
        source_en_negative_scores=(17.0,),
    ).to_dict()
    assert result["positive_doc_id"] == "p"
    assert result["negative_doc_ids"] == ["n"]
    assert result["positive_is_synthetic"] is True
    assert result["source_en_positive_score"] == 25.0


def test_disagreement_is_explicit() -> None:
    primary = score_group(
        TokenScorer(), example_id="x", query="a b", positive="a b", negatives=["a"]
    )

    class ReverseScorer(TokenScorer):
        name = "shadow"

        def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
            return [-value for value in super().score_pairs(pairs)]

    shadow = score_group(
        ReverseScorer(), example_id="x", query="a b", positive="a b", negatives=["a"]
    )
    report = disagreement([primary], [shadow])
    assert report["positive_winner_disagreement_rate"] == 1
    assert report["disagreed_example_ids"] == ["x"]


def test_focus_marks_low_margin_ambiguous() -> None:
    result = assign_focus(
        TokenScorer(), "kot", "Kot śpi. Kot je. Pies biegnie.", ambiguity_margin=0.1
    )
    assert result.focus_sentence_id == 0
    assert result.focus_bucket == "beginning"
    assert result.focus_is_ambiguous
