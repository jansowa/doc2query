"""Group-level inference and retrieval statistics."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from doc2query.evaluation.retrieval import CANDIDATE_POOL_RANKING
from doc2query.reranker.base import PairScorer


@dataclass(frozen=True)
class GroupScore:
    example_id: str
    judge: str
    positive_score: float
    hardest_negative_score: float
    pool_margin: float
    pool_rank: int
    pool_candidate_count: int
    pool_recall_at_1: float
    pool_recall_at_5: float | None
    pool_mrr: float
    pool_ndcg_at_10: float
    pool_near_zero_margin: bool
    pool_all_scores_close: bool
    document_scores: tuple[float, ...]
    query_id: str = ""
    positive_doc_id: str = ""
    negative_doc_ids: tuple[str, ...] = ()
    positive_index: int = 0
    positive_is_synthetic: bool = False
    source_en_positive_score: float | None = None
    source_en_negative_scores: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["protocol"] = CANDIDATE_POOL_RANKING
        result["document_scores"] = list(self.document_scores)
        result["negative_doc_ids"] = list(self.negative_doc_ids)
        result["source_en_negative_scores"] = list(self.source_en_negative_scores)
        return result


def score_group(
    scorer: PairScorer,
    *,
    example_id: str,
    query: str,
    positive: str,
    negatives: list[str],
    near_zero_threshold: float = 0.1,
    query_id: str = "",
    positive_doc_id: str = "",
    negative_doc_ids: tuple[str, ...] = (),
    positive_index: int = 0,
    positive_is_synthetic: bool = False,
    source_en_positive_score: float | None = None,
    source_en_negative_scores: tuple[float, ...] = (),
) -> GroupScore:
    if len(negatives) < 1:
        raise ValueError("at least one hard negative is required")
    scores = scorer.score_pairs([(query, positive), *((query, text) for text in negatives)])
    if len(scores) != len(negatives) + 1 or not all(math.isfinite(x) for x in scores):
        raise ValueError("scorer returned invalid scores")
    positive_score = scores[0]
    hardest = max(scores[1:])
    rank = 1 + sum(score >= positive_score for score in scores[1:])
    spread = max(scores) - min(scores)
    return GroupScore(
        example_id=example_id,
        judge=scorer.name,
        positive_score=positive_score,
        hardest_negative_score=hardest,
        pool_margin=positive_score - hardest,
        pool_rank=rank,
        pool_candidate_count=len(scores),
        pool_recall_at_1=float(rank <= 1),
        pool_recall_at_5=float(rank <= 5) if len(scores) >= 5 else None,
        pool_mrr=1.0 / rank,
        pool_ndcg_at_10=(1.0 / math.log2(rank + 1)) if rank <= 10 else 0.0,
        pool_near_zero_margin=positive_score - hardest <= near_zero_threshold,
        pool_all_scores_close=spread <= near_zero_threshold,
        document_scores=tuple(scores),
        query_id=query_id or example_id,
        positive_doc_id=positive_doc_id,
        negative_doc_ids=negative_doc_ids,
        positive_index=positive_index,
        positive_is_synthetic=positive_is_synthetic,
        source_en_positive_score=source_en_positive_score,
        source_en_negative_scores=source_en_negative_scores,
    )
