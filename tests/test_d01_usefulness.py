from __future__ import annotations

from pathlib import Path

import numpy as np
from conftest import require_local_artifacts

from doc2query.evaluation.d01_usefulness import (
    D01UsefulnessContract,
    _Candidate,
    _difficulty_report,
    _select_groups,
)


def _candidate(
    identity: str,
    *,
    role: str,
    margin: float,
    corpus_hit: float = 1.0,
    copy_risk: bool = False,
    lemmas: frozenset[str] = frozenset(),
) -> _Candidate:
    natural_margin = 2.0
    return _Candidate(
        identity=identity,
        evaluation_id=identity,
        group_id="g1",
        example_id="q1",
        doc_id="d1",
        role=role,
        experiment_id=role,
        text=identity,
        requested_form="full_question" if role == "controlled" else "uncontrolled",
        requested_intent="fact_lookup" if role == "controlled" else "uncontrolled",
        natural_margin=natural_margin,
        margin_excess=margin - natural_margin,
        copy_risk=copy_risk,
        content_lemmas=lemmas,
        metrics={
            "pool_recall_at_1": 1.0,
            "corpus_round_trip_at_20": corpus_hit,
            "sentence_level_source_hit": 1.0,
            "format_valid": 1.0,
            "shadow_pool_recall_at_1": 1.0,
            "pool_margin": margin,
            "shadow_pool_margin": margin,
            "copy_density": 0.2,
            "judge_rank_disagreement": 0.0,
        },
        corpus_effective_candidate_count=10,
        corpus_candidate_count=1000,
    )


def test_usefulness_contract_is_retrospective_and_shadow_reserved() -> None:
    require_local_artifacts()
    contract = D01UsefulnessContract.load(
        Path("configs/evaluation/d01b_usefulness_hybrid_v1.yaml")
    )
    assert contract.payload["evaluation"]["promotion_eligible"] is False
    assert contract.payload["evaluation"]["shadow_reserved_from_selection"] is True


def test_safe_anchor_selector_uses_grounded_diverse_control_but_rejects_regression() -> None:
    candidates = [
        _candidate("b1", role="baseline", margin=5.0, lemmas=frozenset({"wspolny", "a"})),
        _candidate("b2", role="baseline", margin=5.0, lemmas=frozenset({"wspolny", "b"})),
        _candidate("b3", role="baseline", margin=5.0, lemmas=frozenset({"wspolny", "c"})),
        _candidate("b4", role="baseline", margin=5.0, lemmas=frozenset({"wspolny", "d"})),
        _candidate("c1", role="controlled", margin=2.0, lemmas=frozenset({"nowy", "fakt"})),
        _candidate("c2", role="controlled", margin=2.0, corpus_hit=0.0),
        _candidate("c3", role="controlled", margin=2.0, copy_risk=True),
        _candidate("c4", role="controlled", margin=-5.0),
    ]
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )
    selected, _objectives, changed = _select_groups(
        candidates,
        embeddings,
        margin_scale=2.0,
        weights={
            "natural_margin_alignment": 0.35,
            "semantic_diversity": 0.30,
            "lexical_diversity": 0.10,
            "corpus_specificity": 0.15,
            "low_copy_density": 0.10,
        },
    )
    identities = {item.identity for item in selected["g1"]}
    assert "c1" in identities
    assert "c2" not in identities
    assert "c3" not in identities
    assert changed == 1


def test_difficulty_report_does_not_equate_larger_margin_with_more_utility() -> None:
    candidates = [
        _candidate("b1", role="baseline", margin=5.0),
        _candidate("c1", role="controlled", margin=2.0),
    ]
    report = _difficulty_report(candidates)
    assert report["baseline/all"]["margin_excess"]["mean"] == 3.0
    assert report["baseline/all"]["easier_than_natural_rate"] == 1.0
    assert report["controlled/all"]["margin_excess"]["mean"] == 0.0
