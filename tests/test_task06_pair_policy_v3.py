from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from doc2query.preferences.pair_policy_v3 import (
    REQUIRED_UNANIMOUS_VOTES,
    BundleCandidate,
    BundleGroup,
    comparison_item,
    confirmation_votes,
    load_bundle,
    run_group_tournament,
    run_tournaments,
)
from doc2query.preferences.pair_selector_v3 import JudgeEndpoint


def _candidate(index: int, *, chosen_ok: bool = True, rejected_ok: bool = True) -> BundleCandidate:
    return BundleCandidate(
        candidate_id=f"c{index}",
        candidate_index=index,
        query=f"zapytanie numer {index}",
        admissible_as_chosen=chosen_ok,
        admissible_as_rejected=rejected_ok,
    )


def _group(count: int = 4, **kwargs: Any) -> BundleGroup:
    return BundleGroup(
        group_id="g1",
        cohort_id="same_prompt_expansion_v1",
        passage="Pasaż o orangutanach na Borneo i Sumatrze.",
        candidates=tuple(_candidate(index, **kwargs) for index in range(count)),
    )


def _ordering_comparator(
    order: list[str],
) -> Callable[[BundleGroup, BundleCandidate, BundleCandidate], str]:
    """Sędzia zgodny z zadanym rankingiem: wcześniejszy w liście jest lepszy."""
    rank = {name: position for position, name in enumerate(order)}

    def compare(group: BundleGroup, left: BundleCandidate, right: BundleCandidate) -> str:
        return "A" if rank[left.candidate_id] < rank[right.candidate_id] else "B"

    return compare


def test_tournament_finds_best_worst_and_second_worst() -> None:
    group = _group(4)
    outcome = run_group_tournament(group, _ordering_comparator(["c2", "c0", "c3", "c1"]))
    assert outcome["best"] == "c2"
    assert outcome["worst"] == "c1"
    assert outcome["second_worst"] == "c3"
    # k-1 + k-2 + k-3 porównań przy k=4
    assert outcome["comparisons"] == 3 + 2 + 1


def test_tournament_uses_only_admissible_candidates() -> None:
    group = BundleGroup(
        group_id="g1",
        cohort_id="c",
        passage="p",
        candidates=(
            _candidate(0),
            _candidate(1, chosen_ok=False, rejected_ok=False),
            _candidate(2),
        ),
    )
    outcome = run_group_tournament(group, _ordering_comparator(["c0", "c2"]))
    assert outcome["rejected_pool_size"] == 2
    assert {outcome["best"], outcome["worst"]} == {"c0", "c2"}
    assert outcome["second_worst"] is None


def test_group_without_any_clean_candidate_is_skipped_before_any_call() -> None:
    """14,8% grup nie ma czystego kandydata; turniej na nich byłby wydatkiem bez wyniku."""
    group = BundleGroup(
        group_id="g1", cohort_id="c", passage="p",
        candidates=tuple(_candidate(i, chosen_ok=False) for i in range(3)),
    )
    calls = {"n": 0}

    def counting(group: BundleGroup, left: BundleCandidate, right: BundleCandidate) -> str:
        calls["n"] += 1
        return "A"

    outcome = run_group_tournament(group, counting)
    assert outcome["paired"] is False
    assert outcome["reason"] == "no_admissible_chosen"
    assert calls["n"] == 0, "nie wolno wykonać ani jednego porównania"


def test_leader_is_ranked_only_among_clean_candidates() -> None:
    """Lider musi spełniać pełny kontrakt czystości, nie tylko format."""
    group = BundleGroup(
        group_id="g1", cohort_id="c", passage="p",
        candidates=(
            _candidate(0, chosen_ok=False),  # najlepszy w rankingu, ale nieczysty
            _candidate(1),
            _candidate(2),
        ),
    )
    outcome = run_group_tournament(group, _ordering_comparator(["c0", "c1", "c2"]))
    assert outcome["best"] == "c1", "nieczysty kandydat nie może zostać chosen"
    assert outcome["chosen_pool_size"] == 2
    assert outcome["rejected_pool_size"] == 3
    assert outcome["worst"] in {"c0", "c2"}


def test_group_with_one_admissible_candidate_is_not_paired() -> None:
    group = BundleGroup(
        group_id="g1", cohort_id="c", passage="p",
        candidates=(_candidate(0), _candidate(1, chosen_ok=False, rejected_ok=False)),
    )
    outcome = run_group_tournament(group, _ordering_comparator(["c0"]))
    assert outcome["paired"] is False
    assert outcome["reason"] == "too_few_admissible"


def test_tie_does_not_move_the_leader() -> None:
    """Przy remisie lider zostaje: inaczej wynik zależałby od kolejności losowo."""
    group = _group(3)

    def always_tie(group: BundleGroup, left: BundleCandidate, right: BundleCandidate) -> str:
        return "tie"

    outcome = run_group_tournament(group, always_tie)
    assert outcome["best"] == "c0"
    assert outcome["worst"] == "c1"


def test_confirmation_requires_six_votes_from_six() -> None:
    unanimous = {rubric: {"ab": "A", "ba": "A"} for rubric in
                 ("R1_grounding", "R2_retrieval_usefulness", "R3_holistic")}
    result = confirmation_votes(unanimous)
    assert result["votes_for_first"] == REQUIRED_UNANIMOUS_VOTES
    assert result["unanimous"] is True
    assert result["position_flips"] == 0

    one_flip = dict(unanimous)
    one_flip["R2_retrieval_usefulness"] = {"ab": "A", "ba": "B"}
    partial = confirmation_votes(one_flip)
    assert partial["votes_for_first"] == 4
    assert partial["unanimous"] is False
    assert partial["position_flips"] == 1

    reversed_votes = {rubric: {"ab": "B", "ba": "B"} for rubric in unanimous}
    against = confirmation_votes(reversed_votes)
    assert against["unanimous"] is False
    assert against["unanimous_against"] is True


def test_confirmation_needs_every_rubric() -> None:
    incomplete = {"R3_holistic": {"ab": "A", "ba": "A"}}
    assert confirmation_votes(incomplete)["complete"] is False


def test_comparison_item_is_blind() -> None:
    group = _group(2)
    item = comparison_item(group, group.candidates[0], group.candidates[1])
    assert item.item_id == "g1|c0|c1"
    assert item.query_first == group.candidates[0].query
    for leak in ("chosen", "rejected", "score", "margin"):
        assert leak not in json.dumps(item.metadata)


# --- pełny przebieg z fałszywym serwerem ----------------------------------------


def _server(better_first: bool = True) -> Any:
    """Serwer zawsze wskazujący pierwsze zapytanie w promptcie."""
    calls = {"n": 0}

    def call(payload: Mapping[str, Any]) -> dict[str, Any]:
        calls["n"] += 1
        verdict = "A" if better_first else "B"
        return {
            "choices": [
                {"message": {"content": json.dumps({"better": verdict, "confidence": 0.9})}}
            ]
        }

    call.calls = calls  # type: ignore[attr-defined]
    return call


def _bundle(tmp_path: Path, groups: int = 2) -> Path:
    path = tmp_path / "tournament_bundle.jsonl"
    rows = []
    for index in range(groups):
        rows.append(
            json.dumps(
                {
                    "group_id": f"g{index}",
                    "cohort_id": "same_prompt_expansion_v1",
                    "passage": f"pasaż {index}",
                    "candidates": [
                        {
                            "candidate_id": f"g{index}c{slot}",
                            "candidate_index": slot,
                            "query": f"zapytanie {index}-{slot}",
                            "admissible_as_chosen": True,
                            "admissible_as_rejected": True,
                        }
                        for slot in range(3)
                    ],
                }
            )
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_tournament_run_is_resumable_through_the_comparison_cache(tmp_path: Path) -> None:
    groups = load_bundle(_bundle(tmp_path, groups=2))
    endpoint = JudgeEndpoint(base_url="http://x/v1", api_key="k", model="m")
    server = _server()
    first = run_tournaments(
        groups=groups,
        endpoint=endpoint,
        journal_path=tmp_path / "j.jsonl",
        output_path=tmp_path / "outcomes.jsonl",
        transport=server,
        concurrency=2,
        progress_every=0,
    )
    assert first["paired_groups"] == 2
    assert first["failures"] == 0
    assert first["calls_made"] > 0
    second = run_tournaments(
        groups=groups,
        endpoint=endpoint,
        journal_path=tmp_path / "j.jsonl",
        output_path=tmp_path / "outcomes.jsonl",
        transport=_server(),
        concurrency=2,
        progress_every=0,
    )
    assert second["calls_made"] == 0, "wznowienie nie może wykonać ani jednego wywołania"
    assert second["paired_groups"] == 2


def test_position_consistent_server_yields_unanimous_pairs(tmp_path: Path) -> None:
    """Serwer wskazujący zawsze pierwszą pozycję daje position_flip, nie jednomyślność."""
    groups = load_bundle(_bundle(tmp_path, groups=1))
    summary = run_tournaments(
        groups=groups,
        endpoint=JudgeEndpoint(base_url="http://x/v1", api_key="k", model="m"),
        journal_path=tmp_path / "j.jsonl",
        output_path=tmp_path / "outcomes.jsonl",
        transport=_server(),
        concurrency=1,
        progress_every=0,
    )
    assert summary["unanimous_bottom_pairs"] == 0
    outcome = json.loads((tmp_path / "outcomes.jsonl").read_text().splitlines()[0])
    bottom = outcome["confirmations"]["bottom"]
    assert bottom["position_flips"] == 3
    assert bottom["unanimous"] is False


def test_clean_candidate_must_be_format_admissible() -> None:
    with pytest.raises(ValueError, match="dopuszczalny formatem"):
        BundleCandidate(
            candidate_id="c9",
            candidate_index=9,
            query="q",
            admissible_as_chosen=True,
            admissible_as_rejected=False,
        )
