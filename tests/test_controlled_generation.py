from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from doc2query.data.focus_labels import assign_focus, focus_bucket
from doc2query.data.style_labels import label_query
from doc2query.evaluation.format import format_metrics
from doc2query.generation.concepts import covered_concepts, extract_concepts
from doc2query.generation.controlled import generate_query_set
from doc2query.generation.deduplicate import deduplicate_queries
from doc2query.generation.multiquery import parse_multiquery_json
from doc2query.generation.selection import (
    Candidate,
    SelectionStrategy,
    SelectionWeights,
    select_candidates,
)
from doc2query.models.templates import render_controlled_prompt
from doc2query.schemas import (
    EvidenceAnnotation,
    EvidenceType,
    FocusMode,
    QueryControl,
    QueryForm,
    QueryIntent,
)


def test_form_and_intent_are_separate_and_rules_abstain() -> None:
    definition = label_query("Czym jest fotosynteza?")
    assert definition.form == QueryForm.FULL_QUESTION
    assert definition.intent == QueryIntent.DEFINITION
    uncertain = label_query("bardzo rozbudowana fraza bez jednoznacznej funkcji wyszukiwawczej")
    assert uncertain.form == QueryForm.UNKNOWN


def test_focus_assignment_is_confident_only_for_unique_sentence() -> None:
    result = assign_focus(
        "Ile kosztuje bilet?",
        "Pociąg jedzie do Gdańska. Bilet kosztuje 40 zł.",
    )
    assert result.sentence_id == 1
    assert result.bucket == "end"
    assert focus_bucket(0, 1) == "middle"


def test_evidence_schema_rejects_unstable_sentence_ids() -> None:
    assert EvidenceAnnotation(
        evidence_sentence_ids=[1, 2],
        evidence_type=EvidenceType.MULTI_SENTENCE,
        evidence_confidence=0.8,
    ).evidence_sentence_ids == [1, 2]
    with pytest.raises(ValidationError):
        EvidenceAnnotation(evidence_sentence_ids=[2, 1])


@pytest.mark.parametrize(
    "mode", [FocusMode.BUCKET, FocusMode.MARKED_SENTENCE, FocusMode.SENTENCE_ID]
)
def test_focus_control_reaches_prompt(mode: FocusMode) -> None:
    if mode == FocusMode.BUCKET:
        control = QueryControl(
            form=QueryForm.KEYWORD_QUERY,
            intent=QueryIntent.FACT_LOOKUP,
            focus_mode=mode,
            focus_bucket="end",
        )
    else:
        control = QueryControl(
            form=QueryForm.KEYWORD_QUERY,
            intent=QueryIntent.FACT_LOOKUP,
            focus_mode=mode,
            focus_sentence_id=1,
        )
    prompt = render_controlled_prompt(
        "Pierwsze zdanie. Drugie zdanie.",
        control,
    )
    assert "Forma: keyword_query" in prompt
    assert "Intencja: fact_lookup" in prompt
    if mode == FocusMode.MARKED_SENTENCE:
        assert "<FOCUS>Drugie zdanie.</FOCUS>" in prompt
    elif mode == FocusMode.SENTENCE_ID:
        assert "[1] Drugie zdanie." in prompt
    else:
        assert "część pasażu: end" in prompt


def test_deduplication_folds_case_diacritics_and_custom_lemmas() -> None:
    assert deduplicate_queries(["ŁÓDŹ atrakcje", "lodz ATRAKCJE", "Warszawa muzea"]) == [
        "ŁÓDŹ atrakcje",
        "Warszawa muzea",
    ]


def test_generation_retries_duplicates_and_stops_at_limit() -> None:
    outputs = iter(["to samo", "to samo", "inne", "to samo", "to samo", "to samo"])
    controls = [
        QueryControl(form=QueryForm.KEYWORD_QUERY, intent=QueryIntent.FACT_LOOKUP),
        QueryControl(form=QueryForm.FULL_QUESTION, intent=QueryIntent.DEFINITION),
        QueryControl(form=QueryForm.KEYWORD_QUERY, intent=QueryIntent.ENTITY_LOOKUP),
    ]
    result = generate_query_set(
        "To jest wystarczająco długi pasaż.",
        controls,
        lambda _prompt, _seed: next(outputs),
        seed=10,
        max_attempts_per_query=3,
    )
    assert [item.text for item in result.queries] == ["to samo", "inne"]
    assert result.exhausted
    assert result.duplicate_outputs == 4
    assert result.attempts == 6


def test_multiquery_schema_accepts_contract_and_rejects_wrong_shape() -> None:
    payload = {
        "queries": [
            {
                "text": "Czym jest fotosynteza?",
                "form": "full_question",
                "intent": "definition",
                "focus_sentence_id": 0,
            }
        ]
    }
    valid = parse_multiquery_json(json.dumps(payload, ensure_ascii=False))
    assert valid.valid and not valid.repaired
    fenced = parse_multiquery_json(f"```json\n{json.dumps(payload)}\n```")
    assert fenced.valid and fenced.repaired
    assert not parse_multiquery_json('["pytanie"]', allow_minor_repair=False).valid
    assert format_metrics(json.dumps(payload), multi_query_json=True)["format_valid"]


def test_concepts_track_uncovered_aspects() -> None:
    concepts = extract_concepts("Bilet kosztuje 40 zł. Pociąg jedzie do Gdańska.")
    assert "40" in {item.text for item in concepts}
    assert "40" in covered_concepts(concepts, ["cena biletu 40"])


def test_coverage_selector_prefers_diverse_focus_and_form() -> None:
    candidates = [
        Candidate("a", "cena biletu", 1.0, "keyword_query", focus_bucket="beginning"),
        Candidate("b", "jaka jest cena biletu?", 0.99, "keyword_query", focus_bucket="beginning"),
        Candidate(
            "c",
            "Dokąd jedzie pociąg?",
            0.8,
            "full_question",
            focus_bucket="end",
            concepts=frozenset({"gdańsk"}),
        ),
    ]
    assert [
        item.candidate_id
        for item in select_candidates(candidates, count=2, strategy=SelectionStrategy.TOP_N)
    ] == ["a", "b"]
    selected = select_candidates(
        candidates,
        count=2,
        strategy=SelectionStrategy.COVERAGE_AWARE,
        weights=SelectionWeights(
            diversity=0.4,
            focus_coverage=0.4,
            style_coverage=0.3,
            concept_coverage=0.3,
        ),
    )
    assert {item.candidate_id for item in selected} == {"a", "c"}
