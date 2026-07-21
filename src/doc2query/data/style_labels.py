"""High-precision, abstaining form and retrieval-intent labelling rules."""

from __future__ import annotations

import re
from dataclasses import dataclass

from doc2query.schemas import QueryForm, QueryIntent

_WORDS = re.compile(r"[\wąćęłńóśźż-]+", re.IGNORECASE)


@dataclass(frozen=True)
class QueryLabels:
    form: QueryForm
    intent: QueryIntent
    form_confidence: float
    intent_confidence: float
    source: str = "rules_pl_v1"


def label_query(query: str) -> QueryLabels:
    """Label only high-precision patterns and explicitly abstain otherwise."""
    text = " ".join(query.strip().casefold().split())
    words = _WORDS.findall(text)
    if not words:
        return QueryLabels(QueryForm.UNKNOWN, QueryIntent.UNKNOWN, 0.0, 0.0)
    question_openers = (
        "jak ",
        "jaki ",
        "jaka ",
        "jakie ",
        "kto ",
        "co ",
        "czym ",
        "gdzie ",
        "kiedy ",
        "dlaczego ",
        "ile ",
        "czy ",
        "który ",
        "która ",
        "które ",
    )
    is_question = text.endswith("?") or text.startswith(question_openers)
    if is_question:
        form, form_confidence = QueryForm.FULL_QUESTION, 0.95
    elif len(words) <= 6:
        form, form_confidence = QueryForm.KEYWORD_QUERY, 0.9
    else:
        form, form_confidence = QueryForm.UNKNOWN, 0.45

    intent, intent_confidence = QueryIntent.UNKNOWN, 0.0
    if text.startswith(("co to ", "czym jest ", "co oznacza ", "definicja ")):
        intent, intent_confidence = QueryIntent.DEFINITION, 0.98
    elif text.startswith(("jak ", "w jaki sposób ", "instrukcja ")):
        intent, intent_confidence = QueryIntent.PROCEDURE, 0.94
    elif any(marker in text for marker in ("różnica między", "porównanie", "czym różni")):
        intent, intent_confidence = QueryIntent.COMPARISON, 0.97
    elif text.startswith(("kto ", "gdzie ", "nazwa ", "adres ")):
        intent, intent_confidence = QueryIntent.ENTITY_LOOKUP, 0.9
    elif is_question or len(words) <= 6:
        intent, intent_confidence = QueryIntent.FACT_LOOKUP, 0.7
    return QueryLabels(form, intent, form_confidence, intent_confidence)


def intent_applicable(intent: QueryIntent, passage: str) -> bool | None:
    """Conservative passage-only applicability heuristic; None means abstention."""
    text = passage.casefold()
    if intent == QueryIntent.COMPARISON:
        comparison_markers = ("natomiast", "w porównaniu", "różni")
        return True if any(token in text for token in comparison_markers) else None
    if intent == QueryIntent.PROCEDURE:
        procedure_markers = ("należy", "krok", "najpierw", "następnie")
        return True if any(token in text for token in procedure_markers) else None
    if intent == QueryIntent.DEFINITION:
        return True if any(token in text for token in (" oznacza ", " to ", "jest to")) else None
    if intent in {QueryIntent.FACT_LOOKUP, QueryIntent.ENTITY_LOOKUP}:
        return True
    return None
