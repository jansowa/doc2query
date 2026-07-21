"""Strict multi-query JSON contract with deliberately limited syntax repair."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from pydantic import ValidationError

from doc2query.schemas import MultiQueryCompletion

_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class ParsedMultiQuery:
    completion: MultiQueryCompletion | None
    valid: bool
    repaired: bool
    error: str | None


def parse_multiquery_json(text: str, *, allow_minor_repair: bool = True) -> ParsedMultiQuery:
    source = text.strip()
    repaired = False
    if allow_minor_repair and (match := _FENCE.match(source)):
        source, repaired = match.group(1).strip(), True
    if allow_minor_repair and source.endswith(",}"):
        source, repaired = source[:-2] + "}", True
    try:
        payload = json.loads(source)
        completion = MultiQueryCompletion.model_validate(payload)
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        return ParsedMultiQuery(None, False, repaired, str(exc))
    return ParsedMultiQuery(completion, True, repaired, None)


def render_multiquery_prompt(passage: str, *, count: int) -> str:
    if count < 1:
        raise ValueError("query count must be positive")
    return (
        f"Wygeneruj {count} różnych polskich zapytań na podstawie pasażu. "
        "Zwróć wyłącznie obiekt JSON zgodny ze schematem: "
        '{"queries":[{"text":"...","form":"full_question|keyword_query",'
        '"intent":"fact_lookup|definition|entity_lookup|procedure|comparison",'
        '"focus_sentence_id":0}]}. Nie dodawaj Markdown.\n\n'
        f"Pasaż:\n{passage.strip()}\n"
    )
