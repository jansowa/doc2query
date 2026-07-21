"""Backend-independent controlled generation with bounded duplicate retries."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from doc2query.generation.deduplicate import query_key
from doc2query.models.templates import normalize_completion, render_controlled_prompt
from doc2query.schemas import QueryControl

GenerateBackend = Callable[[str, int], str]


@dataclass(frozen=True)
class GeneratedQuery:
    text: str
    control: QueryControl
    seed: int
    attempt: int


@dataclass(frozen=True)
class GenerationBatch:
    queries: tuple[GeneratedQuery, ...]
    attempts: int
    exhausted: bool
    duplicate_outputs: int
    invalid_outputs: int


def generate_one(
    passage: str, control: QueryControl, seed: int, backend: GenerateBackend
) -> GeneratedQuery:
    output = normalize_completion(backend(render_controlled_prompt(passage, control), seed))
    return GeneratedQuery(output, control, seed, 1)


def generate_query_set(
    passage: str,
    controls: Sequence[QueryControl],
    backend: GenerateBackend,
    *,
    seed: int,
    max_attempts_per_query: int = 3,
) -> GenerationBatch:
    if max_attempts_per_query < 1:
        raise ValueError("max_attempts_per_query must be positive")
    accepted: list[GeneratedQuery] = []
    seen: set[str] = set()
    attempts = duplicates = invalid = 0
    for control_index, control in enumerate(controls):
        for attempt in range(1, max_attempts_per_query + 1):
            current_seed = seed + control_index * max_attempts_per_query + attempt - 1
            attempts += 1
            try:
                text = normalize_completion(
                    backend(render_controlled_prompt(passage, control), current_seed)
                )
            except ValueError:
                invalid += 1
                continue
            key = query_key(text)
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            accepted.append(GeneratedQuery(text, control, current_seed, attempt))
            break
    return GenerationBatch(
        tuple(accepted), attempts, len(accepted) < len(controls), duplicates, invalid
    )
