"""Stable prompt templates for passage-to-query SFT baselines."""

from typing import Literal

BaselineName = Literal["b0", "b1"]

_B0 = """Wygeneruj jedno polskie zapytanie wyszukiwawcze, na które można odpowiedzieć \
na podstawie pasażu.

Pasaż:
{passage}

Zapytanie:
"""

_B1 = """Wygeneruj jedno polskie zapytanie wyszukiwawcze, na które można odpowiedzieć \
wyłącznie na podstawie podanego pasażu.
Nie kopiuj długich fragmentów pasażu. Zachowaj konieczne nazwy własne, liczby i terminy.

Pasaż:
{passage}

Zapytanie:
"""


def render_prompt(passage: str, baseline: BaselineName = "b1") -> str:
    """Render a B0/B1 prompt while keeping completion separate."""
    cleaned = passage.strip()
    if not cleaned:
        raise ValueError("passage cannot be empty")
    template = _B0 if baseline == "b0" else _B1
    return template.format(passage=cleaned)


def normalize_completion(query: str) -> str:
    """Return exactly one query without adding chat markup or commentary."""
    if "\n" in query or "\r" in query:
        raise ValueError("query completion must be a single line")
    completion = " ".join(query.strip().split())
    if not completion:
        raise ValueError("query completion cannot be empty")
    return completion
