"""Stable prompt templates for passage-to-query SFT baselines."""

from typing import Literal

from doc2query.schemas import FocusMode, QueryControl

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


def _split_sentences(passage: str) -> list[str]:
    from doc2query.data.focus_labels import split_sentences

    return split_sentences(passage)


def render_controlled_prompt(passage: str, control: QueryControl) -> str:
    """Render a single-query prompt with explicit, non-conflated controls."""
    cleaned = passage.strip()
    if not cleaned:
        raise ValueError("passage cannot be empty")
    sentences = _split_sentences(cleaned)
    focus_instruction = "dowolny fragment pasażu"
    rendered_passage = cleaned
    if control.focus_mode == FocusMode.BUCKET:
        focus_instruction = f"część pasażu: {control.focus_bucket}"
    elif control.focus_mode == FocusMode.MARKED_SENTENCE:
        assert control.focus_sentence_id is not None
        if control.focus_sentence_id >= len(sentences):
            raise ValueError("focus_sentence_id is outside passage")
        marked = list(sentences)
        marked[control.focus_sentence_id] = f"<FOCUS>{marked[control.focus_sentence_id]}</FOCUS>"
        rendered_passage = " ".join(marked)
        focus_instruction = "zdanie oznaczone neutralnymi znacznikami <FOCUS>"
    elif control.focus_mode == FocusMode.SENTENCE_ID:
        assert control.focus_sentence_id is not None
        if control.focus_sentence_id >= len(sentences):
            raise ValueError("focus_sentence_id is outside passage")
        rendered_passage = "\n".join(f"[{index}] {text}" for index, text in enumerate(sentences))
        focus_instruction = f"zdanie [{control.focus_sentence_id}]"
    if control.intent_applicable is True:
        applicability = "tak"
    elif control.intent_applicable is False:
        applicability = "nie"
    else:
        applicability = "nieustalone"
    return (
        "Wygeneruj jedno polskie zapytanie wyszukiwawcze, na które można odpowiedzieć "
        "wyłącznie na podstawie podanego pasażu.\n"
        "Nie kopiuj długich fragmentów pasażu. Zachowaj konieczne nazwy własne, liczby "
        "i terminy. Zwróć wyłącznie zapytanie, bez komentarza i numeracji.\n"
        f"Forma: {control.form.value}\n"
        f"Intencja: {control.intent.value} (adekwatna: {applicability})\n"
        f"Docelowy fragment: {focus_instruction}\n"
        f"Długość: {control.length}\n\n"
        f"Pasaż:\n{rendered_passage}\n\nZapytanie:\n"
    )
