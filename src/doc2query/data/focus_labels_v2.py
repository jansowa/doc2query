"""V2-02: sentence splitting and focus assignment robust to Polish abbreviations.

The frozen v1 splitter (`doc2query.data.focus_labels.split_sentences`) breaks after
every ``.!?`` and the reward-validation corpus measured the consequences: pseudo
sentences like ``.``/``1.``/``Zawartość.``, inflated ``sentence_count``, focus buckets
pointing at headers, and a 26% abstention rate of ``assign_focus`` on the
``good_specific`` class.

This module is a **new, versioned component that lives next to v1**.  Nothing frozen is
modified: every artifact labeled with v1 keeps its meaning, and v2 labels go to new
artifacts with their own provenance (`FOCUS_V2_VERSION`).  The scoring formula and the
abstention semantics of ``assign_focus_v2`` are deliberately identical to v1 — only the
sentence segmentation changes — so any difference between v1 and v2 labels is
attributable to segmentation alone.

Splitting rules (deterministic, documented trade-offs):

1. candidate boundaries after ``.!?`` followed by whitespace (as in v1);
2. a boundary is vetoed when the preceding token is a known Polish/English
   abbreviation (``np.``, ``p.n.e.``, ``m.in.``, …), any single letter with a period
   (initials and ``r.``/``w.``), or a short number with a period (list/date numeration,
   1-3 digits; four-digit years may still end a sentence);
3. a boundary is vetoed when the next character is lowercase (sentence continuation);
4. fragments containing no letter are merged into their neighbour, so ``.`` and ``26.``
   can never become a "sentence" again.
"""

from __future__ import annotations

import re

from doc2query.data.focus_labels import FocusAssignment, focus_bucket
from doc2query.text.normalization import SimplePolishNormalizer

FOCUS_V2_VERSION = "focus-v2:pl-abbrev-v1"

# Skróty, po których kropka niemal nigdy nie kończy zdania. Świadomie NIE ma tu
# "itd."/"itp." — te często stoją na końcu zdania, a ich zawetowanie skleja
# prawdziwe zdania. Wpisy wielokropkowe (p.n.e., m.in.) muszą być dopasowane
# w całości, dlatego token wyciągamy razem z kropkami wewnętrznymi.
_ABBREVIATIONS = frozenset(
    {
        "np",
        "tzw",
        "tzn",
        "tj",
        "m.in",
        "ok",
        "ur",
        "zm",
        "św",
        "płk",
        "gen",
        "dr",
        "prof",
        "hab",
        "mgr",
        "inż",
        "ul",
        "al",
        "os",
        "nr",
        "ks",
        "art",
        "ust",
        "pkt",
        "poz",
        "godz",
        "im",
        "łac",
        "ang",
        "niem",
        "fr",
        "ros",
        "gr",
        "wł",
        "hiszp",
        "p.n.e",
        "n.e",
        "tys",
        "mln",
        "mld",
        "proc",
        "jw",
        "cd",
        "ww",
        "zob",
        "por",
        "red",
        "wyd",
        "przyp",
        "ryc",
        "rys",
        "tab",
        "mr",
        "mrs",
        "ms",
        "st",
        "no",
        "vs",
        "inc",
        "co",
        "ltd",
        "jr",
        "sr",
        "e.g",
        "i.e",
        "a.m",
        "p.m",
        "u.s",
        "u.k",
    }
)
_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_TRAILING_TOKEN = re.compile(r"(\S+)$")
_SHORT_NUMBER = re.compile(r"^\d{1,3}$")
_SINGLE_LETTER = re.compile(r"^\w$", re.UNICODE)


def _boundary_is_vetoed(before: str, after: str) -> bool:
    """Decide whether a candidate boundary between ``before`` and ``after`` is spurious."""
    if before.endswith((".",)):
        match = _TRAILING_TOKEN.search(before)
        token = (match.group(1) if match else "").rstrip(".")
        token = token.lstrip("([\"'„”")
        lowered = token.casefold()
        if lowered in _ABBREVIATIONS:
            return True
        if _SINGLE_LETTER.match(token):
            return True
        if _SHORT_NUMBER.match(token):
            return True
    next_char = after.lstrip()[:1]
    return bool(next_char) and next_char.islower()


def split_sentences_v2(passage: str) -> list[str]:
    """Split a passage into sentences without manufacturing pseudo-sentences."""
    text = passage.strip()
    if not text:
        return []
    fragments = [fragment for fragment in _BOUNDARY.split(text) if fragment.strip()]
    merged: list[str] = []
    for fragment in fragments:
        if merged and _boundary_is_vetoed(merged[-1], fragment):
            merged[-1] = f"{merged[-1]} {fragment.strip()}"
        else:
            merged.append(fragment.strip())
    # Fragment bez ani jednej litery (".", "1.", "26.") nie jest zdaniem: doklej go
    # do poprzedniego zdania, a gdy otwiera pasaż (numeracja nagłówka) — do następnego.
    cleaned: list[str] = []
    pending_prefix = ""
    for fragment in merged:
        if not any(character.isalpha() for character in fragment):
            if cleaned:
                cleaned[-1] = f"{cleaned[-1]} {fragment}"
            else:
                pending_prefix = f"{pending_prefix} {fragment}".strip()
        else:
            if pending_prefix:
                fragment = f"{pending_prefix} {fragment}"
                pending_prefix = ""
            cleaned.append(fragment)
    if pending_prefix:
        # Pasaż w ogóle bez liter — zwróć go w całości; scoring i tak abstynuje.
        cleaned.append(pending_prefix)
    return cleaned


def assign_focus_v2(
    query: str, passage: str, *, minimum_confidence: float = 0.2
) -> FocusAssignment:
    """v1 scoring and abstention semantics on v2 sentences.

    The formula is byte-for-byte the v1 one (content-lemma overlap fraction, unique
    winner above ``minimum_confidence``), so differences against v1 labels isolate the
    effect of segmentation.
    """
    sentences = split_sentences_v2(passage)
    query_terms = set(SimplePolishNormalizer().analyze(query).content_lemmas)
    if not sentences or not query_terms:
        return FocusAssignment(None, None, 0.0, ())
    scores = tuple(
        len(query_terms & set(SimplePolishNormalizer().analyze(sentence).content_lemmas))
        / len(query_terms)
        for sentence in sentences
    )
    best = max(scores)
    winners = [index for index, score in enumerate(scores) if score == best]
    if best < minimum_confidence or len(winners) != 1:
        return FocusAssignment(None, None, best, scores)
    sentence_id = winners[0]
    return FocusAssignment(sentence_id, focus_bucket(sentence_id, len(sentences)), best, scores)
