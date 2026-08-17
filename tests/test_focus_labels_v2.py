from __future__ import annotations

from doc2query.data.focus_labels import assign_focus, split_sentences
from doc2query.data.focus_labels_v2 import (
    FOCUS_V2_VERSION,
    assign_focus_v2,
    split_sentences_v2,
)


def test_abbreviations_measured_in_the_reward_corpus_no_longer_split() -> None:
    """Dokładnie te przypadki zgłosiły podsesje korpusu walidacyjnego."""
    passage = (
        "Rzym założono w 753 r. p.n.e. według legendy. Skróty takie jak np. ten nie "
        "kończą zdania. Firma Apple Inc. powstała w garażu."
    )

    v1 = split_sentences(passage)
    v2 = split_sentences_v2(passage)

    assert len(v1) > 3  # stary splitter produkuje pseudo-zdania
    assert v2 == [
        "Rzym założono w 753 r. p.n.e. według legendy.",
        "Skróty takie jak np. ten nie kończą zdania.",
        "Firma Apple Inc. powstała w garażu.",
    ]


def test_numeration_and_orphan_dots_are_never_sentences() -> None:
    passage = "1. Wstęp do tematu jest krótki. 2. Rozwinięcie zawiera szczegóły. 26."

    v2 = split_sentences_v2(passage)

    assert v2 == [
        "1. Wstęp do tematu jest krótki.",
        "2. Rozwinięcie zawiera szczegóły. 26.",
    ]
    assert all(any(ch.isalpha() for ch in sentence) for sentence in v2)


def test_initials_and_single_letters_do_not_split() -> None:
    passage = "Książkę napisał J. Kowalski w XIX w. Był to przełom epoki."

    v2 = split_sentences_v2(passage)

    assert v2 == ["Książkę napisał J. Kowalski w XIX w. Był to przełom epoki."]


def test_lowercase_continuation_vetoes_the_boundary() -> None:
    passage = "Populacja wynosiła 1291 według spisu z 2010 r. i 1343 w 2013 r. Miasto rośnie."

    v2 = split_sentences_v2(passage)

    assert v2 == [
        "Populacja wynosiła 1291 według spisu z 2010 r. i 1343 w 2013 r. Miasto rośnie."
    ]


def test_four_digit_years_may_still_end_a_sentence() -> None:
    passage = "Wojna zakończyła się w 1945. Odbudowa trwała dekadę."

    v2 = split_sentences_v2(passage)

    assert v2 == ["Wojna zakończyła się w 1945.", "Odbudowa trwała dekadę."]


def test_regular_prose_is_split_identically_to_v1() -> None:
    passage = (
        "Koronawirusy wywołują choroby układu oddechowego. Zakażenie przenosi się drogą "
        "kropelkową. Objawy obejmują gorączkę i kaszel."
    )

    assert split_sentences_v2(passage) == split_sentences(passage)


def test_empty_and_letterless_passages_are_handled() -> None:
    assert split_sentences_v2("") == []
    assert split_sentences_v2("   ") == []
    assert split_sentences_v2("1. 2. 3.") == ["1. 2. 3."]


def test_assign_focus_v2_keeps_v1_scoring_semantics_on_identical_segmentation() -> None:
    passage = (
        "Koronawirusy wywołują choroby układu oddechowego. Zakażenie przenosi się drogą "
        "kropelkową. Okres wylęgania wynosi czternaście dni."
    )
    query = "ile dni trwa okres wylęgania"

    v1 = assign_focus(query, passage)
    v2 = assign_focus_v2(query, passage)

    assert v1.bucket == v2.bucket == "end"
    assert v1.confidence == v2.confidence
    assert v1.scores == v2.scores


def test_assign_focus_v2_recovers_a_label_v1_loses_to_pseudo_sentences() -> None:
    """Numeracja rozcina pasaż w v1 tak, że focus ląduje w złym kubełku."""
    passage = (
        "1. Historia miasta sięga średniowiecza i wiąże się z handlem solą. "
        "2. Współczesna gospodarka opiera się na turystyce górskiej. "
        "3. Kultura regionu słynie z drewnianej architektury sakralnej."
    )
    query = "na czym opiera się współczesna gospodarka miasta"

    v1 = assign_focus(query, passage)
    v2 = assign_focus_v2(query, passage)

    assert len(split_sentences(passage)) == 6  # 3 pseudo-zdania numeracji
    assert len(split_sentences_v2(passage)) == 3
    assert v2.bucket == "middle"
    assert v2.sentence_id == 1
    # v1 na sześciu "zdaniach" przypisuje środek listy do innego kubełka niż treść.
    assert (v1.sentence_id, v1.bucket) != (v2.sentence_id, v2.bucket)


def test_abstention_semantics_are_unchanged() -> None:
    v2 = assign_focus_v2("zupełnie inne słowa", "Jedno krótkie zdanie o czymś innym.")

    assert v2.bucket is None
    assert v2.sentence_id is None


def test_version_marker_is_pinned() -> None:
    assert FOCUS_V2_VERSION == "focus-v2:pl-abbrev-v1"
