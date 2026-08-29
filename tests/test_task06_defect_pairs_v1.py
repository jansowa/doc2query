"""Testy składania par z wadami: filtry są rozstrzygane lokalnie, nie przez LLM.

Sedno: verdict LLM jest daną wejściową, a nie wyrocznią. Testy pilnują, że
`copy_phrasing` rozstrzyga wyłącznie najdłuższy wspólny ciąg słów, że wymagania
answerability per klasa obowiązują, że brak jednomyślnego potwierdzenia odrzuca
parę i że na (grupę, klasę) wypada najwyżej jedna para z pierwszeństwem
kandydata organicznego.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from doc2query.preferences.defect_pairs_v1 import (
    assemble_defect_pairs,
    class_reject,
    deterministic_reject,
    longest_common_run,
    shortcut_audit,
)
from doc2query.utils.records import JsonlWriter

PASSAGE = (
    "Sandpoint w stanie Idaho liczy 7397 mieszkańców według danych z 2014 roku. "
    "Miasto leży nad jeziorem Pend Oreille i żyje z turystyki."
)
CHOSEN = "ilu mieszkańców ma Sandpoint w Idaho"


def test_longest_common_run_counts_contiguous_words() -> None:
    assert longest_common_run("liczy 7397 mieszkańców według danych z 2014", PASSAGE) == 7
    assert longest_common_run("ilu mieszkańców ma Sandpoint", PASSAGE) == 1
    assert longest_common_run("", PASSAGE) == 0


def test_deterministic_reject_catches_equivalence_and_form() -> None:
    assert deterministic_reject(
        "ilu mieszkańców ma Sandpoint w Idaho", CHOSEN, "full_question"
    ) == ("equivalent_to_chosen")
    assert (
        deterministic_reject("populacja mieszkańców miasta Sandpoint", CHOSEN, "full_question")
        == "form_violation"
    )
    # Krótka fraza wypada wcześniej na stosunku długości, nie na formie.
    assert deterministic_reject("populacja Sandpoint", CHOSEN, "full_question") == "length_ratio"
    assert deterministic_reject(
        "jaka jest mediana dochodu w Sandpoint", CHOSEN, "full_question"
    ) is (None)
    assert deterministic_reject("co", CHOSEN, "full_question") == "length"


def test_class_reject_uses_lcs_not_llm_label() -> None:
    copied = "liczy 7397 mieszkańców według danych z 2014"
    # Etykieta LLM nie ma tu nic do rzeczy: liczy się długość wspólnego ciągu.
    assert class_reject("copy_phrasing", query=copied, passage=PASSAGE, answerable=True) is None
    assert class_reject("too_general", query=copied, passage=PASSAGE, answerable=True) == (
        "reclassified_as_copy_phrasing"
    )
    assert (
        class_reject("copy_phrasing", query="jaka jest populacja", passage=PASSAGE, answerable=True)
        == "lcs_too_short"
    )
    assert (
        class_reject(
            "not_answerable",
            query="jaka jest mediana dochodu w Sandpoint",
            passage=PASSAGE,
            answerable=True,
        )
        == "answerable_true"
    )
    assert (
        class_reject(
            "not_answerable",
            query="jaka jest mediana dochodu w Sandpoint",
            passage=PASSAGE,
            answerable=False,
        )
        is None
    )
    assert (
        class_reject(
            "not_answerable", query="jakie buty kupić na zimę", passage=PASSAGE, answerable=False
        )
        == "off_topic"
    )
    # Kotwica encji: jedna wspólna nazwa własna wystarcza, udział nie decyduje.
    assert (
        class_reject(
            "not_answerable",
            query="jaka jest mediana dochodu gospodarstw w Sandpoint",
            passage=PASSAGE,
            answerable=False,
        )
        is None
    )


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    with JsonlWriter(path) as writer:
        for row in rows:
            writer.write(row)


def _fixture(tmp_path: Path, *, journal_rows: list[dict[str, Any]]) -> dict[str, Path]:
    groups = tmp_path / "groups.jsonl"
    _write(
        groups,
        [
            {
                "group_id": "g1",
                "preference_id": "p1",
                "passage": PASSAGE,
                "controls": "Forma: full_question",
                "form": "full_question",
                "chosen": {"candidate_id": "c-win", "query": CHOSEN},
                "current_rejected": {"candidate_id": "c-bad", "query": "definicja czegoś innego"},
                "others": [
                    {"candidate_id": "c-org", "query": "jaka jest mediana dochodu w Sandpoint"},
                    {
                        "candidate_id": "c-two",
                        "query": "czy Sandpoint leży nad jeziorem Pend Oreille",
                    },
                ],
            }
        ],
    )
    v3 = tmp_path / "v3.jsonl"
    _write(
        v3,
        [
            {
                "pair_id": "p1",
                "prompt": f"Pasaż:\n{PASSAGE}\n\nZapytanie:\n",
                "passage": PASSAGE,
                "passage_cluster_id": "cluster-1",
                "doc_id": "doc-1",
                "cohort_id": "cohort-1",
                "chosen": CHOSEN,
                "chosen_candidate_id": "c-win",
                "chosen_components": {"pool_margin": 3.5},
                "rejected": "definicja czegoś innego",
                "rejected_candidate_id": "c-bad",
            }
        ],
    )
    journal = tmp_path / "verdicts.journal.jsonl"
    _write(journal, journal_rows)
    return {"groups": groups, "v3": v3, "journal": journal}


def _base_journal() -> list[dict[str, Any]]:
    return [
        {"key": "g1::answerable::chosen", "verdict": {"answerable": True}},
        {"key": "g1::classify::c-org", "verdict": {"class": "not_answerable"}},
        {"key": "g1::answerable::c-org", "verdict": {"answerable": False}},
        {"key": "g1::confirm::c-org", "votes": ["chosen", "chosen"], "unanimous_chosen": True},
    ]


def test_assembles_one_pair_per_group_and_class(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, journal_rows=_base_journal())
    summary = assemble_defect_pairs(
        groups_path=paths["groups"],
        journal_path=paths["journal"],
        v3_pairs_path=paths["v3"],
        output_dir=tmp_path / "out",
    )
    assert summary["pairs"] == 1
    pair = json.loads((tmp_path / "out" / "pairs.jsonl").read_text(encoding="utf-8").strip())
    assert pair["defect_class"] == "not_answerable"
    assert pair["negative_population"] == "mined_organic"
    assert pair["pair_class"] == "defect"
    assert pair["chosen"] == CHOSEN
    assert pair["prompt"].startswith("Pasaż:")
    assert pair["rejected_measurements"]["answerable"] is False
    assert pair["final_tests_used"] == []


def test_missing_confirmation_or_answerability_rejects(tmp_path: Path) -> None:
    rows = [row for row in _base_journal() if not row["key"].startswith("g1::confirm")]
    paths = _fixture(tmp_path, journal_rows=rows)
    summary = assemble_defect_pairs(
        groups_path=paths["groups"],
        journal_path=paths["journal"],
        v3_pairs_path=paths["v3"],
        output_dir=tmp_path / "out",
    )
    assert summary["pairs"] == 0
    assert summary["counters"]["reject_no_confirmation"] == 1


def test_chosen_not_answerable_drops_whole_group(tmp_path: Path) -> None:
    rows = [
        {"key": "g1::answerable::chosen", "verdict": {"answerable": False}},
        *[row for row in _base_journal() if row["key"] != "g1::answerable::chosen"],
    ]
    paths = _fixture(tmp_path, journal_rows=rows)
    summary = assemble_defect_pairs(
        groups_path=paths["groups"],
        journal_path=paths["journal"],
        v3_pairs_path=paths["v3"],
        output_dir=tmp_path / "out",
    )
    assert summary["pairs"] == 0
    assert summary["counters"]["group_chosen_not_answerable"] == 1


def test_organic_candidate_wins_over_mutation(tmp_path: Path) -> None:
    rows = [
        *_base_journal(),
        {
            "key": "g1::mutate::not_answerable",
            "verdict": {"query": "jaka jest cena biletu w Sandpoint"},
        },
        {
            "key": "g1::answerable::g1::mutated::not_answerable",
            "verdict": {"answerable": False},
        },
        {
            "key": "g1::confirm::g1::mutated::not_answerable",
            "votes": ["chosen", "chosen"],
            "unanimous_chosen": True,
        },
    ]
    paths = _fixture(tmp_path, journal_rows=rows)
    assemble_defect_pairs(
        groups_path=paths["groups"],
        journal_path=paths["journal"],
        v3_pairs_path=paths["v3"],
        output_dir=tmp_path / "out",
    )
    pairs = [
        json.loads(line)
        for line in (tmp_path / "out" / "pairs.jsonl").read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]
    assert len(pairs) == 1
    assert pairs[0]["negative_population"] == "mined_organic"


def test_output_directory_is_never_overwritten(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, journal_rows=_base_journal())
    (tmp_path / "out").mkdir()
    with pytest.raises(FileExistsError):
        assemble_defect_pairs(
            groups_path=paths["groups"],
            journal_path=paths["journal"],
            v3_pairs_path=paths["v3"],
            output_dir=tmp_path / "out",
        )


def test_shortcut_audit_flags_separable_surface_features() -> None:
    separable = [
        {
            "passage": PASSAGE,
            "chosen": "ilu mieszkańców ma Sandpoint w Idaho dokładnie w roku",
            "rejected": "populacja",
        }
        for _ in range(20)
    ]
    result = shortcut_audit(separable)
    assert result["auc"] is not None
    assert result["auc"] > 0.8, "trywialnie rozdzielne pary muszą podnieść AUC"
    assert shortcut_audit(separable[:2])["auc"] is None
