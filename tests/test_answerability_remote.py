from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from doc2query.preferences.answerability_judge import JudgeItem, judge_item_id
from doc2query.preferences.answerability_remote import (
    REMOTE_JOURNAL_SCHEMA,
    apply_acceptance_criteria,
    load_remote_journal,
    packet_items_preview,
    remote_identity,
    write_packet,
)


def _audit_item(query: str, passage: str, role: str, labels: tuple[bool, ...]) -> JudgeItem:
    models = ("openai/gpt-oss-120b", "qwen/qwen3.6-27b")
    return JudgeItem(
        item_id=judge_item_id(query, passage),
        query=query,
        passage=passage,
        metadata={
            "source": "groq_audit",
            "role": role,
            "groq_answerable": {model: labels[i] for i, model in enumerate(models[: len(labels)])},
        },
    )


def _corpus_item(query: str, passage: str, label: str, expected: str) -> JudgeItem:
    return JudgeItem(
        item_id=judge_item_id(query, passage),
        query=query,
        passage=passage,
        metadata={"source": "reward_corpus", "label": label, "expected": expected},
    )


def _write_journal(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def _verdict_row(item: JudgeItem, verdict: str, **overrides: Any) -> dict[str, Any]:
    row = {
        "schema": REMOTE_JOURNAL_SCHEMA,
        "event": "verdict",
        "item_id": item.item_id,
        "verdict": verdict,
        "prompt_version": "task06-answerability-pl-v1",
        "model": "vendor/Judge-27B-FP8",
        "served_model": {"data": [{"id": "vendor/Judge-27B-FP8"}]},
    }
    row.update(overrides)
    return row


def test_packet_deduplicates_passages_and_carries_no_labels(tmp_path: Path) -> None:
    shared = "Pasaż o Rzymie założonym w 753 r. p.n.e."
    items = [
        _audit_item("kiedy powstał rzym", shared, "chosen", (True, True)),
        _audit_item("data założenia rzymu", shared, "rejected", (True, False)),
        _corpus_item("stolica polski", "Inny pasaż o Warszawie.", "ungrounded", "no"),
    ]

    manifest = write_packet(items, tmp_path / "packet")

    assert manifest["item_count"] == 3
    assert manifest["passage_count"] == 2  # pasaż współdzielony zapisany raz
    assert manifest["labels_included"] is False
    body = (tmp_path / "packet" / "items.jsonl").read_text(encoding="utf-8")
    for forbidden in ("groq_answerable", "expected", "ungrounded", "chosen", "rejected"):
        assert forbidden not in body
    preview = packet_items_preview(tmp_path / "packet")
    assert {row["passage"] for row in preview} == {shared, "Inny pasaż o Warszawie."}


def test_packet_refuses_duplicate_item_ids(tmp_path: Path) -> None:
    item = _audit_item("pytanie", "pasaż", "chosen", (True, True))

    with pytest.raises(ValueError, match="unique IDs"):
        write_packet([item, item], tmp_path / "packet")


def test_journal_import_refuses_items_outside_the_packet(tmp_path: Path) -> None:
    items = [_audit_item("pytanie", "pasaż", "chosen", (True, True))]
    manifest = write_packet(items, tmp_path / "packet")
    stranger = _audit_item("obce pytanie", "obcy pasaż", "chosen", (True, True))
    journal = tmp_path / "verdicts.jsonl"
    _write_journal(journal, [_verdict_row(items[0], "yes"), _verdict_row(stranger, "no")])

    with pytest.raises(ValueError, match="outside the packet"):
        load_remote_journal(journal, manifest, items)


def test_journal_import_refuses_mixed_models_and_wrong_prompt(tmp_path: Path) -> None:
    items = [
        _audit_item("a", "pasaż a", "chosen", (True, True)),
        _audit_item("b", "pasaż b", "chosen", (True, True)),
    ]
    manifest = write_packet(items, tmp_path / "packet")
    journal = tmp_path / "mixed.jsonl"
    _write_journal(
        journal,
        [
            _verdict_row(items[0], "yes"),
            _verdict_row(items[1], "no", model="inny/model"),
        ],
    )
    with pytest.raises(ValueError, match="mixes judge models"):
        load_remote_journal(journal, manifest, items)

    other = tmp_path / "prompt.jsonl"
    _write_journal(other, [_verdict_row(items[0], "yes", prompt_version="inna-wersja")])
    with pytest.raises(ValueError, match="unknown prompt version"):
        load_remote_journal(other, manifest, items)


def test_journal_import_refuses_contradicting_itself(tmp_path: Path) -> None:
    items = [_audit_item("pytanie", "pasaż", "chosen", (True, True))]
    manifest = write_packet(items, tmp_path / "packet")
    journal = tmp_path / "verdicts.jsonl"
    _write_journal(journal, [_verdict_row(items[0], "yes"), _verdict_row(items[0], "no")])

    with pytest.raises(ValueError, match="disagrees with itself"):
        load_remote_journal(journal, manifest, items)


def test_journal_import_refuses_a_drifted_item_set(tmp_path: Path) -> None:
    items = [_audit_item("pytanie", "pasaż", "chosen", (True, True))]
    manifest = write_packet(items, tmp_path / "packet")
    journal = tmp_path / "verdicts.jsonl"
    _write_journal(journal, [_verdict_row(items[0], "yes")])
    drifted = [*items, _audit_item("nowe", "nowy pasaż", "chosen", (True, True))]

    with pytest.raises(ValueError, match="do not match the packet"):
        load_remote_journal(journal, manifest, drifted)


def _calibration_fixture(
    *, judge_yes_on_no_class: bool = False, uncertain: int = 0
) -> tuple[list[JudgeItem], dict[str, dict[str, Any]]]:
    items: list[JudgeItem] = []
    verdicts: dict[str, dict[str, Any]] = {}
    # 20 stron konsensusu: 15 yes, 5 no; sędzia trafia wszystkie
    for index in range(20):
        answerable = index < 15
        item = _audit_item(f"q{index}", f"pasaż {index}", "chosen", (answerable, answerable))
        items.append(item)
        verdicts[item.item_id] = _verdict_row(item, "yes" if answerable else "no")
    for index in range(10):
        label, expected = ("ungrounded", "no") if index < 5 else ("good_specific", "yes")
        item = _corpus_item(f"c{index}", f"korpus {index}", label, expected)
        items.append(item)
        verdict = "yes" if expected == "yes" else "no"
        if label == "ungrounded" and judge_yes_on_no_class:
            verdict = "yes"
        verdicts[item.item_id] = _verdict_row(item, verdict)
    for index in range(5):
        item = _corpus_item(f"alt{index}", f"alt {index}", "good_alternative", "yes")
        items.append(item)
        verdicts[item.item_id] = _verdict_row(item, "yes")
    for index in range(uncertain):
        item = _audit_item(f"u{index}", f"niepewny {index}", "chosen", (True, True))
        items.append(item)
        verdicts[item.item_id] = _verdict_row(item, "uncertain")
    return items, verdicts


def test_acceptance_passes_when_all_three_criteria_hold() -> None:
    items, verdicts = _calibration_fixture()

    result = apply_acceptance_criteria(items, verdicts, {"agreement_with_groq": {}})

    assert result["k1_consensus"]["accuracy"] == pytest.approx(1.0)
    assert result["k1_consensus"]["balanced_accuracy"] == pytest.approx(1.0)
    assert result["k2_constructed_classes"]["met"] is True
    assert result["k3_abstention"]["met"] is True
    assert result["accepted"] is True
    assert result["status"] == "accepted_as_axis_a_answerability_signal"
    assert result["manual_review_required"] is True


def test_acceptance_fails_when_the_ungrounded_class_is_not_rejected() -> None:
    """Sanity z konstrukcji jest osobnym warunkiem: zgodność z Groq go nie zastąpi."""
    items, verdicts = _calibration_fixture(judge_yes_on_no_class=True)

    result = apply_acceptance_criteria(items, verdicts, {})

    assert result["k1_consensus"]["met"] is True
    assert result["k2_constructed_classes"]["met"] is False
    assert result["accepted"] is False
    assert result["status"] == "rejected_axis_a_without_answerability_filter"


def test_acceptance_fails_on_too_much_abstention() -> None:
    items, verdicts = _calibration_fixture(uncertain=40)

    result = apply_acceptance_criteria(items, verdicts, {})

    assert result["k3_abstention"]["uncertain_share"] > 0.25
    assert result["k3_abstention"]["met"] is False
    assert result["accepted"] is False


def test_uncertain_sides_leave_the_consensus_denominator() -> None:
    """`uncertain` nie jest defektem, więc nie może liczyć się jako pomyłka w K1."""
    items, verdicts = _calibration_fixture(uncertain=5)

    result = apply_acceptance_criteria(items, verdicts, {})

    assert result["k1_consensus"]["decided_consensus_sides"] == 20
    assert result["k1_consensus"]["accuracy"] == pytest.approx(1.0)


def test_identity_records_that_the_pin_is_weaker_than_a_digest() -> None:
    _, verdicts = _calibration_fixture()

    identity = remote_identity(verdicts)

    assert identity["model"] == "vendor/Judge-27B-FP8"
    assert identity["digest_pinning"] == "weaker_than_ollama_path"


def _cohort(directory: Path, rows: list[dict[str, Any]], eligible: list[dict[str, Any]]) -> None:
    gate = directory / "diversity_gate"
    scoring = directory / "d01_controlled" / "scoring"
    gate.mkdir(parents=True)
    scoring.mkdir(parents=True)
    (gate / "group_verdicts.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in eligible), encoding="utf-8"
    )
    (scoring / "per_generation.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def _scored(candidate_id: str, group: str, query: str, passage: str) -> dict[str, Any]:
    return {
        "evaluation_id": candidate_id,
        "evaluation_group_id": group,
        "generated": query,
        "positive": {"doc_id": "1", "text": passage},
        "final_tests_used": [],
    }


def test_candidate_pool_takes_only_eligible_representatives(tmp_path: Path) -> None:
    from doc2query.preferences.answerability_remote import candidate_pool_items

    cohort = tmp_path / "same_prompt_expansion_v1"
    _cohort(
        cohort,
        [
            _scored("c1", "g1", "reprezentant", "pasaż g1"),
            _scored("c2", "g1", "nie reprezentant", "pasaż g1"),
            _scored("c3", "g2", "grupa odrzucona", "pasaż g2"),
        ],
        [
            {"group_id": "g1", "eligible": True, "representative_candidate_ids": ["c1"]},
            {"group_id": "g2", "eligible": False, "representative_candidate_ids": ["c3"]},
        ],
    )

    items = candidate_pool_items([cohort])

    assert [item.query for item in items] == ["reprezentant"]
    assert items[0].metadata["source"] == "candidate_pool"
    assert items[0].metadata["candidate_id"] == "c1"


def test_candidate_pool_deduplicates_identical_query_passage_pairs(tmp_path: Path) -> None:
    from doc2query.preferences.answerability_remote import candidate_pool_items

    first = tmp_path / "same_prompt_expansion_v1"
    second = tmp_path / "same_prompt_expansion_v2"
    for name, candidate in ((first, "c1"), (second, "c2")):
        _cohort(
            name,
            [_scored(candidate, "g1", "to samo pytanie", "ten sam pasaż")],
            [{"group_id": "g1", "eligible": True, "representative_candidate_ids": [candidate]}],
        )

    items = candidate_pool_items([first, second])

    # Ten sam (zapytanie, pasaż) to jeden werdykt: item_id jest hashem treści.
    assert len(items) == 1


def test_candidate_pool_refuses_a_final_test_leak(tmp_path: Path) -> None:
    from doc2query.preferences.answerability_remote import candidate_pool_items

    cohort = tmp_path / "same_prompt_expansion_v1"
    row = _scored("c1", "g1", "pytanie", "pasaż")
    row["final_tests_used"] = ["test"]
    _cohort(
        cohort,
        [row],
        [{"group_id": "g1", "eligible": True, "representative_candidate_ids": ["c1"]}],
    )

    with pytest.raises(ValueError, match="final-test usage"):
        candidate_pool_items([cohort])


def test_import_skips_out_of_schema_events_but_keeps_verdicts(tmp_path: Path) -> None:
    """Odpowiedzi poza schematem zostają w journalu jako dowód, ale nie są werdyktami."""
    items = [
        _audit_item("a", "pasaż a", "chosen", (True, True)),
        _audit_item("b", "pasaż b", "chosen", (True, True)),
    ]
    manifest = write_packet(items, tmp_path / "packet")
    journal = tmp_path / "verdicts.jsonl"
    _write_journal(
        journal,
        [
            _verdict_row(items[0], "yes"),
            {
                "schema": REMOTE_JOURNAL_SCHEMA,
                "event": "out_of_schema",
                "item_id": items[1].item_id,
                "prompt_version": "task06-answerability-pl-v1",
                "model": "vendor/Judge-27B-FP8",
                "attempts": 2,
                "error": "werdykt poza schematem: 'verdict'",
                "content": '{"verdict": "verdict"}',
            },
        ],
    )

    verdicts = load_remote_journal(journal, manifest, items)

    assert set(verdicts) == {items[0].item_id}


def _verdict_map(pairs: dict[str, str], **extra: Any) -> dict[str, dict[str, Any]]:
    return {
        item_id: {"verdict": verdict, "prompt_version": "task06-answerability-pl-v1", **extra}
        for item_id, verdict in pairs.items()
    }


def test_ab_gate_accepts_when_agreement_is_high_and_drift_symmetric() -> None:
    from doc2query.preferences.answerability_remote import compare_journal_verdicts

    baseline = _verdict_map({f"i{n}": "yes" for n in range(100)})
    candidate = dict(baseline)
    # dwie zmiany w przeciwnych kierunkach: szum, nie dryf
    candidate["i0"] = {**candidate["i0"], "verdict": "no"}
    baseline["i1"] = {**baseline["i1"], "verdict": "no"}

    result = compare_journal_verdicts(baseline, candidate)

    assert result["compared_items"] == 100
    assert result["agreement"] == pytest.approx(0.98)
    assert result["systematic_drift"] is False
    assert result["accepted"] is True


def test_ab_gate_rejects_when_agreement_below_threshold() -> None:
    from doc2query.preferences.answerability_remote import compare_journal_verdicts

    baseline = _verdict_map({f"i{n}": "yes" for n in range(100)})
    candidate = dict(baseline)
    for index in range(5):
        candidate[f"i{index}"] = {**candidate[f"i{index}"], "verdict": "uncertain"}

    result = compare_journal_verdicts(baseline, candidate)

    assert result["agreement"] == pytest.approx(0.95)
    assert result["accepted"] is False
    assert result["status"] == "batching_rejected_keep_single_requests"


def test_ab_gate_catches_one_sided_drift_even_at_high_agreement() -> None:
    """Kluczowy przypadek: zgodność zdaje próg, ale migracje idą tylko w jedną stronę."""
    from doc2query.preferences.answerability_remote import compare_journal_verdicts

    baseline = _verdict_map({f"i{n}": "yes" for n in range(1000)})
    candidate = dict(baseline)
    for index in range(15):  # 98,5% zgodności, ale wszystkie zmiany yes->no
        candidate[f"i{index}"] = {**candidate[f"i{index}"], "verdict": "no"}

    result = compare_journal_verdicts(baseline, candidate)

    assert result["agreement"] >= 0.98
    assert result["systematic_drift"] is True
    assert result["accepted"] is False


def test_provenance_exposes_a_mixed_instrument() -> None:
    from doc2query.preferences.answerability_remote import journal_provenance

    verdicts = {
        "a": {"prompt_version": "task06-answerability-pl-v2-batched", "batch_size": 4},
        "b": {"prompt_version": "task06-answerability-pl-v1", "batch_size": 4, "fallback": True},
    }

    provenance = journal_provenance(verdicts)

    assert provenance["single_instrument"] is False
    assert provenance["fallback_verdicts"] == 1
    assert provenance["fallback_share"] == pytest.approx(0.5)


def test_import_accepts_batched_prompt_version(tmp_path: Path) -> None:
    items = [_audit_item("a", "pasaż a", "chosen", (True, True))]
    manifest = write_packet(items, tmp_path / "packet")
    journal = tmp_path / "verdicts.jsonl"
    _write_journal(
        journal,
        [_verdict_row(items[0], "yes", prompt_version="task06-answerability-pl-v2-batched")],
    )

    verdicts = load_remote_journal(journal, manifest, items)

    assert set(verdicts) == {items[0].item_id}


def test_import_still_refuses_an_unknown_prompt_version(tmp_path: Path) -> None:
    items = [_audit_item("a", "pasaż a", "chosen", (True, True))]
    manifest = write_packet(items, tmp_path / "packet")
    journal = tmp_path / "verdicts.jsonl"
    _write_journal(journal, [_verdict_row(items[0], "yes", prompt_version="cos-innego-v9")])

    with pytest.raises(ValueError, match="unknown prompt version"):
        load_remote_journal(journal, manifest, items)
