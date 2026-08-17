from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from doc2query.preferences.answerability_judge import (
    CONTRACT,
    SYSTEM_PROMPT,
    JudgeItem,
    analyze_calibration,
    calibration_items_from_reward_corpus,
    judge_item_id,
    load_judge_config,
    load_judgments,
    parse_verdict,
    run_judgments,
    verify_pinned_model,
)

CONFIG_PATH = Path("configs/preferences/task06_answerability_judge_v1.json")
PASSAGE = (
    "Koronawirusy to duża rodzina wirusów wywołujących choroby układu oddechowego. "
    "Okres wylęgania wynosi od dwóch do czternastu dni."
)


def _pinned_config(tmp_path: Path, *, digest: str = "sha256:abc") -> Path:
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["judge"]["model_digest"] = digest
    path = tmp_path / "judge.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def _item(index: int, query: str, **metadata: Any) -> JudgeItem:
    passage = f"{PASSAGE} Wariant {index}."
    return JudgeItem(
        item_id=judge_item_id(query, passage),
        query=query,
        passage=passage,
        metadata=metadata or {"source": "test"},
    )


class _Backend:
    """Fake ollama: /api/tags for pinning, /api/chat with a verdict policy."""

    def __init__(self, *, digest: str = "sha256:abc", verdict: str = "yes") -> None:
        self.digest = digest
        self.verdict = verdict
        self.chat_calls: list[dict[str, Any]] = []

    def __call__(self, url: str, payload: Mapping[str, Any], timeout: float) -> dict[str, Any]:
        del timeout
        if url.endswith("/api/tags"):
            return {"models": [{"name": "qwen3.6:27b-q4_K_M", "digest": self.digest}]}
        self.chat_calls.append(dict(payload))
        return {"message": {"content": json.dumps({"verdict": self.verdict})}}


def test_frozen_draft_config_passes_validation_and_pins_the_guards() -> None:
    config = load_judge_config(CONFIG_PATH)

    assert config["judge"]["temperature"] == 0.0
    assert config["decision"]["uncertain_blocks_chosen"] is True
    assert config["decision"]["uncertain_is_not_a_defect"] is True
    assert config["used_for_pair_building"] is False
    assert config["final_tests_used"] == []


def test_generator_family_judge_is_refused(tmp_path: Path) -> None:
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["judge"]["model"] = "bielik:4.5b-instruct"
    path = tmp_path / "judge.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="generator family"):
        load_judge_config(path)


def test_unpinned_digest_refuses_to_judge(tmp_path: Path) -> None:
    config = load_judge_config(CONFIG_PATH)

    with pytest.raises(ValueError, match="pins no model_digest"):
        verify_pinned_model(config, _Backend())

    with pytest.raises(ValueError, match="pins no model_digest"):
        run_judgments(
            [_item(0, "ile dni trwa okres wylęgania")],
            config=config,
            output_dir=tmp_path,
            transport=_Backend(),
        )


def test_digest_mismatch_refuses_to_judge(tmp_path: Path) -> None:
    config = load_judge_config(_pinned_config(tmp_path, digest="sha256:expected"))

    with pytest.raises(ValueError, match="does not match the pinned"):
        run_judgments(
            [_item(0, "ile dni trwa okres wylęgania")],
            config=config,
            output_dir=tmp_path / "out",
            transport=_Backend(digest="sha256:other"),
        )


def test_judgments_are_journaled_and_resume_skips_done_items(tmp_path: Path) -> None:
    config = load_judge_config(_pinned_config(tmp_path))
    items = [_item(index, f"zapytanie {index} o okres wylęgania") for index in range(3)]
    backend = _Backend()

    first = run_judgments(items, config=config, output_dir=tmp_path / "out", transport=backend)
    second_backend = _Backend()
    second = run_judgments(
        items, config=config, output_dir=tmp_path / "out", transport=second_backend
    )

    assert first["judged_count"] == 3
    assert len(backend.chat_calls) == 3
    assert second["counters"]["already_judged"] == 3
    assert second_backend.chat_calls == []


def test_chat_payload_is_deterministic_and_carries_only_passage_and_query(
    tmp_path: Path,
) -> None:
    config = load_judge_config(_pinned_config(tmp_path))
    backend = _Backend()
    run_judgments(
        [_item(0, "ile dni trwa okres wylęgania")],
        config=config,
        output_dir=tmp_path / "out",
        transport=backend,
    )

    payload = backend.chat_calls[0]
    assert payload["options"]["temperature"] == 0.0
    assert payload["format"] == "json"
    assert payload["messages"][0]["content"] == SYSTEM_PROMPT
    user = json.loads(payload["messages"][1]["content"])
    assert set(user) == {"passage", "query"}


def test_invalid_verdicts_fail_closed_per_item(tmp_path: Path) -> None:
    class _Broken(_Backend):
        def __call__(
            self, url: str, payload: Mapping[str, Any], timeout: float
        ) -> dict[str, Any]:
            if url.endswith("/api/tags"):
                return super().__call__(url, payload, timeout)
            return {"message": {"content": "może"}}

    config = load_judge_config(_pinned_config(tmp_path))

    summary = run_judgments(
        [_item(0, "ile dni trwa okres wylęgania")],
        config=config,
        output_dir=tmp_path / "out",
        transport=_Broken(),
        sleep=lambda _s: None,
    )

    assert summary["counters"]["failed_closed"] == 1
    assert summary["judged_count"] == 0
    journal = load_judgments(tmp_path / "out" / "judgments.journal.jsonl")
    assert journal == {}


def test_operator_cap_defers_resumably(tmp_path: Path) -> None:
    config = load_judge_config(_pinned_config(tmp_path))
    items = [_item(index, f"zapytanie {index}") for index in range(3)]

    summary = run_judgments(
        items,
        config=config,
        output_dir=tmp_path / "out",
        transport=_Backend(),
        max_new_judgments=1,
    )

    assert summary["judged_count"] == 1
    assert summary["counters"]["deferred_by_operator_cap"] == 2


def test_parse_verdict_accepts_only_the_frozen_labels() -> None:
    assert parse_verdict('{"verdict": "yes"}') == "yes"
    assert parse_verdict('{"verdict": "NO"}') == "no"
    with pytest.raises(ValueError, match="invalid answerability verdict"):
        parse_verdict('{"verdict": "maybe"}')
    with pytest.raises(json.JSONDecodeError):
        parse_verdict("nie json")


def test_reward_corpus_items_carry_expected_labels(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    records = tmp_path / "records.jsonl"
    corpus.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False)
            for row in (
                {"label": "good_specific", "example_id": "1::7", "query": "pytanie dobre"},
                {"label": "ungrounded", "example_id": "1::7", "query": "pytanie z kosmosu"},
                {"label": "wrong_form", "example_id": "1::7", "query": "pomijane"},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    records.write_text(
        json.dumps(
            {"example_id": "1::7", "positives": [{"doc_id": "7", "text": PASSAGE}]},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    items = calibration_items_from_reward_corpus(corpus, records)

    assert len(items) == 2
    assert {item.metadata["expected"] for item in items} == {"yes", "no"}


def test_calibration_analysis_reports_agreement_and_keeps_uncertain_separate() -> None:
    items = [
        _item(
            0,
            "pytanie a",
            source="groq_audit",
            audit_id="x1",
            role="chosen",
            groq_answerable={"m1": True, "m2": True},
        ),
        _item(
            1,
            "pytanie b",
            source="groq_audit",
            audit_id="x2",
            role="rejected",
            groq_answerable={"m1": False, "m2": True},
        ),
        _item(2, "pytanie c", source="reward_corpus", label="ungrounded", expected="no"),
        _item(3, "pytanie d", source="reward_corpus", label="good_specific", expected="yes"),
    ]
    judgments = {
        items[0].item_id: {"verdict": "yes"},
        items[1].item_id: {"verdict": "no"},
        items[2].item_id: {"verdict": "no"},
        items[3].item_id: {"verdict": "uncertain"},
    }

    report = analyze_calibration(items, judgments)

    assert report["contract"] == CONTRACT
    assert report["judged_items"] == 4
    assert report["agreement_with_groq"]["m1"]["rate"] == pytest.approx(1.0)
    assert report["agreement_with_groq"]["m2"]["rate"] == pytest.approx(0.5)
    # Konsensus tylko tam, gdzie oba modele Groq się zgadzają (item 0).
    assert report["agreement_with_groq_consensus"]["count"] == 1
    assert report["constructed_class_accuracy"]["ungrounded"]["rate"] == pytest.approx(1.0)
    # `uncertain` nie wchodzi do żadnej zgodności — jest liczony osobno.
    assert "good_specific" not in report["constructed_class_accuracy"]
    assert report["uncertain_counts"] == {"reward_corpus": 1}
    assert report["used_for_pair_building"] is False
