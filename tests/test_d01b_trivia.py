from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from doc2query.evaluation import d01b_trivia


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _row(query_id: str, score: float = 24.0) -> dict[str, object]:
    return {
        "query_id": query_id,
        "query": f"pytanie {query_id}",
        "pos": [f"odpowiedź {query_id}"],
        "pos_id": [f"p-{query_id}"],
        "pos_scores_stronger_reranker": [score],
        "pos_is_synthetic": [False],
        "neg": [f"negatyw {query_id} {index}" for index in range(10)],
        "neg_id": [f"n-{query_id}-{index}" for index in range(10)],
        "neg_selection_tier": ["strict"] * 10,
        "translation_missing": None,
    }


def test_strict_threshold_and_shape_validation() -> None:
    _query_id, retained = d01b_trivia._validated_metadata(_row("q", 23.5))
    assert retained == []
    _query_id, retained = d01b_trivia._validated_metadata(_row("q", 23.5001))
    assert retained == [0]
    malformed = _row("q")
    malformed["neg"] = ["only one"]
    with pytest.raises(ValueError, match="invalid shape"):
        d01b_trivia._validated_metadata(malformed)
    missing = _row("q")
    missing["translation_missing"] = {"query": True}
    with pytest.raises(ValueError, match="translation"):
        d01b_trivia._validated_metadata(missing)


def test_materialization_is_query_level_and_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "train_pl.jsonl"
    readme = tmp_path / "README.md"
    policy = tmp_path / "policy.md"
    pilot = tmp_path / "pilot.jsonl"
    rows = [_row(f"q-{index}") for index in range(5)]
    _write_jsonl(source, rows)
    readme.write_text("card", encoding="utf-8")
    policy.write_text("prospective", encoding="utf-8")
    _write_jsonl(
        pilot,
        [
            {
                "source_passage_id": "pilot-1",
                "positive": {"text": "całkiem inny pasaż treningowy"},
            }
        ],
    )
    monkeypatch.setattr(
        d01b_trivia, "SOURCE_SHA256", hashlib.sha256(source.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(
        d01b_trivia, "README_SHA256", hashlib.sha256(readme.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(d01b_trivia, "SELECTED_QUERY_COUNT", 3)
    monkeypatch.setattr(d01b_trivia, "SELECTION_SEED", 7)
    monkeypatch.setattr(d01b_trivia, "SOURCE_QUERY_COUNT", 5)
    output = tmp_path / "out"
    manifest = d01b_trivia.prepare_trivia_external_dev(
        source_path=source,
        readme_path=readme,
        policy_path=policy,
        pilot_inputs=(pilot,),
        output_dir=output,
    )
    assert manifest["sets"]["dev_d01b_trivia_external_v1"]["id_count"] == 3
    assert manifest["documents"]["count"] == 33
    assert manifest["authorization"]["model_evaluation"] is False
    assert manifest["final_tests_used"] == []
    assert len((output / "dev.jsonl").read_text(encoding="utf-8").splitlines()) == 3
    with pytest.raises(FileExistsError):
        d01b_trivia.prepare_trivia_external_dev(
            source_path=source,
            readme_path=readme,
            policy_path=policy,
            pilot_inputs=(pilot,),
            output_dir=output,
        )


def test_near_duplicate_positive_blocks_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "train_pl.jsonl"
    readme = tmp_path / "README.md"
    policy = tmp_path / "policy.md"
    pilot = tmp_path / "pilot.jsonl"
    row = _row("q")
    row["pos"] = [
        "jeden dwa trzy cztery pięć sześć siedem osiem dziewięć dziesięć jedenaście dwanaście"
    ]
    _write_jsonl(source, [row])
    readme.write_text("card", encoding="utf-8")
    policy.write_text("prospective", encoding="utf-8")
    _write_jsonl(
        pilot,
        [
            {
                "source_passage_id": "pilot-1",
                "positive": {
                    "text": (
                        "jeden dwa trzy cztery pięć sześć siedem osiem dziewięć dziesięć "
                        "jedenaście dwanaście trzynaście"
                    )
                },
            }
        ],
    )
    monkeypatch.setattr(
        d01b_trivia, "SOURCE_SHA256", hashlib.sha256(source.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(
        d01b_trivia, "README_SHA256", hashlib.sha256(readme.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(d01b_trivia, "SELECTED_QUERY_COUNT", 1)
    monkeypatch.setattr(d01b_trivia, "SOURCE_QUERY_COUNT", 1)
    with pytest.raises(ValueError, match="near-duplicate"):
        d01b_trivia.prepare_trivia_external_dev(
            source_path=source,
            readme_path=readme,
            policy_path=policy,
            pilot_inputs=(pilot,),
            output_dir=tmp_path / "out",
        )
