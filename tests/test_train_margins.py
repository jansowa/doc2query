from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest

from doc2query.reranker.base import FrozenRerankerConfig
from doc2query.reranker.train_margins import score_natural_train_margins


class LengthScorer:
    name = "judge/test"

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [float(len(document)) for _query, document in pairs]


class InterruptingScorer(LengthScorer):
    def __init__(self) -> None:
        self.calls = 0

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.calls += 1
        if self.calls == 2:
            raise KeyboardInterrupt
        return super().score_pairs(pairs)


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, FrozenRerankerConfig]:
    train = tmp_path / "train.jsonl"
    _write(
        train,
        [
            {
                "example_id": "q1",
                "query": "pytanie",
                "positives": [{"doc_id": "p1", "text": "123456"}],
                "hard_negatives": [{"doc_id": "n1", "text": "12"}],
                "metadata": {"split": "train"},
            },
            {
                "example_id": "q2",
                "query": "drugie",
                "positives": [{"doc_id": "p2", "text": "123"}],
                "hard_negatives": [{"doc_id": "n2", "text": "1"}],
                "metadata": {"split": "train"},
            },
        ],
    )
    scores = tmp_path / "dev_scores.jsonl"
    _write(
        scores,
        [
            {
                "schema": "possible_false_negative_dev_scores_v1",
                "judge": "judge/test",
                "query_id": "d1",
                "positive_doc_ids": ["p"],
                "negative_doc_ids": ["n"],
                "positive_scores": [5.0],
                "negative_scores": [2.0],
            }
        ],
    )
    score_sha = hashlib.sha256(scores.read_bytes()).hexdigest()
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "artifact_id": "dev-cal",
                "artifact_fingerprint": "a" * 64,
                "fit_split": "dev",
                "source_scores_sha256": score_sha,
                "tests_used_for_threshold_tuning": [],
                "primary_judge": {"name_or_path": "judge/test", "revision": "b" * 40},
            }
        ),
        encoding="utf-8",
    )
    judge = FrozenRerankerConfig(
        name_or_path="judge/test", revision="b" * 40, license="test", batch_size=2
    )
    return train, scores, calibration, judge


def test_train_margins_are_calibrated_and_resumable(tmp_path: Path) -> None:
    train, scores, calibration, judge = _fixture(tmp_path)
    output = tmp_path / "output"
    first = score_natural_train_margins(
        input_path=train,
        output_dir=output,
        judge=judge,
        calibration_path=calibration,
        calibration_scores_path=scores,
        group_batch_size=1,
        scorer=LengthScorer(),
    )
    second = score_natural_train_margins(
        input_path=train,
        output_dir=output,
        judge=judge,
        calibration_path=calibration,
        calibration_scores_path=scores,
        scorer=LengthScorer(),
    )
    rows = [json.loads(line) for line in (output / "margins.jsonl").read_text().splitlines()]
    assert first["status"] == second["status"] == "complete"
    assert [row["raw_margin"] for row in rows] == [4.0, 2.0]
    assert [row["calibrated_margin_percentile"] for row in rows] == [1.0, 0.0]
    assert len(rows) == 2


def test_train_margin_resume_rejects_non_prefix(tmp_path: Path) -> None:
    train, scores, calibration, judge = _fixture(tmp_path)
    output = tmp_path / "output"
    score_natural_train_margins(
        input_path=train,
        output_dir=output,
        judge=judge,
        calibration_path=calibration,
        calibration_scores_path=scores,
        scorer=LengthScorer(),
    )
    rows = (output / "margins.jsonl").read_text(encoding="utf-8").splitlines()
    (output / "margins.jsonl").write_text(rows[1] + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exact prefix"):
        score_natural_train_margins(
            input_path=train,
            output_dir=output,
            judge=judge,
            calibration_path=calibration,
            calibration_scores_path=scores,
            scorer=LengthScorer(),
        )


def test_train_margin_resume_can_cross_multi_positive_group(tmp_path: Path) -> None:
    train, scores, calibration, judge = _fixture(tmp_path)
    rows = [json.loads(line) for line in train.read_text(encoding="utf-8").splitlines()]
    rows[0]["positives"].append({"doc_id": "p1b", "text": "12345"})
    _write(train, rows)
    output = tmp_path / "output"
    score_natural_train_margins(
        input_path=train,
        output_dir=output,
        judge=judge,
        calibration_path=calibration,
        calibration_scores_path=scores,
        scorer=LengthScorer(),
    )
    journal_rows = (output / "margins.jsonl").read_text(encoding="utf-8").splitlines()
    (output / "manifest.json").unlink()
    (output / "margins.jsonl").write_text(journal_rows[0] + "\n", encoding="utf-8")
    result = score_natural_train_margins(
        input_path=train,
        output_dir=output,
        judge=judge,
        calibration_path=calibration,
        calibration_scores_path=scores,
        group_batch_size=1,
        scorer=LengthScorer(),
    )
    assert result["pair_count"] == 3
    assert len((output / "margins.jsonl").read_text().splitlines()) == 3


def test_train_margin_logs_progress_and_resumes_after_interrupt(tmp_path: Path) -> None:
    train, scores, calibration, judge = _fixture(tmp_path)
    output = tmp_path / "output"
    log = io.StringIO()
    with pytest.raises(KeyboardInterrupt):
        score_natural_train_margins(
            input_path=train,
            output_dir=output,
            judge=judge,
            calibration_path=calibration,
            calibration_scores_path=scores,
            group_batch_size=1,
            progress_every=1,
            scorer=InterruptingScorer(),
            log_stream=log,
        )
    assert len((output / "margins.jsonl").read_text().splitlines()) == 1
    assert "stage=scoring 1/2" in log.getvalue()
    assert "rate=" in log.getvalue()
    assert "eta=" in log.getvalue()
    assert "interrupted durable=1/2" in log.getvalue()

    resumed_log = io.StringIO()
    result = score_natural_train_margins(
        input_path=train,
        output_dir=output,
        judge=judge,
        calibration_path=calibration,
        calibration_scores_path=scores,
        group_batch_size=1,
        progress_every=1,
        scorer=LengthScorer(),
        log_stream=resumed_log,
    )
    assert result["pair_count"] == 2
    assert "stage=scoring resume=1/2 remaining=1" in resumed_log.getvalue()
