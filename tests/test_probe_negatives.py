from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from doc2query.evaluation.corpus import (
    BM25CorpusIndex,
    BM25IndexConfig,
    build_bm25_index,
)
from doc2query.evaluation.embedder_probe import prepare_probe_pairs
from doc2query.evaluation.probe_negatives import (
    NEGATIVE_RECIPE_VERSION,
    NegativeCandidate,
    NegativeRecipe,
    PossibleFalseNegativeCalibration,
    ProbeNegativeBlocker,
    assert_same_negative_contract,
    calibration_artifact_fingerprint,
    select_negative,
)
from doc2query.utils.records import JsonlWriter


class _MockPrimary:
    name = "sdadas/polish-reranker-roberta-v3"

    def score_pairs(self, pairs: Any) -> list[float]:
        return [9.0 if "fałszywy" in passage else 1.0 for _query, passage in pairs]


class _CountingPrimary(_MockPrimary):
    def __init__(self) -> None:
        self.calls = 0
        self.pairs = 0

    def score_pairs(self, pairs: Any) -> list[float]:
        self.calls += 1
        self.pairs += len(pairs)
        return super().score_pairs(pairs)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with JsonlWriter(path) as writer:
        for row in rows:
            writer.write(row)


def _calibration_payload(*, fit_split: str = "dev_intrinsic") -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "possible_false_negative_threshold",
        "artifact_id": "task02-primary-dev-pfn-v1",
        "fit_split": fit_split,
        "fit_dataset_fingerprint": "d" * 64,
        "primary_judge": {
            "name_or_path": "sdadas/polish-reranker-roberta-v3",
            "revision": "e" * 40,
        },
        "score_kind": "raw_pair_logit",
        "comparison_operator": "greater_than_or_equal",
        "threshold": 5.0,
        "selection_method": "fixture labelled-dev operating point",
        "source_scores_sha256": "s" * 64,
    }
    payload["source_scores_sha256"] = "a" * 64
    payload["artifact_fingerprint"] = calibration_artifact_fingerprint(payload)
    return payload


def _write_calibration(tmp_path: Path, *, fit_split: str = "dev_intrinsic") -> tuple[Path, str]:
    payload = _calibration_payload(fit_split=fit_split)
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, str(payload["artifact_fingerprint"])


def _load_calibration(tmp_path: Path) -> PossibleFalseNegativeCalibration:
    path, fingerprint = _write_calibration(tmp_path)
    return PossibleFalseNegativeCalibration.load(
        path,
        expected_id="task02-primary-dev-pfn-v1",
        expected_fingerprint=fingerprint,
    )


def _recipe(
    tmp_path: Path,
    *,
    strategy: str = "hn0_filter",
    policy: str = "drop",
    bm25_index_fingerprint: str | None = None,
) -> NegativeRecipe:
    path, fingerprint = _write_calibration(tmp_path)
    return NegativeRecipe(
        version=NEGATIVE_RECIPE_VERSION,
        strategy=strategy,  # type: ignore[arg-type]
        false_negative_policy=policy,  # type: ignore[arg-type]
        calibration_artifact_path=str(path),
        calibration_artifact_id="task02-primary-dev-pfn-v1",
        calibration_artifact_fingerprint=fingerprint,
        bm25_index_fingerprint=bm25_index_fingerprint,
        bm25_candidates=3,
    )


def test_calibration_is_dev_only_and_fingerprint_pinned(tmp_path: Path) -> None:
    path, fingerprint = _write_calibration(tmp_path, fit_split="test_native_pl")
    with pytest.raises(ProbeNegativeBlocker, match="development split"):
        PossibleFalseNegativeCalibration.load(
            path,
            expected_id="task02-primary-dev-pfn-v1",
            expected_fingerprint=fingerprint,
        )
    path, fingerprint = _write_calibration(tmp_path)
    payload = json.loads(path.read_text())
    payload["threshold"] = 99.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ProbeNegativeBlocker, match="fingerprint"):
        PossibleFalseNegativeCalibration.load(
            path,
            expected_id="task02-primary-dev-pfn-v1",
            expected_fingerprint=fingerprint,
        )


def test_missing_calibration_is_a_fail_closed_blocker() -> None:
    recipe = NegativeRecipe(
        version=NEGATIVE_RECIPE_VERSION,
        strategy="hn0_filter",
        false_negative_policy="drop",
    )
    with pytest.raises(ProbeNegativeBlocker, match="path, artifact ID"):
        recipe.load_calibration()


def test_drop_demote_and_keep_log_policies(tmp_path: Path) -> None:
    calibration = _load_calibration(tmp_path)
    candidates = [
        NegativeCandidate("n-good", "zwykły negatyw", "inherited", 1),
        NegativeCandidate("n-false", "fałszywy negatyw", "inherited", 2),
    ]
    dropped = select_negative(
        example_id="example",
        query="query",
        candidates=candidates,
        recipe=_recipe(tmp_path, policy="drop"),
        scorer=_MockPrimary(),
        calibration=calibration,
    )
    assert dropped.paired is not None and dropped.paired.doc_id == "n-good"
    assert sum(row["possible_false_negative"] is True for row in dropped.audit_rows) == 1

    demoted = select_negative(
        example_id="example",
        query="query",
        candidates=candidates,
        recipe=_recipe(tmp_path, policy="demote"),
        scorer=_MockPrimary(),
        calibration=calibration,
    )
    assert demoted.paired is not None and demoted.paired.doc_id == "n-good"
    assert demoted.demoted is not None and demoted.demoted.doc_id == "n-false"

    kept = select_negative(
        example_id="example",
        query="query",
        candidates=[candidates[1]],
        recipe=_recipe(tmp_path, policy="keep+log"),
        scorer=_MockPrimary(),
        calibration=calibration,
    )
    assert kept.paired is not None and kept.paired.doc_id == "n-false"
    assert kept.audit_rows[0]["action"] == "paired_keep_flagged"

    all_dropped = select_negative(
        example_id="example",
        query="query",
        candidates=[candidates[1]],
        recipe=_recipe(tmp_path, policy="drop"),
        scorer=_MockPrimary(),
        calibration=calibration,
    )
    assert all_dropped.dropped_example and all_dropped.paired is None


def _probe_record() -> dict[str, Any]:
    return {
        "example_id": "q-1",
        "query": "gdzie leży stolica",
        "positives": [{"doc_id": "d-000", "text": "stolica leży nad rzeką"}],
        "hard_negatives": [
            {"doc_id": "d-001", "text": "fałszywy negatyw o stolicy"},
            {"doc_id": "d-002", "text": "zwykły negatyw o mieście"},
        ],
    }


def test_flags_are_reported_separately_for_natural_and_synthetic(tmp_path: Path) -> None:
    calibration = _load_calibration(tmp_path)
    recipe = _recipe(tmp_path)
    natural, _, natural_report, natural_audit = prepare_probe_pairs(
        [_probe_record()],
        query_source="natural",
        negative_recipe=recipe,
        calibration=calibration,
        primary_scorer=_MockPrimary(),
    )
    generations = tmp_path / "generations.jsonl"
    _write_jsonl(
        generations,
        [
            {
                "example_id": "q-1",
                "mode": "deterministic",
                "candidate_index": 0,
                "generated": "syntetyczne query",
            }
        ],
    )
    synthetic, _, synthetic_report, synthetic_audit = prepare_probe_pairs(
        [_probe_record()],
        query_source="synthetic",
        negative_recipe=recipe,
        calibration=calibration,
        primary_scorer=_MockPrimary(),
        synthetic_generations=generations,
        generator_id="W05",
    )
    assert natural and synthetic
    assert natural_report["per_source"]["natural"]["possible_false_negative_rate"] == 0.5
    assert synthetic_report["per_source"]["W05"]["possible_false_negative_count"] == 1
    assert {row["query_source"] for row in natural_audit} == {"natural"}
    assert {row["generator_id"] for row in synthetic_audit} == {"W05"}


def test_filtered_preparation_bulk_scores_across_examples(tmp_path: Path) -> None:
    calibration = _load_calibration(tmp_path)
    scorer = _CountingPrimary()
    records = [_probe_record(), {**_probe_record(), "example_id": "q-2"}]
    rows, _, _, _ = prepare_probe_pairs(
        records,
        query_source="natural",
        negative_recipe=_recipe(tmp_path),
        calibration=calibration,
        primary_scorer=scorer,
    )
    assert len(rows) == 2
    assert scorer.calls == 1
    assert scorer.pairs == 4


def test_hn1_bm25_contract_is_deterministic_and_fingerprint_pinned(tmp_path: Path) -> None:
    documents = tmp_path / "documents.jsonl"
    rows = [
        {"doc_id": "d-000", "text": "stolica leży nad rzeką"},
        {"doc_id": "d-001", "text": "stolica oraz inne miasto"},
        {"doc_id": "d-002", "text": "zwykły negatyw"},
        {"doc_id": "d-003", "text": "kolejny dokument"},
    ]
    _write_jsonl(documents, rows)
    index_dir = tmp_path / "bm25"
    manifest = build_bm25_index(
        documents,
        output_dir=index_dir,
        config=BM25IndexConfig(relevance_score_threshold=0.0),
    )
    recipe = _recipe(
        tmp_path,
        strategy="hn1_bm25",
        bm25_index_fingerprint=str(manifest["index_fingerprint"]),
    )
    calibration = _load_calibration(tmp_path)
    with BM25CorpusIndex(index_dir) as index:
        first = prepare_probe_pairs(
            [_probe_record()],
            query_source="natural",
            negative_recipe=recipe,
            calibration=calibration,
            primary_scorer=_MockPrimary(),
            bm25_index=index,
            documents_path=documents,
        )
    with BM25CorpusIndex(index_dir) as index:
        second = prepare_probe_pairs(
            [_probe_record()],
            query_source="natural",
            negative_recipe=recipe,
            calibration=calibration,
            primary_scorer=_MockPrimary(),
            bm25_index=index,
            documents_path=documents,
        )
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert {row["miner"] for row in first[3]} == {"bm25"}


def test_probe_comparison_rejects_every_negative_contract_drift() -> None:
    base = {
        "negative_contract": {
            "probe_recipe_version": "probe-v1.1-p03",
            "probe_recipe_fingerprint": "p" * 64,
            "negative_recipe_version": NEGATIVE_RECIPE_VERSION,
            "hard_negative_strategy": "hn0_filter",
            "possible_false_negative_policy": "drop",
            "possible_false_negative_threshold": 5.0,
            "calibration_artifact_id": "cal",
            "calibration_artifact_fingerprint": "c" * 64,
            "bm25_index_fingerprint": None,
        }
    }
    assert assert_same_negative_contract(base, base)["possible_false_negative_threshold"] == 5.0
    for field in (
        "probe_recipe_version",
        "probe_recipe_fingerprint",
        "negative_recipe_version",
        "hard_negative_strategy",
        "possible_false_negative_policy",
        "possible_false_negative_threshold",
        "calibration_artifact_id",
        "calibration_artifact_fingerprint",
        "bm25_index_fingerprint",
    ):
        changed = json.loads(json.dumps(base))
        changed["negative_contract"][field] = "different"
        with pytest.raises(ValueError, match=field):
            assert_same_negative_contract(base, changed)


def test_probe_script_preflight_blocks_before_model_loading(tmp_path: Path) -> None:
    output = tmp_path / "blocked"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_probe_embedder.py",
            "--recipe",
            "configs/evaluation/probe_v1.yaml",
            "--train-input",
            str(tmp_path / "unused.jsonl"),
            "--frozen-manifest",
            str(tmp_path / "unused-manifest.json"),
            "--corpus",
            str(tmp_path / "unused-corpus.jsonl"),
            "--query-source",
            "natural",
            "--output-dir",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    blocker = json.loads((output / "p03_preflight.json").read_text())
    assert blocker["status"] == "blocked"
    assert blocker["models_loaded"] is False
    assert blocker["tests_used_for_threshold_tuning"] == []
