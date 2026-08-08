from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from doc2query.preferences.evidence import (
    assemble_candidate_evidence,
    validate_generated_candidates,
)
from doc2query.utils.records import JsonParquetWriter, read_records

PLAN_FINGERPRINT = "a" * 64


def _request(index: int, *, split: str = "train", cluster: str | None = None) -> dict[str, Any]:
    return {
        "request_id": f"request-{index}",
        "plan_id": "plan-v1",
        "plan_fingerprint": PLAN_FINGERPRINT,
        "candidate_index": index,
        "passage_id": f"passage-{index}",
        "passage_cluster_id": cluster or f"cluster-{index}",
        "passage": f"Pasaż numer {index} zawiera sprawdzalny fakt.",
        "source_pair_ids": [f"pair-{index}"],
        "split": split,
        "prompt": f"Pasaż: {index}\nZapytanie:",
        "control": {
            "form": "full_question",
            "intent": "fact_lookup",
            "focus_mode": "bucket",
            "focus_bucket": "middle",
            "length": "medium",
        },
        "temperature": 0.7,
        "top_p": 0.95,
        "max_new_tokens": 64,
        "seed": 42 + index,
    }


def _candidate(request: dict[str, Any], *, query: str | None = None) -> dict[str, Any]:
    index = int(str(request["request_id"]).split("-")[-1])
    return {
        "candidate_id": f"candidate-{index}",
        "request_id": request["request_id"],
        "plan_id": request["plan_id"],
        "plan_fingerprint": request["plan_fingerprint"],
        "passage_id": request["passage_id"],
        "passage_cluster_id": request["passage_cluster_id"],
        "passage": request["passage"],
        "split": request["split"],
        "prompt": request["prompt"],
        "query": query or f"Jakiego faktu dotyczy pasaż numer {index}?",
        "control": request["control"],
        "provenance": {
            "model_id": "generator/model",
            "model_revision": "generator-rev",
            "checkpoint_id": "checkpoint-v1",
            "checkpoint_fingerprint": "checkpoint-sha256",
            "adapter_id": "adapter-v1",
            "adapter_fingerprint": "adapter-sha256",
            "plan_id": request["plan_id"],
            "plan_fingerprint": request["plan_fingerprint"],
            "decoding": {
                "do_sample": True,
                "temperature": request["temperature"],
                "top_p": request["top_p"],
                "max_new_tokens": request["max_new_tokens"],
                "seed": request["seed"],
                "implementation_parameters": {"renormalize_logits": True},
            },
        },
        "attempt": 1,
        "format_valid": True,
        "duplicate_within_request": False,
        "duplicate_candidate_ids": [],
    }


def _identity(request: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": candidate["candidate_id"],
        "request_id": request["request_id"],
        "plan_id": request["plan_id"],
        "plan_fingerprint": request["plan_fingerprint"],
        "passage_id": request["passage_id"],
        "passage_cluster_id": request["passage_cluster_id"],
        "passage": request["passage"],
        "split": request["split"],
    }


def _evidence(request: dict[str, Any], candidate: dict[str, Any]) -> dict[str, dict[str, Any]]:
    identity = _identity(request, candidate)
    return {
        "primary": {
            **identity,
            "judge_role": "primary",
            "judge_id": "primary/model",
            "judge_revision": "primary-rev",
            "raw_score_scale_id": "primary-logit-v1",
            "positive_score": 3.0,
            "max_negative_score": 1.5,
            "margin": 1.5,
            "positive_rank": 1,
            "candidate_count": 11,
            "best_sentence_score": 2.5,
            "all_scores_close": False,
            "scoring_config_fingerprint": "primary-config-sha",
        },
        "shadow": {
            **identity,
            "judge_role": "shadow",
            "judge_id": "shadow/model",
            "judge_revision": "shadow-rev",
            "raw_score_scale_id": "shadow-logit-v1",
            "positive_score": 0.8,
            "max_negative_score": 0.3,
            "margin": 0.5,
            "positive_rank": 1,
            "candidate_count": 11,
            "best_sentence_score": 0.7,
            "all_scores_close": False,
            "scoring_config_fingerprint": "shadow-config-sha",
        },
        "corpus": {
            **identity,
            "retriever_id": "bm25-v1",
            "retriever_revision": "corpus-code-rev",
            "corpus_fingerprint": "corpus-sha",
            "source_rank": 1,
            "candidate_count": 100,
            "reciprocal_rank": 1.0,
            "recall_at_1": True,
            "recall_at_5": True,
            "ndcg_at_10": 1.0,
        },
        "lexical": {
            **identity,
            "content_lemma_jaccard": 0.2,
            "content_lemma_precision": 0.5,
            "content_lemma_recall": 0.3,
            "longest_common_ngram": 2,
            "longest_common_subsequence_ratio": 0.2,
            "entity_preservation": None,
            "number_unit_preservation": 1.0,
            "copy_risk": False,
            "normalization_version": "simple-pl-v1",
        },
        "focus": {
            **identity,
            "requested_focus_mode": "bucket",
            "requested_focus_bucket": "middle",
            "requested_focus_sentence_id": None,
            "assigned_focus_bucket": "middle",
            "assigned_focus_sentence_id": 0,
            "focus_match": True,
            "confidence": 0.8,
            "method_id": "sentence-proxy-v1",
        },
        "style": {
            **identity,
            "requested_form": "full_question",
            "requested_intent": "fact_lookup",
            "predicted_form": "full_question",
            "predicted_intent": "fact_lookup",
            "form_match": True,
            "intent_match": True,
            "confidence": 0.9,
            "classifier_id": "rules-v1",
        },
        "format": {
            **identity,
            "valid": True,
            "empty": False,
            "single_query": True,
            "has_meta_commentary": False,
            "too_long": False,
            "contains_answer": False,
            "violation_codes": [],
            "validator_version": "format-v1",
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _artifacts(tmp_path: Path, count: int = 2) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = {
        name: tmp_path / f"{name}.jsonl"
        for name in (
            "requests",
            "candidates",
            "primary",
            "shadow",
            "corpus",
            "lexical",
            "focus",
            "style",
            "format",
        )
    }
    requests = [_request(index) for index in range(count)]
    candidates = [_candidate(row) for row in requests]
    components = [
        _evidence(request, candidate)
        for request, candidate in zip(requests, candidates, strict=True)
    ]
    _write_jsonl(paths["requests"], requests)
    _write_jsonl(paths["candidates"], candidates)
    for label in ("primary", "shadow", "corpus", "lexical", "focus", "style", "format"):
        _write_jsonl(paths[label], [row[label] for row in components])
    return paths


def _assemble(paths: dict[str, Path], output_dir: Path) -> dict[str, Any]:
    return assemble_candidate_evidence(
        requests_path=paths["requests"],
        candidates_path=paths["candidates"],
        primary_path=paths["primary"],
        shadow_path=paths["shadow"],
        corpus_path=paths["corpus"],
        lexical_path=paths["lexical"],
        focus_path=paths["focus"],
        style_path=paths["style"],
        format_path=paths["format"],
        output_path=output_dir / "evidence.jsonl",
        manifest_path=output_dir / "manifest.json",
        primary_judge_id="primary/model",
        primary_judge_revision="primary-rev",
        shadow_judge_id="shadow/model",
        shadow_judge_revision="shadow-rev",
    )


def test_assembly_is_deterministic_and_never_computes_total_score(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path / "inputs")
    first = _assemble(paths, tmp_path / "first")
    second = _assemble(paths, tmp_path / "second")
    assert (tmp_path / "first/evidence.jsonl").read_bytes() == (
        tmp_path / "second/evidence.jsonl"
    ).read_bytes()
    assert first["artifact_fingerprint"] == second["artifact_fingerprint"]
    assert first["status"] == "evidence_assembled_not_ranked"
    assert first["model_scoring_performed_by_assembler"] is False
    assert first["final_tests_used"] == []
    assert first["counts"] == {"complete": 2, "missing": 0, "orphan": 0, "duplicate": 0}
    rows = list(read_records(tmp_path / "first/evidence.jsonl"))
    assert "total_score" not in json.dumps(rows)


def test_assembler_accepts_mixed_jsonl_parquet_and_writes_parquet(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    paths = _artifacts(tmp_path / "inputs", count=1)
    for label in ("candidates", "primary"):
        parquet_path = paths[label].with_suffix(".parquet")
        with JsonParquetWriter(parquet_path) as writer:
            for row in read_records(paths[label]):
                writer.write(row)
        paths[label] = parquet_path
    manifest = assemble_candidate_evidence(
        requests_path=paths["requests"],
        candidates_path=paths["candidates"],
        primary_path=paths["primary"],
        shadow_path=paths["shadow"],
        corpus_path=paths["corpus"],
        lexical_path=paths["lexical"],
        focus_path=paths["focus"],
        style_path=paths["style"],
        format_path=paths["format"],
        output_path=tmp_path / "output/evidence.parquet",
        manifest_path=tmp_path / "output/manifest.json",
        primary_judge_id="primary/model",
        primary_judge_revision="primary-rev",
        shadow_judge_id="shadow/model",
        shadow_judge_revision="shadow-rev",
    )
    assert len(list(read_records(tmp_path / "output/evidence.parquet"))) == 1
    assert manifest["counts"]["complete"] == 1


@pytest.mark.parametrize("mode", ["missing", "orphan"])
def test_rejects_missing_and_orphan_evidence(tmp_path: Path, mode: str) -> None:
    paths = _artifacts(tmp_path)
    rows = list(read_records(paths["focus"]))
    if mode == "missing":
        rows.pop()
    else:
        orphan = copy.deepcopy(rows[0])
        orphan["candidate_id"] = "orphan"
        rows.append(orphan)
    _write_jsonl(paths["focus"], rows)
    with pytest.raises(ValueError, match=mode):
        _assemble(paths, tmp_path / "output")


def test_rejects_duplicate_ids(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path)
    rows = list(read_records(paths["primary"]))
    _write_jsonl(paths["primary"], [*rows, rows[0]])
    with pytest.raises(ValueError, match="duplicate candidate_id"):
        _assemble(paths, tmp_path / "output")


@pytest.mark.parametrize(
    ("target", "field", "value", "message"),
    [
        ("candidates", "plan_id", "drifted-plan", "plan identity"),
        ("plan_fingerprint", "plan_fingerprint", "b" * 64, "request/candidate drift"),
        ("provenance", "checkpoint_fingerprint", "drifted-checkpoint", "checkpoint"),
        ("primary", "judge_revision", "drifted-primary", "primary judge ID/revision drift"),
        ("shadow", "judge_revision", "drifted-shadow", "shadow judge ID/revision drift"),
    ],
)
def test_rejects_plan_checkpoint_and_judge_revision_drift(
    tmp_path: Path, target: str, field: str, value: str, message: str
) -> None:
    paths = _artifacts(tmp_path)
    if target == "plan_fingerprint":
        rows = list(read_records(paths["candidates"]))
        rows[1][field] = value
        rows[1]["provenance"][field] = value
        path = paths["candidates"]
    elif target == "provenance":
        rows = list(read_records(paths["candidates"]))
        rows[1]["provenance"][field] = value
        path = paths["candidates"]
    else:
        path = paths[target]
        rows = list(read_records(path))
        rows[1][field] = value
    _write_jsonl(path, rows)
    with pytest.raises(ValueError, match=message):
        _assemble(paths, tmp_path / "output")


def test_rejects_incorrect_stored_margin_and_mixed_scales(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path / "margin")
    rows = list(read_records(paths["primary"]))
    rows[0]["margin"] = 999.0
    _write_jsonl(paths["primary"], rows)
    with pytest.raises(ValueError, match="incorrect primary margin"):
        _assemble(paths, tmp_path / "margin-output")

    paths = _artifacts(tmp_path / "scales")
    shadow = list(read_records(paths["shadow"]))
    shadow[0]["raw_score_scale_id"] = "primary-logit-v1"
    _write_jsonl(paths["shadow"], shadow)
    with pytest.raises(ValueError, match="raw score scales are mixed"):
        _assemble(paths, tmp_path / "scale-output")


def test_rejects_passage_and_cluster_split_leakage(tmp_path: Path) -> None:
    requests = [
        _request(0, split="train", cluster="shared"),
        _request(1, split="dev", cluster="shared"),
    ]
    candidates = [_candidate(row) for row in requests]
    with pytest.raises(ValueError, match="cluster shared crosses"):
        validate_generated_candidates(requests, candidates)
    candidates = [_candidate(_request(0))]
    candidates[0]["passage"] = "Podmieniony pasaż"
    with pytest.raises(ValueError, match="passage"):
        validate_generated_candidates([_request(0)], candidates)


def test_absolutely_rejects_test_split_and_normalized_duplicates() -> None:
    test_request = _request(0, split="test")
    with pytest.raises(ValidationError, match="split"):
        validate_generated_candidates([test_request], [_candidate(test_request)])
    requests = [_request(0), _request(1)]
    candidates = [
        _candidate(requests[0], query="Żółta Łódź?"),
        _candidate(requests[1], query="zolta lodz"),
    ]
    with pytest.raises(ValueError, match="duplicate query after normalization"):
        validate_generated_candidates(requests, candidates)
