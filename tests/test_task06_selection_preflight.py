from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import doc2query.preferences.selection_preflight as preflight_module
from doc2query.preferences.selection_preflight import (
    CandidateSelectionPolicyManifest,
    SelectionComponent,
    prepare_preference_selection_preflight,
)
from doc2query.training.dpo import canonical_fingerprint, file_sha256, ordered_ids_fingerprint

HEX = {letter: letter * 64 for letter in "abcdef"}
COMMIT = "1" * 40


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _identity() -> dict[str, Any]:
    return {
        "candidate_id": "candidate-1",
        "request_id": "request-1",
        "plan_id": "plan-v1",
        "plan_fingerprint": HEX["a"],
        "passage_id": "passage-1",
        "passage_cluster_id": "cluster-1",
        "passage": "Pasaż zawiera sprawdzalny fakt.",
        "split": "train",
    }


def _candidate_bundle() -> dict[str, Any]:
    identity = _identity()
    candidate = {
        **identity,
        "prompt": "Pasaż: sprawdzalny fakt.\nZapytanie:",
        "query": "Jaki fakt opisano w pasażu?",
        "control": {
            "form": "full_question",
            "intent": "fact_lookup",
            "focus_mode": "bucket",
            "focus_bucket": "middle",
            "length": "medium",
        },
        "provenance": {
            "model_id": "generator/model",
            "model_revision": "generator-rev",
            "checkpoint_id": "checkpoint-v1",
            "checkpoint_fingerprint": "checkpoint-sha256",
            "adapter_id": "adapter-v1",
            "adapter_fingerprint": "adapter-sha256",
            "plan_id": "plan-v1",
            "plan_fingerprint": HEX["a"],
            "decoding": {
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.95,
                "max_new_tokens": 64,
                "seed": 42,
            },
        },
        "format_valid": True,
        "duplicate_within_request": False,
        "duplicate_candidate_ids": [],
    }
    return {
        "contract_version": "task06-candidate-evidence-v1",
        "candidate": candidate,
        "primary_judge": {
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
            "scoring_config_fingerprint": HEX["b"],
        },
        "shadow_judge": {
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
            "scoring_config_fingerprint": HEX["c"],
        },
        "corpus_retrieval": {
            **identity,
            "retriever_id": "bm25-v1",
            "retriever_revision": "rev-v1",
            "corpus_fingerprint": HEX["d"],
            "source_rank": 1,
            "candidate_count": 100,
            "reciprocal_rank": 1.0,
            "recall_at_1": True,
            "recall_at_5": True,
            "ndcg_at_10": 1.0,
        },
        "lexical_copy": {
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
            "method_id": "focus-v1",
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
            "classifier_id": "style-v1",
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


def _context() -> dict[str, Any]:
    candidate_ids = ["candidate-1"]
    return {
        "dataset_id": "dataset-v1",
        "dataset_fingerprint": HEX["a"],
        "split_id": "train",
        "split_fingerprint": canonical_fingerprint(
            {"split_id": "train", "candidate_ids": candidate_ids}
        ),
        "cohort_id": "candidate-cohort-v1",
        "cohort_fingerprint": canonical_fingerprint(
            [
                {
                    "candidate_id": "candidate-1",
                    "passage_id": "passage-1",
                    "passage_cluster_id": "cluster-1",
                }
            ]
        ),
        "candidate_ids_fingerprint": ordered_ids_fingerprint(candidate_ids),
        "candidate_count": 1,
    }


FIELDS = {
    "primary": "margin",
    "shadow": "margin",
    "corpus_retrieval": "reciprocal_rank",
    "lexical_copy": "content_lemma_jaccard",
    "focus": "focus_match",
    "style": "form_match",
    "format": "valid",
}


def _artifact(path: Path, source_manifest_sha: str) -> dict[str, Any]:
    payload = {
        "path": path.name,
        "sha256": file_sha256(path),
        "record_count": 1,
        "record_count_method": "jsonl",
        "provenance": {
            "source_task": "Task 06",
            "source_manifest_sha256": source_manifest_sha,
            "producer_git_commit": COMMIT,
        },
    }
    payload["artifact_fingerprint"] = canonical_fingerprint(payload)
    return payload


def _refingerprint(payload: dict[str, Any], key: str = "manifest_fingerprint") -> None:
    payload.pop(key, None)
    payload[key] = canonical_fingerprint(payload)


def _materialize(tmp_path: Path) -> dict[str, Any]:
    evidence = tmp_path / "candidate_evidence.jsonl"
    _write_jsonl(evidence, [_candidate_bundle()])
    evidence_manifest = tmp_path / "candidate_evidence_manifest.json"
    _write_json(
        evidence_manifest,
        {
            "contract_versions": {"bundle": "task06-candidate-evidence-v1"},
            "status": "evidence_assembled_not_ranked",
            "output_sha256": file_sha256(evidence),
            "artifact_fingerprint": file_sha256(evidence),
            "counts": {"complete": 1, "missing": 0, "orphan": 0, "duplicate": 0},
            "model_scoring_performed_by_assembler": False,
            "final_tests_used": [],
        },
    )
    evidence_manifest_sha = file_sha256(evidence_manifest)
    context = _context()
    calibration_paths: list[Path] = []
    calibration_payloads: dict[str, dict[str, Any]] = {}
    for index, component in enumerate(FIELDS):
        artifact_path = tmp_path / f"{component}_calibration.jsonl"
        _write_jsonl(artifact_path, [{"calibration_record_id": f"record-{index}"}])
        metric = {
            "metric_id": f"{component}-metric-v1",
            "source_field": FIELDS[component],
            "direction": "max",
            "normalization_definition_fingerprint": HEX["d"],
            "threshold_definition_fingerprint": HEX["e"],
            "calibration_fingerprint": HEX["f"],
        }
        payload = {
            "schema_version": 1,
            "contract": "task06-component-calibration-evidence-v1",
            "calibration_id": f"{component}-calibration-v1",
            "component": component,
            "context": context,
            "calibration_cohort_id": "calibration-cohort-v1",
            "calibration_cohort_fingerprint": HEX["f"],
            "evidence_definition_fingerprint": HEX["b"],
            "metrics": [metric],
            "source_candidate_evidence_manifest_sha256": evidence_manifest_sha,
            "producer_git_commit": COMMIT,
            "artifact": _artifact(artifact_path, evidence_manifest_sha),
            "final_tests_used": [],
        }
        _refingerprint(payload)
        manifest_path = tmp_path / f"{component}_calibration_manifest.json"
        _write_json(manifest_path, payload)
        calibration_paths.append(manifest_path)
        calibration_payloads[component] = payload

    human_artifact = tmp_path / "human_panel.jsonl"
    _write_jsonl(human_artifact, [{"blind_record_id": "blind-1", "preference": "A"}])
    human = {
        "schema_version": 1,
        "contract": "task06-human-preference-calibration-evidence-v1",
        "panel_id": "blind-panel-v1",
        "panel_version": "1.0.0",
        "blinded": True,
        "context": context,
        "panel_cohort_fingerprint": HEX["d"],
        "annotator_protocol_fingerprint": HEX["e"],
        "criteria_definition_fingerprint": HEX["f"],
        "source_candidate_evidence_manifest_sha256": evidence_manifest_sha,
        "producer_git_commit": COMMIT,
        "artifact": _artifact(human_artifact, evidence_manifest_sha),
        "sample_size": 1,
        "agreement": {
            "metric_id": "krippendorff-alpha-v1",
            "definition_fingerprint": HEX["a"],
            "value": 0.8,
            "sample_size": 1,
            "ci": {"lower": 0.7, "upper": 0.9, "confidence_level": 0.95},
        },
        "final_tests_used": [],
    }
    _refingerprint(human)
    human_path = tmp_path / "human_manifest.json"
    _write_json(human_path, human)

    components = {}
    for component, calibration in calibration_payloads.items():
        components[component] = {
            "component": component,
            "evidence_definition_fingerprint": HEX["b"],
            "calibration_manifest_fingerprint": calibration["manifest_fingerprint"],
            "metrics": [
                {
                    "metric_id": f"{component}-metric-v1",
                    "source_field": FIELDS[component],
                    "direction": "max",
                    "normalization": {
                        "method_id": "externally-frozen-normalizer-v1",
                        "definition_fingerprint": HEX["d"],
                        "parameters": {"center": 0.0, "scale": 1.0},
                    },
                    "weight": 1.0,
                    "thresholds": {"accept": 0.5},
                    "threshold_definition_fingerprint": HEX["e"],
                    "calibration_fingerprint": HEX["f"],
                }
            ],
        }
    policy = {
        "schema_version": 1,
        "contract": "task06-candidate-selection-policy-v1",
        "policy_id": "selection-policy-v1",
        "policy_version": "1.0.0",
        "status": "policy_frozen_not_applied",
        "candidate_evidence_bundle_sha256": file_sha256(evidence),
        "candidate_evidence_bundle_fingerprint": file_sha256(evidence),
        "candidate_evidence_manifest_sha256": evidence_manifest_sha,
        "context": context,
        "components": components,
        "pairing": {
            "minimum_score_margin": 0.25,
            "margin_definition_fingerprint": HEX["a"],
            "near_miss": {
                "definition_id": "near-miss-v1",
                "definition_fingerprint": HEX["b"],
                "lower_quantile": 0.5,
                "upper_quantile": 0.8,
            },
            "bottom": {
                "definition_id": "bottom-v1",
                "definition_fingerprint": HEX["c"],
                "lower_quantile": 0.0,
                "upper_quantile": 0.2,
            },
            "max_pairs_per_passage": 1,
            "max_rejected_per_chosen": 2,
        },
        "human_calibration_manifest_fingerprint": human["manifest_fingerprint"],
        "producer_git_commit": COMMIT,
        "final_tests_used": [],
    }
    _refingerprint(policy)
    policy_path = tmp_path / "selection_policy.json"
    _write_json(policy_path, policy)
    return {
        "candidate_evidence_path": evidence,
        "candidate_evidence_manifest_path": evidence_manifest,
        "policy_manifest_path": policy_path,
        "calibration_manifest_paths": calibration_paths,
        "human_manifest_path": human_path,
        "output_dir": tmp_path / "preflight_bundle",
        "policy": policy,
    }


def _run(inputs: dict[str, Any], *, output: Path | None = None) -> Any:
    kwargs = {key: value for key, value in inputs.items() if key != "policy"}
    if output is not None:
        kwargs["output_dir"] = output
    return prepare_preference_selection_preflight(**kwargs)


def _contains_selection_key(value: Any) -> bool:
    forbidden = {"total_score", "rank", "chosen", "rejected"}
    if isinstance(value, dict):
        return bool(forbidden & set(value)) or any(
            _contains_selection_key(item) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_selection_key(item) for item in value)
    return False


def test_complete_evidence_is_deterministic_and_never_selects(tmp_path: Path) -> None:
    inputs = _materialize(tmp_path)
    first = _run(inputs)
    first_payload = json.loads((inputs["output_dir"] / "manifest.json").read_text())
    second_output = tmp_path / "preflight_bundle_2"
    second = _run(inputs, output=second_output)
    second_payload = json.loads((second_output / "manifest.json").read_text())
    assert first.bundle_fingerprint == second.bundle_fingerprint
    assert first_payload == second_payload
    preflight = json.loads((inputs["output_dir"] / "preflight.json").read_text())
    assert preflight["component_coverage"] == [component.value for component in SelectionComponent]
    assert preflight["selection_outputs"] == []
    assert not _contains_selection_key(preflight)


def test_missing_calibration_and_human_evidence_are_reported(tmp_path: Path) -> None:
    inputs = _materialize(tmp_path)
    inputs["calibration_manifest_paths"] = []
    with pytest.raises(ValueError, match="missing calibration evidence"):
        _run(inputs)
    inputs = _materialize(tmp_path / "human")
    inputs["human_manifest_path"] = tmp_path / "human" / "missing-human.json"
    with pytest.raises(ValueError, match="human calibration evidence"):
        _run(inputs)


@pytest.mark.parametrize("drift", ["sha256", "record_count", "fingerprint", "provenance"])
def test_hash_count_fingerprint_and_provenance_drift_are_rejected(
    tmp_path: Path, drift: str
) -> None:
    inputs = _materialize(tmp_path)
    calibration_path = inputs["calibration_manifest_paths"][0]
    payload = json.loads(calibration_path.read_text())
    if drift == "sha256":
        payload["artifact"]["sha256"] = HEX["a"]
    elif drift == "record_count":
        payload["artifact"]["record_count"] = 2
    elif drift == "fingerprint":
        payload["evidence_definition_fingerprint"] = HEX["e"]
    else:
        payload["artifact"]["provenance"]["producer_git_commit"] = "2" * 40
    if drift in {"sha256", "record_count"}:
        artifact = payload["artifact"]
        artifact.pop("artifact_fingerprint")
        artifact["artifact_fingerprint"] = canonical_fingerprint(artifact)
    _refingerprint(payload)
    _write_json(calibration_path, payload)
    with pytest.raises((ValueError, ValidationError), match=r"drift|fingerprint|record-count"):
        _run(inputs)


@pytest.mark.parametrize(
    "field",
    ["dataset_fingerprint", "split_fingerprint", "cohort_fingerprint", "candidate_ids_fingerprint"],
)
def test_dataset_split_cohort_and_candidate_id_drift_are_rejected(
    tmp_path: Path, field: str
) -> None:
    inputs = _materialize(tmp_path)
    calibration_path = inputs["calibration_manifest_paths"][0]
    payload = json.loads(calibration_path.read_text())
    payload["context"][field] = HEX["e"]
    _refingerprint(payload)
    _write_json(calibration_path, payload)
    with pytest.raises(ValueError, match="dataset/split/cohort or candidate-ID drift"):
        _run(inputs)


def test_missing_score_component_is_rejected(tmp_path: Path) -> None:
    inputs = _materialize(tmp_path)
    inputs["calibration_manifest_paths"] = inputs["calibration_manifest_paths"][:-1]
    with pytest.raises(ValueError, match="component coverage mismatch"):
        _run(inputs)


def test_missing_candidate_score_component_is_rejected(tmp_path: Path) -> None:
    inputs = _materialize(tmp_path)
    bundle = _candidate_bundle()
    bundle.pop("shadow_judge")
    _write_jsonl(inputs["candidate_evidence_path"], [bundle])
    with pytest.raises(ValidationError, match="shadow_judge"):
        _run(inputs)


@pytest.mark.parametrize(
    ("field", "value"),
    [("weight", float("nan")), ("weight", float("inf")), ("threshold", float("-inf"))],
)
def test_non_finite_weights_and_thresholds_are_rejected(field: str, value: float) -> None:
    payload = _materialize_policy_payload()
    metric = payload["components"]["primary"]["metrics"][0]
    if field == "weight":
        metric["weight"] = value
    else:
        metric["thresholds"]["accept"] = value
    _refingerprint(payload)
    with pytest.raises(ValidationError, match="finite"):
        CandidateSelectionPolicyManifest.model_validate(payload)


def _materialize_policy_payload() -> dict[str, Any]:
    context = _context()
    components = {
        component: {
            "component": component,
            "evidence_definition_fingerprint": HEX["b"],
            "calibration_manifest_fingerprint": HEX["c"],
            "metrics": [
                {
                    "metric_id": f"{component}-metric-v1",
                    "source_field": field,
                    "direction": "max",
                    "normalization": {
                        "method_id": "normalizer-v1",
                        "definition_fingerprint": HEX["d"],
                        "parameters": {"center": 0.0},
                    },
                    "weight": 1.0,
                    "thresholds": {"accept": 0.5},
                    "threshold_definition_fingerprint": HEX["e"],
                    "calibration_fingerprint": HEX["f"],
                }
            ],
        }
        for component, field in FIELDS.items()
    }
    payload = {
        "schema_version": 1,
        "contract": "task06-candidate-selection-policy-v1",
        "policy_id": "policy-v1",
        "policy_version": "1",
        "status": "policy_frozen_not_applied",
        "candidate_evidence_bundle_sha256": HEX["a"],
        "candidate_evidence_bundle_fingerprint": HEX["a"],
        "candidate_evidence_manifest_sha256": HEX["b"],
        "context": context,
        "components": components,
        "pairing": {
            "minimum_score_margin": 0.25,
            "margin_definition_fingerprint": HEX["a"],
            "near_miss": {
                "definition_id": "near",
                "definition_fingerprint": HEX["b"],
                "lower_quantile": 0.5,
                "upper_quantile": 0.8,
            },
            "bottom": {
                "definition_id": "bottom",
                "definition_fingerprint": HEX["c"],
                "lower_quantile": 0.0,
                "upper_quantile": 0.2,
            },
            "max_pairs_per_passage": 1,
            "max_rejected_per_chosen": 2,
        },
        "human_calibration_manifest_fingerprint": HEX["d"],
        "producer_git_commit": COMMIT,
        "final_tests_used": [],
    }
    _refingerprint(payload)
    return payload


def test_incomparable_metric_definitions_are_rejected(tmp_path: Path) -> None:
    inputs = _materialize(tmp_path)
    path = inputs["calibration_manifest_paths"][0]
    payload = json.loads(path.read_text())
    payload["metrics"][0]["direction"] = "min"
    _refingerprint(payload)
    _write_json(path, payload)
    policy = json.loads(inputs["policy_manifest_path"].read_text())
    policy["components"]["primary"]["calibration_manifest_fingerprint"] = payload[
        "manifest_fingerprint"
    ]
    _refingerprint(policy)
    _write_json(inputs["policy_manifest_path"], policy)
    with pytest.raises(ValueError, match="not comparable"):
        _run(inputs)


def test_final_test_path_is_rejected_before_read(monkeypatch: pytest.MonkeyPatch) -> None:
    forbidden = Path("results/final_test/candidate_evidence.jsonl")

    def forbidden_read(*_args: Any, **_kwargs: Any) -> str:
        raise AssertionError("final test was read")

    monkeypatch.setattr(Path, "read_text", forbidden_read)
    with pytest.raises(ValueError, match="was not opened"):
        prepare_preference_selection_preflight(
            candidate_evidence_path=forbidden,
            candidate_evidence_manifest_path=Path("evidence.json"),
            policy_manifest_path=Path("policy.json"),
            calibration_manifest_paths=[Path("calibration.json")],
            human_manifest_path=Path("human.json"),
            output_dir=Path("output"),
        )


def test_overwrite_refusal_and_staging_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _materialize(tmp_path)
    inputs["output_dir"].mkdir()
    with pytest.raises(FileExistsError):
        _run(inputs)
    inputs["output_dir"].rmdir()

    def fail_replace(_source: Path, _target: Path) -> None:
        raise OSError("synthetic publish failure")

    monkeypatch.setattr("doc2query.preferences.selection_preflight.os.replace", fail_replace)
    with pytest.raises(OSError, match="synthetic publish failure"):
        _run(inputs)
    assert not inputs["output_dir"].exists()
    assert not list(tmp_path.glob(".preflight_bundle.staging-*"))


def test_module_and_script_have_no_model_or_reranker_imports() -> None:
    paths = [Path(preflight_module.__file__), Path("scripts/prepare_task06_selection_preflight.py")]
    forbidden = {"torch", "transformers", "tokenizers", "trl", "peft", "reranker"}
    imported: set[str] = set()
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
                imported.update(part for part in forbidden if part in node.module.split("."))
    assert imported.isdisjoint(forbidden)


def test_all_execution_flags_remain_false(tmp_path: Path) -> None:
    inputs = _materialize(tmp_path)
    manifest = _run(inputs).model_dump(mode="json")
    preflight = json.loads((inputs["output_dir"] / "preflight.json").read_text())
    fields = (
        "generation_started",
        "scoring_started",
        "calibration_computed",
        "selection_started",
        "preferences_built",
        "model_loading_performed",
    )
    for payload in (manifest, preflight):
        assert all(payload[field] is False for field in fields)
        assert payload["final_tests_used"] == []
