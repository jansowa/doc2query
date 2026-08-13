from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import doc2query.training.comparison_preflight as comparison_module
from doc2query.training.comparison_preflight import prepare_task07_comparison_preflight
from doc2query.training.dpo import DPOArm, canonical_fingerprint, file_sha256

HEX = {letter: letter * 64 for letter in "abcdef"}
COMMIT = "1" * 40
CATEGORIES = ("cost", "human", "intrinsic", "probe_extrinsic", "shadow_independent")
ROLES = {
    "cost": "cost_records",
    "human": "human_panel",
    "intrinsic": "primary_metrics",
    "probe_extrinsic": "probe_metrics",
    "shadow_independent": "shadow_metrics",
}


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


def _fingerprint_payload(payload: dict[str, Any], field: str) -> None:
    payload.pop(field, None)
    payload[field] = canonical_fingerprint(payload)


def _model_stack() -> dict[str, Any]:
    return {
        "base_model": {
            "model_id": "base/model",
            "revision": "base-rev",
            "artifact_fingerprint": HEX["a"],
        },
        "sft_adapter": {
            "adapter_id": "sft/adapter",
            "adapter_revision": "adapter-rev",
            "adapter_fingerprint": HEX["b"],
            "base_model_fingerprint": HEX["a"],
        },
        "tokenizer": {
            "tokenizer_id": "base/tokenizer",
            "revision": "tokenizer-rev",
            "tokenizer_fingerprint": HEX["c"],
        },
    }


def _provenance() -> dict[str, Any]:
    return {
        "dataset_id": "dataset-v1",
        "dataset_fingerprint": HEX["d"],
        "selection_policy_id": "selection-v1",
        "selection_policy_fingerprint": HEX["e"],
    }


def _preference(index: int, split: str) -> dict[str, Any]:
    suffix = f"{split}-{index}"
    return {
        "preference_id": f"pref-{suffix}",
        "prompt": f"Pasaż {suffix}\nZapytanie:",
        "chosen": f"Dobre pytanie {suffix}?",
        "rejected": f"Gorsze pytanie {suffix}?",
        "score_margin": 0.5,
        "chosen_candidate_id": f"chosen-{suffix}",
        "rejected_candidate_id": f"rejected-{suffix}",
        "passage_id": f"passage-{suffix}",
        "passage_cluster_id": f"cluster-{suffix}",
        "split": split,
        "provenance": _provenance(),
    }


def _control(row: dict[str, Any], *, weighted: bool) -> dict[str, Any]:
    result = {
        "preference_id": row["preference_id"],
        "prompt": row["prompt"],
        "completion": row["chosen"],
        "candidate_id": row["chosen_candidate_id"],
        "passage_id": row["passage_id"],
        "passage_cluster_id": row["passage_cluster_id"],
        "split": row["split"],
        "provenance": row["provenance"],
    }
    if weighted:
        result |= {
            "sample_weight": 1.25,
            "weight_policy_id": "weights-v1",
            "weight_policy_fingerprint": HEX["f"],
        }
    return result


def _budget(arm: str, *, tokens: int = 100_000) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "arm": arm,
        "cohort_fingerprint": HEX["a"],
        "seeds": [42, 43],
        "target_token_budget": tokens,
        "target_optimizer_steps": 250,
        "train_example_count": 2,
        "prompt_chosen_tokens_per_cohort": 40,
        "dpo_pair_tokens_per_cohort": 70 if arm == "dpo" else None,
        "weight_policy_id": "weights-v1" if arm == "score_weighted_continued_sft" else None,
        "weight_policy_fingerprint": (HEX["f"] if arm == "score_weighted_continued_sft" else None),
    }
    return payload


def _artifact(path: Path, *, launch_hash: str, count: int = 1) -> dict[str, Any]:
    payload = {
        "path": path.name,
        "sha256": file_sha256(path),
        "record_count": count,
        "record_count_method": "jsonl" if path.suffix == ".jsonl" else "single_json",
        "provenance": {
            "source_task": "Task 07",
            "source_manifest_sha256": launch_hash,
            "producer_git_commit": COMMIT,
        },
    }
    payload["artifact_fingerprint"] = canonical_fingerprint(payload)
    return payload


@dataclass
class ComparisonFixture:
    root: Path
    protocol: Path
    handoff: Path
    selection: Path
    launch: Path
    outcomes: list[Path]

    def kwargs(self, output_dir: Path) -> dict[str, Any]:
        return {
            "protocol_manifest_path": self.protocol,
            "task06_handoff_manifest_path": self.handoff,
            "task06_selection_preflight_manifest_path": self.selection,
            "task07_launch_manifest_path": self.launch,
            "outcome_manifest_paths": self.outcomes,
            "output_dir": output_dir,
        }


def _materialize_fixture(root: Path) -> ComparisonFixture:
    data_dir = root / "handoff"
    train = [_preference(0, "train"), _preference(1, "train")]
    dev = [_preference(0, "dev")]
    rows = {
        "preference_train": train,
        "preference_dev": dev,
        "continued_sft_train": [_control(row, weighted=False) for row in train],
        "continued_sft_dev": [_control(row, weighted=False) for row in dev],
        "weighted_sft_train": [_control(row, weighted=True) for row in train],
        "weighted_sft_dev": [_control(row, weighted=True) for row in dev],
    }
    for name, records in rows.items():
        _write_jsonl(data_dir / f"{name}.jsonl", records)
    handoff_payload: dict[str, Any] = {
        "schema_version": 1,
        "contract": "task06-preference-data-for-task07-v1",
        **_provenance(),
        "artifacts": {
            name: {
                "path": f"{name}.jsonl",
                "sha256": file_sha256(data_dir / f"{name}.jsonl"),
                "record_count": len(records),
            }
            for name, records in rows.items()
        },
        "automatic_thresholds_created": False,
        "relabeling_performed": False,
        "final_tests_used": [],
    }
    _fingerprint_payload(handoff_payload, "manifest_fingerprint")
    handoff = data_dir / "manifest.json"
    _write_json(handoff, handoff_payload)

    selection_dir = root / "selection"
    preflight_payload = {"candidate_ids": ["candidate-1"]}
    _write_json(selection_dir / "preflight.json", preflight_payload)
    selection_payload: dict[str, Any] = {
        "schema_version": 1,
        "contract": "task06-preference-selection-preflight-bundle-v1",
        "status": "ready_for_future_preference_selection_not_selected",
        "policy_id": "selection-v1",
        "policy_fingerprint": HEX["e"],
        "context": {
            "dataset_id": "dataset-v1",
            "dataset_fingerprint": HEX["d"],
            "split_id": "train",
            "split_fingerprint": HEX["a"],
            "cohort_id": "candidate-cohort-v1",
            "cohort_fingerprint": HEX["b"],
            "candidate_ids_fingerprint": HEX["c"],
            "candidate_count": 1,
        },
        "preflight": {
            "path": "preflight.json",
            "sha256": file_sha256(selection_dir / "preflight.json"),
            "candidate_count": 1,
            "component_count": 7,
        },
        "generation_started": False,
        "scoring_started": False,
        "calibration_computed": False,
        "selection_started": False,
        "preferences_built": False,
        "model_loading_performed": False,
        "final_tests_used": [],
    }
    _fingerprint_payload(selection_payload, "bundle_fingerprint")
    selection = selection_dir / "manifest.json"
    _write_json(selection, selection_payload)

    launch_payload: dict[str, Any] = {
        "schema_version": 1,
        "contract": "task07-model-free-launch-bundle-v1",
        "status": "ready_for_model_smoke_not_trained",
        "plan_id": "plan-v1",
        "plan_fingerprint": HEX["b"],
        "dataset_id": "dataset-v1",
        "dataset_fingerprint": HEX["d"],
        "selection_policy_id": "selection-v1",
        "selection_policy_fingerprint": HEX["e"],
        "weight_policy_id": "weights-v1",
        "weight_policy_fingerprint": HEX["f"],
        "cohort_fingerprint": HEX["a"],
        "start_model": _model_stack(),
        "reference_model": _model_stack(),
        "arms": {
            arm.value: {
                "arm": arm.value,
                "train_input": {
                    DPOArm.DPO: "preference_train",
                    DPOArm.CONTINUED_SFT: "continued_sft_train",
                    DPOArm.SCORE_WEIGHTED_CONTINUED_SFT: "weighted_sft_train",
                }[arm],
                "dev_input": {
                    DPOArm.DPO: "preference_dev",
                    DPOArm.CONTINUED_SFT: "continued_sft_dev",
                    DPOArm.SCORE_WEIGHTED_CONTINUED_SFT: "weighted_sft_dev",
                }[arm],
                "budget": _budget(arm.value),
                "reference_logprobs_input": (
                    "reference_logprob_records" if arm == DPOArm.DPO else None
                ),
            }
            for arm in DPOArm
        },
        "input_hashes": {
            name: (file_sha256(handoff) if name == "task06_manifest" else HEX["c"])
            for name in (
                "task06_manifest",
                "preference_train",
                "preference_dev",
                "continued_sft_train",
                "continued_sft_dev",
                "weighted_sft_train",
                "weighted_sft_dev",
                "dpo_plan",
                "token_length_manifest",
                "token_length_records",
                "reference_logprob_manifest",
                "reference_logprob_records",
            )
        },
        "input_record_counts": {
            "preference_train": 2,
            "preference_dev": 1,
            "continued_sft_train": 2,
            "continued_sft_dev": 1,
            "weighted_sft_train": 2,
            "weighted_sft_dev": 1,
            "token_length_records": 3,
            "reference_logprob_records": 2,
        },
        "model_loading_performed": False,
        "tokenizer_loading_performed": False,
        "reference_logprobs_computed": False,
        "training_started": False,
        "evaluation_started": False,
        "final_tests_used": [],
    }
    _fingerprint_payload(launch_payload, "bundle_fingerprint")
    launch = root / "launch" / "manifest.json"
    _write_json(launch, launch_payload)
    launch_hash = file_sha256(launch)

    metrics = []
    for category in CATEGORIES:
        metrics.append(
            {
                "name": f"{category}_metric",
                "category": category,
                "direction": "min" if category == "cost" else "max",
                "unit": "seconds" if category == "cost" else "ratio",
                "definition_fingerprint": HEX["a"],
                "ci_definition_fingerprint": HEX["b"],
                "confidence_level": 0.95,
                "minimum_sample_size": 100,
                "ci_required": True,
                "sample_size_required": True,
            }
        )
    model_stack_fingerprint = canonical_fingerprint(_model_stack())
    config_comparison_fingerprints = {
        arm.value: canonical_fingerprint({"arm": arm.value, "learning_rate": 1e-5})
        for arm in DPOArm
    }
    protocol_payload: dict[str, Any] = {
        "schema_version": 1,
        "contract": "task07-comparison-protocol-v1",
        "status": "protocol_frozen_not_applied",
        "protocol_id": "task07-comparison-v1",
        "protocol_version": "1.0.0",
        "required_seeds": [42, 43],
        "arms": {
            arm.value: {
                "arm": arm.value,
                "config_comparison_fingerprint": config_comparison_fingerprints[arm.value],
                "budget": _budget(arm.value),
            }
            for arm in DPOArm
        },
        "task06_handoff": {
            "sha256": file_sha256(handoff),
            "fingerprint": handoff_payload["manifest_fingerprint"],
        },
        "task06_selection_preflight": {
            "sha256": file_sha256(selection),
            "fingerprint": selection_payload["bundle_fingerprint"],
        },
        "task07_launch_bundle": {
            "sha256": launch_hash,
            "fingerprint": launch_payload["bundle_fingerprint"],
        },
        "dataset": {"identity_id": "dataset-v1", "fingerprint": HEX["d"]},
        "selection_policy": {"identity_id": "selection-v1", "fingerprint": HEX["e"]},
        "weight_policy": {"identity_id": "weights-v1", "fingerprint": HEX["f"]},
        "cohort": {"identity_id": "cohort-v1", "fingerprint": HEX["a"]},
        "plan": {"identity_id": "plan-v1", "fingerprint": HEX["b"]},
        "model_stack": _model_stack(),
        "model_stack_fingerprint": model_stack_fingerprint,
        "tokenizer_fingerprint": HEX["c"],
        "required_metrics": metrics,
        "required_artifact_roles": {category: [ROLES[category]] for category in CATEGORIES},
        "guardrails": [
            {
                "guardrail_id": "grounding-floor",
                "metric_key": "intrinsic:intrinsic_metric",
                "operator": "ge",
                "threshold": 0.4,
                "definition_fingerprint": HEX["c"],
            }
        ],
        "final_tests_used": [],
    }
    _fingerprint_payload(protocol_payload, "manifest_fingerprint")
    protocol = root / "protocol.json"
    _write_json(protocol, protocol_payload)

    outcomes: list[Path] = []
    for arm in DPOArm:
        for seed in (42, 43):
            run_dir = root / "outcomes" / f"{arm.value}-{seed}"
            config = {"arm": arm.value, "learning_rate": 1e-5, "seed": seed}
            config_path = run_dir / "config.json"
            _write_json(config_path, config)
            evidence: dict[str, Any] = {}
            for category in CATEGORIES:
                artifact_path = run_dir / f"{ROLES[category]}.jsonl"
                _write_jsonl(artifact_path, [{"record_id": f"{category}-1"}])
                evidence[category] = {
                    "category": category,
                    "artifacts": {
                        ROLES[category]: _artifact(artifact_path, launch_hash=launch_hash)
                    },
                    "metrics": [
                        {
                            "name": f"{category}_metric",
                            "category": category,
                            "direction": "min" if category == "cost" else "max",
                            "value": 1.0,
                            "unit": "seconds" if category == "cost" else "ratio",
                            "definition_fingerprint": HEX["a"],
                            "ci_definition_fingerprint": HEX["b"],
                            "ci": {"lower": 0.9, "upper": 1.1, "confidence_level": 0.95},
                            "sample_size": 100,
                        }
                    ],
                }
            outcome_payload: dict[str, Any] = {
                "schema_version": 1,
                "contract": "task07-arm-outcome-evidence-v1",
                "run_id": f"{arm.value}-{seed}",
                "arm": arm.value,
                "seed": seed,
                "run_status": "completed",
                "producer_git_commit": COMMIT,
                "config": {
                    "artifact": _artifact(config_path, launch_hash=launch_hash),
                    "format": "json",
                    "fingerprint": canonical_fingerprint(config),
                    "comparison_fingerprint": config_comparison_fingerprints[arm.value],
                },
                "task06_handoff_fingerprint": handoff_payload["manifest_fingerprint"],
                "task06_selection_preflight_fingerprint": selection_payload["bundle_fingerprint"],
                "task07_launch_bundle_fingerprint": launch_payload["bundle_fingerprint"],
                "dataset": protocol_payload["dataset"],
                "selection_policy": protocol_payload["selection_policy"],
                "weight_policy": protocol_payload["weight_policy"],
                "cohort": protocol_payload["cohort"],
                "plan": protocol_payload["plan"],
                "model_stack": _model_stack(),
                "model_stack_fingerprint": model_stack_fingerprint,
                "tokenizer_fingerprint": HEX["c"],
                "budget": _budget(arm.value),
                "evidence": evidence,
                "final_tests_used": [],
            }
            _fingerprint_payload(outcome_payload, "manifest_fingerprint")
            outcome_path = run_dir / "manifest.json"
            _write_json(outcome_path, outcome_payload)
            outcomes.append(outcome_path)
    return ComparisonFixture(root, protocol, handoff, selection, launch, outcomes)


def _rewrite_outcome(path: Path, mutate: Any) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    _fingerprint_payload(payload, "manifest_fingerprint")
    _write_json(path, payload)


def _contains_forbidden_result(value: Any) -> bool:
    forbidden = {"ranking", "winner", "selected_arm", "promotion", "continue_stop_decision"}
    if isinstance(value, dict):
        return any(
            key in forbidden or _contains_forbidden_result(item) for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_result(item) for item in value)
    return False


@pytest.fixture
def comparison_files(tmp_path: Path) -> ComparisonFixture:
    return _materialize_fixture(tmp_path / "fixture")


def test_complete_evidence_is_deterministic_and_not_compared(
    comparison_files: ComparisonFixture, tmp_path: Path
) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = prepare_task07_comparison_preflight(**comparison_files.kwargs(first_dir))
    second_kwargs = comparison_files.kwargs(second_dir)
    second_kwargs["outcome_manifest_paths"] = list(reversed(comparison_files.outcomes))
    second = prepare_task07_comparison_preflight(**second_kwargs)
    assert first == second
    assert (first_dir / "comparison_preflight.json").read_bytes() == (
        second_dir / "comparison_preflight.json"
    ).read_bytes()
    assert first.status == "ready_for_future_task07_comparison_not_compared"
    payload = json.loads((first_dir / "comparison_preflight.json").read_text())
    assert not _contains_forbidden_result(payload)
    assert all("value" not in item for item in payload["evidence_index"])


@pytest.mark.parametrize("fault", ["missing", "extra"])
def test_missing_or_extra_arm_seed_fails_closed(
    comparison_files: ComparisonFixture, tmp_path: Path, fault: str
) -> None:
    paths = list(comparison_files.outcomes)
    if fault == "missing":
        paths.pop()
    else:
        source = paths[0]
        extra = source.parent.parent / "extra" / "manifest.json"
        extra.parent.mkdir()
        extra.write_bytes(source.read_bytes())
        _rewrite_outcome(extra, lambda payload: payload.__setitem__("seed", 44))
        paths.append(extra)
    kwargs = comparison_files.kwargs(tmp_path / "output")
    kwargs["outcome_manifest_paths"] = paths
    with pytest.raises(ValueError, match="arm x seed coverage mismatch"):
        prepare_task07_comparison_preflight(**kwargs)


@pytest.mark.parametrize("fault", ["sha256", "record_count", "fingerprint", "provenance"])
def test_artifact_integrity_and_provenance_drift_fails_closed(
    comparison_files: ComparisonFixture, tmp_path: Path, fault: str
) -> None:
    outcome = comparison_files.outcomes[0]

    def mutate(payload: dict[str, Any]) -> None:
        artifact = payload["evidence"]["intrinsic"]["artifacts"]["primary_metrics"]
        if fault == "fingerprint":
            artifact["artifact_fingerprint"] = HEX["f"]
            return
        if fault == "provenance":
            artifact["provenance"]["source_manifest_sha256"] = HEX["f"]
        elif fault == "sha256":
            artifact["sha256"] = HEX["f"]
        else:
            artifact["record_count"] = 2
        descriptor = dict(artifact)
        descriptor.pop("artifact_fingerprint")
        artifact["artifact_fingerprint"] = canonical_fingerprint(descriptor)

    _rewrite_outcome(outcome, mutate)
    with pytest.raises(ValueError, match=r"fingerprint|SHA-256|record-count|provenance"):
        prepare_task07_comparison_preflight(**comparison_files.kwargs(tmp_path / "output"))


@pytest.mark.parametrize(
    "fault", ["handoff_hash", "handoff_count", "selection_hash", "launch_fingerprint"]
)
def test_source_manifest_integrity_drift_fails_closed(
    comparison_files: ComparisonFixture, tmp_path: Path, fault: str
) -> None:
    if fault == "handoff_hash":
        artifact = comparison_files.handoff.parent / "preference_train.jsonl"
        artifact.write_text(artifact.read_text() + " \n", encoding="utf-8")
    elif fault == "handoff_count":
        payload = json.loads(comparison_files.handoff.read_text())
        payload["artifacts"]["preference_train"]["record_count"] += 1
        _fingerprint_payload(payload, "manifest_fingerprint")
        _write_json(comparison_files.handoff, payload)
    elif fault == "selection_hash":
        artifact = comparison_files.selection.parent / "preflight.json"
        artifact.write_text(artifact.read_text() + " ", encoding="utf-8")
    else:
        payload = json.loads(comparison_files.launch.read_text())
        payload["bundle_fingerprint"] = HEX["f"]
        _write_json(comparison_files.launch, payload)
    with pytest.raises(ValueError, match=r"sha256|SHA-256|record count|fingerprint"):
        prepare_task07_comparison_preflight(**comparison_files.kwargs(tmp_path / "output"))


def test_config_drift_fails_closed(comparison_files: ComparisonFixture, tmp_path: Path) -> None:
    outcome = comparison_files.outcomes[0]
    payload = json.loads(outcome.read_text())
    config_path = outcome.parent / payload["config"]["artifact"]["path"]
    config = json.loads(config_path.read_text())
    config["learning_rate"] = 2e-5
    _write_json(config_path, config)
    config_artifact = payload["config"]["artifact"]
    config_artifact["sha256"] = file_sha256(config_path)
    descriptor = dict(config_artifact)
    descriptor.pop("artifact_fingerprint")
    config_artifact["artifact_fingerprint"] = canonical_fingerprint(descriptor)
    payload["config"]["fingerprint"] = canonical_fingerprint(config)
    payload["config"]["comparison_fingerprint"] = canonical_fingerprint(
        {key: value for key, value in config.items() if key != "seed"}
    )
    _fingerprint_payload(payload, "manifest_fingerprint")
    _write_json(outcome, payload)
    with pytest.raises(ValueError, match="frozen arm configuration"):
        prepare_task07_comparison_preflight(**comparison_files.kwargs(tmp_path / "output"))


@pytest.mark.parametrize(
    "drift",
    ["dataset", "cohort", "plan", "selection", "weight", "model", "tokenizer", "budget"],
)
def test_frozen_identity_and_budget_drift_fails_closed(
    comparison_files: ComparisonFixture, tmp_path: Path, drift: str
) -> None:
    outcome = comparison_files.outcomes[0]

    def mutate(payload: dict[str, Any]) -> None:
        if drift in {"dataset", "cohort", "plan"}:
            payload[drift]["fingerprint"] = HEX["f"]
        elif drift == "selection":
            payload["selection_policy"]["fingerprint"] = HEX["f"]
        elif drift == "weight":
            payload["weight_policy"]["fingerprint"] = HEX["a"]
        elif drift == "model":
            payload["model_stack"]["base_model"]["artifact_fingerprint"] = HEX["f"]
            payload["model_stack"]["sft_adapter"]["base_model_fingerprint"] = HEX["f"]
            payload["model_stack_fingerprint"] = canonical_fingerprint(payload["model_stack"])
        elif drift == "tokenizer":
            payload["model_stack"]["tokenizer"]["tokenizer_fingerprint"] = HEX["f"]
            payload["tokenizer_fingerprint"] = HEX["f"]
            payload["model_stack_fingerprint"] = canonical_fingerprint(payload["model_stack"])
        else:
            payload["budget"]["target_token_budget"] += 1

    _rewrite_outcome(outcome, mutate)
    with pytest.raises(ValueError, match=r"drift"):
        prepare_task07_comparison_preflight(**comparison_files.kwargs(tmp_path / "output"))


def test_incomparable_metric_definition_fails_closed(
    comparison_files: ComparisonFixture, tmp_path: Path
) -> None:
    _rewrite_outcome(
        comparison_files.outcomes[0],
        lambda payload: payload["evidence"]["intrinsic"]["metrics"][0].__setitem__(
            "definition_fingerprint", HEX["f"]
        ),
    )
    with pytest.raises(ValueError, match="metric definition"):
        prepare_task07_comparison_preflight(**comparison_files.kwargs(tmp_path / "output"))


@pytest.mark.parametrize("field", ["ci", "sample_size"])
def test_missing_ci_or_sample_size_fails_closed(
    comparison_files: ComparisonFixture, tmp_path: Path, field: str
) -> None:
    _rewrite_outcome(
        comparison_files.outcomes[0],
        lambda payload: payload["evidence"]["intrinsic"]["metrics"][0].pop(field),
    )
    with pytest.raises(ValueError, match=field):
        prepare_task07_comparison_preflight(**comparison_files.kwargs(tmp_path / "output"))


@pytest.mark.parametrize("category", ["human", "shadow_independent", "probe_extrinsic"])
def test_missing_mandatory_evidence_category_fails_closed(
    comparison_files: ComparisonFixture, tmp_path: Path, category: str
) -> None:
    _rewrite_outcome(
        comparison_files.outcomes[0], lambda payload: payload["evidence"].pop(category)
    )
    with pytest.raises(ValueError, match=r"category coverage.*missing"):
        prepare_task07_comparison_preflight(**comparison_files.kwargs(tmp_path / "output"))


def test_final_test_path_is_rejected_before_read(monkeypatch: pytest.MonkeyPatch) -> None:
    forbidden = Path("results/final_test/outcome_manifest.json")

    def forbidden_read(*_args: Any, **_kwargs: Any) -> str:
        raise AssertionError("final-test path was read")

    monkeypatch.setattr(Path, "read_text", forbidden_read)
    with pytest.raises(ValueError, match="was not opened"):
        prepare_task07_comparison_preflight(
            protocol_manifest_path=Path("protocol.json"),
            task06_handoff_manifest_path=Path("handoff.json"),
            task06_selection_preflight_manifest_path=Path("selection.json"),
            task07_launch_manifest_path=Path("launch.json"),
            outcome_manifest_paths=[forbidden],
            output_dir=Path("output"),
        )


def test_overwrite_refusal_and_staging_cleanup(
    comparison_files: ComparisonFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    prepare_task07_comparison_preflight(**comparison_files.kwargs(output))
    snapshot = (output / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        prepare_task07_comparison_preflight(**comparison_files.kwargs(output))
    assert (output / "manifest.json").read_bytes() == snapshot

    def fail_replace(_source: Path, _target: Path) -> None:
        raise OSError("synthetic publish failure")

    monkeypatch.setattr(vars(comparison_module)["os"], "replace", fail_replace)
    failed = tmp_path / "failed"
    with pytest.raises(OSError, match="synthetic publish failure"):
        prepare_task07_comparison_preflight(**comparison_files.kwargs(failed))
    assert not failed.exists()
    assert not list(tmp_path.glob(".failed.staging-*"))


def test_module_and_script_have_no_model_or_reranker_imports() -> None:
    paths = [
        Path(comparison_module.__file__),
        Path("scripts/prepare_task07_comparison_preflight.py"),
    ]
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


def test_all_execution_and_decision_flags_are_false(
    comparison_files: ComparisonFixture, tmp_path: Path
) -> None:
    output = tmp_path / "output"
    bundle = prepare_task07_comparison_preflight(**comparison_files.kwargs(output)).model_dump(
        mode="json"
    )
    preflight = json.loads((output / "comparison_preflight.json").read_text())
    for payload in (bundle, preflight):
        assert payload["comparison_started"] is False
        assert payload["selection_performed"] is False
        assert payload["promotion_performed"] is False
        assert payload["model_loading_performed"] is False
        assert payload["training_started"] is False
        assert payload["evaluation_started"] is False
        assert payload["final_tests_used"] == []
        assert not _contains_forbidden_result(payload)
