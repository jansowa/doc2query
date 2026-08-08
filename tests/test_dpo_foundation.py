from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from doc2query.training.dpo import (
    DPOPlanManifest,
    PreferenceRecord,
    ReferenceLogprobRecord,
    canonical_fingerprint,
    file_sha256,
    ordered_ids_fingerprint,
    plan_dpo_controls,
    sigmoid_dpo_loss,
    validate_dpo_dataset,
    validate_reference_logprobs,
)

HEX_A = "a" * 64
HEX_B = "b" * 64
HEX_C = "c" * 64
HEX_D = "d" * 64
HEX_E = "e" * 64
HEX_F = "f" * 64


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _provenance(dataset_fingerprint: str = HEX_A) -> dict[str, Any]:
    return {
        "dataset_id": "task06-fixture",
        "dataset_fingerprint": dataset_fingerprint,
        "selection_policy_id": "frozen-selection-v1",
        "selection_policy_fingerprint": HEX_B,
    }


def _preference(index: int, split: str, *, dataset_fingerprint: str = HEX_A) -> dict[str, Any]:
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
        "provenance": _provenance(dataset_fingerprint),
    }


def _control(preference: dict[str, Any], *, weighted: bool) -> dict[str, Any]:
    row = {
        "preference_id": preference["preference_id"],
        "prompt": preference["prompt"],
        "completion": preference["chosen"],
        "candidate_id": preference["chosen_candidate_id"],
        "passage_id": preference["passage_id"],
        "passage_cluster_id": preference["passage_cluster_id"],
        "split": preference["split"],
        "provenance": preference["provenance"],
    }
    if weighted:
        row |= {
            "sample_weight": 1.25,
            "weight_policy_id": "frozen-weights-v1",
            "weight_policy_fingerprint": HEX_C,
        }
    return row


@dataclass
class DatasetFixture:
    root: Path
    rows: dict[str, list[dict[str, Any]]]

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def path(self, name: str) -> Path:
        return self.root / f"{name}.jsonl"

    def materialize(self, *, dataset_fingerprint: str = HEX_A) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for name, rows in self.rows.items():
            _write_jsonl(self.path(name), rows)
        payload: dict[str, Any] = {
            "schema_version": 1,
            "contract": "task06-preference-data-for-task07-v1",
            "dataset_id": "task06-fixture",
            "dataset_fingerprint": dataset_fingerprint,
            "selection_policy_id": "frozen-selection-v1",
            "selection_policy_fingerprint": HEX_B,
            "artifacts": {
                name: {
                    "path": path.name,
                    "sha256": file_sha256(path),
                    "record_count": len(self.rows[name]),
                }
                for name in self.rows
                if (path := self.path(name))
            },
            "automatic_thresholds_created": False,
            "relabeling_performed": False,
            "final_tests_used": [],
        }
        payload["manifest_fingerprint"] = canonical_fingerprint(payload)
        _write_json(self.manifest_path, payload)

    def kwargs(self) -> dict[str, Path]:
        return {
            "task06_manifest_path": self.manifest_path,
            "preference_train_path": self.path("preference_train"),
            "preference_dev_path": self.path("preference_dev"),
            "continued_sft_train_path": self.path("continued_sft_train"),
            "continued_sft_dev_path": self.path("continued_sft_dev"),
            "weighted_sft_train_path": self.path("weighted_sft_train"),
            "weighted_sft_dev_path": self.path("weighted_sft_dev"),
        }


@pytest.fixture
def dataset_files(tmp_path: Path) -> DatasetFixture:
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
    fixture = DatasetFixture(tmp_path / "dataset", rows)
    fixture.materialize()
    return fixture


def _model_stack() -> dict[str, Any]:
    return {
        "base_model": {
            "model_id": "base/model",
            "revision": "base-revision",
            "artifact_fingerprint": HEX_D,
        },
        "sft_adapter": {
            "adapter_id": "sft/adapter",
            "adapter_revision": "adapter-revision",
            "adapter_fingerprint": HEX_E,
            "base_model_fingerprint": HEX_D,
        },
        "tokenizer": {
            "tokenizer_id": "base/tokenizer",
            "revision": "tokenizer-revision",
            "tokenizer_fingerprint": HEX_F,
        },
    }


def _write_token_evidence(tmp_path: Path, dataset: Any) -> tuple[Path, Path]:
    records_path = tmp_path / "token_lengths.jsonl"
    all_preferences = dataset.preference_train + dataset.preference_dev
    rows = [
        {
            "preference_id": row.preference_id,
            "split": row.split,
            "prompt_tokens": 10 + index,
            "chosen_tokens": 4,
            "rejected_tokens": 5,
            "prompt_chosen_tokens": 14 + index,
            "prompt_rejected_tokens": 15 + index,
        }
        for index, row in enumerate(all_preferences)
    ]
    _write_jsonl(records_path, rows)
    manifest_path = tmp_path / "token_lengths_manifest.json"
    payload: dict[str, Any] = {
        "contract": "task07-model-free-token-lengths-v1",
        "records": {
            "path": records_path.name,
            "sha256": file_sha256(records_path),
            "record_count": len(rows),
        },
        "tokenizer": _model_stack()["tokenizer"],
        "dataset_fingerprint": dataset.provenance.dataset_fingerprint,
        "ordered_preference_ids_fingerprint": ordered_ids_fingerprint(
            [row.preference_id for row in all_preferences]
        ),
        "model_loading_performed": False,
        "final_tests_used": [],
    }
    payload["artifact_fingerprint"] = canonical_fingerprint(payload)
    _write_json(manifest_path, payload)
    return manifest_path, records_path


def _write_plan_config(path: Path) -> None:
    payload = {
        "plan_id": "task07-fixture-plan",
        "seeds": [42, 43],
        "beta": 0.1,
        "loss_type": "sigmoid",
        "learning_rate": 1e-5,
        "max_length": 768,
        "max_prompt_length": 640,
        "target_token_budget": 100_000,
        "target_optimizer_steps": 250,
        "start_model": _model_stack(),
        "reference_model": _model_stack(),
        "weight_policy_id": "frozen-weights-v1",
        "weight_policy_fingerprint": HEX_C,
        "arms": ["dpo", "continued_sft", "score_weighted_continued_sft"],
    }
    _write_json(path, payload)


def _make_plan(tmp_path: Path, dataset_files: DatasetFixture) -> tuple[Any, Path, Any]:
    dataset = validate_dpo_dataset(**dataset_files.kwargs())
    token_manifest, token_records = _write_token_evidence(tmp_path, dataset)
    config = tmp_path / "plan_config.json"
    _write_plan_config(config)
    output = tmp_path / "plan.json"
    plan = plan_dpo_controls(
        config_path=config,
        token_length_manifest_path=token_manifest,
        token_length_records_path=token_records,
        output_path=output,
        dataset=dataset,
    )
    return plan, output, dataset


def _write_reference_artifact(tmp_path: Path, plan: Any, dataset: Any) -> tuple[Path, Path]:
    records_path = tmp_path / "reference_logprobs.jsonl"
    rows = [
        {
            "preference_id": row.preference_id,
            "position": index,
            "chosen_logprob": -2.0 - index,
            "rejected_logprob": -3.0 - index,
        }
        for index, row in enumerate(dataset.preference_train)
    ]
    _write_jsonl(records_path, rows)
    manifest_path = tmp_path / "reference_logprobs_manifest.json"
    payload: dict[str, Any] = {
        "contract": "task07-precomputed-reference-logprobs-v1",
        "records": {
            "path": records_path.name,
            "sha256": file_sha256(records_path),
            "record_count": len(rows),
        },
        "ordered_preference_ids_fingerprint": ordered_ids_fingerprint(
            [row["preference_id"] for row in rows]
        ),
        "dataset_fingerprint": plan.dataset_fingerprint,
        "plan_fingerprint": plan.plan_fingerprint,
        "reference_model": plan.reference_model.model_dump(mode="json"),
        "tokenizer_fingerprint": plan.reference_model.tokenizer.tokenizer_fingerprint,
        "final_tests_used": [],
    }
    payload["artifact_fingerprint"] = canonical_fingerprint(payload)
    _write_json(manifest_path, payload)
    return records_path, manifest_path


def _rehash_reference_manifest(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["artifact_fingerprint"] = canonical_fingerprint(
        {key: value for key, value in payload.items() if key != "artifact_fingerprint"}
    )
    _write_json(path, payload)


def test_plan_is_deterministic_matched_and_explicitly_not_started(
    tmp_path: Path, dataset_files: DatasetFixture
) -> None:
    plan, _, dataset = _make_plan(tmp_path, dataset_files)
    token_manifest = tmp_path / "token_lengths_manifest.json"
    token_records = tmp_path / "token_lengths.jsonl"
    second = plan_dpo_controls(
        config_path=tmp_path / "plan_config.json",
        token_length_manifest_path=token_manifest,
        token_length_records_path=token_records,
        output_path=tmp_path / "second_plan.json",
        dataset=dataset,
    )
    assert plan.plan_fingerprint == second.plan_fingerprint
    assert plan.status == "planned_not_trained"
    assert plan.model_loading_performed is False
    assert plan.training_started is False
    assert plan.reference_logprobs_computed is False
    assert plan.final_tests_used == []
    assert len(plan.arms) == 3


def test_duplicate_preference_id_fails_closed(dataset_files: DatasetFixture) -> None:
    duplicate = copy.deepcopy(dataset_files.rows["preference_train"][0])
    duplicate["passage_id"] = "another-passage"
    duplicate["passage_cluster_id"] = "another-cluster"
    dataset_files.rows["preference_dev"] = [duplicate]
    dataset_files.rows["continued_sft_dev"] = [_control(duplicate, weighted=False)]
    dataset_files.rows["weighted_sft_dev"] = [_control(duplicate, weighted=True)]
    dataset_files.materialize()
    with pytest.raises(ValueError, match="duplicate preference_id"):
        validate_dpo_dataset(**dataset_files.kwargs())


@pytest.mark.parametrize("margin", [0.0, -0.1, float("nan")])
def test_nonpositive_or_nan_margin_is_rejected(margin: float) -> None:
    row = _preference(0, "train")
    row["score_margin"] = margin
    with pytest.raises(ValidationError, match="score_margin"):
        PreferenceRecord.model_validate(row)


def test_normalized_identical_pair_is_rejected() -> None:
    row = _preference(0, "train")
    row["chosen"] = "  PYTANIE   O POMPIE  "
    row["rejected"] = "pytanie o pompie"
    with pytest.raises(ValidationError, match="Task 06 normalization"):
        PreferenceRecord.model_validate(row)


def test_continued_sft_must_exactly_match_chosen(dataset_files: DatasetFixture) -> None:
    dataset_files.rows["continued_sft_train"][0]["completion"] = "Inne pytanie"
    dataset_files.materialize()
    with pytest.raises(ValueError, match="does not exactly match chosen"):
        validate_dpo_dataset(**dataset_files.kwargs())


def test_missing_and_orphan_control_rows_fail_closed(dataset_files: DatasetFixture) -> None:
    dataset_files.rows["continued_sft_train"] = [
        dataset_files.rows["continued_sft_train"][0],
        {
            **dataset_files.rows["continued_sft_train"][1],
            "preference_id": "orphan-pref",
        },
    ]
    dataset_files.materialize()
    with pytest.raises(ValueError, match=r"missing=.*orphan="):
        validate_dpo_dataset(**dataset_files.kwargs())


@pytest.mark.parametrize("leak_field", ["passage_id", "passage_cluster_id"])
def test_train_dev_leakage_is_rejected(dataset_files: DatasetFixture, leak_field: str) -> None:
    leaked = dataset_files.rows["preference_train"][0][leak_field]
    dataset_files.rows["preference_dev"][0][leak_field] = leaked
    for name in ("continued_sft_dev", "weighted_sft_dev"):
        dataset_files.rows[name][0][leak_field] = leaked
    dataset_files.materialize()
    with pytest.raises(ValueError, match="leakage"):
        validate_dpo_dataset(**dataset_files.kwargs())


def test_test_split_is_absolutely_rejected(dataset_files: DatasetFixture) -> None:
    for name in ("preference_train", "continued_sft_train", "weighted_sft_train"):
        dataset_files.rows[name][0]["split"] = "test"
    dataset_files.materialize()
    with pytest.raises(ValueError, match="test split is absolutely forbidden"):
        validate_dpo_dataset(**dataset_files.kwargs())


def test_task06_file_hash_drift_is_rejected(dataset_files: DatasetFixture) -> None:
    with dataset_files.path("preference_train").open("a", encoding="utf-8") as handle:
        handle.write(" \n")
    with pytest.raises(ValueError, match="sha256 differs"):
        validate_dpo_dataset(**dataset_files.kwargs())


def test_mismatched_arm_budget_is_rejected(tmp_path: Path, dataset_files: DatasetFixture) -> None:
    _, plan_path, _ = _make_plan(tmp_path, dataset_files)
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    payload["arms"]["continued_sft"]["target_optimizer_steps"] += 1
    with pytest.raises(ValidationError, match="budgets are not matched"):
        DPOPlanManifest.model_validate(payload)


@pytest.mark.parametrize("case", ["dataset", "model", "adapter", "tokenizer", "plan"])
def test_reference_identity_drift_is_rejected(
    tmp_path: Path, dataset_files: DatasetFixture, case: str
) -> None:
    plan, plan_path, dataset = _make_plan(tmp_path, dataset_files)
    records_path, manifest_path = _write_reference_artifact(tmp_path, plan, dataset)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if case == "dataset":
        payload["dataset_fingerprint"] = HEX_B
    elif case == "model":
        payload["reference_model"]["base_model"]["artifact_fingerprint"] = HEX_A
        payload["reference_model"]["sft_adapter"]["base_model_fingerprint"] = HEX_A
    elif case == "adapter":
        payload["reference_model"]["sft_adapter"]["adapter_fingerprint"] = HEX_A
    elif case == "tokenizer":
        payload["reference_model"]["tokenizer"]["tokenizer_fingerprint"] = HEX_A
        payload["tokenizer_fingerprint"] = HEX_A
    else:
        payload["plan_fingerprint"] = HEX_A
    _write_json(manifest_path, payload)
    _rehash_reference_manifest(manifest_path)
    with pytest.raises(ValueError, match="different dataset, plan or model stack"):
        validate_reference_logprobs(
            records_path=records_path,
            manifest_path=manifest_path,
            plan_path=plan_path,
            dataset=dataset,
        )


@pytest.mark.parametrize("fault", ["duplicate", "missing", "orphan"])
def test_reference_id_coverage_is_exact(
    tmp_path: Path, dataset_files: DatasetFixture, fault: str
) -> None:
    plan, plan_path, dataset = _make_plan(tmp_path, dataset_files)
    records_path, manifest_path = _write_reference_artifact(tmp_path, plan, dataset)
    rows = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    if fault == "duplicate":
        rows[1]["preference_id"] = rows[0]["preference_id"]
    elif fault == "missing":
        rows.pop()
    else:
        rows[1]["preference_id"] = "orphan"
    _write_jsonl(records_path, rows)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["records"]["sha256"] = file_sha256(records_path)
    manifest["records"]["record_count"] = len(rows)
    manifest["ordered_preference_ids_fingerprint"] = ordered_ids_fingerprint(
        [row["preference_id"] for row in rows]
    )
    _write_json(manifest_path, manifest)
    _rehash_reference_manifest(manifest_path)
    match = "duplicate preference_id" if fault == "duplicate" else "coverage mismatch"
    with pytest.raises(ValueError, match=match):
        validate_reference_logprobs(
            records_path=records_path,
            manifest_path=manifest_path,
            plan_path=plan_path,
            dataset=dataset,
        )


def test_reference_order_change_is_rejected(tmp_path: Path, dataset_files: DatasetFixture) -> None:
    plan, plan_path, dataset = _make_plan(tmp_path, dataset_files)
    records_path, manifest_path = _write_reference_artifact(tmp_path, plan, dataset)
    rows = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    rows.reverse()
    for index, row in enumerate(rows):
        row["position"] = index
    _write_jsonl(records_path, rows)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["records"]["sha256"] = file_sha256(records_path)
    manifest["ordered_preference_ids_fingerprint"] = ordered_ids_fingerprint(
        [row["preference_id"] for row in rows]
    )
    _write_json(manifest_path, manifest)
    _rehash_reference_manifest(manifest_path)
    with pytest.raises(ValueError, match="order differs"):
        validate_reference_logprobs(
            records_path=records_path,
            manifest_path=manifest_path,
            plan_path=plan_path,
            dataset=dataset,
        )


def test_reference_logprobs_must_be_finite() -> None:
    with pytest.raises(ValidationError, match="finite"):
        ReferenceLogprobRecord(
            preference_id="pref", position=0, chosen_logprob=float("nan"), rejected_logprob=-1
        )


def test_sigmoid_dpo_loss_decreases_with_better_relative_chosen_advantage() -> None:
    baseline = sigmoid_dpo_loss(-2.0, -2.0, -2.0, -2.0, beta=0.1)
    improved = sigmoid_dpo_loss(-1.0, -3.0, -2.0, -2.0, beta=0.1)
    assert improved < baseline
