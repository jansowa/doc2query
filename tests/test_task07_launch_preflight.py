from __future__ import annotations

import builtins
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import doc2query.training.dpo as dpo_module
import doc2query.training.launch as launch_module
from doc2query.training.dpo import (
    canonical_fingerprint,
    file_sha256,
    ordered_ids_fingerprint,
    plan_dpo_controls,
    validate_dpo_dataset,
)
from doc2query.training.launch import prepare_task07_launch
from doc2query.utils.records import read_records as raw_read_records

HEX_A = "a" * 64
HEX_B = "b" * 64
HEX_C = "c" * 64
HEX_D = "d" * 64
HEX_E = "e" * 64
HEX_F = "f" * 64


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _provenance() -> dict[str, Any]:
    return {
        "dataset_id": "task07-launch-fixture",
        "dataset_fingerprint": HEX_A,
        "selection_policy_id": "selection-v1",
        "selection_policy_fingerprint": HEX_B,
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
            "weight_policy_fingerprint": HEX_C,
        }
    return result


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


@dataclass
class LaunchFixture:
    root: Path
    rows: dict[str, list[dict[str, Any]]]

    @property
    def task06_manifest(self) -> Path:
        return self.root / "data" / "manifest.json"

    def data_path(self, name: str) -> Path:
        return self.root / "data" / f"{name}.jsonl"

    @property
    def token_manifest(self) -> Path:
        return self.root / "evidence" / "token_manifest.json"

    @property
    def token_records(self) -> Path:
        return self.root / "evidence" / "token_records.jsonl"

    @property
    def plan_path(self) -> Path:
        return self.root / "plan.json"

    @property
    def reference_manifest(self) -> Path:
        return self.root / "evidence" / "reference_manifest.json"

    @property
    def reference_records(self) -> Path:
        return self.root / "evidence" / "reference_records.jsonl"

    def dataset_kwargs(self) -> dict[str, Path]:
        return {
            "task06_manifest_path": self.task06_manifest,
            "preference_train_path": self.data_path("preference_train"),
            "preference_dev_path": self.data_path("preference_dev"),
            "continued_sft_train_path": self.data_path("continued_sft_train"),
            "continued_sft_dev_path": self.data_path("continued_sft_dev"),
            "weighted_sft_train_path": self.data_path("weighted_sft_train"),
            "weighted_sft_dev_path": self.data_path("weighted_sft_dev"),
        }

    def launch_kwargs(self, output_dir: Path) -> dict[str, Path]:
        return self.dataset_kwargs() | {
            "plan_path": self.plan_path,
            "token_length_manifest_path": self.token_manifest,
            "token_length_records_path": self.token_records,
            "reference_logprob_manifest_path": self.reference_manifest,
            "reference_logprob_records_path": self.reference_records,
            "output_dir": output_dir,
        }

    def materialize_data(self) -> None:
        self.task06_manifest.parent.mkdir(parents=True, exist_ok=True)
        for name, rows in self.rows.items():
            _write_jsonl(self.data_path(name), rows)
        payload: dict[str, Any] = {
            "schema_version": 1,
            "contract": "task06-preference-data-for-task07-v1",
            **_provenance(),
            "artifacts": {
                name: {
                    "path": self.data_path(name).name,
                    "sha256": file_sha256(self.data_path(name)),
                    "record_count": len(rows),
                }
                for name, rows in self.rows.items()
            },
            "automatic_thresholds_created": False,
            "relabeling_performed": False,
            "final_tests_used": [],
        }
        payload["manifest_fingerprint"] = canonical_fingerprint(payload)
        _write_json(self.task06_manifest, payload)

    def materialize_all(self) -> None:
        self.materialize_data()
        evidence = self.token_manifest.parent
        evidence.mkdir(parents=True, exist_ok=True)
        dataset = validate_dpo_dataset(**self.dataset_kwargs())
        all_preferences = dataset.preference_train + dataset.preference_dev
        token_rows = [
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
        _write_jsonl(self.token_records, token_rows)
        token_payload: dict[str, Any] = {
            "contract": "task07-model-free-token-lengths-v1",
            "records": {
                "path": self.token_records.name,
                "sha256": file_sha256(self.token_records),
                "record_count": len(token_rows),
            },
            "tokenizer": _model_stack()["tokenizer"],
            "dataset_fingerprint": HEX_A,
            "ordered_preference_ids_fingerprint": ordered_ids_fingerprint(
                [row.preference_id for row in all_preferences]
            ),
            "model_loading_performed": False,
            "final_tests_used": [],
        }
        token_payload["artifact_fingerprint"] = canonical_fingerprint(token_payload)
        _write_json(self.token_manifest, token_payload)

        config = self.root / "plan_config.json"
        _write_json(
            config,
            {
                "plan_id": "launch-fixture-plan",
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
                "weight_policy_id": "weights-v1",
                "weight_policy_fingerprint": HEX_C,
                "arms": ["dpo", "continued_sft", "score_weighted_continued_sft"],
            },
        )
        plan = plan_dpo_controls(
            config_path=config,
            token_length_manifest_path=self.token_manifest,
            token_length_records_path=self.token_records,
            output_path=self.plan_path,
            dataset=dataset,
        )
        reference_rows = [
            {
                "preference_id": row.preference_id,
                "position": index,
                "chosen_logprob": -2.0 - index,
                "rejected_logprob": -3.0 - index,
            }
            for index, row in enumerate(dataset.preference_train)
        ]
        _write_jsonl(self.reference_records, reference_rows)
        reference_payload: dict[str, Any] = {
            "contract": "task07-precomputed-reference-logprobs-v1",
            "records": {
                "path": self.reference_records.name,
                "sha256": file_sha256(self.reference_records),
                "record_count": len(reference_rows),
            },
            "ordered_preference_ids_fingerprint": ordered_ids_fingerprint(
                [str(row["preference_id"]) for row in reference_rows]
            ),
            "dataset_fingerprint": plan.dataset_fingerprint,
            "plan_fingerprint": plan.plan_fingerprint,
            "reference_model": plan.reference_model.model_dump(mode="json"),
            "tokenizer_fingerprint": plan.reference_model.tokenizer.tokenizer_fingerprint,
            "final_tests_used": [],
        }
        reference_payload["artifact_fingerprint"] = canonical_fingerprint(reference_payload)
        _write_json(self.reference_manifest, reference_payload)

    def rehash_reference(self) -> None:
        payload = json.loads(self.reference_manifest.read_text(encoding="utf-8"))
        payload["records"]["sha256"] = file_sha256(self.reference_records)
        payload["records"]["record_count"] = len(
            self.reference_records.read_text(encoding="utf-8").splitlines()
        )
        rows = [json.loads(line) for line in self.reference_records.read_text().splitlines()]
        payload["ordered_preference_ids_fingerprint"] = ordered_ids_fingerprint(
            [row["preference_id"] for row in rows]
        )
        payload["artifact_fingerprint"] = canonical_fingerprint(
            {key: value for key, value in payload.items() if key != "artifact_fingerprint"}
        )
        _write_json(self.reference_manifest, payload)


@pytest.fixture
def launch_files(tmp_path: Path) -> LaunchFixture:
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
    fixture = LaunchFixture(tmp_path / "inputs", rows)
    fixture.materialize_all()
    return fixture


def _rehash_plan(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    payload["plan_fingerprint"] = canonical_fingerprint(
        {key: value for key, value in payload.items() if key != "plan_fingerprint"}
    )
    _write_json(path, payload)
    return payload


def test_launch_manifest_is_deterministic_three_arm_and_matched(
    launch_files: LaunchFixture, tmp_path: Path
) -> None:
    first = prepare_task07_launch(**launch_files.launch_kwargs(tmp_path / "launch-one"))
    second = prepare_task07_launch(**launch_files.launch_kwargs(tmp_path / "launch-two"))
    assert first == second
    assert first.status == "ready_for_model_smoke_not_trained"
    assert set(first.arms) == {"dpo", "continued_sft", "score_weighted_continued_sft"}
    matched = {
        (
            tuple(arm.budget.seeds),
            arm.budget.target_token_budget,
            arm.budget.target_optimizer_steps,
            arm.budget.cohort_fingerprint,
        )
        for arm in first.arms.values()
    }
    assert len(matched) == 1
    assert len(first.input_hashes) == 12
    assert first.model_loading_performed is False
    assert first.tokenizer_loading_performed is False
    assert first.reference_logprobs_computed is False
    assert first.training_started is False
    assert first.evaluation_started is False
    assert first.final_tests_used == []


@pytest.mark.parametrize("drift", ["dataset", "selection", "weight_policy"])
def test_dataset_selection_and_weight_policy_drift_fail_closed(
    launch_files: LaunchFixture, tmp_path: Path, drift: str
) -> None:
    if drift in {"dataset", "selection"}:
        field = "dataset_fingerprint" if drift == "dataset" else "selection_policy_fingerprint"
        launch_files.rows["preference_train"][0]["provenance"][field] = HEX_F
        launch_files.materialize_data()
    else:
        launch_files.rows["weighted_sft_train"][0]["weight_policy_fingerprint"] = HEX_F
        launch_files.materialize_data()
    with pytest.raises(ValueError, match=r"provenance|weight policy"):
        prepare_task07_launch(**launch_files.launch_kwargs(tmp_path / "launch"))


@pytest.mark.parametrize("drift", ["model", "adapter", "tokenizer", "plan"])
def test_model_adapter_tokenizer_and_plan_drift_fail_closed(
    launch_files: LaunchFixture, tmp_path: Path, drift: str
) -> None:
    payload = json.loads(launch_files.plan_path.read_text(encoding="utf-8"))
    if drift == "model":
        payload["reference_model"]["base_model"]["artifact_fingerprint"] = HEX_A
        payload["reference_model"]["sft_adapter"]["base_model_fingerprint"] = HEX_A
    elif drift == "adapter":
        payload["reference_model"]["sft_adapter"]["adapter_fingerprint"] = HEX_A
    elif drift == "tokenizer":
        for name in ("start_model", "reference_model"):
            payload[name]["tokenizer"]["tokenizer_fingerprint"] = HEX_A
    else:
        payload["plan_fingerprint"] = HEX_A
        _write_json(launch_files.plan_path, payload)
        with pytest.raises(ValueError, match="plan fingerprint"):
            prepare_task07_launch(**launch_files.launch_kwargs(tmp_path / "launch"))
        return
    _write_json(launch_files.plan_path, payload)
    _rehash_plan(launch_files.plan_path)
    with pytest.raises(ValueError, match=r"reference model|tokenizer|different dataset"):
        prepare_task07_launch(**launch_files.launch_kwargs(tmp_path / "launch"))


@pytest.mark.parametrize("fault", ["missing", "orphan", "duplicate", "reordered"])
def test_reference_preference_id_coverage_and_order_fail_closed(
    launch_files: LaunchFixture, tmp_path: Path, fault: str
) -> None:
    rows = [json.loads(line) for line in launch_files.reference_records.read_text().splitlines()]
    if fault == "missing":
        rows.pop()
    elif fault == "orphan":
        rows[-1]["preference_id"] = "orphan"
    elif fault == "duplicate":
        rows[-1]["preference_id"] = rows[0]["preference_id"]
    else:
        rows.reverse()
        for position, row in enumerate(rows):
            row["position"] = position
    _write_jsonl(launch_files.reference_records, rows)
    launch_files.rehash_reference()
    with pytest.raises(ValueError, match=r"coverage|duplicate|order"):
        prepare_task07_launch(**launch_files.launch_kwargs(tmp_path / "launch"))


def test_reordered_control_preference_ids_fail_closed(
    launch_files: LaunchFixture, tmp_path: Path
) -> None:
    launch_files.rows["continued_sft_train"].reverse()
    launch_files.materialize_data()
    with pytest.raises(ValueError, match="order differs"):
        prepare_task07_launch(**launch_files.launch_kwargs(tmp_path / "launch"))


@pytest.mark.parametrize("fault", ["sha256", "record_count"])
def test_hash_and_record_count_drift_fail_closed(
    launch_files: LaunchFixture, tmp_path: Path, fault: str
) -> None:
    if fault == "sha256":
        launch_files.data_path("preference_train").write_text(
            launch_files.data_path("preference_train").read_text() + " \n"
        )
    else:
        payload = json.loads(launch_files.task06_manifest.read_text(encoding="utf-8"))
        payload["artifacts"]["preference_train"]["record_count"] += 1
        payload["manifest_fingerprint"] = canonical_fingerprint(
            {key: value for key, value in payload.items() if key != "manifest_fingerprint"}
        )
        _write_json(launch_files.task06_manifest, payload)
    with pytest.raises(ValueError, match=r"sha256|record count"):
        prepare_task07_launch(**launch_files.launch_kwargs(tmp_path / "launch"))


@pytest.mark.parametrize("field", ["passage_id", "passage_cluster_id"])
def test_train_dev_leakage_fails_closed(
    launch_files: LaunchFixture, tmp_path: Path, field: str
) -> None:
    leaked = launch_files.rows["preference_train"][0][field]
    for name in ("preference_dev", "continued_sft_dev", "weighted_sft_dev"):
        launch_files.rows[name][0][field] = leaked
    launch_files.materialize_data()
    with pytest.raises(ValueError, match="leakage"):
        prepare_task07_launch(**launch_files.launch_kwargs(tmp_path / "launch"))


def test_test_artifact_is_never_read_and_embedded_test_is_forbidden(
    launch_files: LaunchFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    test_artifact = launch_files.root / "data" / "preference_test.jsonl"
    test_artifact.write_text("deliberately invalid JSON\n", encoding="utf-8")
    opened: list[Path] = []

    def recording_reader(path: Path) -> Any:
        opened.append(path)
        return raw_read_records(path)

    monkeypatch.setattr(dpo_module, "read_records", recording_reader)
    prepare_task07_launch(**launch_files.launch_kwargs(tmp_path / "valid-launch"))
    assert test_artifact not in opened

    for name in ("preference_train", "continued_sft_train", "weighted_sft_train"):
        launch_files.rows[name][0]["split"] = "test"
    launch_files.materialize_data()
    with pytest.raises(ValueError, match=r"test|literal"):
        prepare_task07_launch(**launch_files.launch_kwargs(tmp_path / "invalid-launch"))


def test_output_refusal_and_staging_cleanup(
    launch_files: LaunchFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "launch"
    prepare_task07_launch(**launch_files.launch_kwargs(output))
    snapshot = (output / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        prepare_task07_launch(**launch_files.launch_kwargs(output))
    assert (output / "manifest.json").read_bytes() == snapshot

    def interrupted_write(_path: Path, _value: Any) -> None:
        raise RuntimeError("synthetic write interruption")

    monkeypatch.setattr(launch_module, "write_json", interrupted_write)
    interrupted = tmp_path / "interrupted"
    with pytest.raises(RuntimeError, match="write interruption"):
        prepare_task07_launch(**launch_files.launch_kwargs(interrupted))
    assert not interrupted.exists()
    assert list(tmp_path.glob(".interrupted.staging-*")) == []


def test_preflight_never_imports_model_or_training_dependencies(
    launch_files: LaunchFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_import = builtins.__import__
    forbidden = {"torch", "transformers", "tokenizers", "trl", "peft", "bitsandbytes"}

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".", maxsplit=1)[0] in forbidden:
            raise AssertionError(f"model or training dependency imported: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    manifest = prepare_task07_launch(**launch_files.launch_kwargs(tmp_path / "launch"))
    assert manifest.training_started is False
