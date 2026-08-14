from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from doc2query.preferences.same_prompt_cohort import freeze_same_prompt_expansion_cohort
from doc2query.preferences.task06_smoke import generate_same_prompt_expansion
from doc2query.utils.records import JsonParquetWriter, write_json

PAIR_COUNT = 24
COHORT_SIZE = 4


def _pairs(count: int = PAIR_COUNT) -> list[dict[str, Any]]:
    return [
        {
            "pair_id": f"{index}::{1000 + index}",
            "example_id": str(index),
            "doc_id": str(1000 + index),
            "negative_doc_ids": [str(9000 + index * 20 + offset) for offset in range(10)],
            "split": "train",
        }
        for index in range(count)
    ]


def _train_records(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "example_id": row["example_id"],
            "query": f"naturalne zapytanie {row['example_id']}",
            "positives": [
                {"doc_id": row["doc_id"], "text": f"pasaż {row['doc_id']}", "metadata": {}}
            ],
            "hard_negatives": [
                {"doc_id": doc_id, "text": f"negatyw {doc_id}", "metadata": {}}
                for doc_id in reversed(row["negative_doc_ids"])
            ],
            "metadata": {"split": "train", "source": "speakleash/msmarco_pl"},
        }
        for row in pairs
    ]


def _prior_ids(path: Path, clusters: list[str], *, contract: str) -> str:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "contract": contract,
        "status": "ids_frozen_before_text_materialization",
        "records": [
            {
                "pair_id": f"prior::{cluster}",
                "example_id": f"prior-{cluster}",
                "doc_id": cluster,
                "cluster_id": cluster,
            }
            for cluster in clusters
        ],
        "quality_fields_used": [],
        "final_tests_used": [],
    }
    write_json(path, payload)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repository(tmp_path: Path, *, prior_clusters: list[str] | None = None) -> Path:
    root = tmp_path / "repo"
    (root / "configs/preferences").mkdir(parents=True)
    (root / "data/processed/v1").mkdir(parents=True)
    (root / "artifacts/task06").mkdir(parents=True)
    pairs = _pairs()
    pairs_path = root / "data/processed/v1/doc2query_train.parquet"
    with JsonParquetWriter(pairs_path) as writer:
        for row in pairs:
            writer.write(row)
    dedup_path = root / "data/processed/v1/dedup_map.parquet"
    with JsonParquetWriter(dedup_path) as writer:
        for row in pairs:
            writer.write({"doc_id": row["doc_id"], "cluster_id": row["doc_id"]})
    source_path = root / "data/processed/v1/train.parquet"
    with JsonParquetWriter(source_path) as writer:
        for row in _train_records(pairs):
            writer.write(row)
    split_path = root / "data/processed/v1/split_manifest.json"
    write_json(split_path, {"version": "v1", "positive_canonical_leakage": 0})

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    design_path = root / "configs/preferences/task06_candidate_execution_design_v1.yaml"
    write_json(
        design_path,
        {
            "schema_version": 1,
            "contract": "task06-candidate-execution-design-v1",
            "final_tests_used": [],
            "data": {
                "source_train_pairs": "data/processed/v1/doc2query_train.parquet",
                "dedup_map": "data/processed/v1/dedup_map.parquet",
                "split_manifest": "data/processed/v1/split_manifest.json",
                "sha256": {
                    "source_train_pairs": digest(pairs_path),
                    "dedup_map": digest(dedup_path),
                    "split_manifest": digest(split_path),
                },
            },
            "adapter_training_exclusion": {"selection_seed": 42, "max_pairs": 8},
        },
    )
    prior_path = root / "artifacts/task06/prior_cohort.ids.json"
    prior_sha = _prior_ids(
        prior_path,
        prior_clusters if prior_clusters is not None else ["1000", "1001"],
        contract="task06-candidate-pilot-v1",
    )
    write_json(
        root / "configs/preferences/task06_same_prompt_expansion_v2.yaml",
        {
            "schema_version": 1,
            "contract": "task06-same-prompt-preference-expansion-v2",
            "status": "frozen_ready_for_cohort_freeze",
            "final_tests_used": [],
            "design": {
                "config": "configs/preferences/task06_candidate_execution_design_v1.yaml",
                "config_sha256": digest(design_path),
                "read_only": True,
            },
            "cohort": {
                "split": "train",
                "selection": "sha256_cluster_first_quality_blind_v2",
                "selection_seed": 20260814,
                "passage_count": COHORT_SIZE,
                "min_hard_negatives": 10,
                "source_records": "data/processed/v1/train.parquet",
                "source_records_sha256": digest(source_path),
                "exclude_adapter_training_clusters": True,
                "exclude_prior_cohort_ids": [
                    {"path": "artifacts/task06/prior_cohort.ids.json", "sha256": prior_sha}
                ],
            },
            "authorization": {
                "cohort_freeze_authorized": True,
                "generation_authorized": True,
                "scoring_authorized": True,
                "tentative_pair_build_authorized": False,
                "final_tests_used": [],
            },
        },
    )
    return root


def _config(root: Path) -> Path:
    return root / "configs/preferences/task06_same_prompt_expansion_v2.yaml"


def _rewrite_config(root: Path, mutate: Any) -> None:
    path = _config(root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    write_json(path, payload)


def test_cohort_freeze_is_quality_blind_and_disjoint(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    output = root / "artifacts/task06/same_prompt_expansion_v2"
    manifest = freeze_same_prompt_expansion_cohort(_config(root), output)
    assert manifest["status"] == "materialized_after_quality_blind_id_freeze"
    assert manifest["record_count"] == COHORT_SIZE
    assert manifest["cluster_count"] == COHORT_SIZE
    assert manifest["quality_fields_used_for_selection"] == []
    assert manifest["excluded_prior_cluster_count"] == 2
    assert manifest["excluded_adapter_training_cluster_count"] == 8
    assert manifest["prior_cluster_overlap_count"] == 0
    assert manifest["generation_started"] is False
    assert manifest["scoring_started"] is False
    assert manifest["diversity_gate_applied"] is False
    assert manifest["pairs_built"] is False
    assert manifest["model_loading_performed"] is False
    assert manifest["final_tests_used"] == []

    ids = json.loads((output / "cohort.ids.json").read_text(encoding="utf-8"))
    assert ids["status"] == "ids_frozen_before_text_materialization"
    assert len(ids["records"]) == COHORT_SIZE
    clusters = {row["cluster_id"] for row in ids["records"]}
    assert clusters.isdisjoint({"1000", "1001"})

    records = [
        json.loads(line)
        for line in (output / "cohort.records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["pair_id"] for row in records] == [row["pair_id"] for row in ids["records"]]
    assert all(row["split"] == "train" for row in records)
    assert all(len(row["positives"]) == 1 for row in records)
    assert all(len(row["hard_negatives"]) >= 10 for row in records)
    assert all(row["metadata"]["task06_same_prompt_expansion_v2"] is True for row in records)
    first = records[0]
    expected_negatives = next(
        row["negative_doc_ids"] for row in _pairs() if row["pair_id"] == first["pair_id"]
    )
    assert [value["doc_id"] for value in first["hard_negatives"]] == expected_negatives


def test_cohort_partitions_are_disjoint_and_leave_the_default_untouched(tmp_path: Path) -> None:
    unpartitioned = _repository(tmp_path / "whole")
    reference = freeze_same_prompt_expansion_cohort(
        _config(unpartitioned), unpartitioned / "artifacts/task06/same_prompt_expansion_v2"
    )
    assert "partition" not in reference  # absent keeps v2 byte-identical

    selected: list[set[str]] = []
    for index in range(2):
        root = _repository(tmp_path / f"part{index}")
        _rewrite_config(
            root,
            lambda payload, index=index: payload["cohort"].update(
                {"partition": {"index": index, "count": 2}, "passage_count": 2}
            ),
        )
        manifest = freeze_same_prompt_expansion_cohort(
            _config(root), root / "artifacts/task06/same_prompt_expansion_v2"
        )
        assert manifest["partition"] == {"index": index, "count": 2}
        ids = json.loads(
            (root / "artifacts/task06/same_prompt_expansion_v2/cohort.ids.json").read_text(
                encoding="utf-8"
            )
        )
        selected.append({row["cluster_id"] for row in ids["records"]})
    assert selected[0].isdisjoint(selected[1])
    assert manifest["eligible_pair_count"] < reference["eligible_pair_count"]


@pytest.mark.parametrize(
    "partition",
    [{"index": 2, "count": 2}, {"index": -1, "count": 4}, {"index": 0, "count": 1}],
)
def test_cohort_partition_is_validated(tmp_path: Path, partition: dict[str, int]) -> None:
    root = _repository(tmp_path)
    _rewrite_config(root, lambda payload: payload["cohort"].__setitem__("partition", partition))
    with pytest.raises(ValueError, match=r"cohort\.partition requires"):
        freeze_same_prompt_expansion_cohort(
            _config(root), root / "artifacts/task06/same_prompt_expansion_v2"
        )


def test_cohort_freeze_is_deterministic_and_idempotent(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    output = root / "artifacts/task06/same_prompt_expansion_v2"
    first = freeze_same_prompt_expansion_cohort(_config(root), output)
    second = freeze_same_prompt_expansion_cohort(_config(root), output)
    assert first == second

    other = _repository(tmp_path / "second")
    repeated = freeze_same_prompt_expansion_cohort(
        _config(other), other / "artifacts/task06/same_prompt_expansion_v2"
    )
    assert repeated["ids_fingerprint"] == first["ids_fingerprint"]
    assert repeated["records_sha256"] == first["records_sha256"]


def test_cohort_freeze_rejects_drifted_and_unauthorized_inputs(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    output = root / "artifacts/task06/same_prompt_expansion_v2"
    _rewrite_config(root, lambda payload: payload["design"].__setitem__("config_sha256", "a" * 64))
    with pytest.raises(ValueError, match="execution design drifted"):
        freeze_same_prompt_expansion_cohort(_config(root), output)

    root = _repository(tmp_path / "auth")
    _rewrite_config(
        root,
        lambda payload: payload["authorization"].__setitem__("cohort_freeze_authorized", False),
    )
    with pytest.raises(ValueError, match="not authorized"):
        freeze_same_prompt_expansion_cohort(
            _config(root), root / "artifacts/task06/same_prompt_expansion_v2"
        )

    root = _repository(tmp_path / "pairs")
    _rewrite_config(
        root,
        lambda payload: payload["authorization"].__setitem__(
            "tentative_pair_build_authorized", True
        ),
    )
    with pytest.raises(ValueError, match="must not authorize pair building"):
        freeze_same_prompt_expansion_cohort(
            _config(root), root / "artifacts/task06/same_prompt_expansion_v2"
        )

    root = _repository(tmp_path / "prior")
    _rewrite_config(
        root,
        lambda payload: payload["cohort"]["exclude_prior_cohort_ids"][0].__setitem__(
            "sha256", "b" * 64
        ),
    )
    with pytest.raises(ValueError, match="ID manifest drifted"):
        freeze_same_prompt_expansion_cohort(
            _config(root), root / "artifacts/task06/same_prompt_expansion_v2"
        )

    root = _repository(tmp_path / "records")
    _rewrite_config(
        root, lambda payload: payload["cohort"].__setitem__("source_records_sha256", "c" * 64)
    )
    with pytest.raises(ValueError, match="canonical train records drifted"):
        freeze_same_prompt_expansion_cohort(
            _config(root), root / "artifacts/task06/same_prompt_expansion_v2"
        )

    root = _repository(tmp_path / "exclusion")
    _rewrite_config(
        root,
        lambda payload: payload["cohort"].__setitem__("exclude_adapter_training_clusters", False),
    )
    with pytest.raises(ValueError, match="must be excluded"):
        freeze_same_prompt_expansion_cohort(
            _config(root), root / "artifacts/task06/same_prompt_expansion_v2"
        )


def test_cohort_freeze_fails_closed_when_legal_pool_is_too_small(tmp_path: Path) -> None:
    root = _repository(tmp_path, prior_clusters=[str(1000 + index) for index in range(20)])
    with pytest.raises(RuntimeError, match="insufficient legal cluster-unique"):
        freeze_same_prompt_expansion_cohort(
            _config(root), root / "artifacts/task06/same_prompt_expansion_v2"
        )


def _v2_generation_config(root: Path) -> None:
    """Complete the frozen v2 config with the generator block the runner needs."""
    _rewrite_config(
        root,
        lambda payload: payload.__setitem__(
            "generator",
            {
                "role": "d01_controlled",
                "experiment_id": "TASK06-PREFERENCE-D01-SAME-PROMPT-V2",
                "config": "configs/experiments/d01b_scale_pilot_d01_4_5b_s42.yaml",
                "adapter": "runs/D01-4.5B-STYLE-50K-S42/adapter",
                "prompts_per_passage": 1,
                "candidates_per_prompt": 8,
                "exact_same_prompt_required": True,
                "controls": [
                    {"form": "full_question", "intent": "fact_lookup", "focus": "beginning"}
                ]
                * 4,
                "decoding": [
                    {"slot": index, "temperature": 0.6, "top_p": 0.97, "seed": 7600 + index}
                    for index in range(8)
                ],
                "max_new_tokens": 64,
                "generation_batch_size": 8,
            },
        ),
    )


def test_v2_generation_requires_a_previously_frozen_cohort(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    _v2_generation_config(root)
    output = root / "artifacts/task06/same_prompt_expansion_v2"
    with pytest.raises(ValueError, match="requires a previously frozen cohort"):
        generate_same_prompt_expansion(_config(root), output)


def test_v2_generation_refuses_unauthorized_or_drifted_cohort(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    output = root / "artifacts/task06/same_prompt_expansion_v2"
    freeze_same_prompt_expansion_cohort(_config(root), output)
    _v2_generation_config(root)
    manifest_path = output / "cohort.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    drifted = {**manifest, "records_sha256": "d" * 64}
    write_json(manifest_path, drifted)
    with pytest.raises(ValueError, match="cohort drifted"):
        generate_same_prompt_expansion(_config(root), output)

    write_json(manifest_path, {**manifest, "pairs_built": True})
    with pytest.raises(ValueError, match="not pre-generation"):
        generate_same_prompt_expansion(_config(root), output)

    write_json(manifest_path, manifest)
    _rewrite_config(
        root,
        lambda payload: payload["authorization"].__setitem__("generation_authorized", False),
    )
    with pytest.raises(ValueError, match="generation is not authorized"):
        generate_same_prompt_expansion(_config(root), output)


def test_v2_generation_accepts_the_frozen_cohort_and_stops_at_model_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repository(tmp_path)
    output = root / "artifacts/task06/same_prompt_expansion_v2"
    generation_config = root / "configs/experiments/d01b_scale_pilot_d01_4_5b_s42.yaml"
    generation_config.parent.mkdir(parents=True)
    generation_config.write_text("experiment_id: D01-TEST\n", encoding="utf-8")
    _v2_generation_config(root)
    freeze_same_prompt_expansion_cohort(_config(root), output)

    class _Reached(RuntimeError):
        pass

    def _reached(*_args: Any, **_kwargs: Any) -> Any:
        raise _Reached("model loading reached after validation")

    monkeypatch.setattr("doc2query.preferences.task06_smoke.load_config", lambda _path: object())
    monkeypatch.setattr("doc2query.preferences.task06_smoke.load_tokenizer", _reached)
    monkeypatch.setattr("doc2query.preferences.task06_smoke.load_generator", _reached)
    with pytest.raises(_Reached):
        generate_same_prompt_expansion(_config(root), output)
    identity = json.loads(
        (output / "d01_controlled/generations.jsonl.identity.json").read_text(encoding="utf-8")
    )
    assert identity["contract"] == "task06-same-prompt-preference-expansion-v2"
    assert identity["exact_same_prompt_required"] is True
    assert identity["final_tests_used"] == []
    assert len(identity["decoding"]) == 8
    assert not (output / "d01_controlled/generations.jsonl").exists()


def _fake_generation_backend(
    monkeypatch: pytest.MonkeyPatch, calls: list[int], *, fail_after: int | None = None
) -> None:
    """Replace model loading and decoding with deterministic, CPU-only stubs."""

    def fake_generate(
        _model: Any,
        _tokenizer: Any,
        prompts: list[list[int]],
        *,
        seeds: list[int],
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ) -> list[str]:
        calls.append(len(prompts))
        if fail_after is not None and len(calls) > fail_after:
            raise KeyboardInterrupt("simulated interruption")
        return [f"zapytanie {seed} t{temperature}" for seed in seeds]

    monkeypatch.setattr(
        "doc2query.preferences.task06_smoke.generate_text_batch_seeded", fake_generate
    )
    training = type("T", (), {"max_length": 512, "min_prompt_tokens": 16})()
    monkeypatch.setattr(
        "doc2query.preferences.task06_smoke.load_config",
        lambda _path: type("C", (), {"training": training})(),
    )
    monkeypatch.setattr(
        "doc2query.preferences.task06_smoke.load_tokenizer", lambda _config: object()
    )
    monkeypatch.setattr(
        "doc2query.preferences.task06_smoke.load_generator",
        lambda _config, for_training: (
            type("M", (), {"eval": lambda self: None})(),
            type("P", (), {"label": "bf16"})(),
        ),
    )
    monkeypatch.setattr("peft.PeftModel.from_pretrained", lambda model, *_a, **_k: model)
    monkeypatch.setattr(
        "doc2query.preferences.task06_smoke.render_controlled_prompt",
        lambda passage, control: f"prompt::{passage}::{control.form.value}",
    )
    monkeypatch.setattr(
        "doc2query.preferences.task06_smoke._prompt_ids", lambda *_a, **_k: [1, 2, 3]
    )
    monkeypatch.setattr("torch.cuda.max_memory_allocated", lambda: 0)
    monkeypatch.setattr("torch.cuda.max_memory_reserved", lambda: 0)


def test_v2_generation_resumes_after_an_interruption_without_losing_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repository(tmp_path)
    output = root / "artifacts/task06/same_prompt_expansion_v2"
    generation_config = root / "configs/experiments/d01b_scale_pilot_d01_4_5b_s42.yaml"
    generation_config.parent.mkdir(parents=True)
    generation_config.write_text("experiment_id: D01-TEST\n", encoding="utf-8")
    _v2_generation_config(root)
    _rewrite_config(
        root, lambda payload: payload["generator"].__setitem__("generation_batch_size", 2)
    )
    freeze_same_prompt_expansion_cohort(_config(root), output)
    total_rows = COHORT_SIZE * 8
    journal = output / "d01_controlled/generations.jsonl.journal.jsonl"

    interrupted: list[int] = []
    _fake_generation_backend(monkeypatch, interrupted, fail_after=5)
    with pytest.raises(KeyboardInterrupt):
        generate_same_prompt_expansion(_config(root), output)
    durable = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    assert len(durable) == 10
    assert not (output / "d01_controlled/generations.jsonl").exists()

    resumed: list[int] = []
    _fake_generation_backend(monkeypatch, resumed)
    summary = generate_same_prompt_expansion(_config(root), output)
    assert summary["status"] == "same_prompt_generation_complete"
    assert summary["generation_count"] == total_rows
    assert summary["resumed_generation_count"] == 10
    assert sum(resumed) == total_rows - 10
    rows = [
        json.loads(line)
        for line in (output / "d01_controlled/generations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(rows) == total_rows
    assert rows[:10] == durable
    assert all(row["experiment_id"] == "TASK06-PREFERENCE-D01-SAME-PROMPT-V2" for row in rows)
    assert all(row["final_tests_used"] == [] for row in rows)
    assert len({row["evaluation_id"] for row in rows}) == total_rows

    repeated: list[int] = []
    _fake_generation_backend(monkeypatch, repeated)
    again = generate_same_prompt_expansion(_config(root), output)
    assert again == summary
    assert repeated == []


def _prepared_v2_repo(tmp_path: Path) -> tuple[Path, Path]:
    root = _repository(tmp_path)
    output = root / "artifacts/task06/same_prompt_expansion_v2"
    generation_config = root / "configs/experiments/d01b_scale_pilot_d01_4_5b_s42.yaml"
    generation_config.parent.mkdir(parents=True)
    generation_config.write_text("experiment_id: D01-TEST\n", encoding="utf-8")
    _v2_generation_config(root)
    return root, output


def test_v2_generation_retries_a_malformed_completion_on_a_new_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, output = _prepared_v2_repo(tmp_path)
    freeze_same_prompt_expansion_cohort(_config(root), output)
    calls: list[int] = []
    _fake_generation_backend(monkeypatch, calls)

    def flaky(model: Any, tokenizer: Any, prompts: list[list[int]], **kwargs: Any) -> list[str]:
        calls.append(len(prompts))
        return [
            "wiersz pierwszy\nwiersz drugi"
            if index == 0 and seed < 7_000_000
            else f"zapytanie {seed}"
            for index, seed in enumerate(kwargs["seeds"])
        ]

    monkeypatch.setattr("doc2query.preferences.task06_smoke.generate_text_batch_seeded", flaky)
    summary = generate_same_prompt_expansion(_config(root), output)
    rows = [
        json.loads(line)
        for line in (output / "d01_controlled/generations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    retried = [row for row in rows if row["attempt"] > 1]
    assert retried, "the malformed first slot of every batch must be resampled"
    assert all(row["invalid_attempts"] == 1 for row in retried)
    assert all(row["format_repair"] == "none" for row in retried)
    assert all("\n" not in row["generated"] and row["generated"] for row in rows)
    assert all(row["seed"] >= 7_000_000 for row in retried)
    assert summary["retried_row_count"] == len(retried)
    assert summary["invalid_completion_count"] == len(retried)
    assert summary["format_repair_counts"] == {
        "none": len(rows),
        "first_line": 0,
        "empty": 0,
    }
    assert summary["max_attempts_per_slot"] == 4


def test_v2_generation_repairs_only_after_exhausting_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, output = _prepared_v2_repo(tmp_path)
    freeze_same_prompt_expansion_cohort(_config(root), output)
    attempts: list[int] = []
    _fake_generation_backend(monkeypatch, attempts)

    def always_multiline(
        model: Any, tokenizer: Any, prompts: list[list[int]], **kwargs: Any
    ) -> list[str]:
        attempts.append(len(prompts))
        return [f"  \n zapytanie {seed} \nogon do odrzucenia" for seed in kwargs["seeds"]]

    monkeypatch.setattr(
        "doc2query.preferences.task06_smoke.generate_text_batch_seeded", always_multiline
    )
    summary = generate_same_prompt_expansion(_config(root), output)
    rows = [
        json.loads(line)
        for line in (output / "d01_controlled/generations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert summary["generation_count"] == COHORT_SIZE * 8
    assert summary["format_repair_counts"]["first_line"] == len(rows)
    assert all(row["attempt"] == 4 for row in rows)
    assert all(row["invalid_attempts"] == 4 for row in rows)
    assert all(row["generated"].startswith("zapytanie ") for row in rows)
    assert all("ogon" not in row["generated"] for row in rows)


def test_v2_generation_rejects_an_out_of_contract_batch_size(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    output = root / "artifacts/task06/same_prompt_expansion_v2"
    generation_config = root / "configs/experiments/d01b_scale_pilot_d01_4_5b_s42.yaml"
    generation_config.parent.mkdir(parents=True)
    generation_config.write_text("experiment_id: D01-TEST\n", encoding="utf-8")
    _v2_generation_config(root)
    _rewrite_config(
        root, lambda payload: payload["generator"].__setitem__("generation_batch_size", 16)
    )
    freeze_same_prompt_expansion_cohort(_config(root), output)
    with pytest.raises(ValueError, match="batch size must be between 1 and 8"):
        generate_same_prompt_expansion(_config(root), output)


def test_v2_generation_detects_config_drift_after_cohort_freeze(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    output = root / "artifacts/task06/same_prompt_expansion_v2"
    freeze_same_prompt_expansion_cohort(_config(root), output)
    _v2_generation_config(root)
    _rewrite_config(root, lambda payload: payload["cohort"].__setitem__("passage_count", 3))
    with pytest.raises(ValueError, match="config drifted from the frozen cohort"):
        generate_same_prompt_expansion(_config(root), output)


def test_cohort_freeze_refuses_incomplete_prior_state(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    output = root / "artifacts/task06/same_prompt_expansion_v2"
    output.mkdir(parents=True)
    (output / "cohort.records.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(FileExistsError, match="incomplete"):
        freeze_same_prompt_expansion_cohort(_config(root), output)
