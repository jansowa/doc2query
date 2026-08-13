from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from doc2query.evaluation.d01_probe import (
    preflight_d01b_probe_dev_confirm,
    preflight_d01b_probe_dev_screen,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    return _write(path, "".join(json.dumps(row) + "\n" for row in rows))


def _fixture(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    _write(root / "AGENTS.md", "fixture\n")
    comparison = root / "configs/evaluation/comparison_contract_v1.yaml"
    comparison.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path("configs/evaluation/comparison_contract_v1.yaml"), comparison)
    comparison_adr = root / "reports/decisions/task04_p04_statistical_budget_contract.md"
    comparison_adr.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        Path("reports/decisions/task04_p04_statistical_budget_contract.md"), comparison_adr
    )
    probe = root / "configs/evaluation/probe_v1.yaml"
    shutil.copyfile(Path("configs/evaluation/probe_v1.yaml"), probe)
    judge = _write(root / "judge.yaml", "name_or_path: judge\n")
    adr = _write(root / "reports/decisions/probe.md", "frozen before training\n")
    amendment = _write(root / "reports/decisions/amendment.md", "batch 4 restart\n")
    execution_amendment = _write(
        root / "reports/decisions/execution-amendment.md", "encode batch 32\n"
    )
    corpus = _write(root / "data/documents.parquet", "fixture\n")
    ids = _jsonl(root / "data/dev.ids.jsonl", [{"id": "eval-1"}, {"id": "eval-2"}])
    manifest = _write(
        root / "data/manifest.json",
        json.dumps(
            {
                "sets": {
                    "dev_intrinsic_rank10": {
                        "id_count": 2,
                        "id_list_sha256": "i" * 64,
                        "id_path": str(ids.relative_to(root)),
                    }
                }
            }
        ),
    )
    guardrails = _jsonl(
        root / "guardrails.jsonl",
        [{"example_id": "eval-1"}, {"example_id": "eval-2"}],
    )
    rows = []
    for group in range(496):
        for index in range(4):
            pair_id = f"pair-{group}-{index}"
            rows.append(
                {
                    "pair_id": pair_id,
                    "example_id": pair_id,
                    "query": pair_id,
                    "generated": pair_id,
                    "mode": "deterministic",
                    "candidate_index": 0,
                    "positives": [{"doc_id": f"doc-{group}", "text": "positive"}],
                    "hard_negatives": [{"doc_id": "negative", "text": "negative"}],
                    "source_example_id": f"train-{group}",
                    "source_passage_id": f"doc-{group}",
                }
            )
    control = _jsonl(root / "control.jsonl", rows)
    variant = _jsonl(root / "variant.jsonl", rows)
    source = _write(
        root / "source.json",
        json.dumps(
            {
                "contract": "task05-d01b-prospective-probe-inputs-v1",
                "status": "materialized_and_cpu_validated",
                "training_started": False,
                "final_tests_used": [],
            }
        ),
    )
    config = {
        "schema_version": 1,
        "contract": "task05-d01b-probe-dev-screen-v1",
        "status": "amended_before_restart",
        "final_tests_used": [],
        "adr": {"path": str(adr.relative_to(root)), "sha256": _sha(adr)},
        "amendment": {
            "path": str(amendment.relative_to(root)),
            "sha256": _sha(amendment),
        },
        "execution_amendment": {
            "path": str(execution_amendment.relative_to(root)),
            "sha256": _sha(execution_amendment),
        },
        "source_materialization": {
            "path": str(source.relative_to(root)),
            "sha256": _sha(source),
            "contract": "task05-d01b-prospective-probe-inputs-v1",
            "status": "materialized_and_cpu_validated",
            "full_pair_count": 1984,
        },
        "arms": {
            "control": {
                "id": "D01B-PROBE-W05-DEV-SCREEN-S42-B4",
                "input": str(control.relative_to(root)),
                "sha256": _sha(control),
            },
            "variant": {
                "id": "D01B-PROBE-HYBRID-DEV-SCREEN-S42-B4",
                "input": str(variant.relative_to(root)),
                "sha256": _sha(variant),
            },
        },
        "training": {
            "stage": "dev_screen",
            "seed": 42,
            "max_steps": 500,
            "batch_size": 4,
            "train_prefix_pairs": 1984,
            "train_prefix_unique_passages": 496,
            "queries_per_passage": 4,
            "token_count": 1152000,
            "probe_recipe": str(probe.relative_to(root)),
            "probe_recipe_sha256": _sha(probe),
            "comparison_contract": str(comparison.relative_to(root)),
            "comparison_contract_sha256": _sha(comparison),
            "primary_judge": str(judge.relative_to(root)),
            "primary_judge_sha256": _sha(judge),
        },
        "evaluation": {
            "frozen_manifest": str(manifest.relative_to(root)),
            "frozen_manifest_sha256": _sha(manifest),
            "subset": "dev_intrinsic_rank10",
            "subset_id_count": 2,
            "subset_id_list_sha256": "i" * 64,
            "corpus": str(corpus.relative_to(root)),
            "corpus_sha256": _sha(corpus),
            "natural_guardrails": str(guardrails.relative_to(root)),
            "natural_guardrails_sha256": _sha(guardrails),
            "final_tests_forbidden": True,
        },
        "outputs": {
            "run_root": "runs/task05_d01b_probe_dev_screen_v2_batch4",
            "measurement_root": "reports/measurements/task05/d01b_probe_dev_screen_v2_batch4",
            "log_root": "logs/task05/d01b_probe_dev_screen_v2_batch4",
        },
        "execution": {
            "evaluation_encode_batch_size": 32,
            "retrieval_query_batch_size": 512,
            "retrieval_device": "cuda",
        },
        "authorization": {
            "dev_screen_training": True,
            "dev_confirm": False,
            "four_point_five_b": False,
            "final_tests": False,
        },
    }
    path = root / "configs/evaluation/probe_screen.yaml"
    _write(path, yaml.safe_dump(config, sort_keys=False))
    return path


def test_probe_dev_screen_preflight_accepts_exact_common_prefix(tmp_path: Path) -> None:
    result = preflight_d01b_probe_dev_screen(_fixture(tmp_path))
    assert result["status"] == "verified"
    assert result["arms"]["control"]["prefix_unique_passages"] == 496
    assert result["probe_recipe_fingerprint"] != result["base_probe_recipe_fingerprint"]
    assert result["final_tests_used"] == []


def test_probe_dev_screen_preflight_rejects_final_test_authorization(tmp_path: Path) -> None:
    path = _fixture(tmp_path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["authorization"]["final_tests"] = True
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="authorization scope"):
        preflight_d01b_probe_dev_screen(path)


def _dev_confirm_fixture(tmp_path: Path) -> Path:
    path = _fixture(tmp_path)
    root = path.parents[2]
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    rows = []
    for group in range(1984):
        for index in range(4):
            pair_id = f"confirm-pair-{group}-{index}"
            rows.append(
                {
                    "pair_id": pair_id,
                    "example_id": pair_id,
                    "query": pair_id,
                    "generated": pair_id,
                    "mode": "deterministic",
                    "candidate_index": 0,
                    "positives": [{"doc_id": f"doc-{group}", "text": "positive"}],
                    "hard_negatives": [{"doc_id": "negative", "text": "negative"}],
                    "source_example_id": f"confirm-train-{group}",
                    "source_passage_id": f"doc-{group}",
                }
            )
    control = _jsonl(root / "control.jsonl", rows)
    variant = _jsonl(root / "variant.jsonl", rows)
    source_screen = _write(
        root / "reports/measurements/dev-screen/summary.json",
        json.dumps(
            {
                "contract": "task05-d01b-probe-dev-screen-v1",
                "status": "dev_screen_complete",
                "dev_confirm_authorized": True,
                "four_point_five_b_authorized": False,
                "final_tests_used": [],
                "decision": {
                    "stage": "dev_screen",
                    "status": "eligible",
                    "errors": [],
                },
            },
            sort_keys=True,
        )
        + "\n",
    )

    restart_amendment = _write(
        root / "reports/decisions/dev-confirm-batch2-amendment.md",
        "batch 2 / 4000 steps hardware restart amendment\n",
    )
    config["contract"] = "task05-d01b-probe-dev-confirm-v1"
    config["status"] = "amended_before_restart"
    config.pop("amendment")
    config.pop("execution_amendment")
    config["restart_amendment"] = {
        "path": str(restart_amendment.relative_to(root)),
        "sha256": _sha(restart_amendment),
    }
    config["source_dev_screen"] = {
        "path": str(source_screen.relative_to(root)),
        "sha256": _sha(source_screen),
        "contract": "task05-d01b-probe-dev-screen-v1",
        "status": "dev_screen_complete",
        "required_decision_status": "eligible",
    }
    config["source_materialization"]["full_pair_count"] = 7936
    config["source_materialization"]["full_unique_passage_count"] = 1984
    config["source_materialization"]["queries_per_passage"] = 4
    config["arms"] = {
        "control": {
            "comparison_id": "D01B-PROBE-W05-DEV-CONFIRM-B2",
            "generator_id": "W05-1.5B-50K-8GB",
            "input": str(control.relative_to(root)),
            "sha256": _sha(control),
            "runs": {
                "42": "D01B-PROBE-W05-DEV-CONFIRM-S42-B2",
                "43": "D01B-PROBE-W05-DEV-CONFIRM-S43-B2",
                "44": "D01B-PROBE-W05-DEV-CONFIRM-S44-B2",
            },
        },
        "variant": {
            "comparison_id": "D01B-PROBE-HYBRID-DEV-CONFIRM-B2",
            "generator_id": "D01B-PROSPECTIVE-V3-HYBRID-1.5B-S42",
            "input": str(variant.relative_to(root)),
            "sha256": _sha(variant),
            "runs": {
                "42": "D01B-PROBE-HYBRID-DEV-CONFIRM-S42-B2",
                "43": "D01B-PROBE-HYBRID-DEV-CONFIRM-S43-B2",
                "44": "D01B-PROBE-HYBRID-DEV-CONFIRM-S44-B2",
            },
        },
    }
    config["training"] = {
        **config["training"],
        "stage": "dev_confirm",
        "seeds": [42, 43, 44],
        "budget_fraction": 1.0,
        "max_steps": 4000,
        "batch_size": 2,
        "max_length": 192,
        "train_pair_count": 7936,
        "train_unique_passages": 1984,
        "queries_per_passage": 4,
        "token_count": 4608000,
        "definition_version": "probe-budget-v1",
        "checkpoint_interval_steps": 100,
    }
    config["training"].pop("seed", None)
    config["training"].pop("train_prefix_pairs", None)
    config["training"].pop("train_prefix_unique_passages", None)
    config["evaluation"]["bootstrap_samples"] = 10000
    config["evaluation"]["bootstrap_seed"] = 20260721
    config["evaluation"]["fixed_seed_aggregation"] = "per_query_mean_before_query_resampling"
    config["outputs"] = {
        "run_root": "runs/task05_d01b_probe_dev_confirm_v2_batch2",
        "measurement_root": "reports/measurements/task05/d01b_probe_dev_confirm_v2_batch2",
        "log_root": "logs/task05/d01b_probe_dev_confirm_v2_batch2",
    }
    config["execution"]["evaluation_encode_batch_size"] = 8
    config["authorization"] = {
        "dev_screen_training": False,
        "dev_confirm_training": True,
        "four_point_five_b": False,
        "final_tests": False,
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_probe_dev_confirm_preflight_accepts_full_budget_three_seed_contract(
    tmp_path: Path,
) -> None:
    result = preflight_d01b_probe_dev_confirm(_dev_confirm_fixture(tmp_path))
    assert result["status"] == "verified"
    assert result["seeds"] == [42, 43, 44]
    assert result["arms"]["control"]["full_pair_count"] == 7936
    assert result["arms"]["variant"]["full_unique_passages"] == 1984
    assert result["final_tests_used"] == []


def test_probe_dev_confirm_preflight_requires_eligible_dev_screen(tmp_path: Path) -> None:
    path = _dev_confirm_fixture(tmp_path)
    root = path.parents[2]
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    source_path = root / config["source_dev_screen"]["path"]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["decision"]["status"] = "non_inferior_only"
    source_path.write_text(json.dumps(source, sort_keys=True) + "\n", encoding="utf-8")
    config["source_dev_screen"]["sha256"] = _sha(source_path)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="does not authorize"):
        preflight_d01b_probe_dev_confirm(path)
