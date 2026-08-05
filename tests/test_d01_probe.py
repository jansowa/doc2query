from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from doc2query.evaluation.d01_probe import preflight_d01b_probe_dev_screen


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
        "status": "preregistered_before_training",
        "final_tests_used": [],
        "adr": {"path": str(adr.relative_to(root)), "sha256": _sha(adr)},
        "source_materialization": {
            "path": str(source.relative_to(root)),
            "sha256": _sha(source),
            "contract": "task05-d01b-prospective-probe-inputs-v1",
            "status": "materialized_and_cpu_validated",
            "full_pair_count": 1984,
        },
        "arms": {
            "control": {"input": str(control.relative_to(root)), "sha256": _sha(control)},
            "variant": {"input": str(variant.relative_to(root)), "sha256": _sha(variant)},
        },
        "training": {
            "stage": "dev_screen",
            "seed": 42,
            "max_steps": 250,
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
    assert result["final_tests_used"] == []


def test_probe_dev_screen_preflight_rejects_final_test_authorization(tmp_path: Path) -> None:
    path = _fixture(tmp_path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["authorization"]["final_tests"] = True
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="authorization scope"):
        preflight_d01b_probe_dev_screen(path)
