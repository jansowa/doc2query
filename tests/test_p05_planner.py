from __future__ import annotations

import json
from pathlib import Path

from doc2query.evaluation.p05_planner import build_p05_plan
from doc2query.evaluation.statistical_contract import StatisticalContract, build_budget_manifest


def _artifacts(tmp_path: Path) -> dict[str, Path]:
    values: dict[str, Path] = {}
    for name in (
        "probe_recipe",
        "train_input",
        "frozen_dev_manifest",
        "corpus",
        "primary_judge_config",
        "w05_adapter",
        "w05_synthetic_generations",
        "mixed_50_50_generations",
        "mixed_50_50_manifest",
    ):
        path = tmp_path / "artifacts" / name
        if name == "w05_adapter":
            path.mkdir(parents=True, exist_ok=True)
            (path / "adapter_config.json").write_text("{}\n", encoding="utf-8")
            (path / "adapter_model.safetensors").write_bytes(b"fixture")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            if name == "probe_recipe":
                content = "max_steps: 12\nbatch_size: 1\nmax_length: 40\nnegatives_per_example: 1\n"
            elif name == "mixed_50_50_manifest":
                content = json.dumps(
                    {
                        "natural_pair_count": 200,
                        "synthetic_pair_count": 200,
                        "synthetic_generator_id": "W05-1.5B-50K-8GB",
                        "stage_pair_counts": {
                            "dev_screen": {"natural": 50, "synthetic_w05": 50},
                            "dev_confirm": {"natural": 200, "synthetic_w05": 200},
                        },
                    }
                )
            else:
                content = "fixture\n"
            path.write_text(content, encoding="utf-8")
        values[name] = path
    return values


def _budget() -> dict[str, object]:
    return build_budget_manifest(
        token_count=1440,
        pair_count=400,
        unique_passage_count=400,
        queries_per_passage=1,
    )


def test_p05_plan_has_three_matched_arms_hn0_filter_and_required_controls(
    tmp_path: Path,
) -> None:
    plan = build_p05_plan(
        campaign_audit={"complete": True, "selection_performed": False},
        contract=StatisticalContract.load(Path("configs/evaluation/comparison_contract_v1.yaml")),
        comparison_budget=_budget(),
        artifacts=_artifacts(tmp_path),
        output_root=tmp_path / "planned-runs",
    )
    assert plan["execution_ready"] is True
    assert plan["execution_performed"] is False
    assert len(plan["arms"]) == 3
    assert {arm["description"] for arm in plan["arms"]} == {
        "natural-only gold-data control",
        "W05 synthetic-only",
        "natural+synthetic 50/50 materialized mixture",
    }
    assert all(arm["full_comparison_budget"] == _budget() for arm in plan["arms"])
    assert plan["negative_recipe"] == {
        "strategy": "HN0+filter",
        "false_negative_policy": "drop",
    }
    assert {item["id"] for item in plan["required_unexecuted_controls"]} == {
        "S00-zero-shot",
        "S00-few-shot",
        "S07-polish-seq2seq",
    }
    assert all(
        item["status"] == "required_unexecuted" for item in plan["required_unexecuted_controls"]
    )
    assert len(plan["execution_commands"]) == 12
    assert any("--seed 43" in command for command in plan["execution_commands"])
    assert any("--max-steps 3" in command for command in plan["execution_commands"])
    assert any("--train-limit 100" in command for command in plan["execution_commands"])
    assert plan["final_tests_used"] == []


def test_incomplete_campaign_missing_artifact_and_bad_budget_block_commands(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts(tmp_path)
    artifacts["w05_synthetic_generations"].unlink()
    budget = _budget()
    budget["token_count"] = 1441
    plan = build_p05_plan(
        campaign_audit={"complete": False, "selection_performed": False},
        contract=StatisticalContract.load(Path("configs/evaluation/comparison_contract_v1.yaml")),
        comparison_budget=budget,
        artifacts=artifacts,
    )
    assert plan["execution_ready"] is False
    assert plan["execution_commands"] == []
    assert all(run["command"] is None for arm in plan["arms"] for run in arm["runs"])
    assert any("campaign" in blocker for blocker in plan["blockers"])
    assert any("w05_synthetic_generations" in blocker for blocker in plan["blockers"])
    assert any("divisible" in blocker for blocker in plan["blockers"])
