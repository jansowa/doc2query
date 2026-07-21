"""Plan-only P-05 probe matrix with P-04 and campaign completion gates."""

from __future__ import annotations

import json
import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from doc2query.evaluation.corpus import sha256_file
from doc2query.evaluation.p05_materializer import (
    MATERIALIZATION_SCHEMA_VERSION,
    W05_GENERATOR_ID,
)
from doc2query.evaluation.statistical_contract import (
    BUDGET_DEFINITION_VERSION,
    BUDGET_FIELDS,
    StatisticalContract,
)


def _command(parts: list[str]) -> str:
    return shlex.join(parts)


def _load_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _validate_budget(value: Mapping[str, Any], errors: list[str]) -> dict[str, Any]:
    valid_dimensions = True
    if value.get("definition_version") != BUDGET_DEFINITION_VERSION:
        errors.append(f"budget definition must be {BUDGET_DEFINITION_VERSION}")
        valid_dimensions = False
    for field in BUDGET_FIELDS:
        if not isinstance(value.get(field), int) or int(value[field]) < 1:
            errors.append(f"budget requires positive integer {field}")
            valid_dimensions = False
    if not valid_dimensions:
        return dict(value)
    if value["pair_count"] != value["unique_passage_count"] * value["queries_per_passage"]:
        errors.append("budget pair_count must equal unique_passage_count * queries_per_passage")
    if int(value["pair_count"]) % 2:
        errors.append("50/50 P-05 mixture requires an even pair_count")
    if int(value["pair_count"]) % 8:
        errors.append("50/50 dev_screen prefix requires pair_count divisible by eight")
    if any(int(value[field]) % 4 for field in BUDGET_FIELDS[:3]):
        errors.append(
            "dev_screen 0.25 budget requires token_count, pair_count and "
            "unique_passage_count divisible by four"
        )
    return dict(value)


def _probe_recipe_steps_and_tokens(
    path: Path | None, comparison_budget: Mapping[str, Any], errors: list[str]
) -> tuple[int, int]:
    if path is None or not path.is_file():
        return 0, 0
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        errors.append("probe_recipe must be a YAML mapping")
        return 0, 0
    fields: dict[str, int] = {}
    for field in ("max_steps", "batch_size", "max_length", "negatives_per_example"):
        value = raw.get(field)
        if not isinstance(value, int) or value < 1:
            errors.append(f"probe_recipe requires positive integer {field}")
            return 0, 0
        fields[field] = value
    full_steps = fields["max_steps"]
    full_tokens = (
        full_steps
        * fields["batch_size"]
        * fields["max_length"]
        * (2 + fields["negatives_per_example"])
    )
    if comparison_budget.get("token_count") != full_tokens:
        errors.append("P-04 token_count does not match the frozen probe recipe")
    if full_steps % 4:
        errors.append("dev_screen 0.25 budget requires probe max_steps divisible by four")
    return full_steps, full_tokens


def _validate_w05_adapter(path: Path | None, errors: list[str]) -> None:
    if path is None or not path.is_dir():
        return
    weights = (path / "adapter_model.safetensors").is_file() or (
        path / "adapter_model.bin"
    ).is_file()
    if not (path / "adapter_config.json").is_file() or not weights:
        errors.append("w05_adapter is incomplete")


def _validate_materialization_manifest(
    path: Path | None,
    artifacts: Mapping[str, Path],
    comparison_budget: Mapping[str, Any],
    errors: list[str],
) -> None:
    if path is None or not path.is_file():
        return
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        errors.append("p05_materialization_manifest is invalid JSON")
        return
    if not isinstance(raw, Mapping):
        errors.append("p05_materialization_manifest must be a JSON object")
        return
    if raw.get("schema_version") != MATERIALIZATION_SCHEMA_VERSION:
        errors.append("p05_materialization_manifest has an unsupported schema")
    if raw.get("materialization_id") != "task03-p05-common-cohort-v1":
        errors.append("p05_materialization_manifest has an incompatible materialization_id")
    cohort_fingerprint = raw.get("cohort_fingerprint")
    if (
        not isinstance(cohort_fingerprint, str)
        or len(cohort_fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in cohort_fingerprint)
    ):
        errors.append("p05_materialization_manifest requires a SHA-256 cohort fingerprint")
    if not isinstance(raw.get("seed"), int):
        errors.append("p05_materialization_manifest requires an integer seed")
    if raw.get("final_tests_used") != []:
        errors.append("p05_materialization_manifest must declare final_tests_used=[]")
    if raw.get("comparison_budget") != comparison_budget:
        errors.append("p05_materialization_manifest P-04 budget drift")
    if raw.get("negative_recipe") != {
        "strategy": "HN0+filter",
        "false_negative_policy": "drop",
    }:
        errors.append("p05_materialization_manifest must pin HN0+filter/drop")
    half = int(comparison_budget.get("pair_count", 0)) // 2
    stage_counts = raw.get("stage_pair_counts")
    expected_stages = {
        "dev_screen": {"natural": half // 4, "synthetic_w05": half // 4},
        "dev_confirm": {"natural": half, "synthetic_w05": half},
    }
    if stage_counts != expected_stages:
        errors.append("p05_materialization_manifest does not prove 50/50 halving prefixes")
    outputs = raw.get("outputs")
    if not isinstance(outputs, Mapping):
        errors.append("p05_materialization_manifest requires outputs")
        return
    output_keys = {
        "train_input": "gold_natural",
        "w05_synthetic_generations": "w05_synthetic",
        "mixed_50_50_generations": "mixed_50_50",
    }
    expected_source_counts = {
        "gold_natural": {"natural": int(comparison_budget.get("pair_count", 0))},
        "w05_synthetic": {"synthetic_w05": int(comparison_budget.get("pair_count", 0))},
        "mixed_50_50": {"natural": half, "synthetic_w05": half},
    }
    for artifact_key, output_key in output_keys.items():
        artifact_path = artifacts.get(artifact_key)
        output = outputs.get(output_key)
        if artifact_path is None or not artifact_path.is_file() or not isinstance(output, Mapping):
            errors.append(f"p05_materialization_manifest missing output {output_key}")
            continue
        declared_path = output.get("path")
        if (
            not isinstance(declared_path, str)
            or Path(declared_path).resolve() != artifact_path.resolve()
        ):
            errors.append(f"p05 materialized path drift: {artifact_key}")
        expected_sha = output.get("sha256")
        if not isinstance(expected_sha, str) or expected_sha != sha256_file(artifact_path):
            errors.append(f"p05 materialized SHA-256 drift: {artifact_key}")
        for field in BUDGET_FIELDS:
            if output.get(field) != comparison_budget.get(field):
                errors.append(f"p05 materialized budget drift: {artifact_key}.{field}")
        if output.get("source_counts") != expected_source_counts[output_key]:
            errors.append(f"p05 materialized source-count drift: {artifact_key}")
    synthetic = outputs.get("w05_synthetic")
    inputs = raw.get("inputs")
    w05_input = inputs.get("w05_generations") if isinstance(inputs, Mapping) else None
    if (
        not isinstance(synthetic, Mapping)
        or synthetic.get("source_counts")
        != {"synthetic_w05": int(comparison_budget.get("pair_count", 0))}
        or not isinstance(w05_input, Mapping)
        or w05_input.get("generator_id") != W05_GENERATOR_ID
    ):
        errors.append("p05_materialization_manifest does not pin W05 provenance")


def build_p05_plan(
    *,
    campaign_audit: Mapping[str, Any],
    contract: StatisticalContract,
    comparison_budget: Mapping[str, Any],
    artifacts: Mapping[str, Path],
    output_root: Path = Path("runs/p05_probe_matrix"),
) -> dict[str, Any]:
    """Validate and return commands; never execute, train, generate or open final tests."""
    errors: list[str] = []
    if campaign_audit.get("complete") is not True:
        errors.append("base/instruct campaign audit is incomplete")
    if campaign_audit.get("selection_performed") is not False:
        errors.append("campaign completion audit must not contain a winner selection")
    budget = _validate_budget(comparison_budget, errors)
    required_artifacts = (
        "probe_recipe",
        "train_input",
        "frozen_dev_manifest",
        "corpus",
        "primary_judge_config",
        "w05_adapter",
        "w05_synthetic_generations",
        "mixed_50_50_generations",
        "p05_materialization_manifest",
    )
    normalized_artifacts: dict[str, str] = {}
    for name in required_artifacts:
        path = artifacts.get(name)
        if path is None or not path.exists():
            errors.append(f"missing required P-05 artifact: {name}")
        else:
            normalized_artifacts[name] = str(path)
    full_steps, _ = _probe_recipe_steps_and_tokens(artifacts.get("probe_recipe"), budget, errors)
    _validate_w05_adapter(artifacts.get("w05_adapter"), errors)
    _validate_materialization_manifest(
        artifacts.get("p05_materialization_manifest"), artifacts, budget, errors
    )

    reference = contract.reference()
    arms = (
        ("P05-GOLD-NATURAL", "natural", None, None, "natural-only gold-data control"),
        (
            "P05-W05-SYNTHETIC",
            "synthetic",
            "w05_synthetic_generations",
            "W05-1.5B-50K-8GB",
            "W05 synthetic-only",
        ),
        (
            "P05-MIXED50",
            "synthetic",
            "mixed_50_50_generations",
            "P05-W05-NATURAL-SYNTHETIC-50-50",
            "natural+synthetic 50/50 materialized mixture",
        ),
    )
    stages = (
        ("dev_screen", (42,), 0.25),
        ("dev_confirm", (42, 43, 44), 1.0),
    )
    plan_arms: list[dict[str, Any]] = []
    commands: list[str] = []
    ready = not errors
    for arm_id, source, generations_key, generator_id, description in arms:
        runs: list[dict[str, Any]] = []
        for stage, seeds, fraction in stages:
            stage_steps = int(full_steps * fraction)
            stage_budget = {
                **budget,
                "token_count": int(int(budget.get("token_count", 0)) * fraction),
                "pair_count": int(int(budget.get("pair_count", 0)) * fraction),
                "unique_passage_count": int(int(budget.get("unique_passage_count", 0)) * fraction),
            }
            for seed in seeds:
                output_dir = output_root / stage / f"{arm_id}-S{seed}"
                parts = [
                    "python",
                    "scripts/train_probe_embedder.py",
                    "--recipe",
                    normalized_artifacts.get("probe_recipe", f"<{required_artifacts[0]}>"),
                    "--comparison-contract",
                    "configs/evaluation/comparison_contract_v1.yaml",
                    "--train-input",
                    normalized_artifacts.get("train_input", "<train_input>"),
                    "--frozen-manifest",
                    normalized_artifacts.get("frozen_dev_manifest", "<frozen_dev_manifest>"),
                    "--test-subset",
                    "dev_intrinsic",
                    "--corpus",
                    normalized_artifacts.get("corpus", "<corpus>"),
                    "--query-source",
                    source,
                    "--seed",
                    str(seed),
                    "--max-steps",
                    str(stage_steps),
                    "--train-prefix-limit",
                    str(stage_budget["pair_count"]),
                    "--primary-judge-config",
                    normalized_artifacts.get("primary_judge_config", "<primary_judge_config>"),
                    "--output-dir",
                    str(output_dir),
                ]
                if generations_key is not None:
                    parts.extend(
                        [
                            "--synthetic-generations",
                            normalized_artifacts.get(generations_key, f"<{generations_key}>"),
                            "--generator-id",
                            str(generator_id),
                        ]
                    )
                command = _command(parts)
                runs.append(
                    {
                        "stage": stage,
                        "seed": seed,
                        "budget_fraction": fraction,
                        "max_steps": stage_steps,
                        "comparison_budget": stage_budget,
                        "negative_recipe": "HN0+filter",
                        "false_negative_policy": "drop",
                        "evaluation_sets": ["dev_intrinsic"],
                        "final_tests_used": [],
                        "output_dir": str(output_dir),
                        "command": command if ready else None,
                    }
                )
                if ready:
                    commands.append(command)
        plan_arms.append(
            {
                "arm_id": arm_id,
                "description": description,
                "source_mix": (
                    {"natural": 0.5, "synthetic_w05": 0.5}
                    if arm_id == "P05-MIXED50"
                    else {"natural": 1.0}
                    if source == "natural"
                    else {"synthetic_w05": 1.0}
                ),
                "full_comparison_budget": budget,
                "runs": runs,
            }
        )
    return {
        "schema_version": 1,
        "plan_id": "task03-p05-first-probe-matrix-v1",
        "mode": "plan_only",
        "execution_ready": ready,
        "execution_performed": False,
        "winner_selected": False,
        "statistical_contract": reference,
        "negative_recipe": {"strategy": "HN0+filter", "false_negative_policy": "drop"},
        "full_comparison_budget": budget,
        "required_unexecuted_controls": [
            {"id": "S00-zero-shot", "status": "required_unexecuted"},
            {"id": "S00-few-shot", "status": "required_unexecuted"},
            {"id": "S07-polish-seq2seq", "status": "required_unexecuted"},
        ],
        "artifacts": normalized_artifacts,
        "blockers": sorted(set(errors)),
        "arms": plan_arms,
        "execution_commands": commands,
        "final_tests_used": [],
    }


def write_p05_plan(plan: Mapping[str, Any], output: Path) -> None:
    """Persist a deterministic plan only to the caller-provided path."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_p05_planner_inputs(
    campaign_audit_path: Path, budget_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load explicit planner inputs without inspecting campaign outputs implicitly."""
    return (
        _load_object(campaign_audit_path, "campaign audit"),
        _load_object(budget_path, "comparison budget"),
    )
