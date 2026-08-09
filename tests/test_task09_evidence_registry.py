from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

import doc2query.evaluation.evidence_registry as registry_module
from doc2query.evaluation.evidence_registry import (
    INCOMPLETE_STATUS,
    PARETO_STATUS,
    CampaignEvidenceRequirements,
    MetricCategory,
    MetricDirection,
    MetricRequirement,
    ParetoCandidate,
    build_campaign_evidence_registry,
    compute_pareto_front,
    load_experiment_evidence,
)
from doc2query.training.dpo import canonical_fingerprint, file_sha256

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
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _artifact(path: Path, *, method: str, count: int) -> dict[str, Any]:
    payload = {
        "path": path.name,
        "sha256": file_sha256(path),
        "record_count": count,
        "record_count_method": method,
        "provenance": {
            "source_task": "Task 04",
            "source_manifest_sha256": HEX["a"],
            "producer_git_commit": COMMIT,
        },
    }
    payload["artifact_fingerprint"] = canonical_fingerprint(payload)
    return payload


def _model_stack(*, tokenizer: str = "tokenizer-v1", base: str = "model-v1") -> dict[str, Any]:
    payload: dict[str, Any] = {
        "base_model": {
            "model_id": base,
            "revision": "revision-v1",
            "artifact_fingerprint": HEX["b"],
        },
        "adapter": {
            "adapter_id": "adapter-v1",
            "adapter_revision": "revision-v1",
            "adapter_fingerprint": HEX["c"],
            "base_model_fingerprint": HEX["b"],
        },
        "tokenizer": {
            "tokenizer_id": tokenizer,
            "revision": "revision-v1",
            "tokenizer_fingerprint": HEX["d"],
        },
    }
    payload["fingerprint"] = canonical_fingerprint(payload)
    payload["comparison_fingerprint"] = canonical_fingerprint(
        {
            "base_model": payload["base_model"],
            "adapter": {
                "adapter_id": payload["adapter"]["adapter_id"],
                "base_model_fingerprint": payload["adapter"]["base_model_fingerprint"],
            },
            "tokenizer": payload["tokenizer"],
        }
    )
    return payload


def _budget(*, tokens: int = 1000) -> dict[str, Any]:
    payload = {
        "definition_version": "task09-campaign-budget-v1",
        "token_count": tokens,
        "optimizer_steps": 20,
        "pair_count": 20,
        "unique_passage_count": 10,
        "queries_per_passage": 2,
    }
    payload["fingerprint"] = canonical_fingerprint(payload)
    return payload


def _metrics(*, quality: float, cost: float) -> list[dict[str, Any]]:
    definitions = {
        "intrinsic": HEX["a"],
        "probe_extrinsic": HEX["b"],
        "human": HEX["c"],
        "cost": HEX["d"],
    }
    values = {
        "intrinsic": ("source_mrr", "max", quality, "ratio"),
        "probe_extrinsic": ("corpus_ndcg_at_10", "max", quality, "ratio"),
        "human": ("preference_rate", "max", quality, "ratio"),
        "cost": ("seconds_per_query", "min", cost, "seconds"),
    }
    return [
        {
            "name": name,
            "category": category,
            "direction": direction,
            "value": value,
            "unit": unit,
            "definition_fingerprint": definitions[category],
            "ci": {
                "lower": value - 0.01,
                "upper": value + 0.01,
                "confidence_level": 0.95,
            },
            "sample_size": 100,
        }
        for category, (name, direction, value, unit) in values.items()
    ]


def _requirements(*, seeds: list[int] | None = None) -> CampaignEvidenceRequirements:
    raw = {
        "schema_version": 1,
        "contract": "task09-evidence-requirements-v1",
        "required_seeds": seeds or [42, 43],
        "required_metrics": [
            {
                "name": "source_mrr",
                "category": "intrinsic",
                "direction": "max",
                "unit": "ratio",
                "definition_fingerprint": HEX["a"],
            },
            {
                "name": "corpus_ndcg_at_10",
                "category": "probe_extrinsic",
                "direction": "max",
                "unit": "ratio",
                "definition_fingerprint": HEX["b"],
            },
            {
                "name": "preference_rate",
                "category": "human",
                "direction": "max",
                "unit": "ratio",
                "definition_fingerprint": HEX["c"],
            },
            {
                "name": "seconds_per_query",
                "category": "cost",
                "direction": "min",
                "unit": "seconds",
                "definition_fingerprint": HEX["d"],
            },
        ],
        "required_artifact_roles": ["metrics_records"],
        "require_human_evidence": True,
    }
    return CampaignEvidenceRequirements.model_validate(raw)


def _materialize_manifest(
    root: Path,
    *,
    arm: str,
    seed: int,
    quality: float = 0.5,
    cost: float = 1.0,
) -> Path:
    run_dir = root / f"{arm}-{seed}"
    config_path = run_dir / "config.json"
    config = {"learning_rate": 0.0001, "seed": seed}
    _write_json(config_path, config)
    records_path = run_dir / "metrics.jsonl"
    _write_jsonl(records_path, [{"query_id": "q-1"}, {"query_id": "q-2"}])
    config_artifact = _artifact(config_path, method="single_json", count=1)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "contract": "task09-experiment-evidence-v1",
        "experiment_id": "E09-fixture",
        "arm_id": arm,
        "stage_id": "dev_confirm",
        "run_status": "completed",
        "git_commit": COMMIT,
        "config": {
            "artifact": config_artifact,
            "format": "json",
            "fingerprint": canonical_fingerprint(config),
            "comparison_fingerprint": canonical_fingerprint(
                {key: value for key, value in config.items() if key != "seed"}
            ),
        },
        "dataset_id": "frozen-dev-v1",
        "dataset_fingerprint": HEX["a"],
        "split_id": "dev_intrinsic",
        "split_fingerprint": HEX["b"],
        "cohort_id": "cohort-v1",
        "cohort_fingerprint": HEX["c"],
        "model_stack": _model_stack(),
        "seed": seed,
        "budget": _budget(),
        "probe_recipe_fingerprint": HEX["e"],
        "artifacts": {
            "metrics_records": _artifact(records_path, method="jsonl", count=2),
        },
        "metrics": _metrics(quality=quality, cost=cost),
        "final_tests_used": [],
    }
    payload["manifest_fingerprint"] = canonical_fingerprint(payload)
    manifest_path = run_dir / "evidence_manifest.json"
    _write_json(manifest_path, payload)
    return manifest_path


def _rewrite_manifest(path: Path, mutate: Any) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    payload.pop("manifest_fingerprint", None)
    payload["manifest_fingerprint"] = canonical_fingerprint(payload)
    _write_json(path, payload)


def _registry(path: Path) -> dict[str, Any]:
    value = json.loads((path / "registry.json").read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _contains_forbidden_selection_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            key in {"winner", "scalar_score", "selected_arm"}
            or _contains_forbidden_selection_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_selection_key(item) for item in value)
    return False


def test_manifest_and_registry_are_deterministic(tmp_path: Path) -> None:
    paths = [
        _materialize_manifest(tmp_path / "evidence", arm=arm, seed=seed)
        for arm in ("baseline", "variant")
        for seed in (42, 43)
    ]
    first = tmp_path / "registry-one"
    second = tmp_path / "registry-two"
    build_campaign_evidence_registry(
        evidence_manifest_paths=list(reversed(paths)),
        requirements=_requirements(),
        output_dir=first,
    )
    build_campaign_evidence_registry(
        evidence_manifest_paths=paths,
        requirements=_requirements(),
        output_dir=second,
    )
    assert (first / "registry.json").read_bytes() == (second / "registry.json").read_bytes()
    assert (first / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()


def test_comparable_seeds_are_grouped_and_only_they_are_averaged(tmp_path: Path) -> None:
    paths = [
        _materialize_manifest(tmp_path, arm="baseline", seed=42, quality=0.4),
        _materialize_manifest(tmp_path, arm="baseline", seed=43, quality=0.6),
    ]
    output = tmp_path / "registry"
    build_campaign_evidence_registry(
        evidence_manifest_paths=paths,
        requirements=_requirements(),
        output_dir=output,
    )
    summary = _registry(output)["arm_summaries"][0]
    assert summary["observed_seeds"] == [42, 43]
    assert summary["evidence_complete"] is True
    assert summary["metric_means"]["probe_extrinsic:corpus_ndcg_at_10"] == pytest.approx(0.5)


def test_missing_and_duplicate_runs_are_fail_closed(tmp_path: Path) -> None:
    path = _materialize_manifest(tmp_path, arm="baseline", seed=42)
    output = tmp_path / "missing"
    build_campaign_evidence_registry(
        evidence_manifest_paths=[path], requirements=_requirements(), output_dir=output
    )
    summary = _registry(output)["arm_summaries"][0]
    assert summary["missing_seeds"] == [43]
    assert summary["metric_means"] == {}
    with pytest.raises(ValueError, match="duplicate experiment_id/arm_id/seed"):
        build_campaign_evidence_registry(
            evidence_manifest_paths=[path, path],
            requirements=_requirements(seeds=[42]),
            output_dir=tmp_path / "duplicate",
        )


@pytest.mark.parametrize(
    "drift",
    [
        "sha256",
        "record_count",
        "artifact_fingerprint",
        "config_fingerprint",
        "provenance",
    ],
)
def test_hash_record_count_and_config_fingerprint_drift_are_rejected(
    tmp_path: Path, drift: str
) -> None:
    path = _materialize_manifest(tmp_path, arm="baseline", seed=42)

    def mutate(payload: dict[str, Any]) -> None:
        if drift == "config_fingerprint":
            payload["config"]["fingerprint"] = HEX["f"]
            return
        artifact = payload["artifacts"]["metrics_records"]
        if drift == "provenance":
            artifact["provenance"]["producer_git_commit"] = "2" * 40
            descriptor = dict(artifact)
            descriptor.pop("artifact_fingerprint")
            artifact["artifact_fingerprint"] = canonical_fingerprint(descriptor)
            return
        if drift == "artifact_fingerprint":
            artifact["artifact_fingerprint"] = HEX["f"]
            return
        artifact[drift] = HEX["f"] if drift == "sha256" else 3
        descriptor = dict(artifact)
        descriptor.pop("artifact_fingerprint")
        artifact["artifact_fingerprint"] = canonical_fingerprint(descriptor)

    _rewrite_manifest(path, mutate)
    with pytest.raises(ValueError, match=r"drift|fingerprint mismatch|provenance commit"):
        load_experiment_evidence(path)


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("dataset_fingerprint", "dataset_fingerprint"),
        ("split_fingerprint", "split_fingerprint"),
        ("cohort_fingerprint", "cohort_fingerprint"),
        ("probe_recipe_fingerprint", "probe_recipe_fingerprint"),
    ],
)
def test_dataset_split_cohort_and_probe_drift_block_cross_arm_pareto(
    tmp_path: Path, field: str, expected: str
) -> None:
    paths = [
        _materialize_manifest(tmp_path, arm=arm, seed=seed)
        for arm in ("baseline", "variant")
        for seed in (42, 43)
    ]
    variant = next(path for path in paths if "variant-42" in str(path))
    variant_43 = next(path for path in paths if "variant-43" in str(path))
    for path in (variant, variant_43):
        _rewrite_manifest(path, lambda payload: payload.__setitem__(field, HEX["f"]))
    output = tmp_path / "registry"
    build_campaign_evidence_registry(
        evidence_manifest_paths=paths, requirements=_requirements(), output_dir=output
    )
    review = _registry(output)["stage_reviews"][0]
    assert expected in review["cross_arm_drift"]["variant"]
    assert review["pareto"]["status"] == INCOMPLETE_STATUS


def test_budget_drift_blocks_cross_arm_pareto(tmp_path: Path) -> None:
    paths = [
        _materialize_manifest(tmp_path, arm=arm, seed=seed)
        for arm in ("baseline", "variant")
        for seed in (42, 43)
    ]

    def mutate(payload: dict[str, Any]) -> None:
        payload["budget"] = _budget(tokens=2000)

    for path in paths[2:]:
        _rewrite_manifest(path, mutate)
    output = tmp_path / "registry"
    build_campaign_evidence_registry(
        evidence_manifest_paths=paths, requirements=_requirements(), output_dir=output
    )
    review = _registry(output)["stage_reviews"][0]
    assert "budget" in review["cross_arm_drift"]["variant"]
    assert review["pareto"]["status"] == INCOMPLETE_STATUS


@pytest.mark.parametrize("identity", ["model", "tokenizer"])
def test_model_and_tokenizer_drift_block_seed_aggregation(tmp_path: Path, identity: str) -> None:
    paths = [_materialize_manifest(tmp_path, arm="baseline", seed=seed) for seed in (42, 43)]

    def mutate(payload: dict[str, Any]) -> None:
        payload["model_stack"] = (
            _model_stack(base="model-v2")
            if identity == "model"
            else _model_stack(tokenizer="tokenizer-v2")
        )

    _rewrite_manifest(paths[1], mutate)
    output = tmp_path / "registry"
    build_campaign_evidence_registry(
        evidence_manifest_paths=paths, requirements=_requirements(), output_dir=output
    )
    summary = _registry(output)["arm_summaries"][0]
    assert "seed_comparability_drift" in summary["issues"]
    assert summary["metric_means"] == {}


def test_metric_definition_drift_and_missing_ci_are_reported(tmp_path: Path) -> None:
    paths = [_materialize_manifest(tmp_path, arm="baseline", seed=seed) for seed in (42, 43)]

    def mutate(payload: dict[str, Any]) -> None:
        metric = payload["metrics"][0]
        metric["definition_fingerprint"] = HEX["f"]
        metric["ci"] = None

    _rewrite_manifest(paths[1], mutate)
    output = tmp_path / "registry"
    build_campaign_evidence_registry(
        evidence_manifest_paths=paths, requirements=_requirements(), output_dir=output
    )
    summary = _registry(output)["arm_summaries"][0]
    assert summary["metric_definition_drift"]["43"] == ["intrinsic:source_mrr"]
    assert summary["missing_ci"]["43"] == ["intrinsic:source_mrr"]
    assert summary["evidence_complete"] is False


def test_missing_metrics_human_evidence_and_artifacts_are_reported(tmp_path: Path) -> None:
    paths = [_materialize_manifest(tmp_path, arm="baseline", seed=seed) for seed in (42, 43)]

    def mutate(payload: dict[str, Any]) -> None:
        payload["metrics"] = [
            metric
            for metric in payload["metrics"]
            if metric["category"] not in {"human", "probe_extrinsic"}
        ]
        payload["artifacts"] = {}

    _rewrite_manifest(paths[1], mutate)
    output = tmp_path / "registry"
    build_campaign_evidence_registry(
        evidence_manifest_paths=paths, requirements=_requirements(), output_dir=output
    )
    summary = _registry(output)["arm_summaries"][0]
    assert summary["missing_metrics"]["43"] == [
        "human:preference_rate",
        "probe_extrinsic:corpus_ndcg_at_10",
    ]
    assert summary["missing_human_evidence_seeds"] == [43]
    assert summary["missing_artifacts"]["43"] == ["metrics_records"]
    assert summary["metric_means"] == {}


def test_pareto_front_respects_min_and_max_without_scalar_winner() -> None:
    requirements = [
        MetricRequirement(
            name="quality",
            category=MetricCategory.PROBE_EXTRINSIC,
            direction=MetricDirection.MAX,
            unit="ratio",
            definition_fingerprint=HEX["a"],
        ),
        MetricRequirement(
            name="cost",
            category=MetricCategory.COST,
            direction=MetricDirection.MIN,
            unit="seconds",
            definition_fingerprint=HEX["b"],
        ),
    ]
    fingerprint = HEX["c"]
    result = compute_pareto_front(
        [
            ParetoCandidate(
                arm_id="balanced",
                comparison_fingerprint=fingerprint,
                values={"probe_extrinsic:quality": 0.8, "cost:cost": 2.0},
                evidence_complete=True,
            ),
            ParetoCandidate(
                arm_id="cheap",
                comparison_fingerprint=fingerprint,
                values={"probe_extrinsic:quality": 0.7, "cost:cost": 1.0},
                evidence_complete=True,
            ),
            ParetoCandidate(
                arm_id="dominated",
                comparison_fingerprint=fingerprint,
                values={"probe_extrinsic:quality": 0.6, "cost:cost": 3.0},
                evidence_complete=True,
            ),
        ],
        requirements,
    )
    assert result["status"] == PARETO_STATUS
    assert result["pareto_front_arm_ids"] == ["balanced", "cheap"]
    assert result["dominated_arm_ids"] == ["dominated"]
    assert not _contains_forbidden_selection_key(result)


def test_incomplete_or_incomparable_evidence_is_not_ranked() -> None:
    requirement = MetricRequirement(
        name="quality",
        category=MetricCategory.PROBE_EXTRINSIC,
        direction=MetricDirection.MAX,
        unit="ratio",
        definition_fingerprint=HEX["a"],
    )
    result = compute_pareto_front(
        [
            ParetoCandidate(
                arm_id="one",
                comparison_fingerprint=HEX["a"],
                values={},
                evidence_complete=False,
            ),
            ParetoCandidate(
                arm_id="two",
                comparison_fingerprint=HEX["b"],
                values={"probe_extrinsic:quality": 0.5},
                evidence_complete=True,
            ),
        ],
        [requirement],
    )
    assert result["status"] == INCOMPLETE_STATUS
    assert result["pareto_front_arm_ids"] == []


def test_final_test_path_is_rejected_before_read(monkeypatch: pytest.MonkeyPatch) -> None:
    forbidden = Path("results/final_test/manifest.json")

    def forbidden_read(*_args: Any, **_kwargs: Any) -> str:
        raise AssertionError("final test was read")

    monkeypatch.setattr(Path, "read_text", forbidden_read)
    with pytest.raises(ValueError, match="was not opened"):
        load_experiment_evidence(forbidden)


def test_overwrite_is_refused_and_failed_publish_cleans_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _materialize_manifest(tmp_path, arm="baseline", seed=42)
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        build_campaign_evidence_registry(
            evidence_manifest_paths=[path],
            requirements=_requirements(seeds=[42]),
            output_dir=existing,
        )

    def fail_replace(_source: Path, _target: Path) -> None:
        raise OSError("synthetic publish failure")

    monkeypatch.setattr("doc2query.evaluation.evidence_registry.os.replace", fail_replace)
    output = tmp_path / "failed"
    with pytest.raises(OSError, match="synthetic publish failure"):
        build_campaign_evidence_registry(
            evidence_manifest_paths=[path],
            requirements=_requirements(seeds=[42]),
            output_dir=output,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".failed.staging-*"))


def test_module_and_script_have_no_model_or_reranker_imports() -> None:
    paths = [
        Path(registry_module.__file__),
        Path("scripts/build_task09_evidence_registry.py"),
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


def test_bundle_proves_no_campaign_training_evaluation_or_selection(tmp_path: Path) -> None:
    path = _materialize_manifest(tmp_path, arm="baseline", seed=42)
    output = tmp_path / "registry"
    bundle = build_campaign_evidence_registry(
        evidence_manifest_paths=[path],
        requirements=_requirements(seeds=[42]),
        output_dir=output,
    )
    registry = _registry(output)
    for payload in (bundle, registry):
        assert payload["campaign_started"] is False
        assert payload["model_loading_performed"] is False
        assert payload["training_started"] is False
        assert payload["evaluation_started"] is False
        assert payload["selection_performed"] is False
        assert payload["final_tests_used"] == []
        assert not _contains_forbidden_selection_key(payload)
