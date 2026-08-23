from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml
from conftest import require_local_artifacts

from doc2query.config import load_config
from doc2query.evaluation import d01_campaign, d01_pipeline
from doc2query.evaluation.d01_campaign import (
    D01_RECOVERY_CONTRACT,
    audit_d01_artifacts,
    load_common_cohort,
    validate_baseline_provenance,
)
from doc2query.evaluation.d01_pipeline import (
    D01_COMPARISON_CONTRACT,
    D01_GENERATION_CONTRACT,
    D01_SCORING_CONTRACT,
    assemble_matched_report,
    assert_development_subset,
    evaluation_group_ids,
    generate_frozen_dev,
    generate_frozen_dev_batched,
    materialize_probe_inputs,
)
from doc2query.evaluation.intrinsic import evaluate_intrinsic_records
from doc2query.utils.records import read_records


def _record(index: int) -> dict[str, Any]:
    return {
        "example_id": f"q-{index}",
        "query": f"naturalne pytanie {index}",
        "positives": [
            {
                "doc_id": f"d-{index}",
                "text": "Warszawa jest stolicą Polski. Bilet kosztuje 40 zł.",
                "metadata": {},
            }
        ],
        "hard_negatives": [
            {"doc_id": f"n-{index}-{item}", "text": f"Negatyw {item}", "metadata": {}}
            for item in range(10)
        ],
        "metadata": {"domain": "fixture"},
    }


def _install_frozen_mocks(monkeypatch: pytest.MonkeyPatch, records: list[dict[str, Any]]) -> None:
    monkeypatch.setattr(d01_pipeline, "load_frozen_records", lambda _path, _subset: records)
    monkeypatch.setattr(d01_pipeline, "evaluation_fingerprint", lambda _path, _subset: "f" * 64)
    monkeypatch.setattr(d01_campaign, "load_frozen_records", lambda _path, _subset: records)
    monkeypatch.setattr(d01_campaign, "evaluation_fingerprint", lambda _path, _subset: "f" * 64)
    monkeypatch.setattr(d01_pipeline, "collect_code_provenance", lambda: {"commit": "fixture"})


def test_crash_resume_generation_does_not_regenerate_durable_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = [_record(0), _record(1)]
    _install_frozen_mocks(monkeypatch, records)
    calls: list[int] = []

    def interrupted(_prompt: str, seed: int) -> str:
        calls.append(seed)
        if len(calls) == 5:
            raise RuntimeError("fixture crash")
        return f"zapytanie {seed}"

    output = tmp_path / "generation.jsonl"
    with pytest.raises(RuntimeError, match="fixture crash"):
        generate_frozen_dev(
            Path("configs/experiments/d01_1_5b_style_dev_generation_s42.yaml"),
            frozen_manifest=tmp_path / "manifest.json",
            subset="dev_intrinsic_rank10",
            output_path=output,
            backend=interrupted,
        )
    journal = output.with_suffix(".jsonl.journal.jsonl")
    assert [row["evaluation_group_id"] for row in read_records(journal)] == ["q-0::d-0"]
    resumed_calls: list[int] = []

    def resumed(_prompt: str, seed: int) -> str:
        resumed_calls.append(seed)
        return f"zapytanie {seed}"

    summary = generate_frozen_dev(
        Path("configs/experiments/d01_1_5b_style_dev_generation_s42.yaml"),
        frozen_manifest=tmp_path / "manifest.json",
        subset="dev_intrinsic_rank10",
        output_path=output,
        backend=resumed,
    )
    assert summary["resumed_group_count"] == 1
    assert resumed_calls == [1042, 1045, 1048, 1051]
    assert len(list(read_records(output))) == 8


def test_batched_generation_resumes_only_after_atomic_passage_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = [_record(0), _record(1), _record(2)]
    _install_frozen_mocks(monkeypatch, records)
    calls = 0

    def interrupted(_prompts: Sequence[str], seeds: Sequence[int]) -> list[str]:
        nonlocal calls
        calls += 1
        if calls == 5:
            raise RuntimeError("fixture crash")
        return [f"zapytanie {seed}" for seed in seeds]

    output = tmp_path / "generation.batched.jsonl"
    with pytest.raises(RuntimeError, match="fixture crash"):
        generate_frozen_dev_batched(
            Path("configs/experiments/d01_1_5b_style_dev_generation_s42.yaml"),
            frozen_manifest=tmp_path / "manifest.json",
            subset="dev_intrinsic_rank10",
            output_path=output,
            generation_batch_size=2,
            backend=interrupted,
        )
    journal = output.with_suffix(".jsonl.journal.jsonl")
    journal_rows = list(read_records(journal))
    assert len(journal_rows) == 1
    assert [group["evaluation_group_id"] for group in journal_rows[0]["groups"]] == [
        "q-0::d-0",
        "q-1::d-1",
    ]
    resumed_seeds: list[list[int]] = []

    def resumed(_prompts: Sequence[str], seeds: Sequence[int]) -> list[str]:
        resumed_seeds.append(list(seeds))
        return [f"zapytanie {seed}" for seed in seeds]

    summary = generate_frozen_dev_batched(
        Path("configs/experiments/d01_1_5b_style_dev_generation_s42.yaml"),
        frozen_manifest=tmp_path / "manifest.json",
        subset="dev_intrinsic_rank10",
        output_path=output,
        generation_batch_size=2,
        backend=resumed,
    )
    assert summary["resumed_group_count"] == 2
    assert resumed_seeds == [[2042], [2045], [2048], [2051]]
    assert len(list(read_records(output))) == 12
    with pytest.raises(ValueError, match="identity mismatch"):
        generate_frozen_dev_batched(
            Path("configs/experiments/d01_1_5b_style_dev_generation_s42.yaml"),
            frozen_manifest=tmp_path / "manifest.json",
            subset="dev_intrinsic_rank10",
            output_path=output,
            generation_batch_size=4,
            backend=resumed,
        )


def test_generation_rejects_changed_identity_and_can_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_frozen_mocks(monkeypatch, [_record(0), _record(1)])
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("one", encoding="utf-8")
    output = tmp_path / "generation.jsonl"
    calls = 0

    def crash(_prompt: str, seed: int) -> str:
        nonlocal calls
        calls += 1
        if calls == 5:
            raise RuntimeError("stop")
        return f"query {seed}"

    with pytest.raises(RuntimeError):
        generate_frozen_dev(
            Path("configs/experiments/d01_1_5b_style_dev_generation_s42.yaml"),
            frozen_manifest=tmp_path / "manifest.json",
            subset="dev_intrinsic_rank10",
            output_path=output,
            adapter_path=adapter,
            backend=crash,
        )
    (adapter / "adapter_config.json").write_text("two", encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch"):
        generate_frozen_dev(
            Path("configs/experiments/d01_1_5b_style_dev_generation_s42.yaml"),
            frozen_manifest=tmp_path / "manifest.json",
            subset="dev_intrinsic_rank10",
            output_path=output,
            adapter_path=adapter,
            backend=lambda _prompt, seed: str(seed),
            archive_incompatible=True,
        )
    archives = list((tmp_path / "interrupted-generation").glob("*/archive_manifest.json"))
    assert len(archives) == 1
    assert not output.with_suffix(".jsonl.journal.jsonl").exists()


def test_complete_journal_recovers_summary_without_loading_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_frozen_mocks(monkeypatch, [_record(0)])
    output = tmp_path / "generation.jsonl"
    generate_frozen_dev(
        Path("configs/experiments/d01_1_5b_style_dev_generation_s42.yaml"),
        frozen_manifest=tmp_path / "manifest.json",
        subset="dev_intrinsic_rank10",
        output_path=output,
        backend=lambda _prompt, seed: f"query {seed}",
    )
    output.with_suffix(".jsonl.summary.json").unlink()

    def forbidden_loader(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("completed journal must not load a model")

    monkeypatch.setattr(d01_pipeline, "_model_backend", forbidden_loader)
    recovered = generate_frozen_dev(
        Path("configs/experiments/d01_1_5b_style_dev_generation_s42.yaml"),
        frozen_manifest=tmp_path / "manifest.json",
        subset="dev_intrinsic_rank10",
        output_path=output,
    )
    assert recovered["generation_count"] == 4


def test_evaluation_ids_follow_frozen_order() -> None:
    records = [_record(2), _record(0), _record(1)]
    assert evaluation_group_ids(records) == ["q-2::d-2", "q-0::d-0", "q-1::d-1"]


def test_common_cohort_preserves_frozen_order_and_original_seed_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = [_record(0), _record(1), _record(2)]
    _install_frozen_mocks(monkeypatch, records)
    selected_ids = ["q-0::d-0", "q-2::d-2"]
    cohort = tmp_path / "cohort.json"
    _write_json(
        cohort,
        {
            "contract": D01_RECOVERY_CONTRACT,
            "selected_group_ids": selected_ids,
            "selection_policy": {"policy": "technical_only"},
            "selection_policy_fingerprint": "s" * 64,
            "final_tests_used": [],
        },
    )
    selected, indices, _manifest = load_common_cohort(records, cohort)
    assert evaluation_group_ids(selected) == selected_ids
    assert indices == [0, 2]
    output = tmp_path / "baseline.jsonl"
    calls: list[int] = []

    def backend(_prompt: str, seed: int) -> str:
        calls.append(seed)
        return f"query {seed}"

    summary = generate_frozen_dev(
        Path("configs/experiments/d01_w05_matched_dev_generation_s42.yaml"),
        frozen_manifest=tmp_path / "manifest.json",
        subset="dev_intrinsic_rank10",
        cohort_manifest=cohort,
        output_path=output,
        backend=backend,
    )
    assert calls == [42, 45, 48, 51, 2042, 2045, 2048, 2051]
    assert summary["generation_count"] == 8
    assert summary["identity"]["seed_contract"]["index_basis"] == (
        "original_frozen_subset_position"
    )


@pytest.mark.usefixtures()
def test_w06_matched_provenance_rejects_bs1_and_accepts_bs8() -> None:
    require_local_artifacts()
    adapter = Path("runs/W06-4.5B-INSTRUCT-50K-8GB-BS8-L512/adapter")
    manifest = Path("runs/W06-4.5B-INSTRUCT-50K-8GB-BS8-L512/run_manifest.json")
    with pytest.raises(ValueError, match="not BS8"):
        validate_baseline_provenance(
            config_path=Path("configs/experiments/w06_4_5b_50k_8gb_bs1.yaml"),
            adapter_path=adapter,
            training_manifest_path=manifest,
        )
    result = validate_baseline_provenance(
        config_path=Path("configs/experiments/d01_w06_matched_dev_generation_s42.yaml"),
        adapter_path=adapter,
        training_manifest_path=manifest,
    )
    assert result["training_experiment_id"].endswith("BS8-L512")
    assert result["training_batch_size"] == 8


def test_d01_audit_accepts_legacy_model_and_explicit_slot_gaps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = [_record(0)]
    _install_frozen_mocks(monkeypatch, records)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"fixture")
    config_path = Path("configs/experiments/d01_1_5b_style_dev_generation_s42.yaml")
    output = tmp_path / "generation.jsonl"

    def backend(_prompts: Sequence[str], seeds: Sequence[int]) -> list[str]:
        return ["query 42" if seed in {45, 46, 47} else f"query {seed}" for seed in seeds]

    generate_frozen_dev_batched(
        config_path,
        frozen_manifest=tmp_path / "manifest.json",
        subset="dev_intrinsic_rank10",
        output_path=output,
        generation_batch_size=2,
        adapter_path=adapter,
        backend=backend,
    )
    config = load_config(config_path)
    sft = tmp_path / "sft.json"
    _write_json(
        sft,
        {
            "experiment_id": "D01-training-fixture",
            "adapter_path": str(adapter),
            "dataset_fingerprint": "train-fixture",
            "global_step": 1,
            "model": {
                "name_or_path": config.model.name_or_path,
                "revision": config.model.revision,
                "trust_remote_code": config.model.trust_remote_code,
            },
        },
    )
    generations_before = output.read_bytes()
    report = audit_d01_artifacts(
        frozen_manifest=tmp_path / "manifest.json",
        subset="dev_intrinsic_rank10",
        arms=[
            {
                "id": "fixture",
                "training_experiment_id": "D01-training-fixture",
                "sft_summary": str(sft),
                "adapter": str(adapter),
                "generation_config": str(config_path),
                "generations": str(output),
            }
        ],
        output_json=tmp_path / "audit.json",
        output_markdown=tmp_path / "audit.md",
    )
    assert report["status"] == "verified"
    assert report["arms"][0]["complete_group_count"] == 0
    assert report["arms"][0]["exhausted_group_count"] == 1
    assert report["arms"][0]["sft_model_provenance"] == "legacy_without_architecture"
    assert [row["candidate_slot_index"] for row in read_records(output)] == [0, 2, 3]
    assert output.read_bytes() == generations_before
    legacy_sft = json.loads(sft.read_text(encoding="utf-8"))
    legacy_sft["model"]["revision"] = "wrong-revision"
    _write_json(sft, legacy_sft)
    with pytest.raises(ValueError, match="model provenance mismatch"):
        audit_d01_artifacts(
            frozen_manifest=tmp_path / "manifest.json",
            subset="dev_intrinsic_rank10",
            arms=[
                {
                    "id": "fixture",
                    "training_experiment_id": "D01-training-fixture",
                    "sft_summary": str(sft),
                    "adapter": str(adapter),
                    "generation_config": str(config_path),
                    "generations": str(output),
                }
            ],
            output_json=tmp_path / "rejected-audit.json",
            output_markdown=tmp_path / "rejected-audit.md",
        )


def test_campaign_runner_has_lock_and_requires_one_explicit_phase() -> None:
    script = Path("scripts/run_task05_d01_post_campaign.sh").read_text(encoding="utf-8")
    assert "flock -n 9" in script
    assert "PHASE=${1:-}" in script
    assert "generate-matched-baselines|score|compare" in script
    assert "generation-batched" in script
    assert "uncontrolled.full.jsonl" not in script
    assert "--generation-batch-size 16" in script
    assert "corpus=data/processed/v1/evaluation/corpus-bm25-v1" in script
    assert "corpus=artifacts/task04/p03/bm25_train_v1" not in script
    assert '--quality-contract "$quality"' in script
    assert "--semantic-device cuda" in script
    campaign = yaml.safe_load(
        Path("configs/evaluation/d01_campaign_v2.yaml").read_text(encoding="utf-8")
    )
    assert campaign["scoring"] == {
        "primary": "configs/reranker/primary_polish_roberta_v3_cuda.yaml",
        "shadow": "configs/reranker/shadow_bge_v2_m3.yaml",
        "corpus_index": "data/processed/v1/evaluation/corpus-bm25-v1",
        "expected_corpus_fingerprint": (
            "159af07f3b987fe492f9ff89f494521587a4d023fcb4fc62b69d7295d2a57258"
        ),
        "output_root": "reports/measurements/task05_d01_postprocess_v2/scoring",
    }
    assert campaign["comparison"]["quality_contract"] == (
        "configs/evaluation/d01_copy_semantic_quality_v1.yaml"
    )


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_rows(path: Path, ids: list[str], value: float = 1.0) -> None:
    path.write_text(
        "".join(
            json.dumps(
                {
                    "evaluation_group_id": identifier,
                    "pool_mrr": value,
                    "pool_recall_at_1": value,
                    "format_valid": 1.0,
                }
            )
            + "\n"
            for identifier in ids
        ),
        encoding="utf-8",
    )


def test_matched_comparison_fails_closed_on_budget_difference(tmp_path: Path) -> None:
    baseline_summary = tmp_path / "baseline.json"
    variant_summary = tmp_path / "variant.json"
    common: dict[str, Any] = {
        "status": "measured",
        "test_fingerprint": "f" * 64,
        "source_passage_count": 2,
        "generation_count": 8,
        "final_tests_used": [],
        "generation_contract": {
            "cohort": {"fingerprint": "f" * 64},
            "seed_contract": {"base_seed": 42},
            "max_new_tokens": 64,
            "do_sample": True,
            "temperature": 0.8,
            "top_p": 0.95,
            "target_query_count": 4,
        },
    }
    _write_json(baseline_summary, {**common, "experiment_id": "W05"})
    variant_contract = {**common["generation_contract"], "max_new_tokens": 63}
    _write_json(
        variant_summary,
        {**common, "experiment_id": "D01", "generation_contract": variant_contract},
    )
    baseline_rows = tmp_path / "baseline.jsonl"
    variant_rows = tmp_path / "variant.jsonl"
    _write_rows(baseline_rows, ["a", "b"])
    _write_rows(variant_rows, ["a", "b"])
    report = assemble_matched_report(
        baseline_summary_path=baseline_summary,
        baseline_rows_path=baseline_rows,
        variant_summary_path=variant_summary,
        variant_rows_path=variant_rows,
        comparison_contract_path=Path("configs/evaluation/comparison_contract_v1.yaml"),
        quality_contract_path=Path("configs/evaluation/d01_copy_semantic_quality_v1.yaml"),
        output_json=tmp_path / "report.json",
        output_markdown=tmp_path / "report.md",
        bootstrap_samples=10,
    )
    assert report["status"] == "incomplete"
    assert report["decision"] == "not_measured"
    assert not report["automatic_promotion"]


def test_final_test_subset_is_forbidden() -> None:
    with pytest.raises(ValueError, match="restricted to frozen"):
        assert_development_subset("test_intrinsic_rank10")


class _WordScorer:
    name = "word"

    def __init__(self, reverse: bool = False) -> None:
        self.reverse = reverse

    def score_pairs(self, pairs: Any) -> list[float]:
        values = [
            float(len(set(query.casefold().split()) & set(text.casefold().split())))
            for query, text in pairs
        ]
        return [-value for value in values] if self.reverse else values


def test_control_aggregation_and_primary_shadow_disagreement(tmp_path: Path) -> None:
    source = _record(0)
    generations = []
    for index, (form, intent, query) in enumerate(
        (
            ("full_question", "entity_lookup", "Gdzie jest Warszawa?"),
            ("keyword_query", "fact_lookup", "Warszawa stolica"),
        )
    ):
        generations.append(
            {
                "evaluation_id": f"q-0::d-0::candidate::{index}",
                "evaluation_group_id": "q-0::d-0",
                "experiment_id": "D01",
                "example_id": "q-0",
                "mode": "controlled",
                "candidate_index": index,
                "generated": query,
                "reference": source["query"],
                "positive": source["positives"][0],
                "hard_negatives": source["hard_negatives"],
                "positive_count": 1,
                "metadata": source["metadata"],
                "requested_form": form,
                "requested_intent": intent,
                "intent_applicable": True,
            }
        )
    summary = evaluate_intrinsic_records(
        generations,
        primary=_WordScorer(),
        shadow=_WordScorer(reverse=True),
        output_dir=tmp_path / "score",
        test_fingerprint="f" * 64,
        experiment_id="D01",
        scoring_batch_size=1,
    )
    assert summary["controls"]["form_accuracy"] == 1.0
    assert summary["controls"]["intent_unknown_count"] == 0
    assert summary["judge_disagreement"]["rank_disagreement_rate"] == 1.0
    assert set(summary["slices"]["requested_form"]) == {"full_question", "keyword_query"}


def test_d01_scoring_propagates_signed_cohort_hard_negative_minimum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generations = tmp_path / "generations.jsonl"
    summary_path = tmp_path / "generations.summary.json"
    primary_config = tmp_path / "primary.yaml"
    shadow_config = tmp_path / "shadow.yaml"
    identity: dict[str, Any] = {
        "identity_sha256": "i" * 64,
        "minimum_hard_negatives": 5,
        "cohort": {
            "subset": "dev_intrinsic",
            "fingerprint": "f" * 64,
            "selection_policy": {"minimum_hard_negatives": 5},
        },
        "seed_contract": {},
        "generation": {
            "max_new_tokens": 64,
            "do_sample": True,
            "temperature": 0.8,
            "top_p": 0.95,
            "target_query_count": 1,
            "max_attempts_per_query": 16,
        },
    }
    source = _record(0)
    source["hard_negatives"] = source["hard_negatives"][:5]
    row = {
        "evaluation_id": "q-0::d-0::candidate::0",
        "generation_identity_sha256": identity["identity_sha256"],
        "frozen_subset": "dev_intrinsic",
        "frozen_cohort_fingerprint": "f" * 64,
        "final_tests_used": [],
        "positive": source["positives"][0],
        "hard_negatives": source["hard_negatives"],
    }
    generations.write_text(json.dumps(row) + "\n", encoding="utf-8")
    _write_json(
        summary_path,
        {
            "contract": D01_GENERATION_CONTRACT,
            "experiment_id": "D01-fixture",
            "generation_count": 1,
            "source_passage_count": 1,
            "target_queries_per_passage": 1,
            "attempts": 1,
            "invalid_outputs": 0,
            "duplicate_outputs": 0,
            "exhausted_groups": 0,
            "effective_candidate_count_mean": 1.0,
            "identity": identity,
            "final_tests_used": [],
        },
    )
    primary_config.write_text(
        "name_or_path: fixture-primary\nrevision: fixture-revision\n", encoding="utf-8"
    )
    shadow_config.write_text("name_or_path: fixture-shadow\n", encoding="utf-8")
    captured: dict[str, Any] = {}

    def fake_score(_path: Path, **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "protocols": {"corpus_retrieval": {"status": "measured"}},
            "judges": {"primary_status": "measured", "shadow_status": "measured"},
        }

    monkeypatch.setattr(d01_pipeline, "score_generation_artifact", fake_score)
    result = d01_pipeline.score_d01_artifact(
        generations,
        generation_summary_path=summary_path,
        output_dir=tmp_path / "score",
        primary_config=primary_config,
        shadow_config=shadow_config,
    )
    assert captured["minimum_hard_negatives"] == 5
    assert result["generation_contract"]["minimum_hard_negatives"] == 5

    identity["cohort"]["selection_policy"]["minimum_hard_negatives"] = 6
    _write_json(summary_path, {**json.loads(summary_path.read_text()), "identity": identity})
    with pytest.raises(ValueError, match="hard-negative minima differ"):
        d01_pipeline.score_d01_artifact(
            generations,
            generation_summary_path=summary_path,
            output_dir=tmp_path / "score-drift",
            primary_config=primary_config,
            shadow_config=shadow_config,
        )


def _probe_contracts(
    tmp_path: Path, *, comparison_status: str = "intrinsic_complete"
) -> tuple[Path, ...]:
    generation = tmp_path / "generation.summary.json"
    scoring = tmp_path / "scoring.json"
    comparison = tmp_path / "comparison.json"
    _write_json(
        generation,
        {
            "status": "measured",
            "contract": D01_GENERATION_CONTRACT,
            "experiment_id": "D01",
            "target_queries_per_passage": 4,
            "exhausted_groups": 0,
            "invalid_outputs": 0,
            "identity": {
                "identity_sha256": "i" * 64,
                "adapter": {"artifact_sha256": "a" * 64},
                "cohort": {"fingerprint": "f" * 64},
            },
            "final_tests_used": [],
        },
    )
    _write_json(
        scoring,
        {
            "status": "measured",
            "contract": D01_SCORING_CONTRACT,
            "example_count": 4,
            "source_passage_count": 1,
            "generation_count": 4,
            "generation_identity_sha256": "i" * 64,
            "generation_contract": {
                "max_new_tokens": 64,
                "max_attempts_per_query": 3,
            },
            "primary_judge_name": "sdadas/polish-reranker-roberta-v3",
            "primary_judge_revision": "e6471da541f4e7be33845b6d57248a8d8bde27e8",
            "final_tests_used": [],
        },
    )
    _write_json(
        comparison,
        {
            "status": comparison_status,
            "contract": D01_COMPARISON_CONTRACT,
            "budget_difference": {"matched": True},
            "intrinsic_guardrail_decision": "continue",
            "copy_semantic_quality": {
                "status": "measured",
                "decision": "continue",
                "contract": {
                    "contract": "task05-d01-copy-semantic-quality-v1",
                },
                "final_tests_used": [],
            },
            "variant": {"experiment_id": "D01"},
            "final_tests_used": [],
        },
    )
    return generation, scoring, comparison


def test_probe_materialization_requires_gate_and_exact_k4(tmp_path: Path) -> None:
    generation, scoring, comparison = _probe_contracts(tmp_path, comparison_status="incomplete")
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="complete matched"):
        materialize_probe_inputs(
            generations_path=rows_path,
            generation_summary_path=generation,
            scoring_summary_path=scoring,
            scoring_rows_path=rows_path,
            comparison_report_path=comparison,
            probe_recipe_path=Path("configs/evaluation/probe_v1.yaml"),
            output_path=tmp_path / "probe.jsonl",
        )

    generation, scoring, comparison = _probe_contracts(tmp_path)
    comparison_payload = json.loads(comparison.read_text(encoding="utf-8"))
    comparison_payload.pop("copy_semantic_quality")
    _write_json(comparison, comparison_payload)
    with pytest.raises(ValueError, match="copy/semantic quality gate"):
        materialize_probe_inputs(
            generations_path=rows_path,
            generation_summary_path=generation,
            scoring_summary_path=scoring,
            scoring_rows_path=rows_path,
            comparison_report_path=comparison,
            probe_recipe_path=Path("configs/evaluation/probe_v1.yaml"),
            output_path=tmp_path / "probe.jsonl",
        )

    generation, scoring, comparison = _probe_contracts(tmp_path)
    rows = []
    source = _record(0)
    for index in range(4):
        rows.append(
            {
                "evaluation_id": f"q-0::d-0::candidate::{index}",
                "evaluation_group_id": "q-0::d-0",
                "candidate_index": index,
                "generated": f"query {index}",
                "positive": source["positives"][0],
                "hard_negatives": source["hard_negatives"],
                "example_id": "q-0",
                "doc_id": "d-0",
                "experiment_id": "D01",
                "final_tests_used": [],
                "primary_negative_scores": [0.0] * 10,
            }
        )
    rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest = materialize_probe_inputs(
        generations_path=rows_path,
        generation_summary_path=generation,
        scoring_summary_path=scoring,
        scoring_rows_path=rows_path,
        comparison_report_path=comparison,
        probe_recipe_path=Path("configs/evaluation/probe_v1.yaml"),
        output_path=tmp_path / "probe.jsonl",
    )
    assert manifest["negative_recipe"] == "HN0+filter"
    assert manifest["possible_false_negative_policy"] == "drop"
    assert manifest["final_tests_used"] == []
    assert not manifest["training_started"]
