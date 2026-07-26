from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from doc2query.evaluation import d01_pipeline
from doc2query.evaluation.d01_pipeline import (
    D01_COMPARISON_CONTRACT,
    D01_GENERATION_CONTRACT,
    D01_SCORING_CONTRACT,
    assemble_matched_report,
    assert_development_subset,
    evaluation_group_ids,
    generate_frozen_dev,
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
