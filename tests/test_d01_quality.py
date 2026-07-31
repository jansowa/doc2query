from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from doc2query.evaluation.d01_quality import (
    D01QualityContract,
    evaluate_copy_semantic_quality,
)


class _OrthogonalEncoder:
    def encode(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        del batch_size
        return np.eye(len(texts), dtype=np.float32)


class _UnexpectedEncoder:
    def encode(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        raise AssertionError(f"cache miss for {len(texts)} texts at batch {batch_size}")


def _rows(path: Path, *, copied: bool) -> None:
    passage = "Warszawa jest stolicą Polski i największym miastem kraju położonym nad Wisłą."
    values: list[dict[str, Any]] = []
    for group in range(2):
        for candidate in range(4):
            generated = (
                "Warszawa jest stolicą Polski i największym miastem kraju"
                if copied
                else f"pytanie wariant {candidate}"
            )
            values.append(
                {
                    "evaluation_id": f"group-{group}::candidate::{candidate}",
                    "evaluation_group_id": f"group-{group}",
                    "generated": generated,
                    "reference": "Jakie miasto jest stolicą Polski?",
                    "positive": {"doc_id": str(group), "text": passage},
                    "copy_density": 1.0 if copied else 0.0,
                    "normalized_lcs": 1.0 if copied else 0.0,
                    "longest_copied_ngram": 8 if copied else 0,
                    "word_length": 8 if copied else 3,
                    "pool_recall_at_1": 1.0,
                }
            )
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
        encoding="utf-8",
    )


def test_poldense_contract_pins_symmetric_and_retrieval_prefixes() -> None:
    contract = D01QualityContract.load(Path("configs/evaluation/d01_copy_semantic_quality_v1.yaml"))
    reference = contract.reference()
    assert reference["semantic_model"]["name_or_path"] == "OPI-PIB/PolDense-150M"
    assert reference["semantic_model"]["revision"] == ("b94ea7f951cc480369a85fa9021694eef80c3a00")
    assert reference["similarity_prefix"] == "[sts]: "
    assert reference["retrieval_query_prefix"] == "[query]: "
    assert reference["final_tests_used"] == []


def test_quality_gate_passes_clean_matched_groups_and_writes_blind_audit(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline.jsonl"
    variant = tmp_path / "variant.jsonl"
    _rows(baseline, copied=False)
    _rows(variant, copied=False)
    output = tmp_path / "comparison.json"
    report = evaluate_copy_semantic_quality(
        baseline_rows_path=baseline,
        variant_rows_path=variant,
        baseline_label="baseline",
        variant_label="variant",
        contract_path=Path("configs/evaluation/d01_copy_semantic_quality_v1.yaml"),
        output_json=output,
        bootstrap_samples=100,
        bootstrap_seed=7,
        semantic_device="cpu",
        encoder=_OrthogonalEncoder(),
    )
    assert report["decision"] == "continue"
    assert report["anti_copy"]["baseline_rate"] == 0.0
    assert report["anti_copy"]["variant_rate"] == 0.0
    assert report["semantic_diversity"]["common_clean_group_rate"] == 1.0
    assert (
        report["semantic_diversity"]["paired_bootstrap"]["semantic_cluster_count"]["difference"]
        == 0.0
    )
    assert Path(report["blind_audit"]["blind_csv"]).is_file()
    assert report["blind_audit"]["status"] == "pending_human_review"
    header = Path(report["blind_audit"]["blind_csv"]).read_text(encoding="utf-8").splitlines()[0]
    assert "arm" not in header

    cached = evaluate_copy_semantic_quality(
        baseline_rows_path=baseline,
        variant_rows_path=variant,
        baseline_label="baseline",
        variant_label="variant",
        contract_path=Path("configs/evaluation/d01_copy_semantic_quality_v1.yaml"),
        output_json=output,
        bootstrap_samples=100,
        bootstrap_seed=7,
        semantic_device="cpu",
        encoder=_UnexpectedEncoder(),
    )
    assert cached["semantic_diversity"]["baseline_cache"]["cache_hit"]
    assert cached["semantic_diversity"]["variant_cache"]["cache_hit"]


def test_quality_gate_fails_closed_for_passage_copying_without_loading_encoder(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline.jsonl"
    variant = tmp_path / "variant.jsonl"
    _rows(baseline, copied=False)
    _rows(variant, copied=True)
    report = evaluate_copy_semantic_quality(
        baseline_rows_path=baseline,
        variant_rows_path=variant,
        baseline_label="baseline",
        variant_label="copying-variant",
        contract_path=Path("configs/evaluation/d01_copy_semantic_quality_v1.yaml"),
        output_json=tmp_path / "comparison.json",
        bootstrap_samples=100,
        bootstrap_seed=7,
        semantic_device="cpu",
    )
    assert report["decision"] == "stop"
    assert report["anti_copy"]["variant_rate"] == 1.0
    assert report["semantic_diversity"]["common_clean_group_count"] == 0
    assert (
        report["semantic_diversity"]["guardrails"]["common_clean_group_coverage"]["status"]
        == "failed"
    )
