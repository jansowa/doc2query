from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from doc2query.evaluation import d01_pipeline
from doc2query.evaluation.d01_campaign import D01B_PROSPECTIVE_COHORT_CONTRACT
from doc2query.evaluation.d01_pipeline import generate_frozen_dev_batched
from doc2query.evaluation.d01_prospective import (
    PREREGISTERED_COHORT_CONTRACTS,
    PROSPECTIVE_CONTRACTS,
    _id_list_sha256,
    _selection_rows,
    assert_exact_k_summary,
    assert_scoring_summary,
    evaluate_prospective_gates,
)
from doc2query.utils.records import read_records


def _record(index: int, negatives: int = 5) -> dict[str, Any]:
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
            for item in range(negatives)
        ],
        "metadata": {"domain": "fixture"},
    }


def test_v2_contract_and_prior_cohort_exclusion_are_explicit_and_deterministic() -> None:
    assert "task05-d01b-prospective-1.5b-v2" in PROSPECTIVE_CONTRACTS
    assert "task05-d01b-prospective-cohort-v2" in PREREGISTERED_COHORT_CONTRACTS
    records = [_record(index, negatives=4 if index == 5 else 5) for index in range(8)]
    prior = _selection_rows(
        records,
        excluded={"q-0"},
        minimum_hard_negatives=5,
        seed=20260802,
    )[:2]
    prior_ids = {item[1] for item in prior}
    selected = _selection_rows(
        records,
        excluded={"q-0", *prior_ids},
        minimum_hard_negatives=5,
        seed=20260803,
    )
    selected_ids = [item[1] for item in selected]

    assert not prior_ids.intersection(selected_ids)
    assert "q-0" not in selected_ids
    assert "q-5" not in selected_ids
    assert selected_ids == [
        item[1]
        for item in _selection_rows(
            list(reversed(records)),
            excluded={"q-0", *prior_ids},
            minimum_hard_negatives=5,
            seed=20260803,
        )
    ]
    assert _id_list_sha256(selected_ids) != _id_list_sha256(list(reversed(selected_ids)))


def test_prospective_cohort_allows_five_negatives_without_weakening_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = [_record(0), _record(1)]
    monkeypatch.setattr(d01_pipeline, "load_frozen_records", lambda _path, _subset: records)
    monkeypatch.setattr(d01_pipeline, "evaluation_fingerprint", lambda _path, _subset: "f" * 64)
    monkeypatch.setattr(d01_pipeline, "collect_code_provenance", lambda: {"commit": "fixture"})
    cohort = {
        "contract": D01B_PROSPECTIVE_COHORT_CONTRACT,
        "selected_example_ids": ["q-1", "q-0"],
        "selected_group_ids": ["q-1::d-1", "q-0::d-0"],
        "selection_policy": {"minimum_hard_negatives": 5},
        "selection_policy_fingerprint": "a" * 64,
        "final_tests_used": [],
    }
    cohort_path = tmp_path / "cohort.json"
    cohort_path.write_text(json.dumps(cohort), encoding="utf-8")
    output = tmp_path / "prospective.jsonl"

    summary = generate_frozen_dev_batched(
        Path("configs/experiments/d01b_prospective_w05_1_5b_s42.yaml"),
        frozen_manifest=tmp_path / "manifest.json",
        subset="dev_intrinsic",
        output_path=output,
        cohort_manifest=cohort_path,
        generation_batch_size=2,
        backend=lambda _prompts, seeds: [f"zapytanie {seed}" for seed in seeds],
    )

    assert summary["generation_count"] == 8
    assert summary["identity"]["minimum_hard_negatives"] == 5
    assert [row["example_id"] for row in read_records(output)][:4] == ["q-1"] * 4
    with pytest.raises(ValueError, match="at least 10"):
        d01_pipeline.evaluation_group_ids(records)


def test_prospective_gate_rules_are_ci_based_and_fail_closed() -> None:
    passing = {
        "corpus_round_trip_at_20": {"ci95_low": -0.02, "ci95_high": 0.01},
        "copy_risk_rate": {"ci95_low": -0.02, "ci95_high": 0.0},
    }
    config = {
        "corpus_round_trip_at_20": {"direction": "higher", "noninferiority_margin": 0.02},
        "copy_risk_rate": {"direction": "lower", "maximum_upper_ci": 0.0},
    }
    gates, authorized = evaluate_prospective_gates(passing, config)
    assert authorized is True
    assert {item["status"] for item in gates.values()} == {"passed"}

    passing["copy_risk_rate"] = {"ci95_low": -0.02, "ci95_high": 0.0001}
    gates, authorized = evaluate_prospective_gates(passing, config)
    assert authorized is False
    assert gates["copy_risk_rate"]["status"] == "failed"

    gates, authorized = evaluate_prospective_gates({}, config)
    assert authorized is False
    assert gates["corpus_round_trip_at_20"]["reason"] == "not_measured"


def test_exact_k_and_scoring_validators(tmp_path: Path) -> None:
    generation = tmp_path / "generation.json"
    generation.write_text(
        json.dumps(
            {
                "status": "measured",
                "final_tests_used": [],
                "source_passage_count": 2000,
                "generation_count": 8000,
                "exhausted_groups": 0,
                "target_queries_per_passage": 4,
            }
        ),
        encoding="utf-8",
    )
    assert assert_exact_k_summary(generation)["generation_count"] == 8000
    payload = json.loads(generation.read_text(encoding="utf-8"))
    payload["exhausted_groups"] = 1
    generation.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="exact-K"):
        assert_exact_k_summary(generation)

    scoring = tmp_path / "scoring.json"
    scoring.write_text(
        json.dumps(
            {
                "status": "measured",
                "final_tests_used": [],
                "generation_count": 8000,
                "judges": {"primary_status": "measured", "shadow_status": "measured"},
                "protocols": {"corpus_retrieval": {"status": "measured"}},
            }
        ),
        encoding="utf-8",
    )
    assert assert_scoring_summary(scoring)["status"] == "measured"
    payload = json.loads(scoring.read_text(encoding="utf-8"))
    payload["judges"]["shadow_status"] = "incomplete"
    scoring.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="scoring is incomplete"):
        assert_scoring_summary(scoring)
