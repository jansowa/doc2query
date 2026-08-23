from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from doc2query.preferences.defect_pair_gate_v2_1 import (
    GATE_CONTRACT,
    clopper_pearson_bounds,
    guardrail_fired,
    measure_gate,
    paired_contrast,
    verdict_at_least,
    verdict_at_most,
)

POLICY_PATH = Path("configs/preferences/task06_defect_pair_policy_v2_1.yaml")


# --- reguła przedziałowa --------------------------------------------------------


def test_clopper_pearson_reproduces_the_numbers_the_adr_was_sized_on() -> None:
    """Punkty decyzyjne z §4.2 ADR muszą wychodzić z tej implementacji."""
    # P3: ≤ 16/800 przechodzi, 17/800 już nie.
    assert verdict_at_most(16, 800, 0.031, 0.05) == "pass"
    assert verdict_at_most(17, 800, 0.031, 0.05) == "inconclusive"
    # P2: ≥ 262/800 przechodzi, 261 nie.
    assert verdict_at_least(262, 800, 0.30, 0.05) == "pass"
    assert verdict_at_least(261, 800, 0.30, 0.05) == "inconclusive"
    # Guardrail P1: zapala się dopiero od 51/800.
    assert guardrail_fired(51, 800, 0.05, 0.05) is True
    assert guardrail_fired(50, 800, 0.05, 0.05) is False


def test_point_estimate_below_threshold_is_not_enough_to_pass() -> None:
    """To jest cała nauka z v2.0: punkt pod progiem, przedział na progu."""
    # 26/800 = 3,25% punktowo pod progiem 3,1%? nie - pod 5%, ale nad 3,1%.
    assert verdict_at_most(26, 800, 0.031, 0.05) == "inconclusive"
    lower, upper = clopper_pearson_bounds(26, 800, 0.05)
    assert lower < 0.031 < upper


def test_verdicts_are_three_valued_and_fail_only_when_proven() -> None:
    assert verdict_at_most(200, 800, 0.031, 0.05) == "fail"
    assert verdict_at_least(10, 800, 0.30, 0.05) == "fail"
    assert verdict_at_most(0, 800, 0.031, 0.05) == "pass"


def test_clopper_pearson_edges_and_input_validation() -> None:
    assert clopper_pearson_bounds(0, 10, 0.05)[0] == 0.0
    assert clopper_pearson_bounds(10, 10, 0.05)[1] == 1.0
    with pytest.raises(ValueError, match="niepustej próby"):
        clopper_pearson_bounds(0, 0, 0.05)
    with pytest.raises(ValueError, match="mieścić się w próbie"):
        clopper_pearson_bounds(11, 10, 0.05)
    with pytest.raises(ValueError, match="alpha"):
        clopper_pearson_bounds(1, 10, 0.9)


def test_paired_contrast_is_deterministic_and_bounds_the_difference() -> None:
    observations = [(True, False)] * 60 + [(False, False)] * 40
    first = paired_contrast(observations, replicates=500, seed=20260823)
    second = paired_contrast(observations, replicates=500, seed=20260823)
    assert first == second
    assert first["difference"] == pytest.approx(0.6)
    assert first["ci95_low"] < 0.6 < first["ci95_high"]
    assert first["ci95_low"] > 0.20  # próg P4' z ogromnym zapasem
    with pytest.raises(ValueError, match="co najmniej jednej pary"):
        paired_contrast([], replicates=10, seed=1)


# --- pełny pomiar bramki na syntetycznym audycie --------------------------------


def _export(tmp_path: Path, pair_count: int) -> Path:
    """Zgodny manifest eksportu: pomiar bramki przechodzi przez pełną walidację modelu."""
    from doc2query.preferences.pair_audit_export import BLIND_FIELDS
    from doc2query.training.dpo import canonical_fingerprint

    export_dir = tmp_path / "export"
    export_dir.mkdir()
    artifact = {"path": "x.jsonl", "sha256": "a" * 64, "record_count": pair_count}
    payload: dict[str, Any] = {
        "schema_version": 1,
        "contract": "task06-defect-pair-audit-blind-export-v2-1",
        "status": "blind_export_frozen_not_reviewed",
        "policy_id": "task06-defect-pair-policy-v2.1",
        "policy_sha256": "b" * 64,
        "axis": "A",
        "source_cohorts": ["same_prompt_expansion_v1"],
        "source_pair_manifest_sha256": {"same_prompt_expansion_v1": "c" * 64},
        "population_pair_count": pair_count * 2,
        "population_defect_label_counts": {"judge_unanswerable": pair_count * 2},
        "excluded_pair_count": 0,
        "target_pair_count": pair_count,
        "sampled_pair_count": pair_count,
        "shortfall_pair_count": 0,
        "development_gate_met": True,
        "minimum_pair_count_to_start": 1,
        "powered_sample_delivered": True,
        "seed": 20260823,
        "strata": [
            {
                "cohort_id": "same_prompt_expansion_v1",
                "rejected_defect_label": "judge_unanswerable",
                "requested_form": "full_question",
                "population": pair_count,
                "allocated": pair_count,
            }
        ],
        "sampled_defect_label_counts": {"judge_unanswerable": pair_count},
        "orientation_commitment_salt": "salt",
        "orientation_balance": {"A": pair_count // 2, "B": pair_count - pair_count // 2},
        "audit_ids_fingerprint": "f" * 64,
        "blind_pairs": dict(artifact),
        "machine_key": dict(artifact),
        "sample": dict(artifact),
        "report": {"path": "report.json", "sha256": "d" * 64, "record_count": 1},
        "blind_fields": list(BLIND_FIELDS),
        "margin_used_for_stratification": False,
        "axis_used_for_stratification": False,
        "ratings_collected": False,
        "human_evidence_claimed": False,
        "task07_training_authorized": False,
        "final_tests_used": [],
    }
    payload["manifest_fingerprint"] = canonical_fingerprint(payload)
    (export_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    return export_dir


def _audit(
    tmp_path: Path,
    *,
    pair_count: int,
    unanswerable_chosen: int,
    supports: int,
    contradicts: int,
    status: str = "complete",
    rated: int | None = None,
    drop_second_model: bool = False,
) -> Path:
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir()
    (audit_dir / "analysis.json").write_text(
        json.dumps(
            {
                "status": status,
                "rated_pair_count": pair_count if rated is None else rated,
                "export_policy_id": "task06-defect-pair-policy-v2.1",
            }
        ),
        encoding="utf-8",
    )
    lines: list[str] = []
    for index in range(pair_count):
        chosen_bad = index < unanswerable_chosen
        if index < supports:
            consensus = "consensus_supports_automatic"
        elif index < supports + contradicts:
            consensus = "consensus_contradicts_automatic"
        else:
            consensus = "abstained"
        # Orientacja naprzemienna, żeby test sprawdzał odślepianie ról, nie stałą pozycję.
        automatic = "A" if index % 2 == 0 else "B"
        answerable_chosen = not chosen_bad
        rating = {
            "preference": automatic if consensus != "abstained" else "tie",
            "confidence": 0.9,
            "reason_code": "grounding",
            f"answerable_{'a' if automatic == 'A' else 'b'}": answerable_chosen,
            f"format_valid_{'a' if automatic == 'A' else 'b'}": True,
            # rejected: nieodpowiadalny w 60% par, żeby kontrast P4' był wyraźny
            f"answerable_{'b' if automatic == 'A' else 'a'}": index % 5 >= 3,
            f"format_valid_{'b' if automatic == 'A' else 'a'}": True,
        }
        ratings = {"openai/gpt-oss-120b": rating, "qwen/qwen3.6-27b": dict(rating)}
        if drop_second_model and index == 0:
            ratings.pop("qwen/qwen3.6-27b")
        lines.append(
            json.dumps(
                {
                    "audit_id": f"{index:024d}",
                    "pair_id": f"{index:032d}",
                    "automatic_chosen_option": automatic,
                    "consensus": consensus,
                    "ratings": ratings,
                    "rejected_defect_labels": ["judge_unanswerable"],
                },
                ensure_ascii=False,
            )
        )
    (audit_dir / "pair_verdicts.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return audit_dir


def test_gate_passes_when_every_confirmatory_prediction_clears_its_interval(
    tmp_path: Path,
) -> None:
    export_dir = _export(tmp_path, 800)
    audit_dir = _audit(
        tmp_path, pair_count=800, unanswerable_chosen=20, supports=400, contradicts=10
    )
    result = measure_gate(export_dir=export_dir, audit_dir=audit_dir, policy_path=POLICY_PATH)
    assert result["contract"] == GATE_CONTRACT
    assert result["gate"]["passed"] is True
    assert result["gate"]["blocking"] == []
    assert result["predictions"]["P2"]["verdict"] == "pass"
    assert result["predictions"]["P3"]["verdict"] == "pass"
    for row in result["predictions"]["P4_prime"]["per_model"].values():
        assert row["verdict"] == "pass"
    for row in result["predictions"]["P1"]["per_model"].values():
        assert row["guardrail_fired"] is False
    assert result["task07_training_authorized"] is False
    assert result["final_tests_used"] == []


def test_inconclusive_confirmatory_prediction_blocks_the_gate(tmp_path: Path) -> None:
    """Fail-closed: przedział zawierający próg NIE zdaje bramki."""
    export_dir = _export(tmp_path, 800)
    audit_dir = _audit(
        tmp_path, pair_count=800, unanswerable_chosen=20, supports=400, contradicts=20
    )
    result = measure_gate(export_dir=export_dir, audit_dir=audit_dir, policy_path=POLICY_PATH)
    assert result["predictions"]["P3"]["verdict"] == "inconclusive"
    assert result["gate"]["passed"] is False
    assert "P3:inconclusive" in result["gate"]["blocking"]
    assert result["gate"]["decision"] == "pairs_do_not_proceed_policy_returns_to_design"


def test_guardrail_blocks_only_on_a_proven_violation(tmp_path: Path) -> None:
    export_dir = _export(tmp_path, 800)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    clean = _audit(
        tmp_path / "a", pair_count=800, unanswerable_chosen=45, supports=400, contradicts=10
    )
    result = measure_gate(export_dir=export_dir, audit_dir=clean, policy_path=POLICY_PATH)
    # 45/800 = 5,6% punktowo NAD progiem 5%, ale naruszenie nie jest dowiedzione.
    assert result["predictions"]["P1"]["per_model"]["qwen/qwen3.6-27b"]["share"] > 0.05
    assert all(
        row["guardrail_fired"] is False
        for row in result["predictions"]["P1"]["per_model"].values()
    )
    assert result["gate"]["passed"] is True

    proven = _audit(
        tmp_path / "b", pair_count=800, unanswerable_chosen=80, supports=400, contradicts=10
    )
    fired = measure_gate(export_dir=export_dir, audit_dir=proven, policy_path=POLICY_PATH)
    assert all(
        row["guardrail_fired"] is True
        for row in fired["predictions"]["P1"]["per_model"].values()
    )
    assert fired["gate"]["passed"] is False
    assert any(value.startswith("P1_guardrail") for value in fired["gate"]["blocking"])


def test_gate_refuses_an_unfinished_audit(tmp_path: Path) -> None:
    """Bramki nie wolno podejrzeć po pierwszym oknie dziennego budżetu."""
    export_dir = _export(tmp_path, 800)
    audit_dir = _audit(
        tmp_path,
        pair_count=800,
        unanswerable_chosen=20,
        supports=400,
        contradicts=10,
        status="incomplete_quota_deferred",
        rated=384,
    )
    with pytest.raises(ValueError, match="niedokończonym audycie"):
        measure_gate(export_dir=export_dir, audit_dir=audit_dir, policy_path=POLICY_PATH)


def test_gate_refuses_a_pair_rated_by_one_judge_only(tmp_path: Path) -> None:
    export_dir = _export(tmp_path, 20)
    audit_dir = _audit(
        tmp_path,
        pair_count=20,
        unanswerable_chosen=0,
        supports=10,
        contradicts=0,
        drop_second_model=True,
    )
    with pytest.raises(ValueError, match="ocen obu sędziów"):
        measure_gate(export_dir=export_dir, audit_dir=audit_dir, policy_path=POLICY_PATH)


def test_gate_refuses_an_export_from_another_policy(tmp_path: Path) -> None:
    export_dir = _export(tmp_path, 800)
    from doc2query.training.dpo import canonical_fingerprint

    manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest.pop("manifest_fingerprint")
    manifest["policy_id"] = "task06-defect-pair-policy-v2.0"
    manifest["manifest_fingerprint"] = canonical_fingerprint(manifest)
    (export_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    audit_dir = _audit(
        tmp_path, pair_count=800, unanswerable_chosen=20, supports=400, contradicts=10
    )
    with pytest.raises(ValueError, match="zamrożonej polityki"):
        measure_gate(export_dir=export_dir, audit_dir=audit_dir, policy_path=POLICY_PATH)
