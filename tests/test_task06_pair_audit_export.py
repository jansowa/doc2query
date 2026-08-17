from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from doc2query.preferences.llm_audit import build_dual_llm_request_plan, load_llm_audit_config
from doc2query.preferences.pair_audit_export import (
    BLIND_FIELDS,
    _largest_remainder,
    export_blind_audit_sample,
    gap_band_label,
    verify_orientation_commitments,
)
from doc2query.preferences.pair_policy import build_tentative_pairs, load_pair_policy

POLICY_PATH = Path("configs/preferences/task06_tentative_pair_policy_v1.yaml")
PASSAGE = (
    "Koronawirusy to duża rodzina wirusów wywołujących choroby układu oddechowego u ludzi "
    "i u zwierząt. Zakażenie przenosi się drogą kropelkową oraz przez kontakt z skażonymi "
    "powierzchniami, a typowe objawy obejmują gorączkę, suchy kaszel, ból gardła i "
    "uczucie duszności. Okres wylęgania wynosi zwykle od dwóch do czternastu dni."
)


def _scoring_row(group: int, index: int, margin: float, query: str) -> dict[str, Any]:
    return {
        "evaluation_id": f"{group}::{group}::same-prompt::{index}",
        "evaluation_group_id": f"task06-preference::{group}::{group}",
        "example_id": f"{group}::{group}",
        "doc_id": str(group),
        "candidate_index": index,
        "generated": query,
        "prompt": "Wygeneruj jedno polskie zapytanie wyszukiwawcze.",
        "prompt_sha256": f"{group:064d}",
        "positive": {"doc_id": str(group), "text": PASSAGE},
        "metadata": {"split": "train"},
        "requested_form": "full_question" if group % 2 else "keyword_query",
        "requested_intent": "fact_lookup",
        "requested_focus": "beginning",
        "generation_config": {"seed": index, "temperature": 0.7, "top_p": 0.95},
        "control": {"form": "full_question", "intent": "fact_lookup"},
        "seed": index,
        "pool_margin": margin,
        "pool_rank": 1,
        "pool_positive_score": 9.0,
        "shadow_pool_margin": 10.0 - index,
        "shadow_pool_rank": 1,
        "corpus_round_trip_at_5": 1.0,
        "corpus_round_trip_at_20": 1.0,
        "corpus_round_trip_at_100": 1.0,
        "corpus_possibly_ambiguous_query": False,
        "format_valid": True,
        "has_prefix": False,
        "has_metacomment": False,
        "multiple_query": False,
        "empty": False,
        "copy_density": 0.1,
        "normalized_lcs": 0.1,
        "longest_copied_ngram": 1,
        "content_jaccard": 0.3,
        "word_length": 5,
        "focus_accuracy": None,
        "judge_rank_disagreement": False,
        "primary_judge": "sdadas/polish-reranker-roberta-v3",
        "shadow_judge": "BAAI/bge-reranker-v2-m3",
        "final_tests_used": [],
        "generation_identity_sha256": "b" * 64,
        "frozen_cohort_fingerprint": "c" * 64,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fingerprint(payload: dict[str, Any]) -> str:
    from doc2query.training.dpo import canonical_fingerprint

    return canonical_fingerprint(payload)


def _cohort(root: Path, cohort_id: str, groups: int, *, offset: int = 0) -> Path:
    """Materialize a minimal but contract-valid frozen cohort with a finished gate."""
    cohort = root / cohort_id
    rows: list[dict[str, Any]] = []
    verdicts: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    queries = (
        "jakie objawy wywołuje koronawirus",
        "czym jest rodzina wirusów oddechowych",
        "ile dni trwa okres wylęgania zakażenia",
    )
    margins = (9.0, 5.0, 1.0)
    for group in range(offset + 1, offset + groups + 1):
        group_rows = [
            _scoring_row(group, index, margins[index], queries[index]) for index in range(3)
        ]
        rows.extend(group_rows)
        verdicts.append(
            {
                "group_id": f"task06-preference::{group}::{group}",
                "eligible": True,
                "representative_candidate_ids": [row["evaluation_id"] for row in group_rows],
            }
        )
        records.append({"example_id": f"{group}::{group}", "cluster_id": str(group)})
    _write_jsonl(cohort / "d01_controlled" / "scoring" / "per_generation.jsonl", rows)
    (cohort / "d01_controlled" / "scoring" / "summary.json").write_text(
        json.dumps({"status": "measured", "generation_count": len(rows)}), encoding="utf-8"
    )
    _write_jsonl(cohort / "cohort.records.jsonl", records)
    gate = cohort / "diversity_gate"
    _write_jsonl(gate / "group_verdicts.jsonl", verdicts)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "contract": "task06-same-prompt-diversity-gate-v1",
        "status": "diversity_gate_applied_not_paired",
        "policy_id": "task06-same-prompt-diversity-gate-v1",
        "policy_sha256": "d" * 64,
        "policy_fingerprint": "e" * 64,
        "generations_sha256": "f" * 64,
        "generation_identity_sha256": "b" * 64,
        "frozen_cohort_fingerprint": "c" * 64,
        "split": "train",
        "group_count": groups,
        "candidate_count": len(rows),
        "eligible_group_count": groups,
        "rejected_group_count": 0,
        "group_ids_fingerprint": "1" * 64,
        "eligible_group_ids_fingerprint": "2" * 64,
        "verdicts": {
            "path": "group_verdicts.jsonl",
            "sha256": _sha256(gate / "group_verdicts.jsonl"),
            "record_count": groups,
        },
        "report": {"path": "report.json", "sha256": "3" * 64, "record_count": 1},
        "judge_scores_read": False,
        "candidates_ranked": False,
        "pairs_built": False,
        "model_loading_performed": False,
        "final_tests_used": [],
    }
    manifest["manifest_fingerprint"] = _fingerprint(manifest)
    (gate / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return cohort


@pytest.fixture
def pair_dirs(tmp_path: Path) -> list[Path]:
    dirs = []
    for cohort_id, groups, offset in (
        ("same_prompt_expansion_v1", 6, 0),
        ("same_prompt_expansion_v2", 8, 100),
    ):
        cohort = _cohort(tmp_path, cohort_id, groups, offset=offset)
        build_tentative_pairs(cohort_dir=cohort, policy_path=POLICY_PATH)
        dirs.append(cohort / "tentative_pairs")
    return dirs


def test_blind_rows_expose_only_the_frozen_field_set(pair_dirs: list[Path], tmp_path: Path) -> None:
    manifest = export_blind_audit_sample(
        pair_dirs=pair_dirs, policy_path=POLICY_PATH, output_dir=tmp_path / "audit"
    )

    assert manifest.population_pair_count == 14
    assert manifest.sampled_pair_count == 14
    for line in (tmp_path / "audit" / "blind_pairs.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        assert set(row) == set(BLIND_FIELDS)
        assert "chosen" not in row and "rejected" not in row
        assert "primary_margin_gap" not in row


def test_population_below_target_is_reported_as_a_shortfall(
    pair_dirs: list[Path], tmp_path: Path
) -> None:
    manifest = export_blind_audit_sample(
        pair_dirs=pair_dirs, policy_path=POLICY_PATH, output_dir=tmp_path / "audit"
    )

    assert manifest.target_pair_count == 500
    assert manifest.shortfall_pair_count == 486
    assert manifest.development_gate_met is False


def test_orientation_is_counterbalanced_and_committed_before_review(
    pair_dirs: list[Path], tmp_path: Path
) -> None:
    manifest = export_blind_audit_sample(
        pair_dirs=pair_dirs, policy_path=POLICY_PATH, output_dir=tmp_path / "audit"
    )

    assert manifest.orientation_balance == {"A": 7, "B": 7}
    assert manifest.ratings_collected is False
    assert verify_orientation_commitments(tmp_path / "audit") == 14


def test_tampering_with_the_unblinding_key_breaks_the_commitment(
    pair_dirs: list[Path], tmp_path: Path
) -> None:
    export_blind_audit_sample(
        pair_dirs=pair_dirs, policy_path=POLICY_PATH, output_dir=tmp_path / "audit"
    )
    key_path = tmp_path / "audit" / "machine_key.jsonl"
    rows = [json.loads(line) for line in key_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["automatic_chosen_option"] = "B" if rows[0]["automatic_chosen_option"] == "A" else "A"
    _write_jsonl(key_path, rows)

    with pytest.raises(ValueError, match="commitment mismatch"):
        verify_orientation_commitments(tmp_path / "audit")


def test_export_is_deterministic_across_runs(pair_dirs: list[Path], tmp_path: Path) -> None:
    first = export_blind_audit_sample(
        pair_dirs=pair_dirs, policy_path=POLICY_PATH, output_dir=tmp_path / "one"
    )
    second = export_blind_audit_sample(
        pair_dirs=pair_dirs, policy_path=POLICY_PATH, output_dir=tmp_path / "two"
    )

    assert first.audit_ids_fingerprint == second.audit_ids_fingerprint
    assert first.blind_pairs["sha256"] == second.blind_pairs["sha256"]
    assert first.machine_key["sha256"] == second.machine_key["sha256"]


def test_pairs_from_a_foreign_policy_are_refused(
    pair_dirs: list[Path], tmp_path: Path
) -> None:
    raw = yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))
    raw["policy_id"] = "task06-tentative-pair-policy-v1-forged"
    forged = tmp_path / "forged.yaml"
    forged.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")

    with pytest.raises(ValueError, match="different frozen policy"):
        export_blind_audit_sample(
            pair_dirs=pair_dirs, policy_path=forged, output_dir=tmp_path / "audit"
        )


def test_same_cohort_cannot_be_exported_twice(pair_dirs: list[Path], tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="supplied twice"):
        export_blind_audit_sample(
            pair_dirs=[pair_dirs[0], pair_dirs[0]],
            policy_path=POLICY_PATH,
            output_dir=tmp_path / "audit",
        )


def test_existing_export_is_never_overwritten(pair_dirs: list[Path], tmp_path: Path) -> None:
    target = tmp_path / "audit"
    export_blind_audit_sample(pair_dirs=pair_dirs, policy_path=POLICY_PATH, output_dir=target)

    with pytest.raises(FileExistsError):
        export_blind_audit_sample(pair_dirs=pair_dirs, policy_path=POLICY_PATH, output_dir=target)


def test_gap_bands_cover_the_frozen_range() -> None:
    policy = load_pair_policy(POLICY_PATH)

    assert gap_band_label(1.0, policy) == "[1.0,2.0)"
    assert gap_band_label(3.9, policy) == "[2.0,4.0)"
    assert gap_band_label(120.0, policy) == "[4.0,inf)"
    with pytest.raises(ValueError, match="outside the frozen bands"):
        gap_band_label(0.5, policy)


def test_largest_remainder_allocation_is_exact_and_bounded() -> None:
    assert _largest_remainder([10, 10, 10], 30) == [10, 10, 10]
    assert _largest_remainder([10, 10, 10], 40) == [10, 10, 10]
    assert sum(_largest_remainder([100, 50, 7], 100)) == 100
    allocated = _largest_remainder([100, 50, 7], 100)
    assert all(
        value <= population
        for value, population in zip(allocated, [100, 50, 7], strict=True)
    )


def test_blind_export_feeds_the_frozen_groq_plan(pair_dirs: list[Path], tmp_path: Path) -> None:
    """The export is consumable by the frozen dual-LLM planner without leaking fields."""
    export_blind_audit_sample(
        pair_dirs=pair_dirs, policy_path=POLICY_PATH, output_dir=tmp_path / "audit"
    )
    config = load_llm_audit_config(Path("configs/preferences/task06_groq_preference_audit_v1.json"))
    config["pair_count"] = 14

    plan = build_dual_llm_request_plan(config, tmp_path / "audit" / "blind_pairs.jsonl")

    assert {request.model_id for request in plan} == {
        "openai/gpt-oss-120b",
        "qwen/qwen3.6-27b",
    }
    assert sum(len(request.item_ids) for request in plan) == 28
