from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from doc2query.evaluation.probe_convergence import (
    ProbeRunMetrics,
    _sign_flip_p_value,
    apply_convergence_guardrail,
    load_guardrail,
    load_run_group,
    paired_seed_comparison,
    read_probe_run,
)

GUARDRAIL_PATH = Path("configs/evaluation/task04_m03_probe_convergence_guardrail_v1.yaml")
CORPUS_SIZE = 139782


def _run(arm: str, seed: int, *, ndcg: float, recall: float) -> ProbeRunMetrics:
    return ProbeRunMetrics(
        run_id=f"RUN-{arm.upper()}-S{seed}",
        arm=arm,
        seed=seed,
        metrics={"corpus_ndcg_at_10": ndcg, "corpus_recall_at_100": recall},
        candidate_count=CORPUS_SIZE,
        query_count=8000,
        first_loss=2.0,
        last_loss=0.1,
    )


def _write_run(root: Path, run_id: str, *, seed: int, ndcg: float, recall: float) -> Path:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    metrics = {"corpus_ndcg_at_10": ndcg, "corpus_recall_at_100": recall}
    retrieval: dict[str, Any] = {
        "status": "measured",
        "query_count": 8000,
        "metrics": metrics,
        "metric_candidate_count": {key: CORPUS_SIZE for key in metrics},
    }
    (run_dir / "corpus_retrieval_summary.json").write_text(json.dumps(retrieval), encoding="utf-8")
    (run_dir / "train_summary.json").write_text(
        json.dumps({"status": "measured", "recipe": {"seed": seed}, "first_loss": 2.0,
                    "last_loss": 0.1}),
        encoding="utf-8",
    )
    return run_dir


def test_frozen_contract_forbids_a_loss_based_guardrail() -> None:
    guardrail = load_guardrail(GUARDRAIL_PATH)

    assert guardrail.signal.metric == "corpus_recall_at_100"
    assert guardrail.signal.loss_based_guardrail_permitted is False
    assert guardrail.aggregation.decision_metric == "corpus_ndcg_at_10"
    assert guardrail.aggregation.min_converged_seed_pairs == 5
    assert guardrail.aggregation.superiority_threshold == 0.01
    assert guardrail.aggregation.report_unfiltered_result is True
    assert guardrail.final_tests_used == []


def test_a_loss_based_guardrail_cannot_be_configured(tmp_path: Path) -> None:
    raw = yaml.safe_load(GUARDRAIL_PATH.read_text(encoding="utf-8"))
    raw["signal"]["loss_based_guardrail_permitted"] = True
    path = tmp_path / "guardrail.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")

    with pytest.raises(ValueError):
        load_guardrail(path)


def test_fewer_than_five_seed_pairs_is_refused_by_the_contract(tmp_path: Path) -> None:
    raw = yaml.safe_load(GUARDRAIL_PATH.read_text(encoding="utf-8"))
    raw["aggregation"]["min_converged_seed_pairs"] = 4
    path = tmp_path / "guardrail.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")

    with pytest.raises(ValueError):
        load_guardrail(path)


def test_collapsed_run_is_flagged_and_healthy_runs_are_not() -> None:
    guardrail = load_guardrail(GUARDRAIL_PATH)
    runs = [
        _run("hybrid", seed, ndcg=0.10, recall=0.16) for seed in (42, 43, 44, 45, 46)
    ] + [_run("w06", seed, ndcg=0.08, recall=0.15) for seed in (42, 44, 45, 46)]
    runs.append(_run("w06", 43, ndcg=0.00002, recall=0.00045))

    verdicts = apply_convergence_guardrail(runs, guardrail)

    flagged = [verdict for verdict in verdicts if not verdict.converged]
    assert [verdict.run_id for verdict in flagged] == ["RUN-W06-S43"]
    assert flagged[0].failure_reason == "corpus_recall_at_100_below_floor"
    assert flagged[0].applied_floor == pytest.approx(0.5 * 0.155)


def test_chance_floor_catches_a_comparison_that_collapsed_entirely() -> None:
    guardrail = load_guardrail(GUARDRAIL_PATH)
    runs = [_run("hybrid", seed, ndcg=0.0, recall=0.0004) for seed in (42, 43)]
    runs += [_run("w06", seed, ndcg=0.0, recall=0.0004) for seed in (42, 43)]

    verdicts = apply_convergence_guardrail(runs, guardrail)

    assert all(not verdict.converged for verdict in verdicts)
    assert verdicts[0].applied_floor == pytest.approx(4.0 * 100 / CORPUS_SIZE)


def test_seed_is_dropped_as_a_pair_and_the_unfiltered_result_stays_visible() -> None:
    guardrail = load_guardrail(GUARDRAIL_PATH)
    variant = [
        _run("hybrid", 42, ndcg=0.105, recall=0.196),
        _run("hybrid", 43, ndcg=0.102, recall=0.190),
        _run("hybrid", 44, ndcg=0.074, recall=0.160),
        _run("hybrid", 45, ndcg=0.068, recall=0.159),
        _run("hybrid", 46, ndcg=0.067, recall=0.138),
    ]
    anchor = [
        _run("w06", 42, ndcg=0.078, recall=0.166),
        _run("w06", 43, ndcg=0.00002, recall=0.00045),
        _run("w06", 44, ndcg=0.060, recall=0.141),
        _run("w06", 45, ndcg=0.050, recall=0.127),
        _run("w06", 46, ndcg=0.070, recall=0.156),
    ]

    decision = paired_seed_comparison(variant, anchor, guardrail)

    assert decision["dropped_seeds"] == [43]
    assert decision["converged"]["seed_pair_count"] == 4
    assert decision["unfiltered"]["seed_pair_count"] == 5
    # Cztery pary to poniżej minimum pięciu, więc reguła jest nierozstrzygalna.
    assert decision["status"] == "insufficient_converged_seeds"
    assert decision["promotion_authorized"] is False
    assert decision["unfiltered"]["mean_difference"] > decision["converged"]["mean_difference"]


def test_five_converged_seeds_can_reach_a_superior_verdict() -> None:
    guardrail = load_guardrail(GUARDRAIL_PATH)
    variant = [_run("hybrid", seed, ndcg=0.14, recall=0.16) for seed in (42, 43, 44, 45, 46)]
    anchor = [_run("w06", seed, ndcg=0.08, recall=0.15) for seed in (42, 43, 44, 45, 46)]

    decision = paired_seed_comparison(variant, anchor, guardrail)

    assert decision["dropped_seeds"] == []
    assert decision["status"] == "superior"
    assert decision["converged"]["sign_flip"]["p_value"] == pytest.approx(1 / 32)
    assert decision["promotion_authorized"] is False


def test_five_converged_seeds_below_the_threshold_are_not_superior() -> None:
    guardrail = load_guardrail(GUARDRAIL_PATH)
    variant = [_run("hybrid", seed, ndcg=0.0805, recall=0.16) for seed in (42, 43, 44, 45, 46)]
    anchor = [_run("w06", seed, ndcg=0.08, recall=0.15) for seed in (42, 43, 44, 45, 46)]

    decision = paired_seed_comparison(variant, anchor, guardrail)

    assert decision["status"] == "not_superior"


def test_sign_flip_cannot_reach_alpha_five_percent_with_four_pairs() -> None:
    four = _sign_flip_p_value([0.2, 0.2, 0.2, 0.2], 0.01)
    five = _sign_flip_p_value([0.2, 0.2, 0.2, 0.2, 0.2], 0.01)

    assert four["smallest_attainable_p_value"] == pytest.approx(1 / 16)
    assert four["smallest_attainable_p_value"] > 0.05
    assert five["smallest_attainable_p_value"] == pytest.approx(1 / 32)
    assert five["p_value"] == pytest.approx(1 / 32)


def test_run_reader_refuses_an_unmeasured_run(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, "RUN-A", seed=42, ndcg=0.1, recall=0.16)
    path = run_dir / "corpus_retrieval_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = "running"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="not measured"):
        read_probe_run(run_dir, arm="hybrid")


def test_run_group_include_splits_a_sweep_by_budget(tmp_path: Path) -> None:
    for budget in (1024, 2048):
        for arm in ("HYBRID", "W06"):
            for seed in (42, 43, 44):
                _write_run(
                    tmp_path,
                    f"PROBE-BUDGET-{budget}-{arm}-S{seed}",
                    seed=seed,
                    ndcg=0.1,
                    recall=0.16,
                )

    arms = {"HYBRID": "hybrid", "W06": "w06"}
    whole = load_run_group(tmp_path, arms)
    single = load_run_group(tmp_path, arms, include="PROBE-BUDGET-1024")

    assert len(whole) == 12
    assert len(single) == 6
    assert all("1024" in run.run_id for run in single)


def test_two_runs_of_one_arm_per_seed_are_refused() -> None:
    guardrail = load_guardrail(GUARDRAIL_PATH)
    variant = [
        _run("hybrid", 42, ndcg=0.1, recall=0.16),
        _run("hybrid", 42, ndcg=0.2, recall=0.16),
    ]
    anchor = [_run("w06", 42, ndcg=0.08, recall=0.15)]

    with pytest.raises(ValueError, match="at most one run per seed"):
        paired_seed_comparison(variant, anchor, guardrail)
