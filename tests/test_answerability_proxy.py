from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from doc2query.preferences.answerability_proxy import (
    JUDGE_MODELS,
    MINIMUM_PRECISION,
    MINIMUM_RECALL,
    PROXY_FEATURES,
    ProxyRule,
    RuleAtom,
    build_side_items,
    calibrate_answerability_proxy,
    extract_features,
    score_rule,
    select_rule,
    split_half,
)


def _components(**overrides: float) -> dict[str, Any]:
    base: dict[str, Any] = {feature: 0.0 for feature in PROXY_FEATURES}
    base["corpus_possibly_ambiguous_query"] = False
    base.update(overrides)
    return base


def _sample_row(pair_id: str, chosen_value: float, rejected_value: float) -> dict[str, Any]:
    return {
        "pair_id": pair_id,
        "chosen_components": _components(passage_recall=chosen_value),
        "rejected_components": _components(passage_recall=rejected_value),
    }


def _verdict(
    pair_id: str,
    audit_id: str,
    *,
    answerable_a: tuple[bool, bool],
    answerable_b: tuple[bool, bool],
    models: tuple[str, ...] = JUDGE_MODELS,
) -> dict[str, Any]:
    ratings = {
        model: {
            "answerable_a": answerable_a[index],
            "answerable_b": answerable_b[index],
        }
        for index, model in enumerate(models)
    }
    return {"pair_id": pair_id, "audit_id": audit_id, "ratings": ratings}


def test_split_is_deterministic_and_keeps_both_sides_together() -> None:
    """Podział jest po audit_id, więc obie strony pary dzielą połowę (bez wycieku pasażu)."""
    assert split_half("abc") == split_half("abc")
    halves = {split_half(f"audit-{index}") for index in range(64)}
    assert halves == {"fit", "holdout"}


def test_booleans_map_to_zero_one_and_features_are_exactly_the_frozen_list() -> None:
    features = extract_features(_components(corpus_possibly_ambiguous_query=True))

    assert set(features) == set(PROXY_FEATURES)
    assert features["corpus_possibly_ambiguous_query"] == 1.0


def test_label_requires_consensus_of_both_judges() -> None:
    rows = [
        _sample_row("p1", 1.0, 0.0),
        _sample_row("p2", 1.0, 0.0),
        _sample_row("p3", 1.0, 0.0),
    ]
    keys = [
        {"pair_id": "p1", "automatic_chosen_option": "A"},
        {"pair_id": "p2", "automatic_chosen_option": "B"},
        {"pair_id": "p3", "automatic_chosen_option": "A"},
    ]
    verdicts = [
        # zgoda po obu stronach
        _verdict("p1", "a1", answerable_a=(True, True), answerable_b=(False, False)),
        # sędziowie rozjechani po stronie a -> brak etykiety dla tej strony
        _verdict("p2", "a2", answerable_a=(True, False), answerable_b=(True, True)),
        # tylko jeden sędzia ocenił -> żadnej etykiety
        _verdict(
            "p3",
            "a3",
            answerable_a=(True, True),
            answerable_b=(True, True),
            models=(JUDGE_MODELS[0],),
        ),
    ]

    items, supply = build_side_items(sample_rows=rows, verdict_rows=verdicts, key_rows=keys)

    assert supply["sides_with_two_ratings"] == 4
    assert supply["sides_labelled"] == 3
    assert supply["sides_skipped_judge_split"] == 1
    assert supply["sides_skipped_missing_rating"] == 2
    assert supply["inter_judge_answerability_agreement"] == pytest.approx(3 / 4)
    roles = {(item.pair_id, item.role, item.label) for item in items}
    assert roles == {("p1", "chosen", True), ("p1", "rejected", False), ("p2", "chosen", True)}


def test_role_mapping_follows_the_automatic_option() -> None:
    rows = [_sample_row("p1", 9.0, 1.0)]
    keys = [{"pair_id": "p1", "automatic_chosen_option": "B"}]
    verdicts = [_verdict("p1", "a1", answerable_a=(True, True), answerable_b=(True, True))]

    items, _ = build_side_items(sample_rows=rows, verdict_rows=verdicts, key_rows=keys)

    chosen = next(item for item in items if item.role == "chosen")
    # opcja B jest wyborem automatu, więc strona chosen nosi komponenty chosen_*
    assert chosen.features["passage_recall"] == 9.0


def test_missing_sample_row_is_refused() -> None:
    with pytest.raises(ValueError, match="missing from the export sample"):
        build_side_items(
            sample_rows=[],
            verdict_rows=[
                _verdict("p1", "a1", answerable_a=(True, True), answerable_b=(True, True))
            ],
            key_rows=[{"pair_id": "p1", "automatic_chosen_option": "A"}],
        )


def test_score_rule_counts_the_confusion_matrix() -> None:
    items, _ = build_side_items(
        sample_rows=[_sample_row("p1", 1.0, 0.0), _sample_row("p2", 1.0, 0.0)],
        verdict_rows=[
            _verdict("p1", "a1", answerable_a=(True, True), answerable_b=(False, False)),
            _verdict("p2", "a2", answerable_a=(False, False), answerable_b=(True, True)),
        ],
        key_rows=[
            {"pair_id": "p1", "automatic_chosen_option": "A"},
            {"pair_id": "p2", "automatic_chosen_option": "A"},
        ],
    )
    rule = ProxyRule((RuleAtom("passage_recall", "ge", 0.5),))

    metrics = score_rule(rule, items)

    assert metrics["count"] == 4
    assert metrics["true_positive"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["true_negative"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["precision_yes"] == pytest.approx(0.5)
    assert metrics["balanced_accuracy"] == pytest.approx(0.5)


def _separable_items(count: int = 40) -> list[Any]:
    """Cecha idealnie rozdziela klasy, więc reguła progowa musi zostać znaleziona."""
    sample_rows = []
    verdicts = []
    keys = []
    for index in range(count):
        pair_id = f"p{index}"
        answerable = index % 2 == 0
        value = 10.0 if answerable else 0.0
        sample_rows.append(_sample_row(pair_id, value, value))
        keys.append({"pair_id": pair_id, "automatic_chosen_option": "A"})
        verdicts.append(
            _verdict(
                pair_id,
                f"audit-{index}",
                answerable_a=(answerable, answerable),
                answerable_b=(answerable, answerable),
            )
        )
    items, _ = build_side_items(sample_rows=sample_rows, verdict_rows=verdicts, key_rows=keys)
    return items


def test_selection_finds_a_separating_rule_and_is_deterministic() -> None:
    items = _separable_items()
    fit = [item for item in items if item.half == "fit"]

    first = select_rule(fit)
    second = select_rule(fit)

    assert first["construction_status"] == "selected"
    assert first["rule"] == second["rule"]
    assert first["fit_metrics"]["precision_yes"] >= MINIMUM_PRECISION
    assert first["fit_metrics"]["recall_yes"] >= MINIMUM_RECALL


def test_selection_fails_closed_when_no_rule_reaches_the_frozen_criterion() -> None:
    """Cecha stała nie rozdziela klas 50/50, więc konstrukcja musi zawieść, nie poluzować."""
    sample_rows = []
    verdicts = []
    keys = []
    for index in range(30):
        pair_id = f"p{index}"
        answerable = index % 2 == 0
        sample_rows.append(_sample_row(pair_id, 1.0, 1.0))
        keys.append({"pair_id": pair_id, "automatic_chosen_option": "A"})
        verdicts.append(
            _verdict(
                pair_id,
                f"audit-{index}",
                answerable_a=(answerable, answerable),
                answerable_b=(answerable, answerable),
            )
        )
    items, _ = build_side_items(sample_rows=sample_rows, verdict_rows=verdicts, key_rows=keys)

    selection = select_rule(items)

    assert selection["construction_status"] == "failed_no_admissible_rule_on_fit"
    assert selection["rule"] is None
    assert selection["best_fit_precision_metrics"]["precision_yes"] == pytest.approx(0.5)


def _write_export(directory: Path, items_count: int = 40) -> None:
    (directory / "groq_dual_llm").mkdir(parents=True, exist_ok=True)
    sample_rows = []
    verdicts = []
    keys = []
    for index in range(items_count):
        pair_id = f"p{index}"
        answerable = index % 2 == 0
        value = 10.0 if answerable else 0.0
        sample_rows.append(_sample_row(pair_id, value, value))
        keys.append({"pair_id": pair_id, "automatic_chosen_option": "A"})
        verdicts.append(
            _verdict(
                pair_id,
                f"audit-{index}",
                answerable_a=(answerable, answerable),
                answerable_b=(answerable, answerable),
            )
        )
    for name, rows in (
        ("sample.jsonl", sample_rows),
        ("machine_key.jsonl", keys),
    ):
        (directory / name).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
    (directory / "groq_dual_llm" / "pair_verdicts.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in verdicts), encoding="utf-8"
    )


def test_calibration_publishes_a_pinned_artifact_and_reads_holdout_once(tmp_path: Path) -> None:
    export = tmp_path / "export"
    export.mkdir()
    _write_export(export)

    result = calibrate_answerability_proxy(export_dir=export, output_dir=tmp_path / "out")

    published = json.loads((tmp_path / "out" / "answerability_proxy_v1.json").read_text())
    assert published == result
    assert result["answerability_signal"] == "proxy_v1"
    assert result["final_tests_used"] == []
    assert result["task07_training_authorized"] is False
    assert result["split"]["holdout_reads"] == 1
    assert len(result["label_snapshot"]["sample_sha256"]) == 64
    assert result["status"] == "accepted_as_chosen_side_filter"


def test_failed_construction_never_reads_the_holdout(tmp_path: Path) -> None:
    export = tmp_path / "export"
    (export / "groq_dual_llm").mkdir(parents=True)
    sample_rows = []
    verdicts = []
    keys = []
    for index in range(30):
        pair_id = f"p{index}"
        answerable = index % 2 == 0
        sample_rows.append(_sample_row(pair_id, 1.0, 1.0))
        keys.append({"pair_id": pair_id, "automatic_chosen_option": "A"})
        verdicts.append(
            _verdict(
                pair_id,
                f"audit-{index}",
                answerable_a=(answerable, answerable),
                answerable_b=(answerable, answerable),
            )
        )
    for name, rows in (("sample.jsonl", sample_rows), ("machine_key.jsonl", keys)):
        (export / name).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
    (export / "groq_dual_llm" / "pair_verdicts.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in verdicts), encoding="utf-8"
    )

    result = calibrate_answerability_proxy(export_dir=export, output_dir=tmp_path / "out")

    assert result["holdout"] is None
    assert result["split"]["holdout_reads"] == 0
    assert result["status"] == "rejected_axis_a_without_answerability_filter"
