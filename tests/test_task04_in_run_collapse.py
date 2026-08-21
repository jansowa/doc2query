"""CPU tests for the frozen M-03 in-run collapse detection and reseed contract."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from doc2query.evaluation import embedder_probe
from doc2query.evaluation.embedder_probe import (
    ProbeRecipe,
    train_probe,
    train_probe_with_collapse_reseed,
)
from doc2query.evaluation.probe_in_run_collapse import (
    CollapseDetector,
    InRunCollapseDetection,
    ProbeCollapseDetected,
    ProbeCollapseUnresolved,
    build_interim_evaluation_set,
    interim_recall,
    load_collapse_detection,
)
from doc2query.evaluation.probe_negatives import NEGATIVE_RECIPE_VERSION, NegativeRecipe
from doc2query.evaluation.statistical_contract import StatisticalContract

FROZEN_CONTRACT = Path("configs/evaluation/task04_m03_in_run_collapse_detection_v1.yaml")

# The frozen train_summary.json keys of the closed runs S47-S51; a disabled run must not
# add or drop a single one of them (acceptance criterion A1).
FROZEN_TRAIN_SUMMARY_KEYS = {
    "code",
    "comparison_budget",
    "elapsed_seconds",
    "first_loss",
    "last_loss",
    "negative_contract",
    "peak_vram_allocated_bytes",
    "possible_false_negative_report",
    "query_source",
    "recipe",
    "recipe_fingerprint",
    "recipe_version",
    "schema_version",
    "statistical_contract",
    "status",
    "steps",
    "train_examples",
    "train_fingerprint",
}


def _test_contract(**overrides: Any) -> InRunCollapseDetection:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "contract": "task04-m03-in-run-collapse-detection-v1",
        "detector_id": "task04-m03-in-run-collapse-detection-v1",
        "status": "frozen_before_first_new_run",
        "adr": "reports/decisions/task04_m03_in_run_collapse_detection_v1.md",
        "loss_based_guardrail_permitted": False,
        "persistence": {
            "loss_curve_file": "training_loss_curve.jsonl",
            "interim_evaluation_file": "training_interim_evaluation.jsonl",
            "attempt_journal_file": "collapse_detection_journal.jsonl",
        },
        "interim_evaluation": {
            "source": "training_rows_holdin",
            "metric": "train_holdin_recall_at_100",
            "interval_steps": 2,
            "first_check_step": 2,
            "corpus_documents": 100,
            "queries": 8,
            "retrieval_depth": 1,
            "encode_batch_size": 4,
            "restore_rng_state": True,
        },
        "rules": {
            "retrieval_floor": {
                "rule_id": "interim_recall_below_chance_floor",
                "min_chance_multiple": 4.0,
            },
            "loss_direction": {
                "rule_id": "loss_direction_non_decreasing",
                "window_steps": 2,
            },
            "consecutive_hits_required": 2,
        },
        "reseed": {
            "max_attempts": 3,
            "seed_stride": 1000,
            "on_exhausted": "fail_run_with_status_collapse_unresolved",
            "provenance_required": True,
        },
        "final_tests_used": [],
    }
    payload.update(overrides)
    return InRunCollapseDetection.model_validate(payload)


def _retrieval_only_contract() -> InRunCollapseDetection:
    """Isolate the retrieval rule: a toy model's loss never falls, which is not a collapse."""
    return _test_contract(
        rules={
            "retrieval_floor": {
                "rule_id": "interim_recall_below_chance_floor",
                "min_chance_multiple": 4.0,
            },
            "loss_direction": {"rule_id": "loss_direction_non_decreasing", "window_steps": 100},
            "consecutive_hits_required": 2,
        }
    )


def test_frozen_contract_matches_the_adr() -> None:
    contract = load_collapse_detection(FROZEN_CONTRACT)
    assert contract.contract == "task04-m03-in-run-collapse-detection-v1"
    assert contract.loss_based_guardrail_permitted is False
    assert contract.final_tests_used == []
    assert contract.interim_evaluation.interval_steps == 256
    assert contract.interim_evaluation.corpus_documents == 2048
    assert contract.interim_evaluation.retrieval_depth == 100
    assert contract.rules.consecutive_hits_required == 2
    # The floor multiple is not a new calibration: it is the frozen M-03 constant.
    assert contract.rules.retrieval_floor.min_chance_multiple == 4.0
    assert contract.reseed.max_attempts == 3
    assert [contract.attempt_seed(47, index) for index in range(3)] == [47, 1047, 2047]


def test_attempt_seed_refuses_an_index_outside_the_frozen_budget() -> None:
    with pytest.raises(ValueError, match="reseed budget"):
        _test_contract().attempt_seed(42, 3)


def test_interim_set_is_deterministic_and_capped() -> None:
    rows = [
        {
            "example_id": f"q-{index:02d}",
            "query": f"query {index}",
            "positive_doc_id": f"d-{index % 4}",
            "positive": f"passage {index % 4}",
        }
        for index in range(12)
    ]
    contract = _test_contract().interim_evaluation
    evaluation_set = build_interim_evaluation_set(rows, contract)
    assert evaluation_set.documents == [f"passage {index}" for index in range(4)]
    assert evaluation_set.queries == [f"query {index}" for index in range(8)]
    assert evaluation_set.positive_positions == [index % 4 for index in range(8)]
    assert evaluation_set.chance_level(1) == pytest.approx(0.25)
    assert build_interim_evaluation_set(list(reversed(rows)), contract) == evaluation_set


def test_interim_recall_breaks_ties_pessimistically() -> None:
    documents = torch.eye(4)
    queries = torch.eye(4)[[0, 1]]
    assert interim_recall(queries, documents, [0, 1], depth=1) == pytest.approx(1.0)
    assert interim_recall(queries, documents, [2, 3], depth=1) == pytest.approx(0.0)
    assert interim_recall(queries, documents, [2, 3], depth=4) == pytest.approx(1.0)
    # A degenerate encoder maps everything onto one vector: every score ties, and the
    # pessimistic tie-break must report that as a failure, not as a perfect recall.
    collapsed = torch.ones(4, 4)
    assert interim_recall(collapsed[:2], collapsed, [2, 3], depth=1) == pytest.approx(0.0)
    assert interim_recall(collapsed[:2], collapsed, [2, 3], depth=4) == pytest.approx(1.0)


def test_detector_requires_two_consecutive_hits() -> None:
    contract = _test_contract()
    detector = CollapseDetector(contract=contract, chance_level=0.125)
    assert detector.floor == pytest.approx(0.5)
    healthy = [1.0, 0.5]
    first = detector.observe(step=2, recall=0.1, losses=healthy)
    assert first["below_floor"] is True
    assert first["collapse_detected"] is False
    recovered = detector.observe(step=4, recall=0.9, losses=healthy)
    assert recovered["collapse_detected"] is False
    detector.observe(step=6, recall=0.1, losses=healthy)
    fired = detector.observe(step=8, recall=0.1, losses=healthy)
    assert fired["collapse_detected"] is True
    assert fired["rule"] == "interim_recall_below_chance_floor"


def test_detector_loss_direction_is_a_separate_rule() -> None:
    detector = CollapseDetector(contract=_test_contract(), chance_level=0.125)
    rising = [0.5, 0.5, 2.0, 2.0]
    detector.observe(step=2, recall=0.9, losses=rising)
    fired = detector.observe(step=4, recall=0.9, losses=rising)
    assert fired["loss_non_decreasing"] is True
    assert fired["below_floor"] is False
    assert fired["rule"] == "loss_direction_non_decreasing"


def test_detector_ignores_the_loss_rule_before_two_full_windows() -> None:
    detector = CollapseDetector(contract=_test_contract(), chance_level=0.125)
    for step in (2, 4):
        observation = detector.observe(step=step, recall=0.9, losses=[3.0, 3.0, 3.0])
        assert observation["loss_non_decreasing"] is False
        assert observation["collapse_detected"] is False


def test_should_check_follows_the_frozen_interval() -> None:
    detector = CollapseDetector(
        contract=load_collapse_detection(FROZEN_CONTRACT), chance_level=0.05
    )
    assert detector.should_check(128, 1024) is False
    assert detector.should_check(256, 1024) is True
    assert detector.should_check(257, 1024) is False
    assert detector.should_check(768, 1024) is True
    # The final step is never checked: the real evaluation follows immediately.
    assert detector.should_check(1024, 1024) is False


class _TinyBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(1, 2)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.projection(values))

    def save_pretrained(self, path: Path, *, safe_serialization: bool) -> None:
        assert safe_serialization
        path.mkdir(parents=True)
        torch.save(self.state_dict(), path / "model.safetensors")


class _TinyEncoder(torch.nn.Module):
    def __init__(self, _name: str, _revision: str) -> None:
        super().__init__()
        self.backbone = _TinyBackbone()

    def forward(self, encoded: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.nn.functional.normalize(self.backbone(encoded["values"]), dim=-1)


class _TinyTokenizer:
    def save_pretrained(self, path: Path) -> None:
        (path / "tokenizer.json").write_text("{}", encoding="utf-8")


@pytest.fixture
def tiny_probe(monkeypatch: pytest.MonkeyPatch) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import transformers

    monkeypatch.setattr(embedder_probe, "MeanPoolEncoder", _TinyEncoder)
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: _TinyTokenizer(),
    )
    monkeypatch.setattr(
        embedder_probe,
        "_tokenize",
        lambda _tokenizer, texts, _max_length, device, **_kwargs: {
            "values": torch.tensor([[float(len(text) % 7 + 1)] for text in texts], device=device)
        },
    )
    rows = [
        {
            "example_id": f"q-{index}",
            "query": f"query {index}",
            "positive_doc_id": f"d-{index}",
            "positive": f"positive {index}",
            "negative": f"negative {index}",
            "demoted_negative": "",
        }
        for index in range(8)
    ]
    contract = StatisticalContract(
        payload={
            "contract_version": "fixture-v1",
            "adr": {
                "id": "ADR-fixture",
                "version": "v1",
                "path": "reports/adr/fixture.md",
                "sha256": "b" * 64,
            },
        },
        fingerprint="c" * 64,
    )
    keywords: dict[str, Any] = {
        "query_source": "synthetic",
        "train_fingerprint": "d" * 64,
        "negative_contract": {"hard_negative_strategy": "hn0"},
        "false_negative_report": {"status": "not_applicable"},
        "negative_audit_rows": [],
        "statistical_contract": contract,
    }
    return rows, keywords


def _recipe(seed: int = 42, *, steps: int = 8) -> ProbeRecipe:
    return ProbeRecipe(
        model_name_or_path="fixture/encoder",
        revision="a" * 40,
        recipe_version="probe-collapse-fixture-v1",
        negative_recipe=NegativeRecipe(version=NEGATIVE_RECIPE_VERSION, strategy="hn0"),
        max_length=8,
        batch_size=2,
        max_steps=steps,
        seed=seed,
    )


def test_disabled_detection_writes_exactly_the_frozen_artifacts(
    tmp_path: Path, tiny_probe: tuple[list[dict[str, Any]], dict[str, Any]]
) -> None:
    rows, keywords = tiny_probe
    output = tmp_path / "probe"
    summary = train_probe(rows, recipe=_recipe(), output_dir=output, **keywords)
    assert set(summary) == FROZEN_TRAIN_SUMMARY_KEYS
    assert {path.name for path in output.iterdir()} == {
        "model",
        "negative_audit.jsonl",
        "train_summary.json",
    }


def test_enabled_detection_without_collapse_keeps_the_trajectory(
    tmp_path: Path,
    tiny_probe: tuple[list[dict[str, Any]], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows, keywords = tiny_probe
    monkeypatch.setattr(embedder_probe, "_interim_recall_now", lambda *_args, **_kwargs: 1.0)
    disabled = train_probe(rows, recipe=_recipe(), output_dir=tmp_path / "off", **keywords)
    enabled = train_probe(
        rows,
        recipe=_recipe(),
        output_dir=tmp_path / "on",
        collapse_detection=_retrieval_only_contract(),
        **keywords,
    )
    # A4: an enabled run that detects nothing must train exactly like a disabled one.
    assert enabled["first_loss"] == disabled["first_loss"]
    assert enabled["last_loss"] == disabled["last_loss"]
    curve = [
        json.loads(line)
        for line in (tmp_path / "on" / "training_loss_curve.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["step"] for row in curve] == list(range(1, 9))
    assert curve[0]["loss"] == disabled["first_loss"]
    assert curve[-1]["loss"] == disabled["last_loss"]
    interim = [
        json.loads(line)
        for line in (tmp_path / "on" / "training_interim_evaluation.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["step"] for row in interim] == [2, 4, 6]
    assert all(row["collapse_detected"] is False for row in interim)


def test_interim_evaluation_restores_the_rng_state(
    tmp_path: Path, tiny_probe: tuple[list[dict[str, Any]], dict[str, Any]]
) -> None:
    del tmp_path
    rows, _keywords = tiny_probe
    contract = _test_contract()
    model = _TinyEncoder("fixture/encoder", "a" * 40)
    model.train()
    torch.manual_seed(1234)
    before = torch.get_rng_state().clone()
    embedder_probe._interim_recall_now(
        cast(embedder_probe.MeanPoolEncoder, model),
        _TinyTokenizer(),
        build_interim_evaluation_set(rows, contract.interim_evaluation),
        detection=contract,
        max_length=8,
        device=torch.device("cpu"),
    )
    assert torch.equal(torch.get_rng_state(), before)
    assert model.training is True


def test_detected_collapse_aborts_the_attempt_before_evaluation(
    tmp_path: Path,
    tiny_probe: tuple[list[dict[str, Any]], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows, keywords = tiny_probe
    monkeypatch.setattr(embedder_probe, "_interim_recall_now", lambda *_args, **_kwargs: 0.0)
    output = tmp_path / "probe"
    with pytest.raises(ProbeCollapseDetected) as excinfo:
        train_probe(
            rows,
            recipe=_recipe(),
            output_dir=output,
            collapse_detection=_test_contract(),
            **keywords,
        )
    assert excinfo.value.observation["rule"] == "interim_recall_below_chance_floor"
    assert excinfo.value.observation["step"] == 4
    assert not (output / "model").exists()
    assert not (output / "train_summary.json").exists()
    # The evidence of the collapsed attempt survives.
    assert (output / "training_interim_evaluation.jsonl").is_file()
    assert (output / "training_loss_curve.jsonl").is_file()


def test_reseed_records_every_attempt_and_promotes_the_accepted_one(
    tmp_path: Path,
    tiny_probe: tuple[list[dict[str, Any]], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows, keywords = tiny_probe
    seen: list[float] = []

    def recall_by_attempt(*_args: Any, **_kwargs: Any) -> float:
        # The first attempt collapses, the reseeded one is healthy.
        value = 0.0 if len(seen) < 2 else 1.0
        seen.append(value)
        return value

    monkeypatch.setattr(embedder_probe, "_interim_recall_now", recall_by_attempt)
    output = tmp_path / "probe"
    summary, effective = train_probe_with_collapse_reseed(
        rows,
        recipe=_recipe(seed=47),
        output_dir=output,
        collapse_detection=_retrieval_only_contract(),
        **keywords,
    )
    provenance = summary["collapse_detection"]
    assert effective.seed == 1047
    assert provenance["requested_seed"] == 47
    assert provenance["effective_seed"] == 1047
    assert provenance["attempt_count"] == 2
    assert provenance["detection_count"] == 1
    assert provenance["loss_based_guardrail_permitted"] is False
    outcomes = [attempt["outcome"] for attempt in provenance["attempts"]]
    assert outcomes == ["collapsed", "completed"]
    assert provenance["attempts"][0]["seed"] == 47
    assert provenance["attempts"][0]["detected_at_step"] == 4
    assert (output / "model" / "model.safetensors").is_file()
    assert (output / "collapse_detection_journal.jsonl").is_file()
    # The collapsed attempt keeps its own journals as evidence.
    collapsed_dir = output / "collapse_attempts" / "attempt-00-seed-47"
    assert (collapsed_dir / "training_interim_evaluation.jsonl").is_file()
    assert not (collapsed_dir / "training_checkpoint.pt").exists()
    stored = json.loads((output / "train_summary.json").read_text(encoding="utf-8"))
    assert stored["recipe"]["seed"] == 1047
    assert stored["collapse_detection"]["attempt_count"] == 2


def test_exhausted_attempts_fail_the_run_without_a_measured_artifact(
    tmp_path: Path,
    tiny_probe: tuple[list[dict[str, Any]], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows, keywords = tiny_probe
    monkeypatch.setattr(embedder_probe, "_interim_recall_now", lambda *_args, **_kwargs: 0.0)
    output = tmp_path / "probe"
    with pytest.raises(ProbeCollapseUnresolved) as excinfo:
        train_probe_with_collapse_reseed(
            rows,
            recipe=_recipe(seed=50),
            output_dir=output,
            collapse_detection=_test_contract(),
            **keywords,
        )
    assert [attempt["seed"] for attempt in excinfo.value.attempts] == [50, 1050, 2050]
    assert not (output / "train_summary.json").exists()
    assert not (output / "model").exists()


def test_reseed_resumes_without_repeating_a_collapsed_attempt(
    tmp_path: Path,
    tiny_probe: tuple[list[dict[str, Any]], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows, keywords = tiny_probe
    monkeypatch.setattr(embedder_probe, "_interim_recall_now", lambda *_args, **_kwargs: 0.0)
    output = tmp_path / "probe"
    contract = _test_contract(
        rules={
            "retrieval_floor": {
                "rule_id": "interim_recall_below_chance_floor",
                "min_chance_multiple": 4.0,
            },
            "loss_direction": {"rule_id": "loss_direction_non_decreasing", "window_steps": 100},
            "consecutive_hits_required": 2,
        },
        reseed={
            "max_attempts": 1,
            "seed_stride": 1000,
            "on_exhausted": "fail_run_with_status_collapse_unresolved",
            "provenance_required": True,
        },
    )
    with pytest.raises(ProbeCollapseUnresolved):
        train_probe_with_collapse_reseed(
            rows,
            recipe=_recipe(seed=47),
            output_dir=output,
            collapse_detection=contract,
            **keywords,
        )
    monkeypatch.setattr(embedder_probe, "_interim_recall_now", lambda *_args, **_kwargs: 1.0)
    summary, effective = train_probe_with_collapse_reseed(
        rows,
        recipe=_recipe(seed=47),
        output_dir=output,
        collapse_detection=_retrieval_only_contract(),
        **keywords,
    )
    # Attempt 0 stays journalled as collapsed and is not retried.
    assert effective.seed == 1047
    assert summary["collapse_detection"]["attempt_count"] == 2
    assert summary["collapse_detection"]["detection_count"] == 1


def test_reseeded_recipe_is_recovered_from_a_promoted_summary(tmp_path: Path) -> None:
    output = tmp_path / "probe"
    output.mkdir()
    recipe = _recipe(seed=47)
    (output / "train_summary.json").write_text(
        json.dumps({"recipe": asdict(recipe) | {"seed": 1047}}), encoding="utf-8"
    )
    assert embedder_probe._reseeded_recipe(output, recipe).seed == 1047
    assert embedder_probe._reseeded_recipe(tmp_path / "missing", recipe).seed == 47
