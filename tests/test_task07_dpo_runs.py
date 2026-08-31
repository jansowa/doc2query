"""Testy pętli trzech ramion Task 07 na modelu-zabawce (CPU, bez sieci).

Pokrywają to, na czym stoi porównywalność ramion i uczciwość manifestu:
budżet (kroki i tokeny), zgodność kohorty z planem, różnicę strat między
ramionami, wznowienie z checkpointu wraz ze stanem AdamW oraz odmowy — inny plan,
brak logprobów referencji, model bez parametrów trenowalnych.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from doc2query.training.dpo import (
    DPOArm,
    DPOPlanManifest,
    ValidatedDPODataset,
    canonical_fingerprint,
    ordered_ids_fingerprint,
)
from doc2query.training.dpo_runs import (
    CHECKPOINT_DIR,
    MANIFEST_NAME,
    dev_metrics,
    train_arm,
)


class ToyTokenizer:
    """Tokenizator słowo-hash: bez sieci, bez plików, zgodny z wywołaniem HF."""

    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        del add_special_tokens
        return {"input_ids": [2 + (sum(token.encode("utf-8")) % 29) for token in text.split()]}

    def save_pretrained(self, path: str | Path) -> None:
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "toy_tokenizer.json").write_text("{}\n", encoding="utf-8")


class ToyModel(torch.nn.Module):
    """Mały model przyczynowy z `save_pretrained`, żeby nie ładować niczego z dysku."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(11)
        self.inner = LlamaForCausalLM(  # type: ignore[no-untyped-call]
            LlamaConfig(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=4,
                max_position_embeddings=256,
            )
        )

    def forward(self, input_ids: torch.Tensor) -> Any:
        return self.inner(input_ids=input_ids)

    def save_pretrained(self, path: str | Path) -> None:
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "adapter_model.json").write_text("{}\n", encoding="utf-8")


def _provenance() -> dict[str, str]:
    return {
        "dataset_id": "task06-judge-selected-pairs-v3",
        "dataset_fingerprint": "a" * 64,
        "selection_policy_id": "task06-judge-selected-pair-policy-v3",
        "selection_policy_fingerprint": "b" * 64,
    }


def _dataset(train: int = 8, dev: int = 3) -> ValidatedDPODataset:
    def preference(index: int, split: str) -> dict[str, Any]:
        return {
            "preference_id": f"pref-{index:03d}",
            "prompt": f"Pasaż numer {index}. Wygeneruj jedno polskie zapytanie.",
            "chosen": f"jakie objawy wywołuje wirus numer {index}",
            "rejected": f"ile kosztuje bilet numer {index}",
            "score_margin": 1.0,
            "chosen_candidate_id": f"c-{index}",
            "rejected_candidate_id": f"r-{index}",
            "passage_id": f"doc-{index}",
            "passage_cluster_id": f"cluster-{index}",
            "split": split,
            "provenance": _provenance(),
        }

    def control(index: int, split: str, weight: float | None) -> dict[str, Any]:
        row: dict[str, Any] = {
            "preference_id": f"pref-{index:03d}",
            "prompt": f"Pasaż numer {index}. Wygeneruj jedno polskie zapytanie.",
            "completion": f"jakie objawy wywołuje wirus numer {index}",
            "candidate_id": f"c-{index}",
            "passage_id": f"doc-{index}",
            "passage_cluster_id": f"cluster-{index}",
            "split": split,
            "provenance": _provenance(),
        }
        if weight is not None:
            row["sample_weight"] = weight
            row["weight_policy_id"] = "toy-weights"
            row["weight_policy_fingerprint"] = "7" * 64
        return row

    train_ids = list(range(train))
    dev_ids = list(range(train, train + dev))
    return ValidatedDPODataset.model_validate(
        {
            "preference_train": [preference(index, "train") for index in train_ids],
            "preference_dev": [preference(index, "dev") for index in dev_ids],
            "continued_sft_train": [control(index, "train", None) for index in train_ids],
            "continued_sft_dev": [control(index, "dev", None) for index in dev_ids],
            "weighted_sft_train": [
                control(index, "train", 0.5 + index / train) for index in train_ids
            ],
            "weighted_sft_dev": [control(index, "dev", 1.0) for index in dev_ids],
            "provenance": _provenance(),
            "input_hashes": {"preference_train": "c" * 64},
        }
    )


def _plan(dataset: ValidatedDPODataset, *, steps: int = 2) -> DPOPlanManifest:
    train_ids = [row.preference_id for row in dataset.preference_train]
    cohort = ordered_ids_fingerprint(train_ids)
    stack = {
        "base_model": {"model_id": "toy", "revision": "r1", "artifact_fingerprint": "d" * 64},
        "sft_adapter": {
            "adapter_id": "toy-sft",
            "adapter_revision": "runs/toy",
            "adapter_fingerprint": "e" * 64,
            "base_model_fingerprint": "d" * 64,
        },
        "tokenizer": {"tokenizer_id": "toy", "revision": "r1", "tokenizer_fingerprint": "f" * 64},
    }

    def budget(arm: str, **extra: Any) -> dict[str, Any]:
        return {
            "arm": arm,
            "cohort_fingerprint": cohort,
            "seeds": [42],
            "target_token_budget": 10_000,
            "target_optimizer_steps": steps,
            "train_example_count": len(train_ids),
            "prompt_chosen_tokens_per_cohort": 500,
            **extra,
        }

    payload: dict[str, Any] = {
        "contract": "task07-dpo-plan-v1",
        "status": "planned_not_trained",
        "plan_id": "toy-plan",
        "dataset_fingerprint": "a" * 64,
        "cohort_fingerprint": cohort,
        "token_length_artifact_fingerprint": "9" * 64,
        "input_hashes": {"config": "8" * 64},
        "start_model": stack,
        "reference_model": stack,
        "beta": 0.1,
        "loss_type": "sigmoid",
        "learning_rate": 1e-3,
        "max_length": 64,
        "max_prompt_length": 48,
        "arms": {
            "dpo": budget("dpo", dpo_pair_tokens_per_cohort=900),
            "continued_sft": budget("continued_sft"),
            "score_weighted_continued_sft": budget(
                "score_weighted_continued_sft",
                weight_policy_id="toy-weights",
                weight_policy_fingerprint="7" * 64,
            ),
        },
        "model_loading_performed": False,
        "training_started": False,
        "reference_logprobs_computed": False,
        "final_tests_used": [],
    }
    payload["plan_fingerprint"] = canonical_fingerprint(payload)
    return DPOPlanManifest.model_validate(payload)


def _reference(dataset: ValidatedDPODataset) -> dict[str, tuple[float, float]]:
    return {row.preference_id: (-8.0, -9.0) for row in dataset.preference_train}


def _run(
    arm: DPOArm,
    tmp_path: Path,
    *,
    steps: int = 2,
    dataset: ValidatedDPODataset | None = None,
    model: ToyModel | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    data = dataset if dataset is not None else _dataset()
    return train_arm(
        arm=arm,
        dataset=data,
        plan=_plan(data, steps=steps),
        reference=_reference(data),
        model=model if model is not None else ToyModel(),
        tokenizer=ToyTokenizer(),
        output_dir=tmp_path / arm.value,
        batch_size=1,
        gradient_accumulation_steps=2,
        checkpoint_every=0,
        progress_every=0,
        **kwargs,
    )


@pytest.mark.parametrize(
    "arm", [DPOArm.DPO, DPOArm.CONTINUED_SFT, DPOArm.SCORE_WEIGHTED_CONTINUED_SFT]
)
def test_every_arm_runs_the_planned_budget_and_writes_a_manifest(
    arm: DPOArm, tmp_path: Path
) -> None:
    result = _run(arm, tmp_path, steps=3)
    assert result["contract"] == "task07-dpo-run-v1"
    assert result["arm"] == arm.value
    assert result["completed_optimizer_steps"] == 3
    assert result["target_optimizer_steps"] == 3
    assert result["tokens_consumed"] > 0
    assert result["training_started"] is True
    assert result["task07_training_authorized"] is True
    assert result["final_tests_used"] == []
    manifest = json.loads((tmp_path / arm.value / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["artifact_fingerprint"] == result["artifact_fingerprint"]
    assert (tmp_path / arm.value / "adapter" / "adapter_model.json").is_file()
    history = (tmp_path / arm.value / "history.jsonl").read_text(encoding="utf-8").strip()
    assert len(history.split("\n")) == 3


def test_dpo_consumes_more_tokens_than_the_controls(tmp_path: Path) -> None:
    """Pary DPO to dwie sekwencje na przykład — budżet tokenów musi to pokazywać."""
    dpo = _run(DPOArm.DPO, tmp_path, steps=2)
    control = _run(DPOArm.CONTINUED_SFT, tmp_path, steps=2)
    assert dpo["tokens_consumed"] > control["tokens_consumed"]
    assert dpo["completed_optimizer_steps"] == control["completed_optimizer_steps"]


def test_weighted_arm_differs_from_plain_control(tmp_path: Path) -> None:
    """Bez tej różnicy trzecie ramię byłoby kopią continued SFT, czyli puste."""
    plain = _run(DPOArm.CONTINUED_SFT, tmp_path, steps=2)
    weighted = _run(DPOArm.SCORE_WEIGHTED_CONTINUED_SFT, tmp_path, steps=2)
    plain_history = [
        json.loads(line)
        for line in (tmp_path / "continued_sft" / "history.jsonl")
        .read_text(encoding="utf-8")
        .strip()
        .split("\n")
    ]
    weighted_history = [
        json.loads(line)
        for line in (tmp_path / "score_weighted_continued_sft" / "history.jsonl")
        .read_text(encoding="utf-8")
        .strip()
        .split("\n")
    ]
    assert plain_history[0]["mean_weight"] == 1.0
    assert weighted_history[0]["mean_weight"] != 1.0
    assert plain["dev"]["end"] != weighted["dev"]["end"]


def test_resume_continues_from_checkpoint_with_optimizer_state(tmp_path: Path) -> None:
    data = _dataset()
    plan = _plan(data, steps=4)
    output = tmp_path / "resumable"
    first = train_arm(
        arm=DPOArm.DPO,
        dataset=data,
        plan=plan,
        reference=_reference(data),
        model=ToyModel(),
        tokenizer=ToyTokenizer(),
        output_dir=output,
        gradient_accumulation_steps=2,
        max_steps=2,
        checkpoint_every=1,
        progress_every=0,
    )
    assert first["completed_optimizer_steps"] == 2
    state = json.loads((output / CHECKPOINT_DIR / "state.json").read_text(encoding="utf-8"))
    assert state["optimizer_step"] == 1
    assert state["arm"] == "dpo"
    assert (output / CHECKPOINT_DIR / "optimizer.pt").is_file()

    second = train_arm(
        arm=DPOArm.DPO,
        dataset=data,
        plan=plan,
        reference=_reference(data),
        model=ToyModel(),
        tokenizer=ToyTokenizer(),
        output_dir=output,
        gradient_accumulation_steps=2,
        checkpoint_every=1,
        progress_every=0,
    )
    assert second["resumed_from_step"] == 1
    assert second["completed_optimizer_steps"] == 4
    # dev punktu startowego przenosi się z checkpointu: inaczej „start" znaczyłby
    # co innego przed i po wznowieniu.
    assert second["dev"]["start"] == first["dev"]["start"]


def test_refuses_foreign_plan_and_missing_reference(tmp_path: Path) -> None:
    data = _dataset()
    other = _dataset(train=6)
    with pytest.raises(ValueError, match="kohorta"):
        train_arm(
            arm=DPOArm.DPO,
            dataset=data,
            plan=_plan(other),
            reference=_reference(data),
            model=ToyModel(),
            tokenizer=ToyTokenizer(),
            output_dir=tmp_path / "foreign",
            progress_every=0,
        )
    with pytest.raises(ValueError, match="referencji"):
        train_arm(
            arm=DPOArm.DPO,
            dataset=data,
            plan=_plan(data),
            reference={},
            model=ToyModel(),
            tokenizer=ToyTokenizer(),
            output_dir=tmp_path / "noref",
            gradient_accumulation_steps=1,
            progress_every=0,
        )


def test_refuses_model_without_trainable_parameters(tmp_path: Path) -> None:
    data = _dataset()
    model = ToyModel()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with pytest.raises(ValueError, match="trenowalnych"):
        train_arm(
            arm=DPOArm.CONTINUED_SFT,
            dataset=data,
            plan=_plan(data),
            reference={},
            model=model,
            tokenizer=ToyTokenizer(),
            output_dir=tmp_path / "frozen",
            progress_every=0,
        )


def test_dev_metrics_report_reference_free_and_reference_based_numbers() -> None:
    data = _dataset()
    model = ToyModel()
    tokenizer = ToyTokenizer()
    from doc2query.training.dpo_runs import _weighted_records

    rows = _weighted_records(data, DPOArm.DPO, "dev")
    plain = dev_metrics(model=model, tokenizer=tokenizer, records=rows, max_length=64, beta=0.1)
    assert "policy_margin_accuracy" in plain
    assert "implicit_reward_accuracy" not in plain
    reference = {row.preference_id: (-5.0, -5.0) for row in rows}
    with_reference = dev_metrics(
        model=model,
        tokenizer=tokenizer,
        records=rows,
        max_length=64,
        beta=0.1,
        reference=reference,
    )
    assert with_reference["implicit_reward_accuracy"] == plain["policy_margin_accuracy"]


def test_nll_regularizer_raises_loss_and_is_recorded(tmp_path: Path) -> None:
    """RPO: człon NLL ma podnosić stratę i trafiać do manifestu, nie zmieniając planu."""
    data = _dataset()
    plan = _plan(data, steps=2)
    plain = train_arm(
        arm=DPOArm.DPO,
        dataset=data,
        plan=plan,
        reference=_reference(data),
        model=ToyModel(),
        tokenizer=ToyTokenizer(),
        output_dir=tmp_path / "plain",
        gradient_accumulation_steps=2,
        checkpoint_every=0,
        progress_every=0,
    )
    regularized = train_arm(
        arm=DPOArm.DPO,
        dataset=data,
        plan=plan,
        reference=_reference(data),
        model=ToyModel(),
        tokenizer=ToyTokenizer(),
        output_dir=tmp_path / "rpo",
        gradient_accumulation_steps=2,
        checkpoint_every=0,
        progress_every=0,
        nll_coefficient=1.0,
    )
    assert regularized["nll_coefficient"] == 1.0
    assert regularized["loss_type"] == "sigmoid_plus_nll"
    assert plain["loss_type"] == "sigmoid"
    history = [
        json.loads(line)
        for line in (tmp_path / "rpo" / "history.jsonl").read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]
    assert all("chosen_nll_per_token" in row for row in history)
    # Ten sam model startowy i seed: strata z regularyzatorem musi być większa.
    plain_history = [
        json.loads(line)
        for line in (tmp_path / "plain" / "history.jsonl").read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]
    assert history[0]["loss"] > plain_history[0]["loss"]


def test_nll_regularizer_is_dpo_only(tmp_path: Path) -> None:
    data = _dataset()
    with pytest.raises(ValueError, match="wyłącznie ramienia DPO"):
        train_arm(
            arm=DPOArm.CONTINUED_SFT,
            dataset=data,
            plan=_plan(data),
            reference={},
            model=ToyModel(),
            tokenizer=ToyTokenizer(),
            output_dir=tmp_path / "bad",
            progress_every=0,
            nll_coefficient=0.5,
        )
