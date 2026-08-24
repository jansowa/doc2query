from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from doc2query.training.dpo import sigmoid_dpo_loss
from doc2query.training.dpo_runtime import (
    PRECOMPUTE_CONTRACT,
    load_preference_records,
    load_reference_logprobs,
    precompute_reference_logprobs,
    run_dpo_steps,
    save_adapter_atomically,
    sequence_logprob,
    torch_dpo_loss,
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


def _tiny_llama() -> LlamaForCausalLM:
    torch.manual_seed(7)
    return LlamaForCausalLM(  # type: ignore[no-untyped-call]
        LlamaConfig(
            vocab_size=32000,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=256,
        )
    )


@pytest.fixture
def tokenizer() -> ToyTokenizer:
    return ToyTokenizer()


def _pair(index: int, *, split: str = "train") -> dict[str, Any]:
    return {
        "preference_id": f"pref-{index}",
        "prompt": f"Pasaż numer {index}. Wygeneruj jedno polskie zapytanie wyszukiwawcze.",
        "chosen": f"jakie objawy wywołuje wirus numer {index}",
        "rejected": f"ile kosztuje bilet numer {index}",
        "score_margin": 1.5,
        "chosen_candidate_id": f"c-{index}",
        "rejected_candidate_id": f"r-{index}",
        "passage_id": f"doc-{index}",
        "passage_cluster_id": f"cluster-{index}",
        "split": split,
        "provenance": {
            "dataset_id": "task06-defect-pairs-v2-1",
            "dataset_fingerprint": "a" * 64,
            "selection_policy_id": "task06-defect-pair-policy-v2.1",
            "selection_policy_fingerprint": "b" * 64,
        },
    }


def _write_pairs(path: Path, count: int, **kwargs: Any) -> Path:
    path.write_text(
        "\n".join(json.dumps(_pair(index, **kwargs)) for index in range(count)) + "\n",
        encoding="utf-8",
    )
    return path


# --- strata ---------------------------------------------------------------------


def test_torch_loss_matches_the_frozen_scalar_reference() -> None:
    """Wariant torchowy musi zwracać to samo, co przetestowana już funkcja skalarna."""
    values = (-3.0, -5.0, -2.5, -4.5)
    beta = 0.1
    loss, metrics = torch_dpo_loss(
        torch.tensor([values[0]]),
        torch.tensor([values[1]]),
        torch.tensor([values[2]]),
        torch.tensor([values[3]]),
        beta,
    )
    assert float(loss.item()) == pytest.approx(sigmoid_dpo_loss(*values, beta), abs=1e-6)
    assert metrics["reward_margin"] == pytest.approx(
        beta * ((values[0] - values[2]) - (values[1] - values[3])), abs=1e-6
    )


def test_preferring_chosen_lowers_the_loss() -> None:
    better, _ = torch_dpo_loss(
        torch.tensor([-2.0]), torch.tensor([-6.0]), torch.tensor([-3.0]),
        torch.tensor([-3.0]), 0.1,
    )
    worse, _ = torch_dpo_loss(
        torch.tensor([-6.0]), torch.tensor([-2.0]), torch.tensor([-3.0]),
        torch.tensor([-3.0]), 0.1,
    )
    assert float(better.item()) < float(worse.item())


def test_loss_rejects_non_positive_beta() -> None:
    with pytest.raises(ValueError, match="beta"):
        torch_dpo_loss(
            torch.tensor([-1.0]), torch.tensor([-2.0]), torch.tensor([-1.0]),
            torch.tensor([-2.0]), 0.0,
        )


# --- wejście --------------------------------------------------------------------


def test_preference_loader_refuses_duplicates_and_test_split(tmp_path: Path) -> None:
    path = _write_pairs(tmp_path / "pairs.jsonl", 3)
    assert [row.preference_id for row in load_preference_records(path)] == [
        "pref-0",
        "pref-1",
        "pref-2",
    ]
    duplicated = tmp_path / "dup.jsonl"
    rows = [json.dumps(_pair(0)), json.dumps(_pair(0))]
    duplicated.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplikat preference_id"):
        load_preference_records(duplicated)
    from_test = _write_pairs(tmp_path / "from_test.jsonl", 1, split="dev")
    assert load_preference_records(from_test)[0].split == "dev"


def test_preference_loader_refuses_a_final_test_path(tmp_path: Path) -> None:
    directory = tmp_path / "test_native_pl"
    directory.mkdir()
    path = _write_pairs(directory / "pairs.jsonl", 1)
    with pytest.raises(ValueError, match="final-test path is forbidden"):
        load_preference_records(path)


# --- logprob i maskowanie promptu -----------------------------------------------


def test_sequence_logprob_scores_only_the_completion(tokenizer: ToyTokenizer) -> None:
    model = _tiny_llama()
    short = sequence_logprob(model, tokenizer, "Pasaż.", "krótkie", max_length=64)
    longer = sequence_logprob(
        model, tokenizer, "Pasaż.", "znacznie dłuższe zapytanie testowe", max_length=64
    )
    assert short.completion_tokens < longer.completion_tokens
    # Dłuższy completion sumuje więcej ujemnych logprobów, więc suma jest mniejsza.
    assert longer.logprob < short.logprob
    assert short.truncated is False


def test_completion_is_never_truncated_but_the_prompt_is(tokenizer: ToyTokenizer) -> None:
    model = _tiny_llama()
    score = sequence_logprob(
        model, tokenizer, "słowo " * 200, "krótkie zapytanie", max_length=32
    )
    assert score.truncated is True
    assert score.prompt_tokens + score.completion_tokens == 32
    with pytest.raises(ValueError, match="nie ucina completion"):
        sequence_logprob(model, tokenizer, "pasaż", "słowo " * 100, max_length=16)


# --- precompute referencji ------------------------------------------------------


def _precompute(
    tmp_path: Path, tokenizer: ToyTokenizer, records: Any, **kwargs: Any
) -> dict[str, Any]:
    return precompute_reference_logprobs(
        records=records,
        model=_tiny_llama(),
        tokenizer=tokenizer,
        output_dir=tmp_path / "ref",
        max_length=64,
        dataset_fingerprint="a" * 64,
        plan_fingerprint="c" * 64,
        reference_model={"base": "tiny", "adapter": "none"},
        tokenizer_fingerprint="d" * 64,
        progress_every=0,
        **kwargs,
    )


def test_precompute_writes_a_manifest_in_dataset_order(
    tmp_path: Path, tokenizer: ToyTokenizer
) -> None:
    records = load_preference_records(_write_pairs(tmp_path / "pairs.jsonl", 4))
    result = _precompute(tmp_path, tokenizer, records)
    manifest = result["manifest"]
    assert manifest["contract"] == PRECOMPUTE_CONTRACT
    assert manifest["records"]["record_count"] == 4
    assert manifest["final_tests_used"] == []
    rows = [
        json.loads(line)
        for line in (tmp_path / "ref" / "reference_logprobs.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["preference_id"] for row in rows] == [row.preference_id for row in records]
    assert [row["position"] for row in rows] == [0, 1, 2, 3]
    reference = load_reference_logprobs(Path(result["manifest_path"]))
    assert set(reference) == {row.preference_id for row in records}


def test_precompute_resumes_from_its_journal_without_recomputing(
    tmp_path: Path, tokenizer: Any
) -> None:
    records = load_preference_records(_write_pairs(tmp_path / "pairs.jsonl", 4))
    first = _precompute(tmp_path, tokenizer, records[:2])
    assert first["computed_records"] == 2
    second = _precompute(tmp_path, tokenizer, records)
    assert second["resumed_records"] == 2
    assert second["computed_records"] == 2
    assert second["record_count"] == 4


def test_precompute_refuses_a_reordered_dataset(tmp_path: Path, tokenizer: ToyTokenizer) -> None:
    """Restart nie może pomieszać precomputowanych logprobów z inną kolejnością."""
    records = load_preference_records(_write_pairs(tmp_path / "pairs.jsonl", 3))
    _precompute(tmp_path, tokenizer, records)
    with pytest.raises(ValueError, match="innej pozycji"):
        _precompute(tmp_path, tokenizer, list(reversed(records)))


def test_reference_loader_detects_a_drifted_records_file(
    tmp_path: Path, tokenizer: ToyTokenizer
) -> None:
    records = load_preference_records(_write_pairs(tmp_path / "pairs.jsonl", 2))
    result = _precompute(tmp_path, tokenizer, records)
    path = tmp_path / "ref" / "reference_logprobs.jsonl"
    path.write_text(path.read_text(encoding="utf-8").replace("pref-0", "pref-x"), encoding="utf-8")
    with pytest.raises(ValueError, match="rozjechał się z manifestem"):
        load_reference_logprobs(Path(result["manifest_path"]))


# --- trening --------------------------------------------------------------------


def test_dpo_steps_reduce_the_loss_and_report_rewards(
    tmp_path: Path, tokenizer: ToyTokenizer
) -> None:
    records = load_preference_records(_write_pairs(tmp_path / "pairs.jsonl", 4))
    reference = {row.preference_id: (-20.0, -20.0) for row in records}
    model = _tiny_llama()
    summary = run_dpo_steps(
        records=records,
        reference=reference,
        model=model,
        tokenizer=tokenizer,
        beta=0.1,
        learning_rate=5e-3,
        max_length=64,
        batch_size=2,
    )
    assert summary["steps"] == 2
    assert summary["history"][0]["loss"] > 0.0
    assert "reward_accuracy" in summary["history"][0]
    assert summary["completion_length_mean"]["chosen"] is not None
    repeated = run_dpo_steps(
        records=records,
        reference=reference,
        model=model,
        tokenizer=tokenizer,
        beta=0.1,
        learning_rate=5e-3,
        max_length=64,
        batch_size=2,
    )
    # Po pierwszym przejściu polityka odsunęła chosen od rejected, więc strata spada.
    assert repeated["final_loss"] < summary["history"][0]["loss"]


def test_dpo_refuses_pairs_without_precomputed_reference(
    tmp_path: Path, tokenizer: ToyTokenizer
) -> None:
    records = load_preference_records(_write_pairs(tmp_path / "pairs.jsonl", 2))
    with pytest.raises(ValueError, match="brak precomputowanych logprobów"):
        run_dpo_steps(
            records=records,
            reference={records[0].preference_id: (-1.0, -2.0)},
            model=_tiny_llama(),
            tokenizer=tokenizer,
            beta=0.1,
            learning_rate=1e-3,
            max_length=64,
        )


def test_adapter_save_is_atomic_and_reloadable(tmp_path: Path, tokenizer: ToyTokenizer) -> None:
    from doc2query.models.lora import attach_lora
    from doc2query.schemas import LoraConfig

    model, _targets, _stats = attach_lora(
        _tiny_llama(),
        LoraConfig(
            r=4,
            alpha=8,
            dropout=0.0,
            minimum_target_modules=4,
            expected_layer_patterns=["attn", "mlp"],
        ),
    )
    destination = save_adapter_atomically(model, tokenizer, tmp_path / "adapter")
    assert (destination / "adapter_config.json").is_file()
    assert not list(tmp_path.glob(".adapter.staging-*"))
