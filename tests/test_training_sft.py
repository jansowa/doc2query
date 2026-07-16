from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as functional
from peft import PeftModel
from transformers import LlamaConfig, LlamaForCausalLM, TrainingArguments

from doc2query.models.lora import attach_lora, discover_linear_target_modules
from doc2query.schemas import LoraConfig
from doc2query.training.data import (
    IGNORE_INDEX,
    BalancedBatchSampler,
    CompletionOnlyCollator,
    PromptCompletionDataset,
    add_balance_buckets,
    compute_example_weights,
    prepare_datasets,
)
from doc2query.training.sft import (
    CompletionOnlySFTTrainer,
    _resolve_resume_checkpoint,
    checkpoint_is_complete,
    find_latest_complete_checkpoint,
)
from doc2query.training.weighted_sft import weighted_completion_loss
from doc2query.utils.records import JsonlWriter


class ToyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [2 + (sum(token.encode("utf-8")) % 29) for token in text.split()]

    def save_pretrained(self, path: str | Path) -> None:
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "toy_tokenizer.json").write_text("{}\n", encoding="utf-8")


def _examples(count: int = 8) -> list[dict[str, Any]]:
    result = []
    for index in range(count):
        result.append(
            {
                "prompt": f"instrukcja pasaż numer {index} Zapytanie:",
                "completion": f"pytanie {index}",
                "query_style": "rare" if index == 0 else "common",
                "focus_bucket": ["beginning", "middle", "end", None][index % 4],
                "content_lemma_overlap": index / count,
                "passage_word_length": 10 + index,
            }
        )
    return add_balance_buckets(result)


def _tiny_llama() -> LlamaForCausalLM:
    return LlamaForCausalLM(  # type: ignore[no-untyped-call]
        LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
        )
    )


def _lora_settings() -> LoraConfig:
    return LoraConfig(
        r=4,
        alpha=8,
        dropout=0.0,
        minimum_target_modules=4,
        expected_layer_patterns=["attn", "mlp"],
    )


def test_completion_only_masking_excludes_prompt_and_padding() -> None:
    collator = CompletionOnlyCollator(
        ToyTokenizer(), max_length=16, max_completion_tokens=5, min_prompt_tokens=3
    )
    batch = collator(
        [
            {"prompt": "to jest długi prompt", "completion": "krótkie pytanie"},
            {"prompt": "prompt", "completion": "inne pytanie"},
        ]
    )
    for labels, attention in zip(batch["labels"], batch["attention_mask"], strict=True):
        active = labels.ne(IGNORE_INDEX)
        assert active.any()
        assert torch.all(attention[active] == 1)
        first_completion = int(active.nonzero()[0])
        assert torch.all(labels[:first_completion] == IGNORE_INDEX)
        assert torch.all(labels[attention == 0] == IGNORE_INDEX)


def test_truncation_preserves_completion_and_eos() -> None:
    tokenizer = ToyTokenizer()
    collator = CompletionOnlyCollator(
        tokenizer, max_length=8, max_completion_tokens=4, min_prompt_tokens=3
    )
    batch = collator(
        [{"prompt": " ".join(["bardzo"] * 100), "completion": "ważne krótkie pytanie"}]
    )
    completion_labels = batch["labels"][0][batch["labels"][0] != IGNORE_INDEX]
    assert completion_labels.tolist() == [*tokenizer.encode("ważne krótkie pytanie"), 1]
    assert batch["attention_mask"].sum() <= 8


def test_fixed_padding_provides_equal_token_budget() -> None:
    collator = CompletionOnlyCollator(
        ToyTokenizer(),
        max_length=12,
        max_completion_tokens=4,
        min_prompt_tokens=3,
        pad_to_max_length=True,
    )
    batch = collator([{"prompt": "krótki prompt", "completion": "pytanie"}])
    assert batch["input_ids"].shape == (1, 12)
    assert torch.all(batch["labels"][batch["attention_mask"] == 0] == IGNORE_INDEX)


def test_dataset_caps_are_deterministic_by_pair_id(tmp_path: Path) -> None:
    path = tmp_path / "pairs.jsonl"
    with JsonlWriter(path) as writer:
        for index in reversed(range(10)):
            writer.write(
                {
                    "pair_id": f"q{index}::d{index}",
                    "passage": f"Pasaż numer {index}.",
                    "query": f"pytanie {index}",
                    "query_style": "keyword_query",
                    "focus_bucket": "middle",
                    "content_lemma_overlap": index / 10,
                    "passage_word_length": index + 2,
                    "split": "train",
                }
            )
    kwargs = {
        "eval_path": None,
        "train_split": "train",
        "eval_split": "dev",
        "baseline": "b1",
        "strategy": "ordinary",
        "weight_min": 0.25,
        "weight_max": 4.0,
        "seed": 42,
        "batch_size": 1,
        "max_train_examples": 4,
    }
    first = prepare_datasets(path, **kwargs)  # type: ignore[arg-type]
    second = prepare_datasets(path, **kwargs)  # type: ignore[arg-type]
    assert [item["pair_id"] for item in first.train] == [item["pair_id"] for item in second.train]
    assert len(first.train) == 4
    assert first.fingerprint == second.fingerprint


def test_weighted_loss_matches_manual_per_example_calculation() -> None:
    torch.manual_seed(7)
    logits = torch.randn(2, 5, 11)
    labels = torch.tensor(
        [[IGNORE_INDEX, IGNORE_INDEX, 2, 3, 4], [IGNORE_INDEX, 5, 6, IGNORE_INDEX, IGNORE_INDEX]]
    )
    weights = torch.tensor([0.5, 1.5])
    actual = weighted_completion_loss(logits, labels, weights)
    shifted = labels[:, 1:]
    token_losses = functional.cross_entropy(
        logits[:, :-1].transpose(1, 2), shifted, ignore_index=IGNORE_INDEX, reduction="none"
    )
    mask = shifted != IGNORE_INDEX
    per_example = (token_losses * mask).sum(1) / mask.sum(1)
    expected = (per_example * weights).sum() / weights.sum()
    assert torch.allclose(actual, expected)


def test_weights_are_bounded_normalized_and_logged() -> None:
    examples = _examples()
    weights, report = compute_example_weights(examples, minimum=0.5, maximum=2.0)
    assert min(weights) >= 0.5
    assert max(weights) <= 2.0
    assert sum(weights) / len(weights) == pytest.approx(1.0, abs=1e-10)
    assert set(report["bucket_counts"]) == {
        "query_style",
        "focus_bucket",
        "overlap_quantile",
        "passage_length_bucket",
    }


def test_balanced_batch_sampler_is_deterministic_and_oversamples_rare_style() -> None:
    examples = _examples(40)
    sampler = BalancedBatchSampler(examples, batch_size=4, seed=17)
    first = [index for batch in sampler for index in batch]
    second = [index for batch in sampler for index in batch]
    assert first == second
    assert first.count(0) > 1
    sampler.set_epoch(1)
    assert [index for batch in sampler for index in batch] != first


def test_lora_targets_and_trainable_parameter_count_are_plausible() -> None:
    model = _tiny_llama()
    targets = discover_linear_target_modules(model)
    assert {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"} <= set(
        targets
    )
    adapted, selected, stats = attach_lora(model, _lora_settings())
    assert selected
    assert 0 < stats.trainable < stats.total
    assert stats.ratio < 0.25
    assert any(
        "lora" in name for name, parameter in adapted.named_parameters() if parameter.requires_grad
    )


def test_empty_or_incomplete_lora_coverage_stops_run() -> None:
    with pytest.raises(RuntimeError, match="at least"):
        attach_lora(
            _tiny_llama(),
            LoraConfig(
                r=2,
                alpha=4,
                target_modules=["q_proj"],
                minimum_target_modules=2,
                expected_layer_patterns=["attn"],
            ),
        )


def test_save_load_adapter_preserves_logits(tmp_path: Path) -> None:
    torch.manual_seed(11)
    base = _tiny_llama()
    base_state = {name: value.clone() for name, value in base.state_dict().items()}
    adapted, _, _ = attach_lora(base, _lora_settings())
    adapted.eval()
    inputs = torch.tensor([[2, 3, 4, 5]])
    with torch.inference_mode():
        expected = adapted(input_ids=inputs).logits
    adapter_path = tmp_path / "adapter"
    adapted.save_pretrained(adapter_path)
    reloaded_base = _tiny_llama()
    reloaded_base.load_state_dict(base_state)
    reloaded = PeftModel.from_pretrained(reloaded_base, adapter_path)
    reloaded.eval()
    with torch.inference_mode():
        actual = reloaded(input_ids=inputs).logits
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_resume_restores_scheduler_progress(tmp_path: Path) -> None:
    torch.manual_seed(3)
    model, _, _ = attach_lora(_tiny_llama(), _lora_settings())
    collator = CompletionOnlyCollator(
        ToyTokenizer(), max_length=16, max_completion_tokens=4, min_prompt_tokens=3
    )
    dataset = PromptCompletionDataset(_examples(4))
    first_args = TrainingArguments(
        output_dir=str(tmp_path / "run"),
        max_steps=2,
        per_device_train_batch_size=1,
        save_strategy="steps",
        save_steps=2,
        logging_steps=1,
        report_to=[],
        use_cpu=True,
        remove_unused_columns=False,
    )
    trainer = CompletionOnlySFTTrainer(
        model=model, args=first_args, train_dataset=dataset, data_collator=collator
    )
    trainer.train()
    checkpoint = tmp_path / "run" / "checkpoint-2"
    assert (checkpoint / "scheduler.pt").is_file()

    torch.manual_seed(3)
    resumed_model, _, _ = attach_lora(_tiny_llama(), _lora_settings())
    resumed_args = TrainingArguments(
        output_dir=str(tmp_path / "run"),
        max_steps=3,
        per_device_train_batch_size=1,
        save_strategy="no",
        logging_steps=1,
        report_to=[],
        use_cpu=True,
        remove_unused_columns=False,
    )
    resumed = CompletionOnlySFTTrainer(
        model=resumed_model,
        args=resumed_args,
        train_dataset=dataset,
        data_collator=collator,
    )
    resumed.train(resume_from_checkpoint=str(checkpoint))
    assert resumed.state.global_step == 3
    assert resumed.lr_scheduler is not None
    assert resumed.lr_scheduler.last_epoch == 3


def _write_complete_checkpoint(path: Path) -> None:
    path.mkdir(parents=True)
    for name in (
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "adapter_model.safetensors",
    ):
        (path / name).write_bytes(b"test")


def test_resume_if_available_starts_fresh_then_selects_latest_complete_checkpoint(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    identity = {"schema_version": 1, "signature": "same"}
    assert _resolve_resume_checkpoint(run_dir, identity=identity, resume_if_available=True) is None
    assert (run_dir / "resume_identity.json").is_file()
    _write_complete_checkpoint(run_dir / "checkpoint-2")
    _write_complete_checkpoint(run_dir / "checkpoint-10")
    (run_dir / "checkpoint-20").mkdir()
    assert checkpoint_is_complete(run_dir / "checkpoint-10")
    assert not checkpoint_is_complete(run_dir / "checkpoint-20")
    assert find_latest_complete_checkpoint(run_dir) == run_dir / "checkpoint-10"
    assert (
        _resolve_resume_checkpoint(run_dir, identity=identity, resume_if_available=True)
        == run_dir / "checkpoint-10"
    )


def test_resume_if_available_rejects_changed_run_identity(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _resolve_resume_checkpoint(
        run_dir,
        identity={"schema_version": 1, "signature": "original"},
        resume_if_available=True,
    )
    with pytest.raises(RuntimeError, match="identity mismatch"):
        _resolve_resume_checkpoint(
            run_dir,
            identity={"schema_version": 1, "signature": "changed"},
            resume_if_available=True,
        )


def test_resume_without_flag_rejects_nonempty_run_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "foreign.txt").write_text("do not overwrite\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="--resume-if-available"):
        _resolve_resume_checkpoint(
            run_dir,
            identity={"schema_version": 1, "signature": "test"},
            resume_if_available=False,
        )
