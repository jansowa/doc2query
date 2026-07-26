from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from doc2query.config import load_config
from doc2query.evaluation import generator as evaluation_generator
from doc2query.evaluation.generator import generate_evaluation_queries
from doc2query.generation.batching import generate_text_batch, pad_token_sequences
from doc2query.utils.records import read_records


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 9

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        del add_special_tokens
        return [max(1, ord(character) % 64) for character in text]

    def decode(self, values: Any, *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens
        return "".join(chr(int(value)) for value in values if int(value) not in {0, 9})


class _Model(torch.nn.Module):
    def __init__(self, *, encoder_decoder: bool) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(is_encoder_decoder=encoder_decoder)
        self.last_input_ids: torch.Tensor | None = None

    def generate(self, **kwargs: Any) -> torch.Tensor:
        inputs = kwargs["input_ids"]
        self.last_input_ids = inputs.detach().clone()
        candidates = int(kwargs["num_return_sequences"])
        outputs = []
        for row in range(inputs.shape[0]):
            for candidate in range(candidates):
                token = torch.tensor([65 + row * candidates + candidate], dtype=torch.long)
                outputs.append(
                    token
                    if self.config.is_encoder_decoder
                    else torch.cat((inputs[row], token))
                )
        return torch.stack(outputs)


def test_padding_supports_both_model_families() -> None:
    left, left_mask = pad_token_sequences(
        [[1, 2], [3]],
        pad_token_id=0,
        device=torch.device("cpu"),
        padding_side="left",
    )
    right, right_mask = pad_token_sequences(
        [[1, 2], [3]],
        pad_token_id=0,
        device=torch.device("cpu"),
        padding_side="right",
    )
    assert left.tolist() == [[1, 2], [0, 3]]
    assert left_mask.tolist() == [[1, 1], [0, 1]]
    assert right.tolist() == [[1, 2], [3, 0]]
    assert right_mask.tolist() == [[1, 1], [1, 0]]


def test_generation_is_prompt_major_for_causal_and_encoder_decoder() -> None:
    mode = {"do_sample": False, "num_return_sequences": 2}
    for encoder_decoder, expected_padding in (
        (False, [[1, 2], [0, 3]]),
        (True, [[1, 2], [3, 0]]),
    ):
        model = _Model(encoder_decoder=encoder_decoder)
        result = generate_text_batch(
            model,
            _Tokenizer(),
            [[1, 2], [3]],
            mode=mode,
            max_new_tokens=4,
        )
        assert result == ["A", "B", "C", "D"]
        assert model.last_input_ids is not None
        assert model.last_input_ids.tolist() == expected_padding


def _record(identifier: str) -> dict[str, Any]:
    return {
        "example_id": identifier,
        "query": f"Pytanie {identifier}?",
        "positives": [{"doc_id": f"p-{identifier}", "text": f"Pasaż {identifier}."}],
        "hard_negatives": [
            {"doc_id": f"n-{identifier}-{index}", "text": f"Negatyw {index}."}
            for index in range(10)
        ],
        "metadata": {"split": "dev"},
    }


def test_evaluation_generation_batches_and_resumes_exact_prefix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = _Model(encoder_decoder=True)
    monkeypatch.setattr(evaluation_generator, "load_tokenizer", lambda _config: _Tokenizer())
    monkeypatch.setattr(
        evaluation_generator,
        "load_generator",
        lambda _config, **_kwargs: (model, SimpleNamespace(label="fp32")),
    )
    config = load_config(Path("configs/experiments/s07_tiny_smoke.yaml"))
    output = tmp_path / "generations.jsonl"
    modes = [
        {
            "mode": "deterministic",
            "do_sample": False,
            "num_return_sequences": 1,
            "max_new_tokens": 4,
        },
        {
            "mode": "diverse",
            "do_sample": True,
            "num_return_sequences": 2,
            "max_new_tokens": 4,
            "temperature": 0.8,
            "top_p": 0.95,
        },
    ]
    records = [_record(str(index)) for index in range(3)]
    report = generate_evaluation_queries(
        config,
        records,
        adapter_path=None,
        output_path=output,
        modes=modes,
        batch_size=2,
    )
    expected = list(read_records(output))
    assert report["generation_count"] == 9
    assert [row["evaluation_id"] for row in expected[:4]] == [
        "0::deterministic::0",
        "0::diverse::0",
        "0::diverse::1",
        "1::deterministic::0",
    ]
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_text(
        "".join(
            f"{json.dumps(row, ensure_ascii=False, sort_keys=True)}\n"
            for row in expected[:4]
        ),
        encoding="utf-8",
    )
    with partial.open("ab") as handle:
        handle.write(b'{"crash_truncated":')
    output.unlink()
    resumed = generate_evaluation_queries(
        config,
        records,
        adapter_path=None,
        output_path=output,
        modes=modes,
        batch_size=2,
    )
    assert resumed["resumed_generation_count"] == 4
    assert list(read_records(output)) == expected
