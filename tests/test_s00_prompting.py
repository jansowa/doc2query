from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from doc2query.evaluation import s00_prompting
from doc2query.evaluation.s00_prompting import (
    LEGACY_RESUME_IDENTITIES,
    _effective_batch_size,
    _generate_model_batch,
    _left_pad_batch,
    _open_journal,
    encode_prompt,
    generate_s00,
    load_contract,
)
from doc2query.utils.records import JsonlWriter


class ToyTokenizer:
    pad_token_id = 0
    eos_token_id = 9

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(char) for char in text]

    def decode(self, values: list[int], skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return "".join(chr(int(value)) for value in values)


class ToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.last_attention: torch.Tensor | None = None

    def generate(self, **kwargs: Any) -> torch.Tensor:
        inputs = kwargs["input_ids"]
        self.last_attention = kwargs["attention_mask"]
        count = int(kwargs["num_return_sequences"])
        expanded = inputs.repeat_interleave(count, dim=0)
        suffix = torch.arange(65, 65 + expanded.shape[0], dtype=torch.long).unsqueeze(1)
        return torch.cat((expanded, suffix), dim=1)


def _record(identifier: str, query: str, *, split: str = "dev") -> dict[str, Any]:
    return {
        "example_id": identifier,
        "query": query,
        "positives": [{"doc_id": f"p-{identifier}", "text": f"Pasaż {identifier}."}],
        "hard_negatives": [
            {"doc_id": f"n-{identifier}-{index}", "text": f"Negatyw {index}."}
            for index in range(10)
        ],
        "metadata": {"split": split},
    }


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    rows = [_record(str(index), f"Jak działa rzecz {index}?") for index in range(8)]
    rows += [_record(f"k{index}", f"rzecz numer {index}") for index in range(8)]
    source = tmp_path / "dev.jsonl"
    with JsonlWriter(source) as writer:
        for row in rows:
            writer.write(row)
    parent = tmp_path / "parent.json"
    parent.write_text("{}\n", encoding="utf-8")

    def fake_load(_manifest: Path, subset: str) -> list[dict[str, Any]]:
        if subset in {"dev_intrinsic_rank10", "dev_intrinsic"}:
            return rows
        ids_path = tmp_path / "out" / "cohort" / f"{subset}.ids.jsonl"
        ids = {json.loads(line)["id"] for line in ids_path.read_text().splitlines()}
        return [row for row in rows if row["example_id"] in ids]

    monkeypatch.setattr(s00_prompting, "load_frozen_records", fake_load)
    monkeypatch.setattr(s00_prompting, "_sha256_file", lambda _path: "f" * 64)
    manifest = {
        "sets": {
            "dev_intrinsic_rank10": {
                "records_sha256": "a" * 64,
                "source_path": str(source),
                "source_sha256": "b" * 64,
            }
        }
    }
    parent.write_text(json.dumps(manifest), encoding="utf-8")
    contract = yaml.safe_load(Path("configs/evaluation/s00_prompting_v1.yaml").read_text())
    contract.update(
        frozen_manifest=str(parent),
        source_subset_fingerprint="a" * 64,
        target_size=4,
        target_subset="dev_s00_5000",
        output_dir=str(tmp_path / "out"),
    )
    contract["exemplars"]["count"] = 2
    contract["exemplars"]["form_counts"] = {"full_question": 1, "keyword_query": 1}
    path = tmp_path / "contract.yaml"
    path.write_text(yaml.safe_dump(contract), encoding="utf-8")
    monkeypatch.setattr(s00_prompting, "load_contract", lambda _path: contract)
    return path


def test_contract_rejects_final_test_reference(tmp_path: Path) -> None:
    contract = yaml.safe_load(Path("configs/evaluation/s00_prompting_v1.yaml").read_text())
    contract["output_dir"] = "runs/final_test"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(contract), encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden final-test"):
        load_contract(path)


def test_few_shot_encoding_preserves_target_budget() -> None:
    tokenizer = ToyTokenizer()
    exemplars = [
        {"passage": "A" * 20, "query": "Jak A?", "form": "full_question"},
        {"passage": "B" * 20, "query": "hasło B", "form": "keyword_query"},
    ]
    ids, prompt = encode_prompt(
        tokenizer,
        "C" * 1000,
        strategy="few_shot",
        exemplars=exemplars,
        max_prompt_tokens=700,
        min_target_passage_tokens=128,
        max_exemplar_characters=20,
    )
    assert len(ids) == 700
    assert "Przykład 1" in prompt
    assert "Jak A?" in prompt
    assert prompt.endswith("Zapytanie:\n")


def test_left_padding_preserves_causal_prompt_suffix() -> None:
    input_ids, attention = _left_pad_batch(
        [[1, 2, 3], [4]], pad_token_id=0, device=torch.device("cpu")
    )
    assert input_ids.tolist() == [[1, 2, 3], [0, 0, 4]]
    assert attention.tolist() == [[1, 1, 1], [0, 0, 1]]


def test_model_batch_expands_sampling_candidates_in_prompt_order() -> None:
    model = ToyModel()
    generated = _generate_model_batch(
        model,
        ToyTokenizer(),
        [[1, 2, 3], [4]],
        mode={
            "do_sample": True,
            "num_return_sequences": 2,
            "temperature": 0.8,
            "top_p": 0.95,
        },
        max_new_tokens=4,
    )
    assert generated == ["A", "B", "C", "D"]
    assert model.last_attention is not None
    assert model.last_attention.tolist() == [[1, 1, 1], [0, 0, 1]]


def test_legacy_journal_identity_is_accepted(tmp_path: Path) -> None:
    legacy = next(iter(LEGACY_RESUME_IDENTITIES))
    path = tmp_path / "generation.sqlite"
    connection = _open_journal(path, legacy)
    connection.close()
    connection = _open_journal(
        path,
        "new-trajectory-identity",
        compatible_identities=LEGACY_RESUME_IDENTITIES,
    )
    connection.close()


def test_successful_batch_does_not_block_later_growth(tmp_path: Path) -> None:
    path = tmp_path / "generation.sqlite"
    connection = _open_journal(path, "identity")
    try:
        connection.execute(
            "INSERT INTO metadata(key, value) VALUES(?, ?)",
            ("prompt_batch_size:zero_shot:greedy", "8"),
        )
        connection.commit()
        assert (
            _effective_batch_size(
                connection,
                strategy="zero_shot",
                mode="greedy",
                requested=32,
                minimum=1,
            )
            == 32
        )
        connection.execute(
            "INSERT INTO metadata(key, value) VALUES(?, ?)",
            ("oom_batch_ceiling:zero_shot:greedy", "16"),
        )
        connection.commit()
        assert (
            _effective_batch_size(
                connection,
                strategy="zero_shot",
                mode="greedy",
                requested=32,
                minimum=1,
            )
            == 16
        )
    finally:
        connection.close()


def test_mock_generation_resumes_exactly(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    contract = _fixture(tmp_path, monkeypatch)
    with pytest.raises(InterruptedError):
        generate_s00(contract, mock=True, interrupt_after=1)
    result = generate_s00(contract, mock=True)
    assert result["status"] == "complete"
    assert result["resumed_generation_count"] > 0
    assert result["final_tests_used"] == []
    for strategy in ("zero_shot", "few_shot"):
        rows = [
            json.loads(line)
            for line in (tmp_path / "out" / f"{strategy}.generations.jsonl")
            .read_text()
            .splitlines()
        ]
        assert len(rows) == 20
        assert len({row["evaluation_id"] for row in rows}) == 20


def test_mock_generation_halves_batch_after_oom(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    contract = _fixture(tmp_path, monkeypatch)
    result = generate_s00(
        contract,
        batch_size=4,
        min_batch_size=1,
        mock=True,
        mock_oom_above=2,
    )
    assert result["oom_retries"] == 4
    assert set(result["effective_prompt_batch_sizes"].values()) == {2}
    journal = tmp_path / "out" / "generation.sqlite"
    connection = sqlite3.connect(journal)
    try:
        batch_sizes = {
            int(row[0])
            for row in connection.execute(
                "SELECT value FROM metadata WHERE key LIKE 'prompt_batch_size:%'"
            )
        }
    finally:
        connection.close()
    assert batch_sizes == {2}


def test_greedy_and_sampling_use_independent_prompt_batches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    contract = _fixture(tmp_path, monkeypatch)
    result = generate_s00(
        contract,
        greedy_batch_size=4,
        sampling_batch_size=2,
        mock=True,
    )
    assert result["requested_prompt_batch_sizes"] == {"greedy": 4, "sampling": 2}
    assert result["effective_prompt_batch_sizes"] == {
        "zero_shot/greedy": 4,
        "zero_shot/sampling": 2,
        "few_shot/greedy": 4,
        "few_shot/sampling": 2,
    }
