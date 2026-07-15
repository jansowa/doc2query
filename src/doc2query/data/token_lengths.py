"""Optional pinned Hugging Face tokenizer counters for offline data audits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TokenizerSpec:
    label: str
    name_or_path: str
    revision: str

    def __post_init__(self) -> None:
        if len(self.revision) != 40 or any(
            char not in "0123456789abcdef" for char in self.revision
        ):
            raise ValueError("tokenizer revision must be a full commit SHA")


def load_tokenizer_specs(path: Path) -> list[TokenizerSpec]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("tokenizers"), list):
        raise ValueError("tokenizer config requires a tokenizers list")
    return [TokenizerSpec(**item) for item in value["tokenizers"]]


class TokenLengthCounter:
    def __init__(self, spec: TokenizerSpec) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("install the training dependency group for tokenizer audit") from exc
        loader: Any = getattr(AutoTokenizer, "from_" + "pretrained")
        self.spec = spec
        self._tokenizer = loader(
            spec.name_or_path,
            revision=spec.revision,
            trust_remote_code=False,
        )

    def count(self, texts: list[str]) -> list[int]:
        encoded = self._tokenizer(
            texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )
        return [len(token_ids) for token_ids in encoded["input_ids"]]
