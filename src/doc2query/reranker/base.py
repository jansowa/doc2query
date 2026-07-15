"""Public contracts for immutable pair scorers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


class PairScorer(Protocol):
    """A read-only scorer of query/passage pairs."""

    @property
    def name(self) -> str: ...

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]: ...


@dataclass(frozen=True)
class FrozenRerankerConfig:
    name_or_path: str
    revision: str
    license: str
    max_length: int = 8192
    batch_size: int = 8
    device: str = "cpu"
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        if len(self.revision) != 40 or any(c not in "0123456789abcdef" for c in self.revision):
            raise ValueError("reranker revision must be a full 40-character commit SHA")
        if self.trust_remote_code:
            raise ValueError("trust_remote_code is prohibited for frozen judges")
        if self.max_length < 16 or self.batch_size < 1:
            raise ValueError("max_length and batch_size must be positive")
