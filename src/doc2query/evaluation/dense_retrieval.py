"""Exact, batched retrieval over persistent embedding shards.

The implementation deliberately keeps the on-disk shards as the index.  It
therefore has no lossy build step, can resume immediately after corpus
encoding, and uses a matrix-matrix product instead of one matrix-vector scan
per query.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import torch


@dataclass(frozen=True)
class EmbeddingShard:
    path: Path
    start: int
    end: int
    dimension: int


@dataclass(frozen=True)
class ExactRetrievalBatch:
    positive_ranks: list[list[int]]
    hard_negative_win_rates: list[float | None]


class ShardedEmbeddingIndex:
    """A validated, immutable view of ``chunk-*.pt`` embedding shards."""

    def __init__(self, shards: Sequence[EmbeddingShard], *, row_count: int) -> None:
        if not shards or row_count < 1:
            raise ValueError("embedding index must contain at least one row")
        if shards[0].start != 0 or shards[-1].end != row_count:
            raise ValueError("embedding shards do not cover the declared row count")
        for left, right in pairwise(shards):
            if left.end != right.start or left.dimension != right.dimension:
                raise ValueError("embedding shards are not contiguous and dimensionally uniform")
        self.shards = tuple(shards)
        self.row_count = row_count
        self.dimension = shards[0].dimension
        self._ends = tuple(shard.end for shard in shards)

    @classmethod
    def load(cls, cache_dir: Path, *, row_count: int, chunk_size: int) -> ShardedEmbeddingIndex:
        if row_count < 1 or chunk_size < 1:
            raise ValueError("embedding cache row_count and chunk_size must be positive")
        expected_count = math.ceil(row_count / chunk_size)
        paths = [cache_dir / f"chunk-{index:05d}.pt" for index in range(expected_count)]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise ValueError(f"embedding cache is incomplete; first missing shard: {missing[0]}")
        unexpected = sorted(cache_dir.glob("chunk-*.pt"))[expected_count:]
        if unexpected:
            raise ValueError(f"embedding cache has an unexpected shard: {unexpected[0]}")
        shards: list[EmbeddingShard] = []
        dimension: int | None = None
        for index, path in enumerate(paths):
            tensor = _load_tensor(path)
            start = index * chunk_size
            end = min(start + chunk_size, row_count)
            if tensor.shape[0] != end - start:
                raise ValueError(f"embedding shard has wrong row count: {path}")
            if dimension is None:
                dimension = int(tensor.shape[1])
            elif tensor.shape[1] != dimension:
                raise ValueError(f"embedding shard has inconsistent dimension: {path}")
            shards.append(EmbeddingShard(path, start, end, int(tensor.shape[1])))
        return cls(shards, row_count=row_count)

    @property
    def size_bytes(self) -> int:
        return sum(shard.path.stat().st_size for shard in self.shards)

    def shard_for_row(self, row: int) -> EmbeddingShard:
        if row < 0 or row >= self.row_count:
            raise IndexError(f"embedding row is outside the index: {row}")
        return self.shards[bisect.bisect_right(self._ends, row)]

    def lookup(self, rows: Sequence[int]) -> dict[int, torch.Tensor]:
        """Load only shards needed for the requested corpus rows."""
        grouped: dict[Path, list[tuple[int, int]]] = {}
        for row in sorted(set(rows)):
            shard = self.shard_for_row(row)
            grouped.setdefault(shard.path, []).append((row, row - shard.start))
        result: dict[int, torch.Tensor] = {}
        for path, positions in grouped.items():
            tensor = _load_tensor(path)
            for global_row, local_row in positions:
                result[global_row] = tensor[local_row]
        return result


def _load_tensor(path: Path) -> torch.Tensor:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:  # pragma: no cover - compatibility with older supported torch
        value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or not value.is_floating_point():
        raise ValueError(f"invalid embedding shard: {path}")
    return value


def exact_retrieval_batch(
    query_embeddings: torch.Tensor,
    *,
    positive_rows: Sequence[Sequence[int]],
    negative_rows: Sequence[Sequence[int]],
    index: ShardedEmbeddingIndex,
    device: torch.device,
    shard_progress: Callable[[int, int], None] | None = None,
) -> ExactRetrievalBatch:
    """Compute stable exact positive ranks and pool wins with a single shard scan.

    Ties follow the historical contract: corpus rows are ordered by sorted
    ``doc_id`` and an equal-scoring row precedes a positive only when its row
    index is smaller.
    """
    if query_embeddings.ndim != 2 or query_embeddings.shape[0] != len(positive_rows):
        raise ValueError("query embedding batch and positive row groups must align")
    if len(positive_rows) != len(negative_rows):
        raise ValueError("positive and negative row groups must align")
    if query_embeddings.shape[1] != index.dimension:
        raise ValueError("query and corpus embedding dimensions differ")
    if any(not rows for rows in positive_rows):
        raise ValueError("every query must have at least one positive corpus row")

    query_cpu = query_embeddings.detach().to(device="cpu", dtype=torch.float32)
    query_device = query_cpu.to(device)
    wanted = [row for rows in (*positive_rows, *negative_rows) for row in rows]
    vectors = index.lookup(wanted)
    max_positives = max(len(rows) for rows in positive_rows)
    positive_scores = torch.full(
        (len(positive_rows), max_positives),
        float("nan"),
        dtype=torch.float32,
        device=device,
    )
    positive_positions = torch.full(
        (len(positive_rows), max_positives), -1, dtype=torch.int64
    )
    positive_mask = torch.zeros((len(positive_rows), max_positives), dtype=torch.bool)
    for query_index, rows in enumerate(positive_rows):
        for positive_index, row in enumerate(rows):
            positive_positions[query_index, positive_index] = row
            positive_mask[query_index, positive_index] = True
            positive_scores[query_index, positive_index] = torch.dot(
                query_device[query_index], vectors[row].to(device=device, dtype=torch.float32)
            )

    scores_device = positive_scores
    positions_device = positive_positions.to(device)
    ranks = torch.ones_like(positive_positions, device=device)
    with torch.inference_mode():
        for shard_index, shard in enumerate(index.shards, start=1):
            corpus = _load_tensor(shard.path).to(device=device, dtype=torch.float32)
            scores = query_device @ corpus.T
            corpus_positions = torch.arange(shard.start, shard.end, device=device)
            better = scores[:, None, :] > scores_device[:, :, None]
            better &= corpus_positions[None, None, :] != positions_device[:, :, None]
            tied_before = (scores[:, None, :] == scores_device[:, :, None]) & (
                corpus_positions[None, None, :] < positions_device[:, :, None]
            )
            ranks += (better | tied_before).sum(dim=2)
            del corpus, scores, corpus_positions, better, tied_before
            if shard_progress is not None:
                shard_progress(shard_index, len(index.shards))

    rank_rows = [
        [int(value) for value in ranks[index_row, positive_mask[index_row]].cpu().tolist()]
        for index_row in range(len(positive_rows))
    ]
    win_rates: list[float | None] = []
    for query_index, (positives, negatives) in enumerate(
        zip(positive_rows, negative_rows, strict=True)
    ):
        if not negatives:
            win_rates.append(None)
            continue
        positive_values = torch.stack([vectors[row] for row in positives]).to(torch.float32)
        negative_values = torch.stack([vectors[row] for row in negatives]).to(torch.float32)
        query = query_cpu[query_index]
        positive_pair_scores = positive_values @ query
        negative_pair_scores = negative_values @ query
        wins = positive_pair_scores[:, None] > negative_pair_scores[None, :]
        win_rates.append(float(wins.to(torch.float32).mean().item()))
    return ExactRetrievalBatch(rank_rows, win_rates)
