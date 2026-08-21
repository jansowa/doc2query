"""M-03 in-run probe collapse detection and automatic reseed (contract v1).

The within-arm metrology measurement showed that probe collapses are a property of the
training procedure, not of an arm: 5 of 27 runs (18.5%) collapsed while differing only in
their seed.  The guardrail of ``probe_convergence`` can only say so **after** a finished
run, by which time the expensive part of the run — encoding a 139 782 document corpus —
has already been paid.

This module therefore watches a run **while it trains** and lets the caller abort and
reseed.  Three design constraints come straight from the frozen ADR
``reports/decisions/task04_m03_in_run_collapse_detection_v1.md``:

* the interim retrieval signal is built from the run's **training rows**, never from the
  evaluation set and never from the decision metric, so aborting cannot select on the
  outcome variable;
* the loss direction is a cheap fail-fast only.  ``loss_based_guardrail_permitted`` stays
  ``False``: the *level* of the loss correlates with retrieval at ``r = -0.199`` and never
  decides convergence.  Convergence of a finished run is still decided solely by M-03;
* every attempt is journalled, so the number of reseeds can never be hidden.

Thresholds are loaded from an externally frozen contract; this module never invents one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Literal

import torch
import yaml
from pydantic import Field

from doc2query.schemas import StrictModel

DETECTION_CONTRACT = "task04-m03-in-run-collapse-detection-v1"


class PersistenceContract(StrictModel):
    loss_curve_file: Literal["training_loss_curve.jsonl"]
    interim_evaluation_file: Literal["training_interim_evaluation.jsonl"]
    attempt_journal_file: Literal["collapse_detection_journal.jsonl"]


class InterimEvaluationContract(StrictModel):
    source: Literal["training_rows_holdin"]
    metric: Literal["train_holdin_recall_at_100"]
    interval_steps: int = Field(ge=1)
    first_check_step: int = Field(ge=1)
    corpus_documents: int = Field(ge=100)
    queries: int = Field(ge=1)
    retrieval_depth: int = Field(ge=1)
    encode_batch_size: int = Field(ge=1)
    restore_rng_state: Literal[True]


class RetrievalFloorRule(StrictModel):
    rule_id: Literal["interim_recall_below_chance_floor"]
    min_chance_multiple: float = Field(gt=1.0)


class LossDirectionRule(StrictModel):
    rule_id: Literal["loss_direction_non_decreasing"]
    window_steps: int = Field(ge=1)


class RuleContract(StrictModel):
    retrieval_floor: RetrievalFloorRule
    loss_direction: LossDirectionRule
    consecutive_hits_required: int = Field(ge=1)


class ReseedContract(StrictModel):
    max_attempts: int = Field(ge=1)
    seed_stride: int = Field(ge=1)
    on_exhausted: Literal["fail_run_with_status_collapse_unresolved"]
    provenance_required: Literal[True]


class InRunCollapseDetection(StrictModel):
    """The externally frozen in-run detection contract."""

    schema_version: Literal[1]
    contract: Literal["task04-m03-in-run-collapse-detection-v1"]
    detector_id: str = Field(min_length=1)
    status: Literal["frozen_before_first_new_run"]
    adr: str = Field(min_length=1)
    loss_based_guardrail_permitted: Literal[False]
    persistence: PersistenceContract
    interim_evaluation: InterimEvaluationContract
    rules: RuleContract
    reseed: ReseedContract
    final_tests_used: list[str] = Field(max_length=0)

    def attempt_seed(self, requested_seed: int, attempt_index: int) -> int:
        """Deterministic, collision-free reseed: no random and no manual choice."""
        if not 0 <= attempt_index < self.reseed.max_attempts:
            raise ValueError("attempt_index is outside the frozen reseed budget")
        return requested_seed + self.reseed.seed_stride * attempt_index

    def reference(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "detector_id": self.detector_id,
            "adr": self.adr,
            "interval_steps": self.interim_evaluation.interval_steps,
            "consecutive_hits_required": self.rules.consecutive_hits_required,
            "max_attempts": self.reseed.max_attempts,
            "seed_stride": self.reseed.seed_stride,
            "loss_based_guardrail_permitted": False,
        }


def load_collapse_detection(path: Path) -> InRunCollapseDetection:
    """Load a frozen in-run detection contract; thresholds are never derived here."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: in-run collapse contract must be a mapping")
    return InRunCollapseDetection.model_validate(raw)


class ProbeCollapseDetected(Exception):
    """One training attempt was aborted because a frozen detection rule fired twice."""

    def __init__(self, observation: Mapping[str, Any]) -> None:
        self.observation = dict(observation)
        super().__init__(
            f"probe collapse detected at step {self.observation.get('step')} "
            f"by {self.observation.get('rule')}"
        )


class ProbeCollapseUnresolved(Exception):
    """Every reseed attempt collapsed; the run fails instead of reporting a result."""

    def __init__(self, attempts: Sequence[Mapping[str, Any]]) -> None:
        self.attempts = [dict(attempt) for attempt in attempts]
        super().__init__(
            f"probe collapse unresolved after {len(self.attempts)} attempts; "
            "no measured artifact was written"
        )


@dataclass(frozen=True)
class InterimEvaluationSet:
    """A held-in retrieval sanity set built only from the run's own training rows."""

    queries: list[str]
    documents: list[str]
    positive_positions: list[int]

    def chance_level(self, depth: int) -> float:
        """Random-retrieval level of Recall@depth inside this pool."""
        if not self.documents:
            raise ValueError("the interim evaluation pool is empty")
        return min(1.0, depth / len(self.documents))


def build_interim_evaluation_set(
    rows: Sequence[Mapping[str, Any]], contract: InterimEvaluationContract
) -> InterimEvaluationSet:
    """Select the frozen deterministic held-in subset: sorted document ids, then rows."""
    texts: dict[str, str] = {}
    for row in rows:
        doc_id = str(row["positive_doc_id"])
        if doc_id not in texts:
            texts[doc_id] = str(row["positive"])
    document_ids = sorted(texts)[: contract.corpus_documents]
    position_of = {doc_id: position for position, doc_id in enumerate(document_ids)}
    queries: list[str] = []
    positions: list[int] = []
    for row in sorted(rows, key=lambda value: str(value["example_id"])):
        position = position_of.get(str(row["positive_doc_id"]))
        if position is None:
            continue
        queries.append(str(row["query"]))
        positions.append(position)
        if len(queries) == contract.queries:
            break
    if not queries:
        raise ValueError("the interim evaluation set is empty")
    return InterimEvaluationSet(
        queries=queries,
        documents=[texts[doc_id] for doc_id in document_ids],
        positive_positions=positions,
    )


def interim_recall(
    query_vectors: torch.Tensor,
    document_vectors: torch.Tensor,
    positive_positions: Sequence[int],
    *,
    depth: int,
) -> float:
    """Recall@depth of each query's own positive inside the small held-in pool."""
    if query_vectors.shape[0] != len(positive_positions):
        raise ValueError("interim recall needs one positive position per query")
    scores = query_vectors.to(torch.float32) @ document_vectors.to(torch.float32).T
    positives = torch.tensor(list(positive_positions), dtype=torch.long)
    positive_scores = scores.gather(1, positives.unsqueeze(1))
    # Ties are resolved pessimistically, which is the whole point of this signal: a fully
    # collapsed encoder maps every text onto one vector, so every score ties.  Counting only
    # strictly better documents would score that degenerate case as a perfect recall and
    # blind the detector to the failure it exists to catch.
    strictly_better = (scores > positive_scores).sum(dim=1)
    tied = (scores == positive_scores).sum(dim=1) - 1
    worst_case_rank = strictly_better + tied
    return float((worst_case_rank < depth).to(torch.float32).mean())


@dataclass
class CollapseDetector:
    """Applies the two frozen rules to interim observations of a single attempt."""

    contract: InRunCollapseDetection
    chance_level: float
    _streaks: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.chance_level <= 0.0:
            raise ValueError("the interim chance level must be positive")
        self._streaks = {
            self.contract.rules.retrieval_floor.rule_id: 0,
            self.contract.rules.loss_direction.rule_id: 0,
        }

    @property
    def floor(self) -> float:
        return self.chance_level * self.contract.rules.retrieval_floor.min_chance_multiple

    def should_check(self, completed_steps: int, max_steps: int) -> bool:
        interim = self.contract.interim_evaluation
        if completed_steps < interim.first_check_step or completed_steps >= max_steps:
            return False
        return completed_steps % interim.interval_steps == 0

    def observe(self, *, step: int, recall: float, losses: Sequence[float]) -> dict[str, Any]:
        """Record one interim checkpoint and report which rules fired, if any."""
        rules = self.contract.rules
        window = rules.loss_direction.window_steps
        first_window = losses[:window]
        last_window = losses[-window:]
        loss_first = fmean(first_window) if first_window else None
        loss_last = fmean(last_window) if last_window else None
        loss_hit = (
            len(losses) >= 2 * window
            and loss_first is not None
            and loss_last is not None
            and loss_last >= loss_first
        )
        recall_hit = recall < self.floor
        streaks = dict(self._streaks or {})
        hits = {
            rules.retrieval_floor.rule_id: recall_hit,
            rules.loss_direction.rule_id: loss_hit,
        }
        for rule_id, hit in hits.items():
            streaks[rule_id] = streaks[rule_id] + 1 if hit else 0
        self._streaks = streaks
        fired = [
            rule_id
            for rule_id, streak in streaks.items()
            if streak >= rules.consecutive_hits_required
        ]
        return {
            "step": step,
            "train_holdin_recall_at_100": recall,
            "chance_level": self.chance_level,
            "floor": self.floor,
            "below_floor": recall_hit,
            "loss_window_first": loss_first,
            "loss_window_last": loss_last,
            "loss_non_decreasing": loss_hit,
            "consecutive_hits": streaks,
            "collapse_detected": bool(fired),
            "rule": fired[0] if fired else None,
        }
