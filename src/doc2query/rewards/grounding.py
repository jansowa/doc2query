"""Grounding reward built from separately calibrated absolute score and margin."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class Calibrator(Protocol):
    def transform(self, value: float) -> float: ...


@dataclass(frozen=True)
class GroundingReward:
    absolute: float
    margin: float
    total: float


def grounding_reward(
    positive_score: float,
    hardest_negative_score: float,
    *,
    score_calibrator: Calibrator,
    margin_calibrator: Calibrator,
    margin_weight: float = 1.0,
) -> GroundingReward:
    absolute = score_calibrator.transform(positive_score)
    margin = margin_calibrator.transform(positive_score - hardest_negative_score)
    return GroundingReward(
        absolute=absolute, margin=margin, total=absolute + margin_weight * margin
    )
