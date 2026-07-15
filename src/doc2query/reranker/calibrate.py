"""Serializable calibration over frozen model outputs only."""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from statistics import median
from typing import Any, Literal


@dataclass(frozen=True)
class RobustZCalibrator:
    center: float
    scale: float
    kind: Literal["robust_z"] = "robust_z"

    @classmethod
    def fit(cls, values: list[float]) -> RobustZCalibrator:
        if not values:
            raise ValueError("calibration requires values")
        center = median(values)
        mad = median(abs(value - center) for value in values)
        return cls(center=center, scale=max(1.4826 * mad, 1e-8))

    def transform(self, value: float) -> float:
        return (value - self.center) / self.scale

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "center": self.center, "scale": self.scale}


@dataclass(frozen=True)
class PercentileCalibrator:
    sorted_values: tuple[float, ...]
    kind: Literal["percentile"] = "percentile"

    @classmethod
    def fit(cls, values: list[float]) -> PercentileCalibrator:
        if not values:
            raise ValueError("calibration requires values")
        return cls(tuple(sorted(values)))

    def transform(self, value: float) -> float:
        if len(self.sorted_values) == 1:
            return 0.5
        left = bisect.bisect_left(self.sorted_values, value)
        right = bisect.bisect_right(self.sorted_values, value)
        return ((left + right - 1) / 2) / (len(self.sorted_values) - 1)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "sorted_values": list(self.sorted_values)}


def calibrator_from_dict(data: dict[str, Any]) -> RobustZCalibrator | PercentileCalibrator:
    if data.get("kind") == "robust_z":
        return RobustZCalibrator(float(data["center"]), float(data["scale"]))
    if data.get("kind") == "percentile":
        return PercentileCalibrator(tuple(float(x) for x in data["sorted_values"]))
    raise ValueError("unknown calibrator kind")
