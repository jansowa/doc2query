"""Natural-query percentile bands and component dominance diagnostics."""

from __future__ import annotations

import math
from statistics import fmean


def quantile(values: list[float], probability: float) -> float:
    if not values or not 0 <= probability <= 1:
        raise ValueError("invalid quantile input")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def overlap_band_reward(value: float, *, low: float, high: float) -> float:
    """Triangular shoulders, flat optimum in the natural-query band, range [0, 1]."""
    if not 0 < low < high < 1:
        raise ValueError("overlap band must satisfy 0 < low < high < 1")
    if low <= value <= high:
        return 1.0
    if value < low:
        return max(0.0, value / low)
    return max(0.0, (1.0 - value) / (1.0 - high))


def pearson_matrix(rows: list[dict[str, float]]) -> dict[str, dict[str, float | None]]:
    if not rows:
        raise ValueError("correlation requires rows")
    keys = sorted(set.intersection(*(set(row) for row in rows)))
    result: dict[str, dict[str, float | None]] = {}
    for left in keys:
        result[left] = {}
        xs = [row[left] for row in rows]
        mx = fmean(xs)
        for right in keys:
            ys = [row[right] for row in rows]
            my = fmean(ys)
            numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
            denominator = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
            result[left][right] = numerator / denominator if denominator else None
    return result


def dominance_warnings(
    correlations: dict[str, dict[str, float | None]],
    *,
    total_key: str = "total",
    limit: float = 0.95,
) -> list[str]:
    return [
        key
        for key, value in correlations.get(total_key, {}).items()
        if key != total_key and value is not None and abs(value) >= limit
    ]
