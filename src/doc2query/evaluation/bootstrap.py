"""Deterministic paired bootstrap with strict evaluation-fingerprint checks."""

from __future__ import annotations

import random
from statistics import fmean
from typing import Any

from doc2query.evaluation.retrieval import percentile


def assert_same_test_fingerprint(left: dict[str, Any], right: dict[str, Any]) -> str:
    left_value = left.get("test_fingerprint")
    right_value = right.get("test_fingerprint")
    if not left_value or left_value != right_value:
        raise ValueError("run comparison requires identical non-empty test_fingerprint values")
    return str(left_value)


def paired_bootstrap(
    left: dict[str, float],
    right: dict[str, float],
    *,
    samples: int = 2000,
    seed: int = 42,
) -> dict[str, float | int]:
    """Return right-minus-left uncertainty over shared query IDs."""
    if samples < 1:
        raise ValueError("samples must be positive")
    ids = sorted(left.keys() & right.keys())
    if not ids or set(left) != set(right):
        raise ValueError("paired bootstrap requires the same non-empty query IDs")
    differences = [right[key] - left[key] for key in ids]
    rng = random.Random(seed)
    estimates = [
        fmean(differences[rng.randrange(len(differences))] for _ in differences)
        for _ in range(samples)
    ]
    observed = fmean(differences)
    low = percentile(estimates, 0.025)
    high = percentile(estimates, 0.975)
    if low is None or high is None:
        raise RuntimeError("bootstrap unexpectedly produced no estimates")
    return {
        "query_count": len(ids),
        "bootstrap_samples": samples,
        "seed": seed,
        "difference": observed,
        "ci95_low": low,
        "ci95_high": high,
        "variant_win_fraction": sum(value > 0 for value in estimates) / samples,
    }
