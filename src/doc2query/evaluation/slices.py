"""Slice assignment and aggregation without silently turning missing values into zero."""

from __future__ import annotations

from collections import defaultdict
from statistics import fmean
from typing import Any

from doc2query.evaluation.retrieval import distribution


def rank_buckets(values: list[float], labels: tuple[str, ...]) -> list[str]:
    if not labels:
        raise ValueError("at least one bucket label is required")
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    result = [labels[0]] * len(values)
    for rank, index in enumerate(order):
        result[index] = labels[min(len(labels) - 1, rank * len(labels) // len(values))]
    return result


def aggregate_slices(
    rows: list[dict[str, Any]],
    *,
    slice_fields: list[str],
    metric_fields: list[str],
) -> dict[str, Any]:
    groups: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        raw_slices = row.get("slices", {})
        if not isinstance(raw_slices, dict):
            raise ValueError("row slices must be a mapping")
        for field in slice_fields:
            groups[field][str(raw_slices.get(field, "unknown"))].append(row)
    result: dict[str, Any] = {}
    for field, values in groups.items():
        if sum(len(group) for group in values.values()) != len(rows):
            raise RuntimeError(f"slice {field} does not sum to the full population")
        result[field] = {}
        for value, group in sorted(values.items()):
            metrics: dict[str, Any] = {}
            for metric in metric_fields:
                observed = [
                    float(row[metric]) for row in group if isinstance(row.get(metric), (int, float))
                ]
                metrics[metric] = (
                    {"mean": fmean(observed), "distribution": distribution(observed)}
                    if observed
                    else None
                )
            result[field][value] = {"count": len(group), "metrics": metrics}
    return result
