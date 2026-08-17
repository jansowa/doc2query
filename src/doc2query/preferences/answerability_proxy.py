"""Answerability proxy v1: a frozen threshold rule calibrated on Groq audit labels.

The proxy is the *temporary* answerability signal of pair-policy axis A.  Its protocol
is frozen prospectively in `reports/decisions/task06_answerability_proxy_v1.md`
(contract ``task06-answerability-proxy-v1``) and this module implements that ADR and
nothing else:

* labels are **per side** (chosen/rejected) and come from the consensus of both Groq
  judges on ``answerable_a``/``answerable_b``; sides where the judges disagree, or where
  a judge has no rating, carry no label;
* features are only the thirteen already-computed scoring fields listed in the ADR;
* the fit/holdout split is deterministic on ``sha256(audit_id)`` and keeps both sides of
  a pair in the same half, so the passage cannot leak across halves;
* rule selection happens on the fit half only, under the frozen objective; the holdout is
  scored **once** against the frozen acceptance criterion.

The pinned local judge (`answerability_judge.py`) supersedes the proxy by a separate ADR
once the 27B weights exist; every artifact therefore carries
``answerability_signal="proxy_v1"``.  Nothing here builds a pair, trains anything or
touches a frozen threshold.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Literal

from doc2query.evaluation.retrieval import percentile
from doc2query.training.dpo import file_sha256
from doc2query.utils.records import read_records, write_json

PROXY_CONTRACT = "task06-answerability-proxy-v1"
PROXY_ADR = "reports/decisions/task06_answerability_proxy_v1.md"
JUDGE_MODELS = ("openai/gpt-oss-120b", "qwen/qwen3.6-27b")

# Zamrożona w ADR lista cech; wyłącznie pola już policzone w zamrożonym scoringu.
PROXY_FEATURES: tuple[str, ...] = (
    "content_jaccard",
    "copy_density",
    "corpus_margin_to_best_nonpositive",
    "corpus_possibly_ambiguous_query",
    "corpus_round_trip_at_5",
    "longest_copied_ngram",
    "natural_content_jaccard",
    "normalized_lcs",
    "passage_recall",
    "pool_margin",
    "pool_positive_score",
    "query_precision",
    "word_length",
)
THRESHOLD_QUANTILES: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
# Kryteria z §5/§6 ADR: identyczne na fit (wybór) i na holdoucie (akceptacja).
MINIMUM_PRECISION = 0.88
MINIMUM_RECALL = 0.50

Direction = Literal["ge", "le"]
Half = Literal["fit", "holdout"]


@dataclass(frozen=True)
class SideItem:
    """One labelled side of an audited pair."""

    audit_id: str
    pair_id: str
    role: Literal["chosen", "rejected"]
    half: Half
    label: bool
    features: Mapping[str, float]


@dataclass(frozen=True)
class RuleAtom:
    """A single ``feature >= t`` / ``feature <= t`` test."""

    feature: str
    direction: Direction
    threshold: float

    def holds(self, features: Mapping[str, float]) -> bool:
        value = features[self.feature]
        return value >= self.threshold if self.direction == "ge" else value <= self.threshold

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "direction": self.direction,
            "threshold": self.threshold,
        }

    def sort_key(self) -> tuple[str, str, float]:
        return (self.feature, self.direction, self.threshold)


@dataclass(frozen=True)
class ProxyRule:
    """A conjunction of one or two atoms; predicts ``yes`` when every atom holds."""

    atoms: tuple[RuleAtom, ...]

    def predict(self, features: Mapping[str, float]) -> bool:
        return all(atom.holds(features) for atom in self.atoms)

    def as_dict(self) -> dict[str, Any]:
        return {
            "atoms": [atom.as_dict() for atom in self.atoms],
            "expression": " and ".join(
                f"{atom.feature} {'>=' if atom.direction == 'ge' else '<='} {atom.threshold:.6g}"
                for atom in self.atoms
            ),
        }

    def sort_key(self) -> tuple[Any, ...]:
        return (len(self.atoms), tuple(atom.sort_key() for atom in self.atoms))


def split_half(audit_id: str) -> Half:
    """Deterministic 50/50 split on ``sha256(audit_id)``; both sides share the half."""
    digest = hashlib.sha256(audit_id.encode("utf-8")).hexdigest()
    return "fit" if int(digest[:2], 16) < 0x80 else "holdout"


def _feature_value(components: Mapping[str, Any], feature: str) -> float:
    value = components[feature]
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return float(value)


def extract_features(components: Mapping[str, Any]) -> dict[str, float]:
    return {feature: _feature_value(components, feature) for feature in PROXY_FEATURES}


def _consensus_label(ratings: Mapping[str, Mapping[str, Any]], option: str) -> bool | None:
    """``True``/``False`` only when both judges agree; ``None`` otherwise."""
    verdicts = []
    for model in JUDGE_MODELS:
        rating = ratings.get(model)
        if rating is None:
            return None
        verdicts.append(bool(rating[f"answerable_{option}"]))
    if verdicts[0] != verdicts[1]:
        return None
    return verdicts[0]


def build_side_items(
    *,
    sample_rows: Sequence[Mapping[str, Any]],
    verdict_rows: Sequence[Mapping[str, Any]],
    key_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[SideItem], dict[str, Any]]:
    """Join the blind key, the judge verdicts and the per-candidate components."""
    components_by_pair = {str(row["pair_id"]): row for row in sample_rows}
    option_by_pair = {
        str(row["pair_id"]): str(row["automatic_chosen_option"]).upper() for row in key_rows
    }
    items: list[SideItem] = []
    sides_with_two_ratings = 0
    judges_agree = 0
    skipped_missing_rating = 0
    skipped_judge_split = 0
    for verdict in verdict_rows:
        pair_id = str(verdict["pair_id"])
        audit_id = str(verdict["audit_id"])
        ratings = dict(verdict.get("ratings") or {})
        sample = components_by_pair.get(pair_id)
        if sample is None:
            raise ValueError(f"pair {pair_id} is missing from the export sample")
        chosen_option = option_by_pair[pair_id]
        if chosen_option not in {"A", "B"}:
            raise ValueError(f"pair {pair_id} has an unexpected automatic option")
        rated_by_both = all(model in ratings for model in JUDGE_MODELS)
        half = split_half(audit_id)
        for option in ("a", "b"):
            role: Literal["chosen", "rejected"] = (
                "chosen" if option.upper() == chosen_option else "rejected"
            )
            if rated_by_both:
                sides_with_two_ratings += 1
            label = _consensus_label(ratings, option)
            if label is None:
                if rated_by_both:
                    skipped_judge_split += 1
                else:
                    skipped_missing_rating += 1
                continue
            judges_agree += 1
            items.append(
                SideItem(
                    audit_id=audit_id,
                    pair_id=pair_id,
                    role=role,
                    half=half,
                    label=label,
                    features=extract_features(sample[f"{role}_components"]),
                )
            )
    labels = [item.label for item in items]
    supply = {
        "sides_with_two_ratings": sides_with_two_ratings,
        "sides_labelled": len(items),
        "sides_label_yes": sum(labels),
        "sides_label_no": len(labels) - sum(labels),
        "sides_skipped_judge_split": skipped_judge_split,
        "sides_skipped_missing_rating": skipped_missing_rating,
        # Sufit szumu: zgodność sędziów na wszystkich stronach z dwiema ocenami.
        "inter_judge_answerability_agreement": (
            judges_agree / sides_with_two_ratings if sides_with_two_ratings else None
        ),
        "majority_class_baseline": (sum(labels) / len(labels) if labels else None),
    }
    return items, supply


def score_rule(rule: ProxyRule, items: Sequence[SideItem]) -> dict[str, Any]:
    """Confusion matrix and the two frozen criteria for one rule on one set of sides."""
    true_positive = false_positive = true_negative = false_negative = 0
    for item in items:
        predicted = rule.predict(item.features)
        if predicted and item.label:
            true_positive += 1
        elif predicted and not item.label:
            false_positive += 1
        elif not predicted and item.label:
            false_negative += 1
        else:
            true_negative += 1
    predicted_yes = true_positive + false_positive
    actual_yes = true_positive + false_negative
    actual_no = false_positive + true_negative
    total = len(items)
    recall_yes = true_positive / actual_yes if actual_yes else None
    recall_no = true_negative / actual_no if actual_no else None
    balanced = (
        (recall_yes + recall_no) / 2 if recall_yes is not None and recall_no is not None else None
    )
    return {
        "count": total,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "predicted_yes": predicted_yes,
        "precision_yes": true_positive / predicted_yes if predicted_yes else None,
        "recall_yes": recall_yes,
        "recall_no": recall_no,
        "accuracy": (true_positive + true_negative) / total if total else None,
        "balanced_accuracy": balanced,
    }


def _passes(metrics: Mapping[str, Any]) -> bool:
    precision = metrics["precision_yes"]
    recall = metrics["recall_yes"]
    if precision is None or recall is None:
        return False
    return float(precision) >= MINIMUM_PRECISION and float(recall) >= MINIMUM_RECALL


def candidate_atoms(items: Sequence[SideItem]) -> list[RuleAtom]:
    """Every atom of the frozen space: 13 features x 2 directions x 9 fit-half deciles."""
    atoms: list[RuleAtom] = []
    for feature in PROXY_FEATURES:
        values = [item.features[feature] for item in items]
        thresholds = sorted(
            {
                round(threshold, 12)
                for quantile in THRESHOLD_QUANTILES
                if (threshold := percentile(values, quantile)) is not None
            }
        )
        for threshold in thresholds:
            atoms.append(RuleAtom(feature, "ge", threshold))
            atoms.append(RuleAtom(feature, "le", threshold))
    return atoms


def select_rule(fit_items: Sequence[SideItem]) -> dict[str, Any]:
    """Frozen objective: max recall_yes subject to precision_yes >= 0.88 and recall >= 0.50."""
    atoms = candidate_atoms(fit_items)
    rules = [ProxyRule((atom,)) for atom in atoms]
    for left, right in combinations(atoms, 2):
        if left.feature != right.feature:
            rules.append(ProxyRule(tuple(sorted((left, right), key=RuleAtom.sort_key))))
    admissible: list[tuple[ProxyRule, dict[str, Any]]] = []
    best_precision_seen: dict[str, Any] | None = None
    best_rule_seen: ProxyRule | None = None
    for rule in rules:
        metrics = score_rule(rule, fit_items)
        precision = metrics["precision_yes"]
        if precision is not None and (
            best_precision_seen is None
            or (precision, metrics["recall_yes"] or 0.0)
            > (best_precision_seen["precision_yes"], best_precision_seen["recall_yes"] or 0.0)
        ):
            best_precision_seen, best_rule_seen = metrics, rule
        if _passes(metrics):
            admissible.append((rule, metrics))
    if not admissible:
        return {
            "construction_status": "failed_no_admissible_rule_on_fit",
            "rule": None,
            "fit_metrics": None,
            "candidate_rule_count": len(rules),
            "best_fit_precision_rule": None if best_rule_seen is None else best_rule_seen.as_dict(),
            "best_fit_precision_metrics": best_precision_seen,
        }
    winner, winner_metrics = min(
        admissible,
        key=lambda entry: (
            -(entry[1]["recall_yes"] or 0.0),
            -(entry[1]["precision_yes"] or 0.0),
            entry[0].sort_key(),
        ),
    )
    return {
        "construction_status": "selected",
        "rule": winner.as_dict(),
        "fit_metrics": winner_metrics,
        "candidate_rule_count": len(rules),
        "admissible_rule_count": len(admissible),
        "best_fit_precision_rule": None if best_rule_seen is None else best_rule_seen.as_dict(),
        "best_fit_precision_metrics": best_precision_seen,
        "_rule": winner,
    }


def _proportion_ci(
    values: Sequence[bool], *, samples: int = 2000, seed: int = 42
) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "rate": None, "ci95_low": None, "ci95_high": None}
    rng = random.Random(seed)
    population = [1 if value else 0 for value in values]
    count = len(population)
    estimates = [sum(rng.choices(population, k=count)) / count for _ in range(samples)]
    return {
        "count": len(values),
        "rate": sum(population) / count,
        "ci95_low": percentile(estimates, 0.025),
        "ci95_high": percentile(estimates, 0.975),
    }


def evaluate_holdout(rule: ProxyRule, holdout_items: Sequence[SideItem]) -> dict[str, Any]:
    """Score the frozen rule on the holdout exactly once, with CIs for both criteria."""
    metrics = score_rule(rule, holdout_items)
    precision_values = [item.label for item in holdout_items if rule.predict(item.features)]
    recall_values = [rule.predict(item.features) for item in holdout_items if item.label]
    per_role = {
        role: score_rule(rule, [item for item in holdout_items if item.role == role])
        for role in ("chosen", "rejected")
    }
    return {
        "metrics": metrics,
        "precision_yes_ci": _proportion_ci(precision_values),
        "recall_yes_ci": _proportion_ci(recall_values),
        "per_role": per_role,
        "criterion": {
            "minimum_precision_yes": MINIMUM_PRECISION,
            "minimum_recall_yes": MINIMUM_RECALL,
            "precision_met": (metrics["precision_yes"] or 0.0) >= MINIMUM_PRECISION,
            "recall_met": (metrics["recall_yes"] or 0.0) >= MINIMUM_RECALL,
            "accepted": _passes(metrics),
        },
    }


def calibrate_answerability_proxy(*, export_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Run the frozen ADR end to end and publish the calibration artifact."""
    sample_path = export_dir / "sample.jsonl"
    key_path = export_dir / "machine_key.jsonl"
    verdict_path = export_dir / "groq_dual_llm" / "pair_verdicts.jsonl"
    sample_rows = list(read_records(sample_path))
    key_rows = list(read_records(key_path))
    verdict_rows = list(read_records(verdict_path))
    items, supply = build_side_items(
        sample_rows=sample_rows, verdict_rows=verdict_rows, key_rows=key_rows
    )
    fit_items = [item for item in items if item.half == "fit"]
    holdout_items = [item for item in items if item.half == "holdout"]
    selection = select_rule(fit_items)
    rule = selection.pop("_rule", None)
    holdout: dict[str, Any] | None = None
    if rule is not None:
        holdout = evaluate_holdout(rule, holdout_items)
    accepted = bool(holdout and holdout["criterion"]["accepted"])
    result: dict[str, Any] = {
        "schema": "task06-answerability-proxy-result-v1",
        "contract": PROXY_CONTRACT,
        "adr": PROXY_ADR,
        "answerability_signal": "proxy_v1",
        "status": (
            "accepted_as_chosen_side_filter"
            if accepted
            else "rejected_axis_a_without_answerability_filter"
        ),
        "label_snapshot": {
            "sample_sha256": file_sha256(sample_path),
            "machine_key_sha256": file_sha256(key_path),
            "pair_verdicts_sha256": file_sha256(verdict_path),
            "audit_status": "incomplete_quota_deferred_day1_snapshot",
        },
        "label_supply": supply,
        "split": {
            "rule": "fit if int(sha256(audit_id)[:2],16) < 0x80 else holdout",
            "fit_sides": len(fit_items),
            "holdout_sides": len(holdout_items),
            "fit_label_yes": sum(item.label for item in fit_items),
            "holdout_label_yes": sum(item.label for item in holdout_items),
            "holdout_reads": 1 if rule is not None else 0,
        },
        "feature_space": {
            "features": list(PROXY_FEATURES),
            "threshold_quantiles": list(THRESHOLD_QUANTILES),
            "maximum_atoms_per_rule": 2,
        },
        "selection": selection,
        "holdout": holdout,
        "supersedes_nothing": True,
        "superseded_by": "pinned local answerability judge (V2-01) via a separate ADR",
        "human_evidence_claimed": False,
        "task07_training_authorized": False,
        "final_tests_used": [],
    }
    write_json(output_dir / "answerability_proxy_v1.json", result)
    return result
