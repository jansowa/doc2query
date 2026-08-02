"""Retrospective D01 usefulness diagnosis and safe-anchor hybrid selection."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from statistics import fmean
from typing import Any, Protocol, cast

import numpy as np
import yaml

from doc2query.evaluation.bootstrap import paired_bootstrap
from doc2query.evaluation.d01_quality import D01QualityContract, PolDenseSemanticEncoder
from doc2query.evaluation.retrieval import distribution, percentile
from doc2query.text.normalization import SimplePolishNormalizer
from doc2query.utils.records import read_records, write_json

USEFULNESS_CONTRACT = "task05-d01b-usefulness-hybrid-v1"
_FEASIBILITY_METRICS = (
    "pool_recall_at_1",
    "corpus_round_trip_at_20",
    "sentence_level_source_hit",
    "format_valid",
)
_REPORT_METRICS = (
    *_FEASIBILITY_METRICS,
    "shadow_pool_recall_at_1",
    "pool_margin",
    "shadow_pool_margin",
    "copy_density",
    "judge_rank_disagreement",
)
_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _project_root(path: Path) -> Path:
    root = next(
        (parent for parent in path.resolve().parents if (parent / "AGENTS.md").is_file()),
        None,
    )
    if root is None:
        raise ValueError("cannot resolve project root for D01b usefulness contract")
    return root


@dataclass(frozen=True)
class D01UsefulnessContract:
    """Validated retrospective-only selector contract."""

    payload: dict[str, Any]
    fingerprint: str
    path: Path
    project_root: Path

    @classmethod
    def load(cls, path: Path) -> D01UsefulnessContract:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("contract") != USEFULNESS_CONTRACT:
            raise ValueError("unsupported D01b usefulness contract")
        if raw.get("schema_version") != 1 or raw.get("final_tests_used") != []:
            raise ValueError("D01b usefulness contract must be schema v1 and dev-only")
        if raw.get("status") != "retrospective_development_diagnostic_only":
            raise ValueError("D01b existing-dev selector must remain retrospective-only")
        if raw.get("frozen_subset") != "dev_intrinsic_rank10":
            raise ValueError("D01b usefulness contract must use dev_intrinsic_rank10")
        root = _project_root(path)
        for section_name in ("adr", "natural_primary_scores", "copy_semantic_contract"):
            section = raw.get(section_name)
            if not isinstance(section, Mapping):
                raise ValueError(f"D01b usefulness contract omits {section_name}")
            artifact = root / str(section.get("path", ""))
            if not artifact.is_file() or _file_sha256(artifact) != str(section.get("sha256", "")):
                raise ValueError(f"D01b {section_name} fingerprint mismatch")
        natural = cast(Mapping[str, Any], raw["natural_primary_scores"])
        if (
            natural.get("judge") != "sdadas/polish-reranker-roberta-v3"
            or natural.get("revision") != "e6471da541f4e7be33845b6d57248a8d8bde27e8"
            or natural.get("schema") != "possible_false_negative_dev_scores_v1"
            or not bool(natural.get("same_positive_and_intersecting_negatives"))
        ):
            raise ValueError("D01b natural primary reference drifted")
        selection = raw.get("selection")
        evaluation = raw.get("evaluation")
        copy_risk = raw.get("copy_risk")
        semantic = raw.get("semantic_model")
        if not all(
            isinstance(item, Mapping) for item in (selection, evaluation, copy_risk, semantic)
        ):
            raise ValueError("D01b selector sections are incomplete")
        selection = cast(Mapping[str, Any], selection)
        evaluation = cast(Mapping[str, Any], evaluation)
        semantic = cast(Mapping[str, Any], semantic)
        if int(selection.get("candidate_count", 0)) != 8 or int(
            selection.get("selected_count", 0)
        ) != 4:
            raise ValueError("D01b selector requires best-four-of-eight")
        if tuple(selection.get("feasibility_not_below_anchor", [])) != _FEASIBILITY_METRICS:
            raise ValueError("D01b feasibility metrics drifted")
        expected_weights = {
            "natural_margin_alignment": 0.35,
            "semantic_diversity": 0.30,
            "lexical_diversity": 0.10,
            "corpus_specificity": 0.15,
            "low_copy_density": 0.10,
        }
        weights = selection.get("objective_weights")
        if not isinstance(weights, Mapping) or {
            str(key): float(value) for key, value in weights.items()
        } != expected_weights:
            raise ValueError("D01b objective weights drifted from the ADR")
        if (
            not bool(evaluation.get("shadow_reserved_from_selection"))
            or bool(evaluation.get("promotion_eligible"))
            or bool(evaluation.get("probe_materialization_authorized"))
            or not bool(evaluation.get("future_unseen_validation_required"))
        ):
            raise ValueError("D01b retrospective safety policy drifted")
        if (
            semantic.get("name_or_path") != "OPI-PIB/PolDense-150M"
            or semantic.get("revision") != "b94ea7f951cc480369a85fa9021694eef80c3a00"
            or semantic.get("similarity_prefix") != "[sts]: "
            or bool(semantic.get("trust_remote_code"))
        ):
            raise ValueError("D01b PolDense identity drifted")
        return cls(raw, _canonical_sha256(raw), path, root)

    @property
    def natural_scores_path(self) -> Path:
        section = cast(Mapping[str, Any], self.payload["natural_primary_scores"])
        return self.project_root / str(section["path"])

    @property
    def quality_contract_path(self) -> Path:
        section = cast(Mapping[str, Any], self.payload["copy_semantic_contract"])
        return self.project_root / str(section["path"])


@dataclass(frozen=True)
class _NaturalScores:
    positive: Mapping[str, float]
    negative: Mapping[str, float]


@dataclass(frozen=True)
class _Candidate:
    identity: str
    evaluation_id: str
    group_id: str
    example_id: str
    doc_id: str
    role: str
    experiment_id: str
    text: str
    requested_form: str
    requested_intent: str
    natural_margin: float
    margin_excess: float
    copy_risk: bool
    content_lemmas: frozenset[str]
    metrics: Mapping[str, float]
    corpus_effective_candidate_count: int
    corpus_candidate_count: int


class SemanticEncoder(Protocol):
    def encode(self, texts: Sequence[str], *, batch_size: int) -> np.ndarray: ...


def _load_natural_scores(path: Path, expected_judge: str) -> dict[str, _NaturalScores]:
    result: dict[str, _NaturalScores] = {}
    for row in read_records(path):
        if row.get("schema") != "possible_false_negative_dev_scores_v1":
            raise ValueError("natural primary score schema drifted")
        if row.get("judge") != expected_judge:
            raise ValueError("natural and synthetic primary judges differ")
        query_id = str(row["query_id"])
        if query_id in result:
            raise ValueError("natural primary query IDs must be unique")
        positive_ids = cast(Sequence[Any], row["positive_doc_ids"])
        positive_scores = cast(Sequence[Any], row["positive_scores"])
        negative_ids = cast(Sequence[Any], row["negative_doc_ids"])
        negative_scores = cast(Sequence[Any], row["negative_scores"])
        if len(positive_ids) != len(positive_scores) or len(negative_ids) != len(
            negative_scores
        ):
            raise ValueError("natural primary score arrays are misaligned")
        result[query_id] = _NaturalScores(
            positive={
                str(key): float(value)
                for key, value in zip(positive_ids, positive_scores, strict=True)
            },
            negative={
                str(key): float(value)
                for key, value in zip(negative_ids, negative_scores, strict=True)
            },
        )
    if not result:
        raise ValueError("natural primary score artifact is empty")
    return result


def _copy_risk(row: Mapping[str, Any], thresholds: Mapping[str, Any]) -> bool:
    query_words = int(row["word_length"])
    if query_words < int(thresholds["minimum_query_words"]):
        return False
    positive = row.get("positive")
    if not isinstance(positive, Mapping) or not isinstance(positive.get("text"), str):
        raise ValueError("D01b candidate omits positive passage text")
    passage_words = max(1, len(_WORD_RE.findall(str(positive["text"]))))
    joint_overlap = (
        float(row["copy_density"]) > float(thresholds["copy_density"])
        and float(row["normalized_lcs"]) > float(thresholds["normalized_lcs"])
    )
    long_span = float(row["longest_copied_ngram"]) > float(
        thresholds["longest_copied_ngram"]
    )
    passage_like = query_words / passage_words > float(
        thresholds["query_to_passage_length_ratio"]
    )
    return joint_overlap or long_span or passage_like


def _compact_candidates(
    path: Path,
    *,
    role: str,
    natural_scores: Mapping[str, _NaturalScores],
    copy_thresholds: Mapping[str, Any],
) -> list[_Candidate]:
    normalizer = SimplePolishNormalizer()
    candidates: list[_Candidate] = []
    for row in read_records(path):
        if row.get("final_tests_used") != []:
            raise ValueError("D01b candidate used a final-test subset")
        example_id = str(row["example_id"])
        natural = natural_scores.get(example_id)
        if natural is None:
            raise ValueError(f"missing frozen natural primary scores for {example_id}")
        doc_id = str(row["doc_id"])
        positive_score = natural.positive.get(doc_id)
        if positive_score is None:
            raise ValueError(f"natural primary scores omit positive {example_id}:{doc_id}")
        hard_negatives = row.get("hard_negatives")
        if not isinstance(hard_negatives, Sequence):
            raise ValueError("D01b candidate omits inherited hard negatives")
        negative_ids = [
            str(item["doc_id"])
            for item in hard_negatives
            if isinstance(item, Mapping) and str(item["doc_id"]) in natural.negative
        ]
        if not negative_ids:
            raise ValueError(f"no natural/synthetic negative intersection for {example_id}")
        natural_margin = float(positive_score) - max(natural.negative[key] for key in negative_ids)
        text = str(row["generated"])
        metrics = {
            name: float(bool(row[name])) if name == "format_valid" else float(row[name])
            for name in _REPORT_METRICS
        }
        experiment_id = str(row["experiment_id"])
        evaluation_id = str(row["evaluation_id"])
        identity = f"{role}:{experiment_id}:{evaluation_id}"
        candidates.append(
            _Candidate(
                identity=identity,
                evaluation_id=evaluation_id,
                group_id=str(row["evaluation_group_id"]),
                example_id=example_id,
                doc_id=doc_id,
                role=role,
                experiment_id=experiment_id,
                text=text,
                requested_form=str(row.get("requested_form") or "uncontrolled"),
                requested_intent=str(row.get("requested_intent") or "uncontrolled"),
                natural_margin=natural_margin,
                margin_excess=float(row["pool_margin"]) - natural_margin,
                copy_risk=_copy_risk(row, copy_thresholds),
                content_lemmas=frozenset(normalizer.analyze(text).content_lemmas),
                metrics=metrics,
                corpus_effective_candidate_count=int(row["corpus_effective_candidate_count"]),
                corpus_candidate_count=int(row["corpus_candidate_count"]),
            )
        )
    if len({item.identity for item in candidates}) != len(candidates):
        raise ValueError("D01b candidate identities must be unique")
    return candidates


def _embedding_identity(candidates: Sequence[_Candidate], contract: D01UsefulnessContract) -> str:
    digest = hashlib.sha256(contract.fingerprint.encode())
    for candidate in candidates:
        digest.update(candidate.identity.encode())
        digest.update(b"\0")
        digest.update(candidate.text.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _load_or_encode(
    candidates: Sequence[_Candidate],
    *,
    contract: D01UsefulnessContract,
    cache_dir: Path,
    device: str,
    encoder: SemanticEncoder | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    identity = _embedding_identity(candidates, contract)
    array_path = cache_dir / f"candidates.{identity[:16]}.npy"
    manifest_path = cache_dir / f"candidates.{identity[:16]}.json"
    if array_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        values = np.load(array_path)
        if (
            manifest.get("identity_sha256") == identity
            and int(manifest.get("row_count", -1)) == len(candidates)
            and values.shape[0] == len(candidates)
        ):
            return np.asarray(values, dtype=np.float32), {**manifest, "cache_hit": True}
    semantic = cast(Mapping[str, Any], contract.payload["semantic_model"])
    active_encoder = encoder
    if active_encoder is None:
        quality = D01QualityContract.load(contract.quality_contract_path)
        active_encoder = PolDenseSemanticEncoder(quality, device=device)
    values = np.asarray(
        active_encoder.encode(
            [candidate.text for candidate in candidates],
            batch_size=int(semantic["batch_size"]),
        ),
        dtype=np.float32,
    )
    if values.ndim != 2 or values.shape[0] != len(candidates) or not np.isfinite(values).all():
        raise ValueError("D01b semantic encoder returned invalid embeddings")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("D01b semantic encoder returned zero embeddings")
    values = values / norms
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = array_path.with_suffix(".npy.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, array_path)
    manifest = {
        "contract": "task05-d01b-semantic-cache-v1",
        "identity_sha256": identity,
        "row_count": len(candidates),
        "embedding_dimension": int(values.shape[1]),
        "dtype": "float32_normalized",
        "array": str(array_path),
        "cache_hit": False,
        "final_tests_used": [],
    }
    write_json(manifest_path, manifest)
    return values, manifest


def _mean_pairwise_cosine(indices: Sequence[int], embeddings: np.ndarray) -> float:
    pairs = list(combinations(indices, 2))
    return fmean(float(np.dot(embeddings[left], embeddings[right])) for left, right in pairs)


def _lemma_jaccard(left: _Candidate, right: _Candidate) -> float:
    union = left.content_lemmas | right.content_lemmas
    return len(left.content_lemmas & right.content_lemmas) / len(union) if union else 1.0


def _lexical_diversity(items: Sequence[_Candidate]) -> float:
    pairs = list(combinations(items, 2))
    return 1.0 - fmean(_lemma_jaccard(left, right) for left, right in pairs)


def _group_metric(items: Sequence[_Candidate], metric: str) -> float:
    return fmean(float(item.metrics[metric]) for item in items)


def _objective(
    items: Sequence[_Candidate],
    *,
    indices: Sequence[int],
    embeddings: np.ndarray,
    margin_scale: float,
    weights: Mapping[str, Any],
) -> tuple[float, dict[str, float]]:
    alignment = fmean(math.exp(-abs(item.margin_excess) / margin_scale) for item in items)
    mean_cosine = _mean_pairwise_cosine(indices, embeddings)
    semantic_diversity = min(1.0, max(0.0, 1.0 - mean_cosine))
    lexical_diversity = _lexical_diversity(items)
    specificity = fmean(
        1.0
        - math.log1p(item.corpus_effective_candidate_count)
        / math.log1p(item.corpus_candidate_count)
        for item in items
    )
    low_copy = 1.0 - fmean(item.metrics["copy_density"] for item in items)
    components = {
        "natural_margin_alignment": alignment,
        "semantic_diversity": semantic_diversity,
        "lexical_diversity": lexical_diversity,
        "corpus_specificity": specificity,
        "low_copy_density": low_copy,
        "mean_pairwise_cosine": mean_cosine,
    }
    score = sum(float(weights[name]) * components[name] for name in weights)
    return score, components


def _feasible(items: Sequence[_Candidate], anchor: Sequence[_Candidate]) -> bool:
    if sum(item.copy_risk for item in items) > sum(item.copy_risk for item in anchor):
        return False
    return all(
        _group_metric(items, metric) + 1e-12 >= _group_metric(anchor, metric)
        for metric in _FEASIBILITY_METRICS
    )


def _select_groups(
    candidates: Sequence[_Candidate],
    embeddings: np.ndarray,
    *,
    margin_scale: float,
    weights: Mapping[str, Any],
) -> tuple[dict[str, tuple[_Candidate, ...]], dict[str, dict[str, float]], int]:
    by_group: dict[str, list[tuple[int, _Candidate]]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        by_group[candidate.group_id].append((index, candidate))
    selected: dict[str, tuple[_Candidate, ...]] = {}
    objectives: dict[str, dict[str, float]] = {}
    changed = 0
    for group_id in sorted(by_group):
        pool = sorted(by_group[group_id], key=lambda pair: pair[1].identity)
        if len(pool) != 8:
            raise ValueError(f"D01b group {group_id} does not have exactly eight candidates")
        anchor_pairs = [pair for pair in pool if pair[1].role == "baseline"]
        if len(anchor_pairs) != 4:
            raise ValueError(f"D01b group {group_id} does not have four baseline anchors")
        anchor = tuple(pair[1] for pair in anchor_pairs)
        best_items: tuple[_Candidate, ...] | None = None
        best_components: dict[str, float] | None = None
        best_score = -math.inf
        for subset in combinations(pool, 4):
            items = tuple(pair[1] for pair in subset)
            if not _feasible(items, anchor):
                continue
            score, components = _objective(
                items,
                indices=[pair[0] for pair in subset],
                embeddings=embeddings,
                margin_scale=margin_scale,
                weights=weights,
            )
            if score > best_score + 1e-12:
                best_items = items
                best_components = {"score": score, **components}
                best_score = score
        if best_items is None or best_components is None:
            raise RuntimeError("baseline anchor should make every D01b group feasible")
        selected[group_id] = best_items
        objectives[group_id] = best_components
        if {item.identity for item in best_items} != {item.identity for item in anchor}:
            changed += 1
    return selected, objectives, changed


def _difficulty_report(candidates: Sequence[_Candidate]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[_Candidate]] = defaultdict(list)
    for item in candidates:
        grouped[(item.role, item.requested_form, item.requested_intent)].append(item)

    def summarize(items: Sequence[_Candidate]) -> dict[str, Any]:
        excess = [item.margin_excess for item in items]
        return {
            "count": len(items),
            "margin_excess": distribution(excess),
            "easier_than_natural_rate": sum(value > 0 for value in excess) / len(excess),
            "primary_margin": distribution([item.metrics["pool_margin"] for item in items]),
            "natural_margin": distribution([item.natural_margin for item in items]),
            "copy_density": distribution([item.metrics["copy_density"] for item in items]),
            "sentence_level_source_hit": fmean(
                item.metrics["sentence_level_source_hit"] for item in items
            ),
            "corpus_effective_candidate_count": distribution(
                [float(item.corpus_effective_candidate_count) for item in items]
            ),
        }

    result: dict[str, Any] = {}
    for (role, form, intent), items in sorted(grouped.items()):
        result[f"{role}/{form}/{intent}"] = summarize(items)
    for role in ("baseline", "controlled"):
        items = [item for item in candidates if item.role == role]
        result[f"{role}/all"] = summarize(items)
    return result


def _selection_report(
    candidates: Sequence[_Candidate],
    selected: Mapping[str, Sequence[_Candidate]],
    embeddings: np.ndarray,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    index = {item.identity: position for position, item in enumerate(candidates)}
    by_group: dict[str, list[_Candidate]] = defaultdict(list)
    for item in candidates:
        by_group[item.group_id].append(item)
    anchors = {
        group_id: tuple(item for item in items if item.role == "baseline")
        for group_id, items in by_group.items()
    }
    paired: dict[str, Any] = {}
    for metric in _REPORT_METRICS:
        left = {group_id: _group_metric(items, metric) for group_id, items in anchors.items()}
        right = {
            group_id: _group_metric(items, metric) for group_id, items in selected.items()
        }
        paired[metric] = paired_bootstrap(
            left, right, samples=bootstrap_samples, seed=bootstrap_seed
        )
    for metric in ("semantic_diversity", "lexical_diversity"):
        if metric == "semantic_diversity":
            left = {
                group_id: 1.0
                - _mean_pairwise_cosine([index[item.identity] for item in items], embeddings)
                for group_id, items in anchors.items()
            }
            right = {
                group_id: 1.0
                - _mean_pairwise_cosine([index[item.identity] for item in items], embeddings)
                for group_id, items in selected.items()
            }
        else:
            left = {group_id: _lexical_diversity(items) for group_id, items in anchors.items()}
            right = {group_id: _lexical_diversity(items) for group_id, items in selected.items()}
        paired[metric] = paired_bootstrap(
            left, right, samples=bootstrap_samples, seed=bootstrap_seed
        )
    selected_items = [item for items in selected.values() for item in items]
    composition = Counter(
        f"{item.role}/{item.requested_form}/{item.requested_intent}" for item in selected_items
    )
    return {
        "paired_group_bootstrap_selected_minus_anchor": paired,
        "selected_composition": dict(sorted(composition.items())),
        "controlled_selected_rate": sum(item.role == "controlled" for item in selected_items)
        / len(selected_items),
        "selected_copy_risk_rate": sum(item.copy_risk for item in selected_items)
        / len(selected_items),
        "anchor_copy_risk_rate": sum(
            item.copy_risk for items in anchors.values() for item in items
        )
        / len(selected_items),
    }


def _write_selected(
    path: Path,
    selected: Mapping[str, Sequence[_Candidate]],
    objectives: Mapping[str, Mapping[str, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for group_id in sorted(selected):
            for rank, item in enumerate(selected[group_id]):
                handle.write(
                    json.dumps(
                        {
                            "contract": USEFULNESS_CONTRACT,
                            "status": "retrospective_diagnostic_selection",
                            "selection_rank": rank,
                            "evaluation_group_id": group_id,
                            "evaluation_id": item.evaluation_id,
                            "candidate_identity": item.identity,
                            "role": item.role,
                            "experiment_id": item.experiment_id,
                            "generated": item.text,
                            "requested_form": item.requested_form,
                            "requested_intent": item.requested_intent,
                            "natural_margin": item.natural_margin,
                            "synthetic_margin": item.metrics["pool_margin"],
                            "margin_excess": item.margin_excess,
                            "copy_risk": item.copy_risk,
                            "group_objective": dict(objectives[group_id]),
                            "promotion_eligible": False,
                            "probe_materialization_authorized": False,
                            "final_tests_used": [],
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    selection = cast(Mapping[str, Any], report["selection"])
    paired = cast(
        Mapping[str, Mapping[str, Any]],
        selection["paired_group_bootstrap_selected_minus_anchor"],
    )
    lines = [
        "# D01b retrospective usefulness and hybrid-selection diagnostic",
        "",
        f"- Status: `{report['status']}`",
        "- Promotion eligible: `false`",
        "- Probe materialization authorized: `false`",
        f"- Passage groups: `{report['group_count']}`",
        f"- Groups changed from the baseline anchor: `{report['changed_group_count']}`",
        f"- Controlled selected rate: `{float(selection['controlled_selected_rate']):.6f}`",
        "",
        "## Selected minus baseline anchor",
        "",
        "| Metric | Difference | 95% CI |",
        "|---|---:|---:|",
    ]
    for metric, value in paired.items():
        lines.append(
            f"| {metric} | {float(value['difference']):.6f} | "
            f"[{float(value['ci95_low']):.6f}, {float(value['ci95_high']):.6f}] |"
        )
    lines.extend(
        [
            "",
            "## Research-safety interpretation",
            "",
            "This report was designed after inspecting the existing D01 dev aggregate. It is an "
            "engineering diagnostic only and cannot reopen the failed D01 gate. A separate unseen "
            "development validation is required before equal-budget probe materialization.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_usefulness_and_select(
    *,
    contract_path: Path,
    baseline_rows_path: Path,
    controlled_rows_path: Path,
    output_json: Path,
    output_markdown: Path,
    output_selected: Path,
    semantic_cache_dir: Path,
    semantic_device: str = "cuda",
    semantic_encoder: SemanticEncoder | None = None,
) -> dict[str, Any]:
    """Diagnose easy-vs-grounded queries and select a safe-anchor hybrid K=4."""
    contract = D01UsefulnessContract.load(contract_path)
    natural_section = cast(Mapping[str, Any], contract.payload["natural_primary_scores"])
    natural = _load_natural_scores(contract.natural_scores_path, str(natural_section["judge"]))
    copy_thresholds = cast(Mapping[str, Any], contract.payload["copy_risk"])
    baseline = _compact_candidates(
        baseline_rows_path,
        role="baseline",
        natural_scores=natural,
        copy_thresholds=copy_thresholds,
    )
    controlled = _compact_candidates(
        controlled_rows_path,
        role="controlled",
        natural_scores=natural,
        copy_thresholds=copy_thresholds,
    )
    candidates = sorted([*baseline, *controlled], key=lambda item: item.identity)
    baseline_groups = {item.group_id for item in baseline}
    controlled_groups = {item.group_id for item in controlled}
    if baseline_groups != controlled_groups or not baseline_groups:
        raise ValueError("D01b requires identical non-empty baseline/controlled groups")
    if len(baseline) != 4 * len(baseline_groups) or len(controlled) != len(baseline):
        raise ValueError("D01b requires exact K=4 in both source arms")
    natural_margin_by_group: dict[str, float] = {}
    for item in baseline:
        previous = natural_margin_by_group.setdefault(item.group_id, item.natural_margin)
        if abs(previous - item.natural_margin) > 1e-12:
            raise ValueError("natural margin differs within a matched passage group")
    unique_natural_margins = [
        natural_margin_by_group[group_id] for group_id in sorted(baseline_groups)
    ]
    q25 = percentile(unique_natural_margins, 0.25)
    q75 = percentile(unique_natural_margins, 0.75)
    if q25 is None or q75 is None:
        raise RuntimeError("cannot calibrate natural-margin scale")
    margin_scale = max(1e-6, q75 - q25)
    embeddings, cache = _load_or_encode(
        candidates,
        contract=contract,
        cache_dir=semantic_cache_dir,
        device=semantic_device,
        encoder=semantic_encoder,
    )
    selection_section = cast(Mapping[str, Any], contract.payload["selection"])
    weights = cast(Mapping[str, Any], selection_section["objective_weights"])
    selected, objectives, changed = _select_groups(
        candidates,
        embeddings,
        margin_scale=margin_scale,
        weights=weights,
    )
    evaluation = cast(Mapping[str, Any], contract.payload["evaluation"])
    selection_report = _selection_report(
        candidates,
        selected,
        embeddings,
        bootstrap_samples=int(evaluation["bootstrap_samples"]),
        bootstrap_seed=int(evaluation["bootstrap_seed"]),
    )
    _write_selected(output_selected, selected, objectives)
    report = {
        "schema_version": 1,
        "contract": USEFULNESS_CONTRACT,
        "contract_fingerprint": contract.fingerprint,
        "status": "retrospective_exploratory_complete",
        "frozen_subset": "dev_intrinsic_rank10",
        "group_count": len(baseline_groups),
        "candidate_count": len(candidates),
        "selected_count": sum(len(items) for items in selected.values()),
        "changed_group_count": changed,
        "natural_margin_scale_iqr": margin_scale,
        "difficulty_diagnostic": _difficulty_report(candidates),
        "semantic_cache": cache,
        "selection": selection_report,
        "selected_rows": str(output_selected),
        "selected_rows_sha256": _file_sha256(output_selected),
        "selection_uses_shadow": False,
        "promotion_eligible": False,
        "probe_materialization_authorized": False,
        "future_unseen_validation_required": True,
        "final_tests_used": [],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_json, report)
    _write_markdown(output_markdown, report)
    return report
