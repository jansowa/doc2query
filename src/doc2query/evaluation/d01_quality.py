"""Prospective anti-copy and semantic-diversity gates for matched D01 comparisons."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from statistics import fmean
from typing import Any, Protocol, cast

import numpy as np
import yaml

from doc2query.evaluation.bootstrap import paired_bootstrap
from doc2query.evaluation.retrieval import distribution
from doc2query.rewards.calibration import quantile
from doc2query.rewards.lexical import lexical_metrics
from doc2query.text.normalization import SimplePolishNormalizer
from doc2query.utils.records import read_records, write_json

QUALITY_CONTRACT = "task05-d01-copy-semantic-quality-v1"


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_json(temporary, value)
    os.replace(temporary, path)


@dataclass(frozen=True)
class D01QualityContract:
    """Validated, fingerprinted Task-05 quality contract."""

    payload: dict[str, Any]
    fingerprint: str
    path: Path

    @classmethod
    def load(cls, path: Path) -> D01QualityContract:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("contract") != QUALITY_CONTRACT:
            raise ValueError("unsupported D01 copy/semantic quality contract")
        if raw.get("schema_version") != 1 or raw.get("final_tests_used") != []:
            raise ValueError("D01 quality contract must be schema v1 and dev-only")
        if raw.get("frozen_subset") != "dev_intrinsic_rank10":
            raise ValueError("D01 quality calibration must use dev_intrinsic_rank10")
        adr = raw.get("adr")
        if not isinstance(adr, Mapping):
            raise ValueError("D01 quality contract requires an ADR")
        root = next(
            (parent for parent in path.resolve().parents if (parent / "AGENTS.md").is_file()),
            None,
        )
        if root is None:
            raise ValueError("cannot resolve project root for D01 quality ADR")
        adr_path = root / str(adr.get("path", ""))
        if not adr_path.is_file() or _file_sha256(adr_path) != str(adr.get("sha256", "")):
            raise ValueError("D01 quality ADR fingerprint mismatch")
        calibration = raw.get("natural_calibration")
        anti_copy = raw.get("anti_copy")
        semantic = raw.get("semantic_diversity")
        audit = raw.get("blind_audit")
        if not all(isinstance(item, Mapping) for item in (calibration, anti_copy, semantic, audit)):
            raise ValueError("D01 quality contract sections are incomplete")
        assert isinstance(calibration, Mapping)
        assert isinstance(anti_copy, Mapping)
        assert isinstance(semantic, Mapping)
        assert isinstance(audit, Mapping)
        for name in ("copy_density", "normalized_lcs", "longest_copied_ngram"):
            if name not in cast(Sequence[str], calibration.get("upper_tail_metrics", [])):
                raise ValueError(f"natural calibration omits {name}")
        if (
            float(calibration.get("upper_quantile", 0.0)) != 0.95
            or float(calibration.get("length_ratio_upper_quantile", 0.0)) != 0.99
        ):
            raise ValueError("D01 natural copy quantiles drifted from the ADR")
        if (
            int(anti_copy.get("minimum_query_words", 0)) != 4
            or float(anti_copy.get("max_natural_tail_excess", -1.0)) != 0.05
            or float(anti_copy.get("variant_vs_baseline_margin", -1.0)) != 0.02
        ):
            raise ValueError("D01 anti-copy rules drifted from the ADR")
        model = semantic.get("model")
        if not isinstance(model, Mapping) or len(str(model.get("revision", ""))) != 40:
            raise ValueError("semantic model must have a full pinned revision")
        if (
            model.get("name_or_path") != "OPI-PIB/PolDense-150M"
            or model.get("revision") != "b94ea7f951cc480369a85fa9021694eef80c3a00"
            or model.get("license") != "gemma"
        ):
            raise ValueError("semantic model identity drifted from the ADR")
        if bool(model.get("trust_remote_code")):
            raise ValueError("semantic model may not require trust_remote_code")
        if str(semantic.get("similarity_prefix")) != "[sts]: ":
            raise ValueError("query-query similarity must use the PolDense [sts] prefix")
        if str(semantic.get("retrieval_query_prefix")) != "[query]: ":
            raise ValueError("PolDense retrieval queries must use the [query] prefix")
        if (
            float(semantic.get("cluster_cosine_threshold", 0.0)) != 0.85
            or float(semantic.get("minimum_common_clean_group_rate", 0.0)) != 0.80
            or float(semantic.get("cluster_count_noninferiority_margin", -1.0)) != 0.10
            or float(semantic.get("max_pairwise_cosine_margin", -1.0)) != 0.02
        ):
            raise ValueError("semantic diversity guardrails drifted from the ADR")
        if int(audit.get("sample_size", 0)) != 100 or not bool(
            audit.get("automatic_labels_prohibited")
        ):
            raise ValueError("blind anti-copy audit drifted from the ADR")
        return cls(payload=raw, fingerprint=_canonical_sha256(raw), path=path)

    def reference(self) -> dict[str, Any]:
        adr = cast(Mapping[str, Any], self.payload["adr"])
        semantic = cast(Mapping[str, Any], self.payload["semantic_diversity"])
        model = cast(Mapping[str, Any], semantic["model"])
        return {
            "contract": QUALITY_CONTRACT,
            "contract_fingerprint": self.fingerprint,
            "path": str(self.path),
            "adr_id": str(adr["id"]),
            "adr_version": str(adr["version"]),
            "adr_sha256": str(adr["sha256"]),
            "semantic_model": dict(model),
            "similarity_prefix": str(semantic["similarity_prefix"]),
            "retrieval_query_prefix": str(semantic["retrieval_query_prefix"]),
            "final_tests_used": [],
        }


class SemanticEncoder(Protocol):
    """Small injectable surface used by tests and the PolDense backend."""

    def encode(self, texts: Sequence[str], *, batch_size: int) -> np.ndarray: ...


class PolDenseSemanticEncoder:
    """Pinned SentenceTransformers wrapper for symmetric query-query similarity."""

    def __init__(self, contract: D01QualityContract, *, device: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("install sentence-transformers to run D01 semantic gates") from exc
        semantic = cast(Mapping[str, Any], contract.payload["semantic_diversity"])
        model = cast(Mapping[str, Any], semantic["model"])
        dtype = str(model.get("dtype", "float32")) if device != "cpu" else "float32"
        self._prefix = str(semantic["similarity_prefix"])
        self._model = SentenceTransformer(
            str(model["name_or_path"]),
            revision=str(model["revision"]),
            trust_remote_code=bool(model["trust_remote_code"]),
            device=device,
            model_kwargs={
                "dtype": dtype,
                "attn_implementation": str(model.get("attention_implementation", "sdpa")),
            },
        )

    def encode(self, texts: Sequence[str], *, batch_size: int) -> np.ndarray:
        values = self._model.encode(
            [self._prefix + text for text in texts],
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        return np.asarray(values, dtype=np.float32)


@dataclass(frozen=True)
class _CompactRow:
    evaluation_id: str
    group_id: str
    generated: str
    reference: str
    passage: str
    copy_density: float
    normalized_lcs: float
    longest_copied_ngram: float
    query_words: int
    pool_recall_at_1: float


def _compact_rows(path: Path) -> list[_CompactRow]:
    rows: list[_CompactRow] = []
    for row in read_records(path):
        positive = row.get("positive")
        if not isinstance(positive, Mapping) or not isinstance(positive.get("text"), str):
            raise ValueError("quality gate requires the positive passage in every scoring row")
        required = ("copy_density", "normalized_lcs", "longest_copied_ngram", "word_length")
        if any(not isinstance(row.get(field), (int, float)) for field in required):
            raise ValueError("quality gate requires complete lexical-copy metrics")
        rows.append(
            _CompactRow(
                evaluation_id=str(row.get("evaluation_id", "")),
                group_id=str(row.get("evaluation_group_id", "")),
                generated=str(row.get("generated", "")),
                reference=str(row.get("reference", "")),
                passage=str(positive["text"]),
                copy_density=float(row["copy_density"]),
                normalized_lcs=float(row["normalized_lcs"]),
                longest_copied_ngram=float(row["longest_copied_ngram"]),
                query_words=int(row["word_length"]),
                pool_recall_at_1=float(row.get("pool_recall_at_1", 0.0)),
            )
        )
    if not rows or any(not row.evaluation_id or not row.group_id for row in rows):
        raise ValueError("quality gate requires non-empty stable row identities")
    if len({row.evaluation_id for row in rows}) != len(rows):
        raise ValueError("quality gate requires unique evaluation IDs")
    return rows


def _natural_calibration(
    rows: Sequence[_CompactRow], contract: D01QualityContract
) -> tuple[dict[str, Any], dict[str, float], dict[str, int]]:
    config = cast(Mapping[str, Any], contract.payload["natural_calibration"])
    normalizer = SimplePolishNormalizer()
    natural_values: dict[str, list[float]] = defaultdict(list)
    passage_words: dict[str, int] = {}
    seen: set[str] = set()
    for row in rows:
        if row.group_id in seen:
            continue
        seen.add(row.group_id)
        query = normalizer.analyze(row.reference)
        passage = normalizer.analyze(row.passage)
        metrics = lexical_metrics(query, passage)
        passage_count = max(1, len(passage.tokens))
        passage_words[row.group_id] = passage_count
        natural_values["copy_density"].append(metrics.copy_density)
        natural_values["normalized_lcs"].append(metrics.normalized_lcs)
        natural_values["longest_copied_ngram"].append(float(metrics.longest_copied_ngram))
        natural_values["query_to_passage_length_ratio"].append(len(query.tokens) / passage_count)
    upper = float(config["upper_quantile"])
    ratio_upper = float(config["length_ratio_upper_quantile"])
    thresholds = {
        name: quantile(values, ratio_upper if name == "query_to_passage_length_ratio" else upper)
        for name, values in natural_values.items()
    }
    report = {
        "source": "matched natural references from frozen dev; D01 outputs excluded",
        "group_count": len(seen),
        "upper_quantile": upper,
        "length_ratio_upper_quantile": ratio_upper,
        "thresholds": thresholds,
        "distributions": {name: distribution(values) for name, values in natural_values.items()},
        "final_tests_used": [],
    }
    return report, thresholds, passage_words


def _copy_risk(
    row: _CompactRow,
    *,
    thresholds: Mapping[str, float],
    passage_words: int,
    minimum_query_words: int,
) -> bool:
    if row.query_words < minimum_query_words:
        return False
    joint_overlap = (
        row.copy_density > thresholds["copy_density"]
        and row.normalized_lcs > thresholds["normalized_lcs"]
    )
    long_span = row.longest_copied_ngram > thresholds["longest_copied_ngram"]
    passage_like_length = (
        row.query_words / max(1, passage_words) > thresholds["query_to_passage_length_ratio"]
    )
    return joint_overlap or long_span or passage_like_length


def _group_means(rows: Sequence[_CompactRow], values: Sequence[float]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row, value in zip(rows, values, strict=True):
        grouped[row.group_id].append(value)
    return {group_id: fmean(items) for group_id, items in grouped.items()}


def _embedding_identity(rows: Sequence[_CompactRow], contract: D01QualityContract) -> str:
    semantic = cast(Mapping[str, Any], contract.payload["semantic_diversity"])
    model = cast(Mapping[str, Any], semantic["model"])
    digest = hashlib.sha256()
    digest.update(_canonical_sha256(model).encode())
    digest.update(str(semantic["similarity_prefix"]).encode())
    for row in rows:
        digest.update(row.evaluation_id.encode())
        digest.update(b"\0")
        digest.update(row.generated.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _safe_label(label: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", label).strip("-") or "arm"


def _cached_embeddings(
    rows: Sequence[_CompactRow],
    *,
    label: str,
    contract: D01QualityContract,
    cache_dir: Path,
    batch_size: int,
    encoder_holder: list[SemanticEncoder],
    device: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    identity = _embedding_identity(rows, contract)
    stem = f"{_safe_label(label)}.{identity[:16]}"
    array_path = cache_dir / f"{stem}.npy"
    manifest_path = cache_dir / f"{stem}.json"
    if array_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        values = np.load(array_path)
        if (
            manifest.get("identity_sha256") == identity
            and int(manifest.get("row_count", -1)) == len(rows)
            and values.shape[0] == len(rows)
        ):
            return np.asarray(values, dtype=np.float32), {**manifest, "cache_hit": True}
    if not encoder_holder:
        encoder_holder.append(PolDenseSemanticEncoder(contract, device=device))
    values = np.asarray(
        encoder_holder[0].encode([row.generated for row in rows], batch_size=batch_size),
        dtype=np.float32,
    )
    if values.ndim != 2 or values.shape[0] != len(rows) or not np.isfinite(values).all():
        raise ValueError("semantic encoder returned invalid embeddings")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("semantic encoder returned zero embeddings")
    values = values / norms
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = array_path.with_suffix(".npy.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, array_path)
    manifest = {
        "contract": "task05-d01-semantic-cache-v1",
        "identity_sha256": identity,
        "row_count": len(rows),
        "embedding_dimension": int(values.shape[1]),
        "dtype": "float32_normalized",
        "array": str(array_path),
        "cache_hit": False,
        "final_tests_used": [],
    }
    _atomic_json(manifest_path, manifest)
    return values, manifest


def _semantic_group_metrics(
    rows: Sequence[_CompactRow],
    embeddings: np.ndarray,
    *,
    clean_groups: set[str],
    cluster_threshold: float,
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row.group_id in clean_groups:
            grouped[row.group_id].append(index)
    result: dict[str, dict[str, float]] = defaultdict(dict)
    for group_id, indices in grouped.items():
        if len(indices) < 2:
            continue
        cosines = [
            float(np.dot(embeddings[left], embeddings[right]))
            for left, right in combinations(indices, 2)
        ]
        parents = list(range(len(indices)))

        def root(index: int, parents: list[int] = parents) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        for left, right in combinations(range(len(indices)), 2):
            if (
                float(np.dot(embeddings[indices[left]], embeddings[indices[right]]))
                >= cluster_threshold
            ):
                parents[root(right)] = root(left)
        result["mean_pairwise_cosine"][group_id] = fmean(cosines)
        result["max_pairwise_cosine"][group_id] = max(cosines)
        result["semantic_cluster_count"][group_id] = float(
            len({root(index) for index in range(len(indices))})
        )
    return dict(result)


def _residual_group_jaccard(
    rows: Sequence[_CompactRow], clean_groups: set[str]
) -> dict[str, float]:
    normalizer = SimplePolishNormalizer()
    grouped: dict[str, list[set[str]]] = defaultdict(list)
    passage_cache: dict[str, set[str]] = {}
    for row in rows:
        if row.group_id not in clean_groups:
            continue
        passage_lemmas = passage_cache.setdefault(
            row.group_id, set(normalizer.analyze(row.passage).content_lemmas)
        )
        query_lemmas = set(normalizer.analyze(row.generated).content_lemmas)
        grouped[row.group_id].append(query_lemmas - passage_lemmas)
    result: dict[str, float] = {}
    for group_id, values in grouped.items():
        pairs = list(combinations(values, 2))
        if not pairs:
            continue
        scores = [
            len(left & right) / len(left | right) if left | right else 1.0 for left, right in pairs
        ]
        result[group_id] = fmean(scores)
    return result


def _status(passed: bool) -> str:
    return "passed" if passed else "failed"


def _write_blind_audit(
    *,
    baseline: Sequence[_CompactRow],
    variant: Sequence[_CompactRow],
    baseline_risk: Sequence[float],
    variant_risk: Sequence[float],
    output_json: Path,
    contract: D01QualityContract,
) -> dict[str, Any]:
    config = cast(Mapping[str, Any], contract.payload["blind_audit"])
    sample_size = int(config["sample_size"])
    seed = int(config["seed"])
    candidates: list[tuple[str, _CompactRow, float]] = []
    for arm, rows, risks in (
        ("baseline", baseline, baseline_risk),
        ("variant", variant, variant_risk),
    ):
        for row, risk in zip(rows, risks, strict=True):
            if row.pool_recall_at_1 == 1.0:
                score = 10.0 * risk + row.copy_density + row.normalized_lcs
                candidates.append((arm, row, score))
    candidates.sort(key=lambda item: (-item[2], item[1].evaluation_id, item[0]))
    per_arm = max(1, sample_size // 2)
    selected: list[tuple[str, _CompactRow, float]] = []
    for arm in ("baseline", "variant"):
        selected.extend([item for item in candidates if item[0] == arm][:per_arm])
    selected = selected[:sample_size]
    rng = random.Random(seed)
    rng.shuffle(selected)
    csv_path = output_json.with_suffix(".anti_copy_audit.csv")
    key_path = output_json.with_suffix(".anti_copy_audit.key.json")
    instructions_path = output_json.with_suffix(".anti_copy_audit.instructions.md")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "audit_id",
        "passage",
        "query",
        "answerable_yes_no_uncertain",
        "natural_yes_no_uncertain",
        "excessive_copy_yes_no_uncertain",
        "useful_search_query_yes_no_uncertain",
        "evidence_fragment",
        "reviewer_id",
        "notes",
    )
    key_rows = []
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, (arm, row, score) in enumerate(selected, 1):
            audit_id = f"copy-{index:04d}"
            writer.writerow({"audit_id": audit_id, "passage": row.passage, "query": row.generated})
            key_rows.append(
                {
                    "audit_id": audit_id,
                    "arm": arm,
                    "evaluation_id": row.evaluation_id,
                    "selection_score": score,
                    "copy_density": row.copy_density,
                    "normalized_lcs": row.normalized_lcs,
                    "longest_copied_ngram": row.longest_copied_ngram,
                    "final_tests_used": [],
                }
            )
    _atomic_json(
        key_path,
        {
            "contract": "task05-d01-blind-copy-audit-key-v1",
            "seed": seed,
            "sample_size": len(selected),
            "rows": key_rows,
            "final_tests_used": [],
        },
    )
    instructions_path.write_text(
        "# D01 blind anti-copy audit\n\n"
        "Do not open the machine key before submitting ratings. Rate every row as "
        "`yes`, `no`, or `uncertain`; quote the shortest evidence fragment. The automatic "
        "selection targets high-retrieval/high-copy-risk cases and is not a quality label.\n",
        encoding="utf-8",
    )
    return {
        "status": "pending_human_review",
        "sample_size": len(selected),
        "blind_csv": str(csv_path),
        "machine_key": str(key_path),
        "instructions": str(instructions_path),
        "final_tests_used": [],
    }


def evaluate_copy_semantic_quality(
    *,
    baseline_rows_path: Path,
    variant_rows_path: Path,
    baseline_label: str,
    variant_label: str,
    contract_path: Path,
    output_json: Path,
    bootstrap_samples: int,
    bootstrap_seed: int,
    semantic_device: str = "cuda",
    encoder: SemanticEncoder | None = None,
) -> dict[str, Any]:
    """Evaluate natural-calibrated copying and clean-group semantic diversity."""
    contract = D01QualityContract.load(contract_path)
    baseline = _compact_rows(baseline_rows_path)
    variant = _compact_rows(variant_rows_path)
    if [row.evaluation_id for row in baseline] != [row.evaluation_id for row in variant]:
        raise ValueError("copy/semantic quality requires identical matched evaluation IDs")
    for left, right in zip(baseline, variant, strict=True):
        if (
            left.group_id != right.group_id
            or left.reference != right.reference
            or left.passage != right.passage
        ):
            raise ValueError("copy/semantic calibration references differ between arms")
    group_sizes: dict[str, int] = defaultdict(int)
    for row in baseline:
        group_sizes[row.group_id] += 1
    if set(group_sizes.values()) != {4}:
        raise ValueError("copy/semantic quality requires exact K=4 in every matched group")
    calibration, thresholds, passage_words = _natural_calibration(baseline, contract)
    anti_copy = cast(Mapping[str, Any], contract.payload["anti_copy"])
    minimum_words = int(anti_copy["minimum_query_words"])
    baseline_risk = [
        float(
            _copy_risk(
                row,
                thresholds=thresholds,
                passage_words=passage_words[row.group_id],
                minimum_query_words=minimum_words,
            )
        )
        for row in baseline
    ]
    variant_risk = [
        float(
            _copy_risk(
                row,
                thresholds=thresholds,
                passage_words=passage_words[row.group_id],
                minimum_query_words=minimum_words,
            )
        )
        for row in variant
    ]
    natural_rows: list[_CompactRow] = []
    seen: set[str] = set()
    normalizer = SimplePolishNormalizer()
    for row in baseline:
        if row.group_id in seen:
            continue
        seen.add(row.group_id)
        query = normalizer.analyze(row.reference)
        passage = normalizer.analyze(row.passage)
        metrics = lexical_metrics(query, passage)
        natural_rows.append(
            _CompactRow(
                evaluation_id=row.group_id,
                group_id=row.group_id,
                generated=row.reference,
                reference=row.reference,
                passage=row.passage,
                copy_density=metrics.copy_density,
                normalized_lcs=metrics.normalized_lcs,
                longest_copied_ngram=float(metrics.longest_copied_ngram),
                query_words=len(query.tokens),
                pool_recall_at_1=0.0,
            )
        )
    natural_risk = [
        float(
            _copy_risk(
                row,
                thresholds=thresholds,
                passage_words=passage_words[row.group_id],
                minimum_query_words=minimum_words,
            )
        )
        for row in natural_rows
    ]
    natural_rate = fmean(natural_risk)
    baseline_rate = fmean(baseline_risk)
    variant_rate = fmean(variant_risk)
    absolute_limit = natural_rate + float(anti_copy["max_natural_tail_excess"])
    risk_bootstrap = paired_bootstrap(
        _group_means(baseline, baseline_risk),
        _group_means(variant, variant_risk),
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    relative_margin = float(anti_copy["variant_vs_baseline_margin"])
    copy_guardrails = {
        "baseline_natural_tail": {
            "status": _status(baseline_rate <= absolute_limit),
            "rate": baseline_rate,
            "limit": absolute_limit,
        },
        "variant_natural_tail": {
            "status": _status(variant_rate <= absolute_limit),
            "rate": variant_rate,
            "limit": absolute_limit,
        },
        "variant_vs_baseline": {
            "status": _status(float(risk_bootstrap["ci95_high"]) <= relative_margin),
            "margin": relative_margin,
            "bootstrap": risk_bootstrap,
        },
    }
    baseline_group_risk = _group_means(baseline, baseline_risk)
    variant_group_risk = _group_means(variant, variant_risk)
    common_clean_groups = {
        group_id
        for group_id in baseline_group_risk
        if baseline_group_risk[group_id] == 0.0 and variant_group_risk[group_id] == 0.0
    }
    group_count = len(baseline_group_risk)
    clean_rate = len(common_clean_groups) / group_count
    semantic = cast(Mapping[str, Any], contract.payload["semantic_diversity"])
    minimum_clean_rate = float(semantic["minimum_common_clean_group_rate"])
    cluster_threshold = float(semantic["cluster_cosine_threshold"])
    semantic_bootstrap: dict[str, Any] = {}
    residual_bootstrap: dict[str, Any] | None = None
    baseline_cache: dict[str, Any] | None = None
    variant_cache: dict[str, Any] | None = None
    if common_clean_groups:
        cache_dir = output_json.parent / "semantic_cache"
        encoder_holder: list[SemanticEncoder] = [encoder] if encoder is not None else []
        batch_size = int(semantic["batch_size"])
        baseline_clean = [row for row in baseline if row.group_id in common_clean_groups]
        variant_clean = [row for row in variant if row.group_id in common_clean_groups]
        baseline_embeddings, baseline_cache = _cached_embeddings(
            baseline_clean,
            label=baseline_label,
            contract=contract,
            cache_dir=cache_dir,
            batch_size=batch_size,
            encoder_holder=encoder_holder,
            device=semantic_device,
        )
        variant_embeddings, variant_cache = _cached_embeddings(
            variant_clean,
            label=variant_label,
            contract=contract,
            cache_dir=cache_dir,
            batch_size=batch_size,
            encoder_holder=encoder_holder,
            device=semantic_device,
        )
        baseline_semantic = _semantic_group_metrics(
            baseline_clean,
            baseline_embeddings,
            clean_groups=common_clean_groups,
            cluster_threshold=cluster_threshold,
        )
        variant_semantic = _semantic_group_metrics(
            variant_clean,
            variant_embeddings,
            clean_groups=common_clean_groups,
            cluster_threshold=cluster_threshold,
        )
        semantic_bootstrap = {
            metric: paired_bootstrap(
                baseline_semantic[metric],
                variant_semantic[metric],
                samples=bootstrap_samples,
                seed=bootstrap_seed,
            )
            for metric in (
                "mean_pairwise_cosine",
                "max_pairwise_cosine",
                "semantic_cluster_count",
            )
        }
        baseline_residual = _residual_group_jaccard(baseline, common_clean_groups)
        variant_residual = _residual_group_jaccard(variant, common_clean_groups)
        residual_bootstrap = paired_bootstrap(
            baseline_residual,
            variant_residual,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        )
    cluster_bootstrap = semantic_bootstrap.get("semantic_cluster_count")
    cosine_bootstrap = semantic_bootstrap.get("max_pairwise_cosine")
    semantic_guardrails = {
        "common_clean_group_coverage": {
            "status": _status(clean_rate >= minimum_clean_rate),
            "rate": clean_rate,
            "minimum": minimum_clean_rate,
        },
        "cluster_count_noninferiority": {
            "status": _status(
                cluster_bootstrap is not None
                and float(cluster_bootstrap["ci95_low"])
                >= -float(semantic["cluster_count_noninferiority_margin"])
            ),
            "margin": float(semantic["cluster_count_noninferiority_margin"]),
            "bootstrap": cluster_bootstrap,
        },
        "max_cosine_noninferiority": {
            "status": _status(
                cosine_bootstrap is not None
                and float(cosine_bootstrap["ci95_high"])
                <= float(semantic["max_pairwise_cosine_margin"])
            ),
            "margin": float(semantic["max_pairwise_cosine_margin"]),
            "bootstrap": cosine_bootstrap,
        },
    }
    statuses = {
        str(item["status"]) for item in [*copy_guardrails.values(), *semantic_guardrails.values()]
    }
    decision = "continue" if statuses == {"passed"} else "stop"
    audit = _write_blind_audit(
        baseline=baseline,
        variant=variant,
        baseline_risk=baseline_risk,
        variant_risk=variant_risk,
        output_json=output_json,
        contract=contract,
    )
    return {
        "status": "measured",
        "decision": decision,
        "contract": contract.reference(),
        "natural_calibration": calibration,
        "anti_copy": {
            "risk_rule": (
                "minimum length AND ((copy_density>natural_q AND normalized_lcs>natural_q) "
                "OR longest_ngram>natural_q OR query/passage_length>natural_q)"
            ),
            "natural_reference_rate": natural_rate,
            "baseline_rate": baseline_rate,
            "variant_rate": variant_rate,
            "guardrails": copy_guardrails,
        },
        "semantic_diversity": {
            "scope": "intersection of groups with zero anti-copy-risk queries in both arms",
            "common_clean_group_count": len(common_clean_groups),
            "common_clean_group_rate": clean_rate,
            "cluster_cosine_threshold": cluster_threshold,
            "baseline_cache": baseline_cache,
            "variant_cache": variant_cache,
            "paired_bootstrap": semantic_bootstrap,
            "passage_lemma_removed_pairwise_jaccard": residual_bootstrap,
            "guardrails": semantic_guardrails,
            "model_note": (
                "[sts] is used for symmetric query-query similarity; [query] is reserved "
                "for asymmetric query-to-passage retrieval"
            ),
        },
        "blind_audit": audit,
        "final_tests_used": [],
    }
