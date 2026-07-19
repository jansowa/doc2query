"""Versioned hard-negative and possible-false-negative contracts for probe training."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from doc2query.reranker.base import PairScorer

HardNegativeStrategy = Literal["hn0", "hn0_filter", "hn1_bm25"]
FalseNegativePolicy = Literal["drop", "demote", "keep+log"]

CALIBRATION_SCHEMA_VERSION = 1
NEGATIVE_RECIPE_VERSION = "probe-negatives-v1"
CALIBRATION_ARTIFACT_TYPE = "possible_false_negative_threshold"
CALIBRATION_SCORE_KIND = "raw_pair_logit"
CALIBRATION_OPERATOR = "greater_than_or_equal"


class ProbeNegativeBlocker(RuntimeError):
    """A fail-closed P-03 preflight failure that must not be bypassed."""


def _is_sha256(value: str | None) -> bool:
    return bool(
        value and len(value) == 64 and all(character in "0123456789abcdef" for character in value)
    )


def _artifact_payload_fingerprint(payload: Mapping[str, Any]) -> str:
    canonical = {key: value for key, value in payload.items() if key != "artifact_fingerprint"}
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def calibration_artifact_fingerprint(payload: Mapping[str, Any]) -> str:
    """Return the canonical fingerprint stored in a Task 02 threshold artifact."""
    return _artifact_payload_fingerprint(payload)


@dataclass(frozen=True)
class PossibleFalseNegativeCalibration:
    """Validated Task 02 threshold fitted on a development split only."""

    artifact_id: str
    artifact_fingerprint: str
    fit_split: str
    fit_dataset_fingerprint: str
    primary_judge_name: str
    primary_judge_revision: str
    score_kind: str
    threshold: float
    selection_method: str
    source_scores_sha256: str

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        expected_id: str,
        expected_fingerprint: str,
    ) -> PossibleFalseNegativeCalibration:
        if not path.is_file():
            raise ProbeNegativeBlocker(
                "P-03 BLOCKED: the pinned Task 02 possible-false-negative calibration "
                f"artifact does not exist: {path}"
            )
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ProbeNegativeBlocker("P-03 BLOCKED: calibration artifact must be a JSON object")
        actual_fingerprint = _artifact_payload_fingerprint(raw)
        embedded_fingerprint = raw.get("artifact_fingerprint")
        if (
            raw.get("schema_version") != CALIBRATION_SCHEMA_VERSION
            or raw.get("artifact_type") != CALIBRATION_ARTIFACT_TYPE
        ):
            raise ProbeNegativeBlocker(
                "P-03 BLOCKED: unsupported possible-false-negative calibration schema"
            )
        if (
            not _is_sha256(expected_fingerprint)
            or actual_fingerprint != expected_fingerprint
            or embedded_fingerprint != expected_fingerprint
        ):
            raise ProbeNegativeBlocker(
                "P-03 BLOCKED: calibration artifact fingerprint is missing or does not match "
                "the frozen probe recipe"
            )
        artifact_id = str(raw.get("artifact_id", ""))
        if not expected_id or artifact_id != expected_id:
            raise ProbeNegativeBlocker(
                "P-03 BLOCKED: calibration artifact ID does not match the frozen probe recipe"
            )
        fit_split = str(raw.get("fit_split", "")).strip().lower()
        if not fit_split.startswith("dev") or "test" in fit_split:
            raise ProbeNegativeBlocker(
                "P-03 BLOCKED: possible-false-negative threshold must be fitted exclusively "
                "on a development split, never on a final test"
            )
        threshold = raw.get("threshold")
        if not isinstance(threshold, (int, float)) or not math.isfinite(float(threshold)):
            raise ProbeNegativeBlocker("P-03 BLOCKED: calibration artifact has no finite threshold")
        primary = raw.get("primary_judge")
        if not isinstance(primary, dict):
            raise ProbeNegativeBlocker(
                "P-03 BLOCKED: calibration artifact has no pinned primary judge"
            )
        revision = str(primary.get("revision", ""))
        if len(revision) != 40 or any(
            character not in "0123456789abcdef" for character in revision
        ):
            raise ProbeNegativeBlocker(
                "P-03 BLOCKED: calibration artifact must pin a full primary-judge revision"
            )
        if raw.get("score_kind") != CALIBRATION_SCORE_KIND:
            raise ProbeNegativeBlocker(
                "P-03 BLOCKED: calibration score kind is incompatible with probe filtering"
            )
        if raw.get("comparison_operator") != CALIBRATION_OPERATOR:
            raise ProbeNegativeBlocker(
                "P-03 BLOCKED: calibration comparison operator must be greater_than_or_equal"
            )
        dataset_fingerprint = str(raw.get("fit_dataset_fingerprint", ""))
        source_scores_sha256 = str(raw.get("source_scores_sha256", ""))
        if not _is_sha256(dataset_fingerprint) or not _is_sha256(source_scores_sha256):
            raise ProbeNegativeBlocker(
                "P-03 BLOCKED: calibration provenance fingerprints are missing"
            )
        selection_method = str(raw.get("selection_method", "")).strip()
        if not selection_method:
            raise ProbeNegativeBlocker(
                "P-03 BLOCKED: calibration threshold selection method is undocumented"
            )
        return cls(
            artifact_id=artifact_id,
            artifact_fingerprint=actual_fingerprint,
            fit_split=fit_split,
            fit_dataset_fingerprint=dataset_fingerprint,
            primary_judge_name=str(primary.get("name_or_path", "")),
            primary_judge_revision=revision,
            score_kind=CALIBRATION_SCORE_KIND,
            threshold=float(threshold),
            selection_method=selection_method,
            source_scores_sha256=source_scores_sha256,
        )

    def to_manifest(self) -> dict[str, Any]:
        return asdict(self) | {"comparison_operator": CALIBRATION_OPERATOR}


@dataclass(frozen=True)
class NegativeRecipe:
    """Frozen P-03 hard-negative selection and false-negative policy."""

    version: str
    strategy: HardNegativeStrategy
    false_negative_policy: FalseNegativePolicy = "drop"
    calibration_artifact_path: str | None = None
    calibration_artifact_id: str | None = None
    calibration_artifact_fingerprint: str | None = None
    bm25_index_fingerprint: str | None = None
    bm25_candidates: int = 32

    def __post_init__(self) -> None:
        if self.version != NEGATIVE_RECIPE_VERSION:
            raise ValueError(f"negative recipe version must be exactly {NEGATIVE_RECIPE_VERSION!r}")
        if self.strategy not in {"hn0", "hn0_filter", "hn1_bm25"}:
            raise ValueError(f"unsupported hard-negative strategy: {self.strategy}")
        if self.false_negative_policy not in {"drop", "demote", "keep+log"}:
            raise ValueError(f"unsupported false-negative policy: {self.false_negative_policy}")
        if self.bm25_candidates < 1:
            raise ValueError("bm25_candidates must be positive")
        if self.strategy == "hn1_bm25" and not _is_sha256(self.bm25_index_fingerprint):
            raise ProbeNegativeBlocker(
                "P-03 BLOCKED: HN1 BM25 requires a pinned 64-character index fingerprint"
            )

    @property
    def requires_filter(self) -> bool:
        return self.strategy in {"hn0_filter", "hn1_bm25"}

    def load_calibration(self) -> PossibleFalseNegativeCalibration | None:
        if not self.requires_filter:
            return None
        if not (
            self.calibration_artifact_path
            and self.calibration_artifact_id
            and self.calibration_artifact_fingerprint
        ):
            raise ProbeNegativeBlocker(
                "P-03 BLOCKED: filtered probe recipes require a pinned Task 02 dev-only "
                "calibration path, artifact ID and artifact fingerprint"
            )
        return PossibleFalseNegativeCalibration.load(
            Path(self.calibration_artifact_path),
            expected_id=self.calibration_artifact_id,
            expected_fingerprint=self.calibration_artifact_fingerprint,
        )

    def manifest(
        self,
        calibration: PossibleFalseNegativeCalibration | None,
    ) -> dict[str, Any]:
        return {
            "negative_recipe_version": self.version,
            "hard_negative_strategy": self.strategy,
            "possible_false_negative_policy": self.false_negative_policy,
            "possible_false_negative_threshold": (
                calibration.threshold if calibration is not None else None
            ),
            "calibration_artifact_id": (
                calibration.artifact_id if calibration is not None else None
            ),
            "calibration_artifact_fingerprint": (
                calibration.artifact_fingerprint if calibration is not None else None
            ),
            "calibration_fit_split": calibration.fit_split if calibration is not None else None,
            "primary_judge_name": (
                calibration.primary_judge_name if calibration is not None else None
            ),
            "primary_judge_revision": (
                calibration.primary_judge_revision if calibration is not None else None
            ),
            "bm25_index_fingerprint": self.bm25_index_fingerprint,
            "bm25_candidates": self.bm25_candidates if self.strategy == "hn1_bm25" else None,
            "diagnostic_unfiltered": self.strategy == "hn0",
        }


@dataclass(frozen=True)
class NegativeCandidate:
    doc_id: str
    text: str
    miner: str
    miner_rank: int
    miner_score: float | None = None


@dataclass(frozen=True)
class NegativeSelection:
    paired: NegativeCandidate | None
    demoted: NegativeCandidate | None
    audit_rows: tuple[dict[str, Any], ...]
    dropped_example: bool


def _selection_index(example_id: str, count: int) -> int:
    digest = hashlib.sha256(f"probe-v1:{example_id}".encode()).hexdigest()
    return int(digest[:8], 16) % count


def _validate_candidates(candidates: Sequence[NegativeCandidate]) -> tuple[NegativeCandidate, ...]:
    if not candidates:
        raise ValueError("hard-negative selection requires at least one candidate")
    seen: set[str] = set()
    result = []
    for candidate in candidates:
        if not candidate.doc_id or not candidate.text:
            raise ValueError("negative candidates require non-empty doc_id and text")
        if candidate.doc_id in seen:
            raise ValueError(f"duplicate negative candidate doc_id: {candidate.doc_id}")
        seen.add(candidate.doc_id)
        result.append(candidate)
    return tuple(result)


def _score_candidates(
    *,
    query: str,
    candidates: Sequence[NegativeCandidate],
    scorer: PairScorer,
    calibration: PossibleFalseNegativeCalibration,
) -> tuple[list[float], list[bool]]:
    if scorer.name != calibration.primary_judge_name:
        raise ProbeNegativeBlocker(
            "P-03 BLOCKED: runtime primary reranker does not match the calibration artifact"
        )
    scores = scorer.score_pairs([(query, candidate.text) for candidate in candidates])
    if len(scores) != len(candidates) or not all(math.isfinite(value) for value in scores):
        raise ValueError("primary reranker returned invalid possible-false-negative scores")
    flags = [float(score) >= calibration.threshold for score in scores]
    return [float(score) for score in scores], flags


def select_negative(
    *,
    example_id: str,
    query: str,
    candidates: Sequence[NegativeCandidate],
    recipe: NegativeRecipe,
    scorer: PairScorer | None,
    calibration: PossibleFalseNegativeCalibration | None,
) -> NegativeSelection:
    """Select one paired negative and optionally one demoted candidate deterministically."""
    validated = _validate_candidates(candidates)
    if recipe.strategy == "hn0":
        index = _selection_index(example_id, len(validated))
        candidate = validated[index]
        return NegativeSelection(
            paired=candidate,
            demoted=None,
            audit_rows=(
                {
                    "doc_id": candidate.doc_id,
                    "miner": candidate.miner,
                    "miner_rank": candidate.miner_rank,
                    "miner_score": candidate.miner_score,
                    "primary_score": None,
                    "possible_false_negative": None,
                    "action": "unfiltered_hn0",
                },
            ),
            dropped_example=False,
        )
    if scorer is None or calibration is None:
        raise ProbeNegativeBlocker(
            "P-03 BLOCKED: filtered hard-negative strategy requires the frozen primary "
            "reranker and a validated Task 02 calibration artifact"
        )
    scores, flags = _score_candidates(
        query=query,
        candidates=validated,
        scorer=scorer,
        calibration=calibration,
    )
    unflagged = [
        candidate
        for candidate, flagged_value in zip(validated, flags, strict=True)
        if not flagged_value
    ]
    flagged = [candidate for candidate, flag in zip(validated, flags, strict=True) if flag]
    policy = recipe.false_negative_policy
    if policy == "drop" and not unflagged:
        paired = None
        demoted = None
        dropped = True
    elif policy == "drop":
        paired = (
            unflagged[0]
            if recipe.strategy == "hn1_bm25"
            else unflagged[_selection_index(example_id, len(unflagged))]
        )
        demoted = None
        dropped = False
    elif policy == "demote":
        paired = (
            None
            if not unflagged
            else (
                unflagged[0]
                if recipe.strategy == "hn1_bm25"
                else unflagged[_selection_index(example_id, len(unflagged))]
            )
        )
        demoted = (
            None
            if not flagged
            else flagged[_selection_index(f"{example_id}:demoted", len(flagged))]
        )
        dropped = False
    else:
        paired = (
            validated[0]
            if recipe.strategy == "hn1_bm25"
            else validated[_selection_index(example_id, len(validated))]
        )
        demoted = None
        dropped = False
    selected_paired_id = paired.doc_id if paired is not None else None
    selected_demoted_id = demoted.doc_id if demoted is not None else None
    audit = []
    for candidate, score, flagged_value in zip(validated, scores, flags, strict=True):
        if candidate.doc_id == selected_paired_id:
            action = "paired_keep_flagged" if flagged_value else "paired"
        elif candidate.doc_id == selected_demoted_id:
            action = "demoted_to_in_batch"
        elif flagged_value and policy == "drop":
            action = "dropped_false_negative"
        elif flagged_value and policy == "demote":
            action = "demoted_not_selected"
        else:
            action = "not_selected"
        audit.append(
            {
                "doc_id": candidate.doc_id,
                "miner": candidate.miner,
                "miner_rank": candidate.miner_rank,
                "miner_score": candidate.miner_score,
                "primary_score": score,
                "possible_false_negative": flagged_value,
                "action": action,
            }
        )
    return NegativeSelection(
        paired=paired,
        demoted=demoted,
        audit_rows=tuple(audit),
        dropped_example=dropped,
    )


COMPARISON_CONTRACT_FIELDS = (
    "probe_recipe_version",
    "probe_recipe_fingerprint",
    "negative_recipe_version",
    "hard_negative_strategy",
    "possible_false_negative_policy",
    "possible_false_negative_threshold",
    "calibration_artifact_id",
    "calibration_artifact_fingerprint",
    "bm25_index_fingerprint",
)


def assert_same_negative_contract(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject generator/probe comparisons with any P-03 contract drift."""
    left_contract = left.get("negative_contract")
    right_contract = right.get("negative_contract")
    if not isinstance(left_contract, Mapping) or not isinstance(right_contract, Mapping):
        raise ValueError("probe comparison requires a complete P-03 negative_contract")
    mismatches = [
        field
        for field in COMPARISON_CONTRACT_FIELDS
        if left_contract.get(field) != right_contract.get(field)
    ]
    if mismatches:
        raise ValueError(
            "probe comparison rejected due to different P-03 negative contracts: "
            + ", ".join(mismatches)
        )
    return {field: left_contract.get(field) for field in COMPARISON_CONTRACT_FIELDS}


def summarize_false_negative_audit(
    audit_rows: Sequence[Mapping[str, Any]],
    *,
    query_source: str,
    generator_id: str | None,
    input_examples: int,
    output_examples: int,
    policy_dropped_examples: int,
) -> dict[str, Any]:
    """Report counts and rates without mixing natural and synthetic sources."""
    scored = [row for row in audit_rows if isinstance(row.get("possible_false_negative"), bool)]
    flagged = [row for row in scored if row["possible_false_negative"] is True]
    source_id = generator_id if query_source == "synthetic" else query_source
    if query_source == "synthetic" and not source_id:
        raise ValueError("synthetic false-negative reporting requires generator_id")
    return {
        "query_source": query_source,
        "source_id": source_id,
        "generator_id": generator_id,
        "input_examples": input_examples,
        "output_examples": output_examples,
        "policy_dropped_examples": policy_dropped_examples,
        "selection_excluded_examples": max(
            input_examples - policy_dropped_examples - output_examples, 0
        ),
        "candidate_count": len(audit_rows),
        "scored_candidate_count": len(scored),
        "possible_false_negative_count": len(flagged),
        "possible_false_negative_rate": len(flagged) / len(scored) if scored else None,
        "actions": {
            action: sum(row.get("action") == action for row in audit_rows)
            for action in sorted({str(row.get("action")) for row in audit_rows})
        },
        "per_source": {
            str(source_id): {
                "candidate_count": len(audit_rows),
                "scored_candidate_count": len(scored),
                "possible_false_negative_count": len(flagged),
                "possible_false_negative_rate": len(flagged) / len(scored) if scored else None,
            }
        },
    }
