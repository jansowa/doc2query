"""Resumable offline primary-judge margins for natural SFT pairs."""

from __future__ import annotations

import bisect
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, TextIO

from doc2query.reranker.base import FrozenRerankerConfig, PairScorer
from doc2query.reranker.false_negative_calibration import load_query_score_groups, sha256_file
from doc2query.reranker.load import load_frozen_reranker
from doc2query.utils.records import read_durable_jsonl_prefix, read_records

SCHEMA_VERSION = 1
SCHEMA_NAME = "natural_train_primary_margin_v1"


def _duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _log(message: str, stream: TextIO) -> None:
    print(f"[train-margins] {message}", file=stream, flush=True)


def _progress(
    *,
    stage: str,
    completed: int,
    total: int,
    resumed: int,
    started: float,
    stream: TextIO,
) -> None:
    elapsed = time.perf_counter() - started
    newly_completed = completed - resumed
    rate = newly_completed / elapsed if newly_completed > 0 and elapsed > 0 else 0.0
    eta = (total - completed) / rate if rate > 0 else 0.0
    percent = 100.0 * completed / total if total else 100.0
    _log(
        f"stage={stage} {completed:,}/{total:,} ({percent:5.1f}%) "
        f"new={newly_completed:,} elapsed={_duration(elapsed)} "
        f"rate={rate:,.2f}/s eta={_duration(eta)}",
        stream,
    )


def _sha256_with_progress(
    path: Path,
    *,
    stage: str,
    stream: TextIO,
    interval_seconds: float,
) -> str:
    total = path.stat().st_size
    completed = 0
    digest = hashlib.sha256()
    started = time.perf_counter()
    last_log = started
    _log(f"stage={stage} start bytes={total:,} path={path}", stream)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
            completed += len(chunk)
            now = time.perf_counter()
            if now - last_log >= interval_seconds:
                _progress(
                    stage=stage,
                    completed=completed,
                    total=total,
                    resumed=0,
                    started=started,
                    stream=stream,
                )
                last_log = now
    _progress(
        stage=stage,
        completed=completed,
        total=total,
        resumed=0,
        started=started,
        stream=stream,
    )
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _validate_document(value: Any, kind: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not str(value.get("doc_id", "")):
        raise ValueError(f"{kind} must be an object with doc_id")
    if not isinstance(value.get("text"), str) or not value["text"].strip():
        raise ValueError(f"{kind} must contain non-empty text")
    return value


def _calibration_contract(
    calibration_path: Path,
    calibration_scores_path: Path,
    judge: FrozenRerankerConfig,
) -> tuple[dict[str, Any], list[float]]:
    artifact = json.loads(calibration_path.read_text(encoding="utf-8"))
    if not isinstance(artifact, dict):
        raise ValueError("calibration artifact must be an object")
    primary = artifact.get("primary_judge")
    if not isinstance(primary, dict):
        raise ValueError("calibration artifact lacks primary_judge provenance")
    if (
        primary.get("name_or_path") != judge.name_or_path
        or primary.get("revision") != judge.revision
    ):
        raise ValueError("calibration judge identity does not match train scorer")
    fit_split = str(artifact.get("fit_split", "")).lower()
    if not fit_split.startswith("dev") or "test" in fit_split:
        raise ValueError("train margins require a development-only calibration artifact")
    if artifact.get("tests_used_for_threshold_tuning") != []:
        raise ValueError("calibration artifact must not use final tests")
    expected_scores_sha = str(artifact.get("source_scores_sha256", ""))
    if sha256_file(calibration_scores_path) != expected_scores_sha:
        raise ValueError("calibration score fingerprint mismatch")
    groups = load_query_score_groups(
        calibration_scores_path,
        expected_judge=judge.name_or_path,
    )
    margins = sorted(
        positive - max(group.negative_scores)
        for group in groups
        for positive in group.positive_scores
    )
    if not margins:
        raise ValueError("calibration scores contain no natural positive margins")
    return artifact, margins


def _contract(
    *,
    input_path: Path,
    judge: FrozenRerankerConfig,
    calibration_path: Path,
    calibration_scores_path: Path,
    calibration: dict[str, Any],
    dev_margin_count: int,
    input_sha256: str | None = None,
) -> dict[str, Any]:
    identity = {
        "schema_version": SCHEMA_VERSION,
        "schema": SCHEMA_NAME,
        "input_path": str(input_path.resolve()),
        "input_sha256": input_sha256 or sha256_file(input_path),
        "judge": asdict(judge),
        "calibration_path": str(calibration_path.resolve()),
        "calibration_sha256": sha256_file(calibration_path),
        "calibration_scores_path": str(calibration_scores_path.resolve()),
        "calibration_scores_sha256": sha256_file(calibration_scores_path),
        "calibration_artifact_id": calibration["artifact_id"],
        "calibration_artifact_fingerprint": calibration["artifact_fingerprint"],
        "calibration_fit_split": calibration["fit_split"],
        "dev_margin_count": dev_margin_count,
        "calibration_method": "empirical_cdf_of_natural_dev_positive_margin",
        "final_tests_used": [],
    }
    identity["contract_fingerprint"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return identity


def _iter_train_groups(input_path: Path) -> Any:
    for record in read_records(input_path):
        metadata = record.get("metadata")
        split = str(metadata.get("split", "")) if isinstance(metadata, dict) else ""
        if split != "train":
            raise ValueError("offline train-margin input must contain only split=train")
        query_id = str(record.get("example_id", ""))
        query = record.get("query")
        positives = record.get("positives")
        negatives = record.get("hard_negatives")
        if not query_id or not isinstance(query, str) or not query.strip():
            raise ValueError("every train group requires example_id and query")
        if not isinstance(positives, list) or not positives:
            raise ValueError("every train group requires positives")
        if not isinstance(negatives, list) or not negatives:
            raise ValueError("every train group requires hard negatives")
        positive_docs = [_validate_document(value, "positive") for value in positives]
        negative_docs = [_validate_document(value, "hard negative") for value in negatives]
        yield query_id, query, positive_docs, negative_docs


def _record_count_hint(input_path: Path) -> int | None:
    if input_path.suffix != ".parquet":
        return None
    try:
        import pyarrow.parquet as pq  # type: ignore[import-untyped]
    except ImportError:
        return None
    return int(pq.ParquetFile(input_path).metadata.num_rows)


def _expected_pair_ids(
    input_path: Path,
    *,
    progress_every: int = 10_000,
    progress_interval_seconds: float = 10.0,
    stream: TextIO = sys.stderr,
) -> list[str]:
    expected: list[str] = []
    groups = 0
    total_hint = _record_count_hint(input_path)
    started = time.perf_counter()
    last_log = started
    _log(
        f"stage=inventory start groups_total={total_hint if total_hint is not None else 'unknown'}",
        stream,
    )
    for query_id, _query, positives, _negatives in _iter_train_groups(input_path):
        groups += 1
        expected.extend(f"{query_id}::{positive['doc_id']}" for positive in positives)
        now = time.perf_counter()
        if groups % progress_every == 0 or now - last_log >= progress_interval_seconds:
            if total_hint is not None:
                _progress(
                    stage="inventory-groups",
                    completed=groups,
                    total=total_hint,
                    resumed=0,
                    started=started,
                    stream=stream,
                )
            else:
                elapsed = now - started
                rate = groups / elapsed if elapsed > 0 else 0.0
                _log(
                    f"stage=inventory-groups groups={groups:,} pairs={len(expected):,} "
                    f"elapsed={_duration(elapsed)} rate={rate:,.2f}/s eta=unknown",
                    stream,
                )
            last_log = now
    elapsed = time.perf_counter() - started
    _log(
        f"stage=inventory complete groups={groups:,} pairs={len(expected):,} "
        f"elapsed={_duration(elapsed)} rate={groups / elapsed if elapsed else 0.0:,.2f}/s",
        stream,
    )
    return expected


def score_natural_train_margins(
    *,
    input_path: Path,
    output_dir: Path,
    judge: FrozenRerankerConfig,
    calibration_path: Path,
    calibration_scores_path: Path,
    group_batch_size: int = 128,
    progress_every: int = 1_000,
    progress_interval_seconds: float = 10.0,
    scorer: PairScorer | None = None,
    log_stream: TextIO = sys.stderr,
) -> dict[str, Any]:
    """Score all train pairs; resume only an exact, durable prefix."""
    if group_batch_size < 1 or progress_every < 1 or progress_interval_seconds <= 0:
        raise ValueError("batch size, progress interval and progress count must be positive")
    _log(
        f"stage=calibration start judge={judge.name_or_path} revision={judge.revision}",
        log_stream,
    )
    calibration, dev_margins = _calibration_contract(
        calibration_path, calibration_scores_path, judge
    )
    _log(
        f"stage=calibration complete dev_margins={len(dev_margins):,} "
        f"artifact={calibration['artifact_id']}",
        log_stream,
    )
    input_sha256 = _sha256_with_progress(
        input_path,
        stage="fingerprint-input",
        stream=log_stream,
        interval_seconds=progress_interval_seconds,
    )
    contract = _contract(
        input_path=input_path,
        judge=judge,
        calibration_path=calibration_path,
        calibration_scores_path=calibration_scores_path,
        calibration=calibration,
        dev_margin_count=len(dev_margins),
        input_sha256=input_sha256,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = output_dir / "contract.json"
    if contract_path.exists():
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing != contract:
            raise ValueError("existing train-margin contract does not match this invocation")
        _log(
            f"stage=contract resume fingerprint={contract['contract_fingerprint']}", log_stream
        )
    else:
        _atomic_json(contract_path, contract)
        _log(
            f"stage=contract created fingerprint={contract['contract_fingerprint']}", log_stream
        )

    journal_path = output_dir / "margins.jsonl"
    _log(f"stage=journal-read start path={journal_path}", log_stream)
    prior = read_durable_jsonl_prefix(journal_path)
    prior_ids = [str(row.get("pair_id", "")) for row in prior]
    _log(f"stage=journal-read complete durable_rows={len(prior_ids):,}", log_stream)
    expected_ids = _expected_pair_ids(
        input_path,
        progress_interval_seconds=progress_interval_seconds,
        stream=log_stream,
    )
    if len(prior_ids) > len(expected_ids) or prior_ids != expected_ids[: len(prior_ids)]:
        raise ValueError("train-margin journal is not an exact prefix of the frozen input")
    if len(prior_ids) == len(expected_ids):
        _log("stage=scoring already complete; validating manifest", log_stream)
        return finalize_train_margin_artifact(
            output_dir,
            expected_ids=expected_ids,
            progress_interval_seconds=progress_interval_seconds,
            log_stream=log_stream,
        )

    _log(
        f"stage=model-load start device={judge.device} batch={judge.batch_size}", log_stream
    )
    model_started = time.perf_counter()
    model = scorer or load_frozen_reranker(judge)
    _log(
        f"stage=model-load complete elapsed={_duration(time.perf_counter() - model_started)}",
        log_stream,
    )
    started = time.perf_counter()
    completed = len(prior_ids)
    resumed_count = completed
    last_progress_count = completed
    last_progress_time = started
    replayed_prefix_rows = 0
    pending_groups: list[tuple[str, str, list[dict[str, Any]], list[dict[str, Any]]]] = []

    def flush(handle: Any) -> None:
        nonlocal completed, replayed_prefix_rows, last_progress_count, last_progress_time
        pairs = [
            (query, str(document["text"]))
            for _query_id, query, positives, negatives in pending_groups
            for document in [*positives, *negatives]
        ]
        order = sorted(
            range(len(pairs)),
            key=lambda index: len(pairs[index][0].split()) + len(pairs[index][1].split()),
        )
        ordered_scores = model.score_pairs([pairs[index] for index in order])
        if len(ordered_scores) != len(pairs):
            raise ValueError("primary scorer returned the wrong number of scores")
        scores = [0.0] * len(pairs)
        for index, score in zip(order, ordered_scores, strict=True):
            scores[index] = float(score)
        offset = 0
        rows: list[dict[str, Any]] = []
        for query_id, _query, positives, negatives in pending_groups:
            group_scores = scores[offset : offset + len(positives) + len(negatives)]
            offset += len(group_scores)
            negative_scores = group_scores[len(positives) :]
            hardest_index = max(range(len(negative_scores)), key=negative_scores.__getitem__)
            hardest_score = negative_scores[hardest_index]
            for positive, positive_score in zip(
                positives, group_scores[: len(positives)], strict=True
            ):
                margin = positive_score - hardest_score
                rows.append(
                    {
                        "schema": SCHEMA_NAME,
                        "pair_id": f"{query_id}::{positive['doc_id']}",
                        "query_id": query_id,
                        "positive_doc_id": str(positive["doc_id"]),
                        "hardest_negative_doc_id": str(negatives[hardest_index]["doc_id"]),
                        "positive_score": positive_score,
                        "hardest_negative_score": hardest_score,
                        "raw_margin": margin,
                        "calibrated_margin_percentile": bisect.bisect_right(dev_margins, margin)
                        / len(dev_margins),
                        "judge": judge.name_or_path,
                        "judge_revision": judge.revision,
                        "calibration_artifact_id": calibration["artifact_id"],
                        "calibration_artifact_fingerprint": calibration[
                            "artifact_fingerprint"
                        ],
                        "calibration_fit_split": calibration["fit_split"],
                    }
                )
        for row in rows:
            if replayed_prefix_rows:
                replayed_prefix_rows -= 1
                continue
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            completed += 1
        handle.flush()
        os.fsync(handle.fileno())
        pending_groups.clear()
        now = time.perf_counter()
        if (
            completed == len(expected_ids)
            or completed - last_progress_count >= progress_every
            or now - last_progress_time >= progress_interval_seconds
        ):
            _progress(
                stage="scoring",
                completed=completed,
                total=len(expected_ids),
                resumed=resumed_count,
                started=started,
                stream=log_stream,
            )
            last_progress_count = completed
            last_progress_time = now

    # Rescoring at most the one group crossing the prefix is intentionally avoided by
    # consuming whole groups until the next missing pair.
    seen_pairs = 0
    _log(
        f"stage=scoring resume={resumed_count:,}/{len(expected_ids):,} "
        f"remaining={len(expected_ids) - resumed_count:,} group_batch={group_batch_size}; "
        "Ctrl+C is safe (only the active unsynced batch may repeat)",
        log_stream,
    )
    try:
        with journal_path.open("a", encoding="utf-8") as handle:
            for group in _iter_train_groups(input_path):
                group_pairs = len(group[2])
                if seen_pairs < len(prior_ids):
                    if seen_pairs + group_pairs <= len(prior_ids):
                        seen_pairs += group_pairs
                        continue
                    replayed_prefix_rows = len(prior_ids) - seen_pairs
                pending_groups.append(group)
                seen_pairs += group_pairs
                if len(pending_groups) >= group_batch_size:
                    flush(handle)
            if pending_groups:
                flush(handle)
    except KeyboardInterrupt:
        _log(
            f"stage=scoring interrupted durable={completed:,}/{len(expected_ids):,}; "
            "run the same command to resume",
            log_stream,
        )
        raise
    result = finalize_train_margin_artifact(
        output_dir,
        expected_ids=expected_ids,
        progress_interval_seconds=progress_interval_seconds,
        log_stream=log_stream,
    )
    result["scoring_wall_seconds_this_invocation"] = time.perf_counter() - started
    return result


def finalize_train_margin_artifact(
    output_dir: Path,
    *,
    expected_ids: list[str] | None = None,
    progress_interval_seconds: float = 10.0,
    log_stream: TextIO = sys.stderr,
) -> dict[str, Any]:
    """Validate completeness and atomically publish the immutable manifest."""
    contract = json.loads((output_dir / "contract.json").read_text(encoding="utf-8"))
    _log("stage=finalize start: validating journal and input coverage", log_stream)
    margins_path = output_dir / "margins.jsonl"
    rows = read_durable_jsonl_prefix(margins_path)
    expected = expected_ids or _expected_pair_ids(
        Path(contract["input_path"]),
        progress_interval_seconds=progress_interval_seconds,
        stream=log_stream,
    )
    actual = [str(row.get("pair_id", "")) for row in rows]
    if actual != expected:
        raise ValueError(f"margin artifact incomplete: {len(actual)}/{len(expected)} pairs")
    values = [float(row["calibrated_margin_percentile"]) for row in rows]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "pair_count": len(rows),
        "contract_fingerprint": contract["contract_fingerprint"],
        "margins_path": str(margins_path),
        "margins_sha256": _sha256_with_progress(
            margins_path,
            stage="fingerprint-margins",
            stream=log_stream,
            interval_seconds=progress_interval_seconds,
        ),
        "calibrated_margin_percentile_min": min(values),
        "calibrated_margin_percentile_max": max(values),
        "final_tests_used": [],
    }
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError("published train-margin manifest is immutable and no longer matches")
    else:
        _atomic_json(manifest_path, manifest)
    _log(
        f"stage=finalize complete pairs={len(rows):,} sha256={manifest['margins_sha256']}",
        log_stream,
    )
    return manifest
