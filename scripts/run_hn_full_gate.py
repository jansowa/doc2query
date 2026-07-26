#!/usr/bin/env python3
"""Run the inference-only HN0--HN3 gate on a frozen development cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import yaml

from doc2query.evaluation.corpus import load_corpus_index, sha256_file
from doc2query.evaluation.datasets import evaluation_fingerprint, load_frozen_records
from doc2query.evaluation.dense_retrieval import ShardedEmbeddingIndex, _load_tensor
from doc2query.evaluation.embedder_probe import MeanPoolEncoder, _encode
from doc2query.evaluation.hn_gate import (
    ARMS,
    GATE_VERSION,
    assert_dev_only_contract,
    canonical_fingerprint,
    compare_to_reference,
    deduplicate_scoring_pairs,
    pool_metrics,
    positive_aware_keep,
    select_dev_records,
    stable_union,
    summarize_rows,
)
from doc2query.evaluation.probe_negatives import PossibleFalseNegativeCalibration
from doc2query.reranker.base import FrozenRerankerConfig
from doc2query.reranker.load import load_frozen_reranker
from doc2query.utils.records import JsonlWriter, read_records, write_json


def _load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return raw


def _mapping(raw: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"gate config requires mapping: {key}")
    return value


def _judge_config(path: Path) -> FrozenRerankerConfig:
    return FrozenRerankerConfig(**_load_yaml(path))


def _fingerprint_ids(rows: Sequence[Mapping[str, Any]]) -> str:
    return canonical_fingerprint([str(row["example_id"]) for row in rows])


def _pairs_fingerprint(pairs: Sequence[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for query, passage in pairs:
        digest.update(query.encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(passage.encode()).digest())
        digest.update(b"\n")
    return digest.hexdigest()


def _open_journal(path: Path, identity: Mapping[str, Any]) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS rows (ordinal INTEGER PRIMARY KEY, payload TEXT NOT NULL)"
    )
    fingerprint = canonical_fingerprint(dict(identity))
    existing = connection.execute(
        "SELECT value FROM metadata WHERE key='identity_fingerprint'"
    ).fetchone()
    if existing is not None and str(existing[0]) != fingerprint:
        connection.close()
        raise ValueError(f"resume journal identity mismatch: {path}")
    connection.execute(
        "INSERT OR IGNORE INTO metadata(key, value) VALUES('identity_fingerprint', ?)",
        (fingerprint,),
    )
    connection.commit()
    return connection


def _load_or_create_corpus_ids(full_corpus: Path, cache: Path, expected: int) -> list[str]:
    target = cache / "corpus_ids.jsonl"
    if target.is_file():
        ids = [str(json.loads(line)) for line in target.read_text(encoding="utf-8").splitlines()]
    else:
        ids = sorted(str(row["doc_id"]) for row in read_records(full_corpus))
        temporary = target.with_suffix(".scalar.jsonl")
        temporary.write_text("".join(json.dumps(value) + "\n" for value in ids), encoding="utf-8")
        temporary.replace(target)
    if len(ids) != expected or len(set(ids)) != len(ids):
        raise ValueError("dense corpus ID catalog is incomplete or non-unique")
    return ids


def _encode_queries(
    records: Sequence[Mapping[str, Any]],
    model_path: Path,
    *,
    cache_path: Path,
    identity: Mapping[str, Any],
) -> torch.Tensor:
    from transformers import AutoTokenizer

    manifest_path = cache_path.with_suffix(".manifest.json")
    expected_manifest = {
        "schema_version": 1,
        "identity": dict(identity),
        "count": len(records),
    }
    if cache_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        embeddings = torch.load(cache_path, map_location="cpu", weights_only=True)
        if manifest != expected_manifest or not isinstance(embeddings, torch.Tensor):
            raise ValueError("dense query embedding resume cache mismatch")
        if embeddings.ndim != 2 or embeddings.shape[0] != len(records):
            raise ValueError("dense query embedding resume cache has invalid dimensions")
        print("[HN gate] dense query embeddings resumed", file=sys.stderr, flush=True)
        return embeddings
    if cache_path.exists() or manifest_path.exists():
        raise ValueError("partial dense query embedding resume cache")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False
    )
    model = MeanPoolEncoder(str(model_path), "main").to("cpu").eval()
    chunks = []
    for start in range(0, len(records), 16):
        chunks.append(
            _encode(
                model,
                tokenizer,
                [str(row["query"]) for row in records[start : start + 16]],
                max_length=192,
                device=torch.device("cpu"),
            )
        )
        if start == 0 or start % 160 == 0:
            print(
                f"[HN gate] dense query encoding {min(start + 16, len(records))}/{len(records)}",
                file=sys.stderr,
            )
    embeddings = torch.cat(chunks)
    temporary = cache_path.with_suffix(".tmp")
    torch.save(embeddings, temporary)
    temporary.replace(cache_path)
    write_json(manifest_path, expected_manifest)
    return embeddings


def _dense_mine(
    queries: torch.Tensor,
    *,
    cache: Path,
    corpus_ids: Sequence[str],
    train_ids: set[str],
    positives: Sequence[set[str]],
    limit: int,
) -> list[list[dict[str, Any]]]:
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    index = ShardedEmbeddingIndex.load(
        cache,
        row_count=int(manifest["row_count"]),
        chunk_size=int(manifest["chunk_size"]),
    )
    if queries.shape != (len(positives), index.dimension):
        raise ValueError("dense query embeddings do not match the frozen index")
    search_limit = limit + max(len(value) for value in positives)
    best_scores = torch.full((len(queries), search_limit), -float("inf"))
    best_rows = torch.full((len(queries), search_limit), -1, dtype=torch.int64)
    started = time.perf_counter()
    for number, shard in enumerate(index.shards, start=1):
        local_ids = corpus_ids[shard.start : shard.end]
        allowed_local = [offset for offset, doc_id in enumerate(local_ids) if doc_id in train_ids]
        if not allowed_local:
            continue
        vectors = _load_tensor(shard.path)[allowed_local].to(torch.float32)
        global_rows = torch.tensor(
            [shard.start + value for value in allowed_local], dtype=torch.int64
        )
        scores = queries.to(torch.float32) @ vectors.T
        take = min(search_limit, scores.shape[1])
        values, positions = torch.topk(scores, take, dim=1, largest=True, sorted=True)
        rows = global_rows[positions]
        merged_values = torch.cat((best_scores, values), dim=1)
        merged_rows = torch.cat((best_rows, rows), dim=1)
        best_scores, order = torch.topk(
            merged_values, search_limit, dim=1, largest=True, sorted=True
        )
        best_rows = torch.gather(merged_rows, 1, order)
        elapsed = time.perf_counter() - started
        print(
            f"[HN gate] dense shard {number}/{len(index.shards)} elapsed={elapsed:.1f}s",
            file=sys.stderr,
            flush=True,
        )
    output = []
    for query_index in range(len(queries)):
        candidates = []
        rank = 0
        for row, score in zip(
            best_rows[query_index].tolist(), best_scores[query_index].tolist(), strict=True
        ):
            doc_id = corpus_ids[int(row)]
            if doc_id in positives[query_index]:
                continue
            rank += 1
            candidates.append({"doc_id": doc_id, "rank": rank, "score": float(score)})
            if rank == limit:
                break
        if len(candidates) < limit:
            raise ValueError("dense miner did not retain enough train-only non-positive documents")
        output.append(candidates)
    return output


def _load_texts(path: Path, wanted: set[str]) -> dict[str, str]:
    result = {}
    for row in read_records(path):
        doc_id = str(row["doc_id"])
        if doc_id in wanted:
            result[doc_id] = str(row["text"])
    missing = wanted - result.keys()
    if missing:
        raise ValueError(f"mined train documents are missing: {sorted(missing)[:3]}")
    return result


def _score_map(
    scorer: Any,
    pairs: list[tuple[str, str]],
    *,
    label: str,
    journal_path: Path,
    identity: Mapping[str, Any],
) -> list[float]:
    journal_identity = dict(identity) | {
        "label": label,
        "pair_count": len(pairs),
        "pairs_fingerprint": _pairs_fingerprint(pairs),
        "scorer": scorer.config.__dict__,
    }
    connection = _open_journal(journal_path, journal_identity)
    stored = connection.execute("SELECT ordinal, payload FROM rows ORDER BY ordinal").fetchall()
    if [int(row[0]) for row in stored] != list(range(len(stored))):
        connection.close()
        raise ValueError(f"non-contiguous scoring journal: {journal_path}")
    scores = [float(json.loads(str(row[1]))["score"]) for row in stored]
    if len(scores) > len(pairs):
        connection.close()
        raise ValueError(f"scoring journal longer than its contract: {journal_path}")
    if scores:
        print(f"[HN gate] {label} resumed {len(scores)}/{len(pairs)}", file=sys.stderr)
    for start in range(len(scores), len(pairs), 1024):
        chunk = scorer.score_pairs(pairs[start : start + 1024])
        connection.executemany(
            "INSERT INTO rows(ordinal, payload) VALUES (?, ?)",
            (
                (start + offset, json.dumps({"score": float(score)}))
                for offset, score in enumerate(chunk)
            ),
        )
        connection.commit()
        scores.extend(float(value) for value in chunk)
        print(
            f"[HN gate] {label} {min(start + 1024, len(pairs))}/{len(pairs)}",
            file=sys.stderr,
            flush=True,
        )
    connection.close()
    return scores


def _bm25_mine(
    records: Sequence[Mapping[str, Any]],
    positives: Sequence[set[str]],
    *,
    index_path: Path,
    candidate_count: int,
    journal_path: Path,
    identity: Mapping[str, Any],
) -> list[list[dict[str, Any]]]:
    connection = _open_journal(journal_path, identity)
    stored = connection.execute("SELECT ordinal, payload FROM rows ORDER BY ordinal").fetchall()
    if [int(row[0]) for row in stored] != list(range(len(stored))):
        connection.close()
        raise ValueError("non-contiguous BM25 mining journal")
    rows = [json.loads(str(row[1])) for row in stored]
    if len(rows) > len(records):
        connection.close()
        raise ValueError("BM25 mining journal longer than its frozen cohort")
    if rows:
        print(f"[HN gate] BM25 resumed {len(rows)}/{len(records)}", file=sys.stderr)
    bm25 = load_corpus_index(index_path)
    try:
        for index in range(len(rows), len(records)):
            row, excluded = records[index], positives[index]
            result = bm25.search(str(row["query"]), limit=candidate_count + len(excluded))
            values = [
                {"doc_id": value.doc_id, "rank": value.rank, "score": value.score}
                for value in result.documents
                if value.doc_id not in excluded
            ][:candidate_count]
            connection.execute(
                "INSERT INTO rows(ordinal, payload) VALUES (?, ?)",
                (index, json.dumps(values, ensure_ascii=False, sort_keys=True)),
            )
            connection.commit()
            rows.append(values)
            completed = index + 1
            if completed == 1 or completed % 100 == 0:
                print(
                    f"[HN gate] BM25 {completed}/{len(records)}",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        bm25.close()
        connection.close()
    return rows


def run(config_path: Path, root: Path) -> dict[str, Any]:
    raw = _load_yaml(config_path)
    assert_dev_only_contract(raw)
    if raw.get("gate_version") != GATE_VERSION:
        raise ValueError("unsupported hard-negative gate version")
    inputs, judges = _mapping(raw, "inputs"), _mapping(raw, "judges")
    subset = str(raw["evaluation_subset"])
    manifest_path = root / str(inputs["frozen_manifest"])
    all_dev = load_frozen_records(manifest_path, subset)
    records = select_dev_records(all_dev, limit=int(raw["dev_limit"]), seed=int(raw["seed"]))
    output_dir, report_dir = root / str(raw["output_dir"]), root / str(raw["report_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    contract = {
        "gate_version": GATE_VERSION,
        "evaluation_subset": subset,
        "frozen_dev_fingerprint": evaluation_fingerprint(manifest_path, subset),
        "cohort_fingerprint": _fingerprint_ids(records),
        "cohort_count": len(records),
        "seed": int(raw["seed"]),
        "candidate_count": int(raw["candidate_count"]),
        "retained_negatives": int(raw["retained_negatives"]),
        "final_tests_used": [],
        "training_runs": [],
    }
    assert_dev_only_contract(contract)
    write_json(output_dir / "contract.json", contract)

    calibration_path = root / str(inputs["calibration"])
    calibration_raw = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration = PossibleFalseNegativeCalibration.load(
        calibration_path,
        expected_id=str(calibration_raw["artifact_id"]),
        expected_fingerprint=str(calibration_raw["artifact_fingerprint"]),
    )
    candidate_count = int(raw["candidate_count"])
    positives = [{str(document["doc_id"]) for document in row["positives"]} for row in records]

    bm25_path = root / str(inputs["bm25_index"])
    bm25_manifest = json.loads((bm25_path / "manifest.json").read_text(encoding="utf-8"))
    bm25_rows = _bm25_mine(
        records,
        positives,
        index_path=bm25_path,
        candidate_count=candidate_count,
        journal_path=output_dir / "bm25_mining.sqlite",
        identity={
            "stage": "bm25_mining",
            "contract": contract,
            "index_fingerprint": bm25_manifest["index_fingerprint"],
        },
    )

    dense_cache = root / str(inputs["dense_embedding_cache"])
    dense_manifest = json.loads((dense_cache / "manifest.json").read_text(encoding="utf-8"))
    full_corpus = root / str(inputs["full_corpus"])
    if dense_manifest.get("identity", {}).get("corpus_sha256") != sha256_file(full_corpus):
        raise ValueError("dense embedding cache does not match the pinned full corpus")
    corpus_ids = _load_or_create_corpus_ids(
        full_corpus, dense_cache, int(dense_manifest["row_count"])
    )
    train_corpus = root / str(inputs["train_corpus"])
    train_ids = {str(row["doc_id"]) for row in read_records(train_corpus)}
    dense_stage_path = output_dir / "dense_mining.json"
    dense_identity = {
        "stage": "dense_mining",
        "contract": contract,
        "embedding_cache_identity": dense_manifest["identity"],
        "train_corpus_sha256": sha256_file(train_corpus),
    }
    if dense_stage_path.is_file():
        dense_payload = json.loads(dense_stage_path.read_text(encoding="utf-8"))
        if dense_payload.get("identity") != dense_identity:
            raise ValueError("dense mining resume cache identity mismatch")
        dense_rows = dense_payload.get("rows")
        if not isinstance(dense_rows, list) or len(dense_rows) != len(records):
            raise ValueError("dense mining resume cache is incomplete")
        print("[HN gate] dense mining resumed", file=sys.stderr, flush=True)
    else:
        query_embeddings = _encode_queries(
            records,
            root / str(inputs["dense_model"]),
            cache_path=output_dir / "dense_query_embeddings.pt",
            identity=dense_identity,
        )
        dense_rows = _dense_mine(
            query_embeddings,
            cache=dense_cache,
            corpus_ids=corpus_ids,
            train_ids=train_ids,
            positives=positives,
            limit=candidate_count,
        )
        write_json(dense_stage_path, {"identity": dense_identity, "rows": dense_rows})

    wanted = {
        str(value["doc_id"])
        for groups in (bm25_rows, dense_rows)
        for group in groups
        for value in group
    }
    texts = _load_texts(train_corpus, wanted)
    inherited: list[list[dict[str, Any]]] = [
        [
            {
                "doc_id": str(value["doc_id"]),
                "rank": rank,
                "score": None,
                "text": str(value["text"]),
            }
            for rank, value in enumerate(
                sorted(row["hard_negatives"], key=lambda item: str(item["doc_id"])), 1
            )
        ]
        for row in records
    ]
    for groups in (bm25_rows, dense_rows):
        for group in groups:
            for value in group:
                value["text"] = texts[str(value["doc_id"])]

    primary_config = _judge_config(root / str(judges["primary"]))
    shadow_config = _judge_config(root / str(judges["shadow"]))
    if primary_config.name_or_path != calibration.primary_judge_name:
        raise ValueError("primary judge differs from dev calibration")
    primary = load_frozen_reranker(primary_config)
    primary_pairs: list[tuple[str, str]] = []
    offsets: list[tuple[int, int, int, int]] = []
    for row, inherited_group, bm25_group, dense_group in zip(
        records, inherited, bm25_rows, dense_rows, strict=True
    ):
        query = str(row["query"])
        positive = str(row["positives"][0]["text"])
        candidates = inherited_group + bm25_group + dense_group
        start = len(primary_pairs)
        primary_pairs.append((query, positive))
        primary_pairs.extend((query, str(value["text"])) for value in candidates)
        offsets.append((start, len(inherited_group), len(bm25_group), len(dense_group)))
    primary_unique_pairs, primary_inverse = deduplicate_scoring_pairs(primary_pairs)
    print(
        f"[HN gate] primary deduplicated {len(primary_pairs)} -> {len(primary_unique_pairs)} pairs",
        file=sys.stderr,
        flush=True,
    )
    primary_unique_scores = _score_map(
        primary,
        primary_unique_pairs,
        label="primary",
        journal_path=output_dir / "primary_scores_cuda_dedup_v2.sqlite",
        identity=contract,
    )
    primary_scores = [primary_unique_scores[index] for index in primary_inverse]

    retained = int(raw["retained_negatives"])
    arm_candidates: dict[str, list[list[dict[str, Any]]]] = {arm: [] for arm in ARMS}
    for row_index, (start, n0, n1, n2) in enumerate(offsets):
        positive_score = primary_scores[start]
        cursor = start + 1
        scored_groups: list[list[dict[str, Any]]] = []
        for size, values in (
            (n0, inherited[row_index]),
            (n1, bm25_rows[row_index]),
            (n2, dense_rows[row_index]),
        ):
            scored: list[dict[str, Any]] = []
            for value, score in zip(values, primary_scores[cursor : cursor + size], strict=True):
                scored.append(dict(value) | {"primary_score": float(score)})
            scored_groups.append(scored)
            cursor += size
        inherited_scored: list[dict[str, Any]]
        bm25_scored: list[dict[str, Any]]
        dense_scored: list[dict[str, Any]]
        inherited_scored, bm25_scored, dense_scored = scored_groups
        arm_candidates["hn0"].append(inherited_scored[:retained])
        arm_candidates["hn0_filter"].append(
            [value for value in inherited_scored if value["primary_score"] < calibration.threshold][
                :retained
            ]
        )
        arm_candidates["hn1_bm25"].append(
            [value for value in bm25_scored if value["primary_score"] < calibration.threshold][
                :retained
            ]
        )
        arm_candidates["hn2_biencoder"].append(
            [value for value in dense_scored if value["primary_score"] < calibration.threshold][
                :retained
            ]
        )
        union = stable_union(bm25_scored, dense_scored)
        by_id = {str(value["doc_id"]): value for value in bm25_scored + dense_scored}
        arm_candidates["hn3_union_positive_filter"].append(
            [
                by_id[str(value["doc_id"])] | {"miners": value["miners"]}
                for value in union
                if positive_aware_keep(
                    negative_score=float(by_id[str(value["doc_id"])]["primary_score"]),
                    positive_score=float(positive_score),
                    absolute_threshold=calibration.threshold,
                )
            ][:retained]
        )

    legal = [
        index
        for index in range(len(records))
        if all(len(arm_candidates[arm][index]) >= retained for arm in ARMS)
    ]
    if not legal:
        raise ValueError("full HN gate has no common legal dev cohort")
    del primary
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    shadow = load_frozen_reranker(shadow_config)
    shadow_pairs: list[tuple[str, str]] = []
    shadow_offsets: dict[tuple[str, int], int] = {}
    for record_index in legal:
        query = str(records[record_index]["query"])
        for arm in ARMS:
            shadow_offsets[(arm, record_index)] = len(shadow_pairs)
            shadow_pairs.append((query, str(records[record_index]["positives"][0]["text"])))
            shadow_pairs.extend(
                (query, str(value["text"])) for value in arm_candidates[arm][record_index]
            )
    shadow_unique_pairs, shadow_inverse = deduplicate_scoring_pairs(shadow_pairs)
    print(
        f"[HN gate] shadow deduplicated {len(shadow_pairs)} -> {len(shadow_unique_pairs)} pairs",
        file=sys.stderr,
        flush=True,
    )
    shadow_unique_scores = _score_map(
        shadow,
        shadow_unique_pairs,
        label="shadow",
        journal_path=output_dir / "shadow_scores_cuda_dedup_v2.sqlite",
        identity=contract,
    )
    shadow_scores = [shadow_unique_scores[index] for index in shadow_inverse]

    arm_rows: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    for record_index in legal:
        start, n0, n1, n2 = offsets[record_index]
        primary_positive = float(primary_scores[start])
        for arm in ARMS:
            candidates = arm_candidates[arm][record_index]
            primary_negative = [float(value["primary_score"]) for value in candidates]
            shadow_start = shadow_offsets[(arm, record_index)]
            shadow_positive = float(shadow_scores[shadow_start])
            shadow_negative = [
                float(value)
                for value in shadow_scores[shadow_start + 1 : shadow_start + 1 + retained]
            ]
            primary_metric = pool_metrics(primary_positive, primary_negative)
            shadow_metric = pool_metrics(shadow_positive, shadow_negative)
            arm_rows[arm].append(
                {
                    "example_id": str(records[record_index]["example_id"]),
                    **primary_metric,
                    "positive_score": primary_positive,
                    "negative_scores": primary_negative,
                    "shadow_pool_mrr": shadow_metric["pool_mrr"],
                    "shadow_pool_ndcg_at_10": shadow_metric["pool_ndcg_at_10"],
                    "judge_winner_disagreement": (primary_metric["pool_recall_at_1"] > 0)
                    != (shadow_metric["pool_recall_at_1"] > 0),
                    "possible_false_negative_rate": sum(
                        value >= calibration.threshold for value in primary_negative
                    )
                    / len(primary_negative),
                    "negative_provenance": candidates,
                }
            )
    for arm, rows in arm_rows.items():
        with JsonlWriter(output_dir / f"{arm}.jsonl") as writer:
            for row in rows:
                writer.write(row)

    summary = {
        "schema_version": 1,
        "status": "measured",
        "contract": contract,
        "common_legal_query_count": len(legal),
        "common_legal_drop_rate": 1 - len(legal) / len(records),
        "arms": {
            arm: summarize_rows(rows)
            | {
                "judge_winner_disagreement_rate": sum(
                    bool(row["judge_winner_disagreement"]) for row in rows
                )
                / len(rows)
            }
            for arm, rows in arm_rows.items()
        },
        "paired_bootstrap": compare_to_reference(
            arm_rows, samples=int(raw["bootstrap_samples"]), seed=int(raw["seed"])
        ),
        "provenance": {
            "bm25_index_fingerprint": bm25_manifest["index_fingerprint"],
            "dense_cache_identity": dense_manifest["identity"],
            "dense_model_train_summary_sha256": sha256_file(
                root / str(inputs["dense_train_summary"])
            ),
            "dense_mining_corpus": "train-only row mask over immutable full-corpus shards",
            "calibration": calibration.to_manifest(),
            "primary_judge": primary_config.__dict__,
            "shadow_judge": shadow_config.__dict__,
        },
        "final_tests_used": [],
        "training_runs": [],
        "p06_t_modified": False,
    }
    summary["artifact_fingerprint"] = canonical_fingerprint(summary)
    write_json(output_dir / "summary.json", summary)
    write_json(report_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/evaluation/hn_full_gate_v1.yaml")
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    result = run(args.config, args.root.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
