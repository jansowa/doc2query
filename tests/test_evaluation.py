from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from doc2query.evaluation.bootstrap import assert_same_test_fingerprint, paired_bootstrap
from doc2query.evaluation.comparison import compare_generator_runs, rank_variants
from doc2query.evaluation.corpus import (
    BiEncoderCorpusIndex,
    BM25CorpusIndex,
    BM25IndexConfig,
    FrozenBiEncoderConfig,
    backfill_candidate_pools,
    build_biencoder_index,
    build_bm25_index,
    evaluate_round_trip_query,
)
from doc2query.evaluation.datasets import (
    evaluation_fingerprint,
    freeze_evaluation_sets,
    load_frozen_records,
    verify_frozen_manifest,
)
from doc2query.evaluation.diversity import diversity_metrics
from doc2query.evaluation.embedder_probe import prepare_probe_pairs
from doc2query.evaluation.format import format_metrics
from doc2query.evaluation.human import cohen_kappa, fleiss_kappa
from doc2query.evaluation.intrinsic import evaluate_intrinsic_records
from doc2query.evaluation.retrieval import (
    candidate_pool_metrics_from_rank,
    corpus_metrics_from_positive_ranks,
    distribution,
    ndcg,
    validate_recall_cutoffs,
)
from doc2query.evaluation.slices import aggregate_slices
from doc2query.utils.records import JsonlWriter


def _canonical(identifier: str, negative_count: int = 10) -> dict[str, Any]:
    return {
        "example_id": identifier,
        "query": "Gdzie leży Warszawa?",
        "positives": [
            {
                "doc_id": f"p-{identifier}",
                "text": "Warszawa jest stolicą Polski. Miasto leży nad Wisłą.",
                "metadata": {},
            }
        ],
        "hard_negatives": [
            {
                "doc_id": f"n-{identifier}-{index}",
                "text": f"Kraków ma zabytek numer {index}.",
                "metadata": {},
            }
            for index in range(negative_count)
        ],
        "metadata": {"source": "fixture", "split": "test"},
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with JsonlWriter(path) as writer:
        for row in rows:
            writer.write(row)


def test_freeze_and_verify_rank10_subset(tmp_path: Path) -> None:
    dev, test, adversarial = tmp_path / "dev.jsonl", tmp_path / "test.jsonl", tmp_path / "adv.jsonl"
    _write_jsonl(dev, [_canonical("d1"), _canonical("d2", 9)])
    _write_jsonl(test, [_canonical("t1"), _canonical("t2", 9)])
    _write_jsonl(adversarial, [{"case_id": "a1", "passage": "A", "query": "Q"}])
    (tmp_path / "split_manifest.json").write_text("{}\n", encoding="utf-8")
    output = tmp_path / "frozen"
    manifest = freeze_evaluation_sets(
        dev_path=dev,
        test_path=test,
        adversarial_path=adversarial,
        output_dir=output,
        human_panel_size=1,
        generation_panel_size=1,
    )
    rank10 = manifest["sets"]["test_intrinsic_rank10"]
    assert rank10["id_count"] == 1
    assert rank10["excluded_count"] == 1
    assert len(load_frozen_records(output / "manifest.json", "test_intrinsic_rank10")) == 1
    assert verify_frozen_manifest(output / "manifest.json")["verified"]["test_embedder"] == 2
    assert len(evaluation_fingerprint(output / "manifest.json", "test_intrinsic")) == 64
    with pytest.raises(FileExistsError):
        freeze_evaluation_sets(
            dev_path=dev,
            test_path=test,
            adversarial_path=adversarial,
            output_dir=output,
        )


def test_frozen_set_detects_id_tampering(tmp_path: Path) -> None:
    source = tmp_path / "test.jsonl"
    _write_jsonl(source, [_canonical("x")])
    _write_jsonl(tmp_path / "adv.jsonl", [{"case_id": "a", "passage": "a", "query": "q"}])
    (tmp_path / "split_manifest.json").write_text("{}\n", encoding="utf-8")
    output = tmp_path / "frozen"
    freeze_evaluation_sets(
        dev_path=source,
        test_path=source,
        adversarial_path=tmp_path / "adv.jsonl",
        output_dir=output,
        human_panel_size=1,
        generation_panel_size=1,
    )
    (output / "test_intrinsic.ids.jsonl").write_text('{"id":"changed"}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="ID-list"):
        load_frozen_records(output / "manifest.json", "test_intrinsic")


def test_known_retrieval_rankings() -> None:
    first = candidate_pool_metrics_from_rank(1, candidate_count=11)
    third = candidate_pool_metrics_from_rank(3, candidate_count=11)
    assert first["pool_mrr"] == 1.0
    assert third["pool_mrr"] == pytest.approx(1 / 3)
    assert third["pool_recall_at_1"] == 0.0
    assert third["pool_recall_at_5"] == 1.0
    assert third["pool_ndcg_at_10"] == pytest.approx(1 / 2)
    assert third["pool_candidate_count"] == 11
    assert ndcg([0, 1, 0], 10) == pytest.approx(1 / 1.584962500721156)
    multi = corpus_metrics_from_positive_ranks([1, 3], candidate_count=1000)
    assert multi["corpus_recall_at_1"] == 0.5
    assert multi["corpus_recall_at_5"] == 1.0
    assert multi["corpus_map"] == pytest.approx((1 + 2 / 3) / 2)
    assert multi["corpus_candidate_count"] == 1000


def test_protocol_metric_names_are_disjoint_and_recall_cutoff_is_validated() -> None:
    pool = candidate_pool_metrics_from_rank(1, candidate_count=11)
    corpus = corpus_metrics_from_positive_ranks([1], candidate_count=100)
    assert set(pool).isdisjoint(corpus)
    assert all(key.startswith("pool_") for key in pool)
    assert all(key.startswith("corpus_") for key in corpus)
    with pytest.raises(ValueError, match="only 4 documents"):
        candidate_pool_metrics_from_rank(1, candidate_count=4)
    with pytest.raises(ValueError, match="recall@100"):
        corpus_metrics_from_positive_ranks([1], candidate_count=99)
    with pytest.raises(ValueError, match="only 3 documents"):
        validate_recall_cutoffs(3, (1, 5))


def test_duplicates_have_expected_diversity_penalty() -> None:
    duplicate = diversity_metrics(["pompa ciepła", "pompa ciepła"])
    diverse = diversity_metrics(["pompa ciepła", "stolica Polski"])
    assert duplicate["duplicate_rate"] == 0.5
    assert duplicate["distinct_1"] < diverse["distinct_1"]
    assert duplicate["mean_pairwise_lemma_jaccard"] == 1.0
    assert duplicate["mean_pairwise_embedding_cosine"] is None


def test_bootstrap_is_seeded_and_fingerprint_safe() -> None:
    left = {"a": 0.0, "b": 1.0, "c": 0.0}
    right = {"a": 1.0, "b": 1.0, "c": 1.0}
    first = paired_bootstrap(left, right, samples=100, seed=7)
    assert first == paired_bootstrap(left, right, samples=100, seed=7)
    assert first["difference"] == pytest.approx(2 / 3)
    assert_same_test_fingerprint({"test_fingerprint": "x"}, {"test_fingerprint": "x"})
    with pytest.raises(ValueError, match="fingerprint"):
        assert_same_test_fingerprint({"test_fingerprint": "x"}, {"test_fingerprint": "y"})


def test_slices_sum_and_missing_is_not_zero() -> None:
    rows = [
        {"score": 1.0, "missing": None, "slices": {"domain": "a"}},
        {"score": 0.0, "missing": None, "slices": {"domain": "b"}},
    ]
    result = aggregate_slices(rows, slice_fields=["domain"], metric_fields=["score", "missing"])
    assert sum(value["count"] for value in result["domain"].values()) == len(rows)
    assert result["domain"]["a"]["metrics"]["missing"] is None
    assert distribution([]) is None


def test_format_and_agreement_metrics() -> None:
    assert format_metrics("Jak działa pompa ciepła?")["format_valid"]
    assert not format_metrics("Zapytanie: Jak działa pompa?")["format_valid"]
    assert format_metrics('["a", "b"]', multi_query_json=True)["json_valid"]
    assert cohen_kappa(["a", "b"], ["a", "b"]) == 1.0
    assert fleiss_kappa([["a", "a"], ["b", "b"]]) == 1.0


def test_probe_controls_keep_sampling_identical(tmp_path: Path) -> None:
    records = [_canonical("1"), _canonical("2")]
    natural, natural_hash = prepare_probe_pairs(records, query_source="natural")
    copied, copied_hash = prepare_probe_pairs(records, query_source="copy_control")
    assert natural_hash != copied_hash
    assert [row["positive"] for row in natural] == [row["positive"] for row in copied]
    assert [row["negative"] for row in natural] == [row["negative"] for row in copied]
    generations = tmp_path / "generated.jsonl"
    _write_jsonl(
        generations,
        [
            {
                "example_id": "1",
                "mode": "deterministic",
                "candidate_index": 0,
                "generated": "syntetyczne pytanie",
            }
        ],
    )
    synthetic, _ = prepare_probe_pairs(
        records, query_source="synthetic", synthetic_generations=generations
    )
    assert len(synthetic) == 1
    assert synthetic[0]["positive"] == natural[0]["positive"]
    assert synthetic[0]["negative"] == natural[0]["negative"]


def test_ranking_requires_probe_metric() -> None:
    intrinsic_only = {"experiment_id": "reward-winner", "reward": 100.0}
    probe = {
        "experiment_id": "probe",
        "probe_embedder": {"corpus_ndcg_at_10": 0.4, "corpus_mrr_at_10": 0.5},
    }
    assert rank_variants([intrinsic_only, probe]) == [probe]


def test_generator_comparison_requires_measured_same_corpus_index(tmp_path: Path) -> None:
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    payload = {"test_fingerprint": "f" * 64, "protocols": {}}
    left.write_text(json.dumps(payload), encoding="utf-8")
    right.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="measured corpus_retrieval"):
        compare_generator_runs(
            left,
            right,
            left_per_generation_path=tmp_path / "missing-left.jsonl",
            right_per_generation_path=tmp_path / "missing-right.jsonl",
            output_path=tmp_path / "comparison.json",
        )


class _OverlapScorer:
    name = "fixture-overlap"

    def score_pairs(self, pairs: Any) -> list[float]:
        result = []
        for query, passage in pairs:
            query_tokens = set(str(query).lower().split())
            passage_tokens = set(str(passage).lower().split())
            result.append(float(len(query_tokens & passage_tokens)))
        return result


def test_intrinsic_smoke_writes_null_for_unmeasured(tmp_path: Path) -> None:
    source = _canonical("1")
    generation = {
        "evaluation_id": "1::deterministic::0",
        "experiment_id": "fixture",
        "example_id": "1",
        "mode": "deterministic",
        "candidate_index": 0,
        "generated": "Gdzie leży Warszawa?",
        "reference": source["query"],
        "positive": source["positives"][0],
        "hard_negatives": source["hard_negatives"],
        "positive_count": 1,
        "metadata": source["metadata"],
    }
    summary = evaluate_intrinsic_records(
        [generation],
        primary=_OverlapScorer(),
        shadow=None,
        output_dir=tmp_path,
        test_fingerprint="f" * 64,
        experiment_id="fixture",
    )
    assert summary["generation_count"] == 1
    assert summary["judges"]["shadow_status"] == "not_measured"
    assert summary["focus"]["control_accuracy"] is None
    assert summary["diversity"]["semantic_cluster_count"] is None
    assert summary["protocols"]["candidate_pool_ranking"]["metric_prefix"] == "pool_"
    assert summary["protocols"]["corpus_retrieval"]["status"] == "not_measured"
    assert json.loads((tmp_path / "summary.json").read_text())["test_fingerprint"] == "f" * 64


def _corpus_documents(count: int = 100) -> list[dict[str, Any]]:
    return [
        {
            "doc_id": f"d-{index:03d}",
            "text": (
                "Warszawa jest stolicą Polski nad Wisłą."
                if index == 0
                else f"Kraków ma zabytek numer {index}."
            ),
            "metadata": {},
        }
        for index in range(count)
    ]


def test_bm25_corpus_round_trip_records_full_pool_and_fingerprint(tmp_path: Path) -> None:
    documents = tmp_path / "documents.jsonl"
    _write_jsonl(documents, _corpus_documents())
    index_dir = tmp_path / "bm25"
    manifest = build_bm25_index(
        documents,
        output_dir=index_dir,
        config=BM25IndexConfig(
            relevance_score_threshold=1.0,
            ambiguity_candidate_threshold=20,
        ),
    )
    assert manifest["protocol"] == "corpus_retrieval"
    assert manifest["candidate_count"] == 100
    assert len(manifest["index_fingerprint"]) == 64
    with BM25CorpusIndex(index_dir) as index:
        result = evaluate_round_trip_query(
            index,
            query="Gdzie leży Warszawa?",
            positive_doc_ids=("d-000",),
        )
    assert result["corpus_candidate_count"] == 100
    assert result["corpus_round_trip_at_1"] == 1.0
    assert result["corpus_round_trip_at_100"] == 1.0
    assert result["corpus_margin_to_best_nonpositive"] > 0
    assert isinstance(result["corpus_effective_candidate_count"], int)


def test_candidate_pool_backfill_is_deterministic_and_marks_provenance(tmp_path: Path) -> None:
    documents = tmp_path / "documents.jsonl"
    _write_jsonl(documents, _corpus_documents())
    source = _canonical("short", negative_count=2)
    first = backfill_candidate_pools(
        [source],
        documents_path=documents,
        corpus_fingerprint="c" * 64,
    )
    second = backfill_candidate_pools(
        [source],
        documents_path=documents,
        corpus_fingerprint="c" * 64,
    )
    assert first == second
    assert len(first[0]["hard_negatives"]) == 10
    assert first[0]["candidate_pool_backfilled_count"] == 8
    assert all(
        document["metadata"]["candidate_pool_backfill"]["corpus_fingerprint"] == "c" * 64
        for document in first[0]["hard_negatives"][2:]
    )


class _FixtureEncoder:
    def encode(self, texts: Any, *, batch_size: int) -> Any:
        import numpy as np

        del batch_size
        return np.asarray(
            [[1.0, 0.0] if "warszaw" in str(text).lower() else [0.0, 1.0] for text in texts],
            dtype=np.float32,
        )


def test_frozen_auxiliary_biencoder_manifest_and_round_trip(tmp_path: Path) -> None:
    documents = tmp_path / "documents.jsonl"
    _write_jsonl(documents, _corpus_documents())
    config = FrozenBiEncoderConfig(
        model_name_or_path="fixture/encoder",
        revision="a" * 40,
        license="apache-2.0",
        relevance_score_threshold=0.5,
    )
    index_dir = tmp_path / "biencoder"
    manifest = build_biencoder_index(
        documents,
        output_dir=index_dir,
        config=config,
        encoder=_FixtureEncoder(),
    )
    assert manifest["config"]["revision"] == "a" * 40
    assert manifest["config"]["license"] == "apache-2.0"
    index = BiEncoderCorpusIndex(index_dir, encoder=_FixtureEncoder())
    try:
        result = evaluate_round_trip_query(
            index,
            query="Warszawa",
            positive_doc_ids=("d-000",),
        )
    finally:
        index.close()
    assert result["corpus_round_trip_at_1"] == 1.0
    assert result["corpus_candidate_count"] == 100


def test_intrinsic_reports_round_trip_and_pool_margin_correlation(tmp_path: Path) -> None:
    corpus_rows = _corpus_documents()
    documents = tmp_path / "documents.jsonl"
    _write_jsonl(documents, corpus_rows)
    index_dir = tmp_path / "bm25"
    build_bm25_index(
        documents,
        output_dir=index_dir,
        config=BM25IndexConfig(relevance_score_threshold=1.0),
    )
    base = {
        "experiment_id": "fixture",
        "example_id": "1",
        "mode": "diverse",
        "reference": "Gdzie leży Warszawa?",
        "positive": corpus_rows[0],
        "hard_negatives": corpus_rows[1:11],
        "positive_count": 1,
        "metadata": {"source": "fixture"},
    }
    generations = [
        {
            **base,
            "evaluation_id": "1::diverse::0",
            "candidate_index": 0,
            "generated": "Warszawa stolica",
        },
        {
            **base,
            "evaluation_id": "1::diverse::1",
            "candidate_index": 1,
            "generated": "Kraków zabytek",
        },
    ]
    with BM25CorpusIndex(index_dir) as index:
        summary = evaluate_intrinsic_records(
            generations,
            primary=_OverlapScorer(),
            shadow=None,
            output_dir=tmp_path / "evaluation",
            test_fingerprint="f" * 64,
            experiment_id="fixture",
            corpus_index=index,
        )
    corpus = summary["protocols"]["corpus_retrieval"]
    assert corpus["status"] == "measured"
    assert corpus["candidate_count"] == 100
    assert corpus["metrics"]["corpus_round_trip_at_20"] == 0.5
    assert corpus["round_trip_pool_margin_correlation"]["corpus_round_trip_at_20"] == 1.0
