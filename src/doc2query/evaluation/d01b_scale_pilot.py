"""Prospective, fail-closed D01b 4.5B scale-interaction pilot."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import yaml

from doc2query.config import load_config
from doc2query.evaluation.corpus import evaluate_round_trip_queries, load_corpus_index
from doc2query.evaluation.d01_pipeline import _artifact_fingerprint
from doc2query.evaluation.d01_prospective import prepare_prospective_cohort
from doc2query.evaluation.datasets import load_frozen_records
from doc2query.evaluation.embedder_probe import ProbeRecipe
from doc2query.evaluation.p04_decision import evaluate_p04_comparison
from doc2query.evaluation.p05_guardrails import build_dev_screen_report
from doc2query.evaluation.statistical_contract import StatisticalContract
from doc2query.utils.records import read_records, write_json

CONTRACT = "task05-d01b-scale-interaction-4.5b-pilot-v1"
EVALUATION_SUBSET = "dev_d01b_scale_pilot_v1"
EXPECTED_SELECTOR_WEIGHTS = {
    "natural_margin_alignment": 0.35,
    "semantic_diversity": 0.30,
    "lexical_diversity": 0.10,
    "corpus_specificity": 0.15,
    "low_copy_density": 0.10,
}
EXPECTED_METRICS = [
    "pool_recall_at_1",
    "shadow_pool_recall_at_1",
    "corpus_round_trip_at_20",
    "sentence_level_source_hit",
    "format_valid_rate",
    "copy_risk_rate",
    "semantic_diversity",
    "duplicate_rate",
]
EXPECTED_SHARED_MODEL = {
    "name_or_path": "speakleash/Bielik-4.5B-v3.0-Instruct",
    "revision": "4b1220a9d745bdd874c44347075ef25484ef322b",
    "trust_remote_code": False,
}
EXPECTED_DECODING = {
    "seed": 42,
    "do_sample": True,
    "temperature": 0.8,
    "top_p": 0.95,
    "max_new_tokens": 64,
    "max_attempts_per_query": 16,
    "queries_per_arm_per_passage": 4,
    "exact_k_required": True,
    "generation_batch_size": 8,
    "passages_per_arm": 1000,
    "candidate_queries_per_arm": 4000,
    "maximum_new_token_budget_per_arm": 256000,
}
EXPECTED_INTRINSIC_GATES = {
    "pool_recall_at_1": {"direction": "higher", "noninferiority_margin": 0.02},
    "corpus_round_trip_at_20": {
        "direction": "higher",
        "noninferiority_margin": 0.02,
    },
    "sentence_level_source_hit": {
        "direction": "higher",
        "noninferiority_margin": 0.02,
    },
    "format_valid_rate": {"direction": "higher", "noninferiority_margin": 0.005},
    "shadow_pool_recall_at_1": {
        "direction": "higher",
        "noninferiority_margin": 0.02,
    },
    "copy_risk_rate": {"direction": "lower", "maximum_upper_ci": 0.0},
    "semantic_diversity": {"direction": "higher", "noninferiority_margin": 0.0},
    "duplicate_rate": {"direction": "lower", "maximum_upper_ci": 0.0},
}
EXPECTED_P04_THRESHOLDS = {
    "corpus_ndcg_at_10": 0.01,
    "corpus_round_trip_at_20": -0.02,
    "sentence_level_source_hit": -0.02,
    "format_valid_rate": -0.005,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _records_sha256(records: Sequence[Mapping[str, Any]], *, sort_ids: bool = False) -> str:
    values = list(records)
    if sort_ids:
        values.sort(key=lambda row: str(row["example_id"]))
    digest = hashlib.sha256()
    for row in values:
        digest.update(_canonical(row))
        digest.update(b"\n")
    return digest.hexdigest()


def _ids_sha256(ids: Sequence[str], *, sort_ids: bool = False) -> str:
    values = sorted(ids) if sort_ids else list(ids)
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def _root(path: Path) -> Path:
    root = next(
        (parent for parent in path.resolve().parents if (parent / "AGENTS.md").is_file()), None
    )
    if root is None:
        raise ValueError("cannot resolve repository root")
    return root


def _load(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping: {path}")
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_json(temporary, value)
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _assert_pin(root: Path, section: Mapping[str, Any], *, key: str = "path") -> Path:
    path = root / str(section[key])
    if not path.is_file() or _sha256(path) != str(section["sha256"]):
        raise ValueError(f"scale-pilot pin drifted: {path}")
    return path


def _assert_contract_shape(config: Mapping[str, Any]) -> None:
    """Validate scientific and authorization constants before touching model outputs."""
    if (
        config.get("schema_version") != 1
        or config.get("contract") != CONTRACT
        or config.get("status") != "preregistered_before_pilot"
        or config.get("hypothesis_formed_after_1_5b_result") is not True
        or config.get("final_tests_used") != []
    ):
        raise ValueError("invalid D01b scale-pilot contract")
    forbidden = (
        "test_intrinsic",
        "test_embedder",
        "test_native_pl",
        "test_translated_msmarco_pl",
        "test_generator_panel",
        "test_human_panel",
        "test_adversarial",
        "/test.parquet",
    )

    def strings(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, Mapping):
            return [item for nested in value.values() for item in strings(nested)]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [item for nested in value for item in strings(nested)]
        return []

    if any(token in value.lower() for value in strings(config) for token in forbidden):
        raise ValueError("final-test reference is forbidden in the scale-pilot contract")
    if config.get("adr") != {
        "path": "reports/decisions/task05_d01b_scale_interaction_4_5b_pilot_v1.md",
        "sha256": "a3eaea2f7772d642e7125f5822ca5bedf022fd4eb49ef05b98bf178c4baf017d",
    }:
        raise ValueError("scale-pilot ADR pin drifted")
    source = cast(Mapping[str, Any], config.get("source_1_5b_decision", {}))
    if source != {
        "path": "reports/measurements/task05/d01b_probe_dev_confirm_v2_batch2/summary.json",
        "status": "dev_confirm_complete",
        "decision": "non_inferior_only",
        "mean_difference": 0.011187611748928645,
        "ci95": [0.006927431152133765, 0.015391286067850935],
        "minimum_practical_effect": 0.01,
        "retained_for_finalist_freeze": False,
        "four_point_five_b_authorized": False,
        "final_tests_used": [],
    }:
        raise ValueError("completed 1.5B result was reinterpreted or drifted")
    selector = cast(Mapping[str, Any], config.get("selector", {}))
    if (
        selector.get("frozen_commit") != "2164822"
        or selector.get("implementation")
        != {
            "path": "src/doc2query/evaluation/d01_usefulness.py",
            "sha256": "f8bb6ccd491a6e4f3fd721ca6368bcc75a450455082f5d0f6aa94a96e29c764c",
        }
        or selector.get("retrospective_contract")
        != {
            "path": "configs/evaluation/d01b_usefulness_hybrid_v1.yaml",
            "sha256": "0ba63995648c57c5d68ec23c6b0c54008036c914d8d75f241f5a07db8c84abd5",
        }
        or selector.get("candidate_count") != 8
        or selector.get("selected_count") != 4
        or selector.get("anchor") != "all_four_observed_uncontrolled_slots"
        or selector.get("duplicate_candidate_text_allowed") is not True
        or selector.get("enumerate_all_subsets") is not True
        or selector.get("feasibility_not_below_anchor")
        != [
            "pool_recall_at_1",
            "corpus_round_trip_at_20",
            "sentence_level_source_hit",
            "format_valid",
        ]
        or selector.get("copy_risk_count_not_above_anchor") is not True
        or selector.get("objective_weights") != EXPECTED_SELECTOR_WEIGHTS
        or selector.get("natural_margin_scale") != "cohort_iqr_with_floor_1e-6"
        or selector.get("shadow_reserved_from_selection") is not True
        or selector.get("deterministic_tie_break") != "lexicographic_candidate_identity"
    ):
        raise ValueError("D01b v3 selector drifted")
    cohort = cast(Mapping[str, Any], config.get("cohort", {}))
    if (
        cohort.get("manifest")
        != "reports/preregistrations/task05_d01b_scale_interaction_4_5b_pilot_v1.cohort.json"
        or cohort.get("manifest_sha256")
        != "f0b16d021371df62ed5f12a35254b64e244b8b111390ef8003db28db27e05231"
        or cohort.get("source_frozen_manifest")
        != "data/processed/v1/evaluation/task04-v1/manifest.json"
        or cohort.get("source_subset") != "dev_intrinsic"
        or cohort.get("exclude_subset") != "dev_intrinsic_rank10"
        or cohort.get("source_records") != "data/processed/v1/dev.parquet"
        or cohort.get("source_records_sha256")
        != "12b86da10c73707a58a8793111afcca0424981d738667a04c228014780a02a1d"
        or cohort.get("minimum_hard_negatives") != 5
        or cohort.get("available_after_exclusions") != 3674
        or cohort.get("eligible_count") != 3591
        or cohort.get("selected_count") != 1000
        or cohort.get("evaluation_offset") != 1000
        or cohort.get("evaluation_count") != 2000
        or cohort.get("selection_seed") != 20260809
        or cohort.get("selected_id_list_sha256")
        != "931777e0fd51fa7cc12213ae4fd59e54791817cc9fe1212806f90d680e1f1cf9"
        or cohort.get("selected_records_sha256")
        != "a8b9255210633f2632c7744e38e72bbbe1a360a6fd879fbbc89ed2fc694e9fd6"
        or cohort.get("evaluation_id_list_sha256")
        != "52d7a8ec9e63d68d902c306957a4d7d08530d81ade3b82185a70c7415499cd38"
        or cohort.get("evaluation_records_sha256")
        != "b26523ad105fcfe09706aeff364fc59aea138ef145f44f7238118fc458627daa"
        or cohort.get("intersection_with_excluded_subset") != 0
        or cohort.get("intersection_with_prior_cohorts") != 0
    ):
        raise ValueError("scale-pilot cohort or exclusions drifted")
    arms = cast(Mapping[str, Any], config.get("arms", {}))
    if arms.get("shared_model") != EXPECTED_SHARED_MODEL:
        raise ValueError("scale-pilot Bielik model/revision drifted")
    expected_arm_pins = {
        "baseline": {
            "id": "W06-4.5B-INSTRUCT-50K-8GB-BS8-L512",
            "generation_experiment_id": "D01B-SCALE-PILOT-W06-4.5B-S42",
            "generation_config": "configs/experiments/d01b_scale_pilot_w06_4_5b_s42.yaml",
            "generation_config_sha256": (
                "0e6539e535a1a51a382753cb69069452c1ab227f4714bcb53ec81c9a6aa0b53d"
            ),
            "adapter": "runs/W06-4.5B-INSTRUCT-50K-8GB-BS8-L512/adapter",
            "adapter_sha256": ("9253810026385c8749bfbb4de9b3520e1b0a73fd16020c98e728d8ff405d73e2"),
            "training_manifest": "runs/W06-4.5B-INSTRUCT-50K-8GB-BS8-L512/run_manifest.json",
            "training_manifest_sha256": (
                "4a97904a62060a3bd376ad6dfe91963bf75c798d090f191dadc80dfd16912c5b"
            ),
            "preserve_duplicate_slots": True,
        },
        "controlled": {
            "id": "D01-4.5B-STYLE-50K-S42",
            "generation_experiment_id": "D01B-SCALE-PILOT-D01-4.5B-S42",
            "generation_config": "configs/experiments/d01b_scale_pilot_d01_4_5b_s42.yaml",
            "generation_config_sha256": (
                "ae1be0d5b7b4ae5ccbdbaceb7b86693c23e24c4525f90f699294301ac10d83b6"
            ),
            "adapter": "runs/D01-4.5B-STYLE-50K-S42/adapter",
            "adapter_sha256": ("71937228ea977d9d6a89613fe6f802fc3711dba9499a8e23c6c1e4e21e77a867"),
            "training_manifest": "runs/D01-4.5B-STYLE-50K-S42/run_manifest.json",
            "training_manifest_sha256": (
                "0c0d466babe0cab0e4c8e67ae30948cf727cc95a221234ba3b8a32c1e3f6452c"
            ),
            "preserve_duplicate_slots": False,
        },
    }
    if any(arms.get(role) != expected for role, expected in expected_arm_pins.items()):
        raise ValueError("scale-pilot arm identity or adapter pin drifted")
    decoding = cast(Mapping[str, Any], arms.get("decoding", {}))
    if decoding != EXPECTED_DECODING:
        raise ValueError("scale-pilot decoding or generation budget drifted")
    scoring = cast(Mapping[str, Any], config.get("scoring", {}))
    if scoring != {
        "primary": {
            "config": "configs/reranker/primary_polish_roberta_v3_cuda.yaml",
            "name_or_path": "sdadas/polish-reranker-roberta-v3",
            "revision": "e6471da541f4e7be33845b6d57248a8d8bde27e8",
            "trust_remote_code": False,
        },
        "shadow": {
            "config": "configs/reranker/shadow_bge_v2_m3.yaml",
            "name_or_path": "BAAI/bge-reranker-v2-m3",
            "revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
            "trust_remote_code": False,
            "reserved_from_selection": True,
        },
        "corpus_index": {
            "path": "data/processed/v1/evaluation/corpus-bm25-v1",
            "fingerprint": "159af07f3b987fe492f9ff89f494521587a4d023fcb4fc62b69d7295d2a57258",
        },
        "semantic_model": {
            "name_or_path": "OPI-PIB/PolDense-150M",
            "revision": "b94ea7f951cc480369a85fa9021694eef80c3a00",
            "trust_remote_code": False,
            "similarity_prefix": "[sts]: ",
            "normalize_embeddings": True,
            "batch_size": 128,
        },
    }:
        raise ValueError("scale-pilot judge, corpus, or semantic-model pin drifted")
    if config.get("natural_primary_scores") != {
        "path": "artifacts/task02/pfn_dev_v1/primary_scores.jsonl",
        "sha256": "7a1f9f5949a26d3c09780ba30689fb259c1d20cd401a1e8860b8292461eef115",
        "schema": "possible_false_negative_dev_scores_v1",
        "same_positive_and_intersecting_negatives": True,
    }:
        raise ValueError("scale-pilot natural-primary pin drifted")
    if config.get("copy_risk") != {
        "minimum_query_words": 4,
        "copy_density": 0.6,
        "normalized_lcs": 0.8,
        "longest_copied_ngram": 3.0,
        "query_to_passage_length_ratio": 0.3838827838827845,
    }:
        raise ValueError("scale-pilot copy-risk guardrail drifted")
    evaluation = cast(Mapping[str, Any], config.get("evaluation", {}))
    if (
        evaluation.get("required_metrics") != EXPECTED_METRICS
        or evaluation.get("gates") != EXPECTED_INTRINSIC_GATES
        or evaluation.get("comparison") != "hybrid_minus_observed_4_5b_baseline_anchor"
        or evaluation.get("resampling_unit") != "passage_query_group"
        or evaluation.get("bootstrap_samples") != 10_000
        or evaluation.get("bootstrap_seed") != 20_260_721
        or evaluation.get("interval") != "paired_percentile_95"
        or evaluation.get("all_gates_required_for_probe_authorization") is not True
        or evaluation.get("final_tests_forbidden") is not True
        or cast(Mapping[str, Any], cast(Mapping[str, Any], config["scoring"])["shadow"]).get(
            "reserved_from_selection"
        )
        is not True
    ):
        raise ValueError("scale-pilot metrics/bootstrap/shadow boundary drifted")
    probe = cast(Mapping[str, Any], config.get("probe", {}))
    expected_probe = {
        "seed": 42,
        "input_passages": 768,
        "input_pairs": 3072,
        "queries_per_passage": 4,
        "max_steps": 1024,
        "batch_size": 2,
        "max_length": 192,
        "token_count": 1_179_648,
        "evaluation_queries": 2000,
        "bootstrap_samples": 10_000,
        "bootstrap_seed": 20_260_721,
    }
    if any(probe.get(key) != value for key, value in expected_probe.items()):
        raise ValueError("scale-pilot probe budget, seed, or bootstrap drifted")
    if (
        probe.get("recipe") != "configs/evaluation/probe_v1.yaml"
        or probe.get("recipe_sha256")
        != "3ba6114ef37652d2fa49bb788e61a7ff26d6af59ecad50f9697715d53dc1f00b"
        or probe.get("comparison_contract") != "configs/evaluation/comparison_contract_v1.yaml"
        or probe.get("comparison_contract_sha256")
        != "75c2060d8a48b1ceaa554d97b4ec9af607b4b07634dffe29717d35d77c3f5b5c"
        or probe.get("primary_judge")
        != "configs/reranker/primary_polish_roberta_v3_p03_gpu_batch4.yaml"
        or probe.get("primary_judge_sha256")
        != "7c305ea90d186c4977549f309522db1fe4c4f50b419b6fac39fd23a2b3722008"
        or probe.get("model_name_or_path") != "sdadas/polish-reranker-base-ranknet"
        or probe.get("model_revision") != "a7c66d41a8097ca02e75616d0951c941d94ff6a1"
        or probe.get("corpus") != "data/processed/v1/documents.parquet"
        or probe.get("corpus_sha256")
        != "78f9f2be82ba6e42b80e80b4f77c4ff050c3ec783f1fc422f5b21313381deca7"
        or probe.get("checkpoint_interval_steps") != 64
        or probe.get("evaluation_encode_batch_size") != 8
        or probe.get("retrieval_query_batch_size") != 512
        or probe.get("retrieval_device") != "cuda"
    ):
        raise ValueError("scale-pilot probe model or execution recipe drifted")
    if probe.get("p04_thresholds") != EXPECTED_P04_THRESHOLDS:
        raise ValueError("P-04 thresholds drifted")
    multiplicity = cast(Mapping[str, Any], config.get("multiplicity", {}))
    if multiplicity != {
        "family": "D01b_method_by_generator_scale",
        "pilot_claim": "screen_only_no_confirmatory_claim",
        "one_point_five_b_result_unchanged": True,
        "dev_confirm_required_seeds": [42, 43, 44],
        "dev_confirm_primary_interval_minimum": "paired_percentile_two_sided_97_5",
    }:
        raise ValueError("scale-pilot multiplicity control drifted")
    resources = cast(Mapping[str, Any], config.get("resources", {}))
    if (
        resources.get("minimum_free_disk_bytes") != 18_000_000_000
        or resources.get("expected_peak_vram_gib") != 7.5
        or resources.get("expected_incremental_disk_gib") != 17
        or resources.get("eta_hours") != [10, 14]
    ):
        raise ValueError("scale-pilot resource floor or estimate drifted")
    authorization = cast(Mapping[str, Any], config.get("authorization", {}))
    if authorization != {
        "pilot_generation": True,
        "pilot_scoring": True,
        "pilot_probe_training": True,
        "dev_confirm": False,
        "retained_for_finalist_freeze": False,
        "four_point_five_b_full_authorized": False,
        "final_tests": False,
    }:
        raise ValueError("scale-pilot authorization scope drifted")


def _cohorts(
    config: Mapping[str, Any], root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cohort = cast(Mapping[str, Any], config["cohort"])
    records = load_frozen_records(
        root / str(cohort["source_frozen_manifest"]), str(cohort["source_subset"])
    )
    rank10 = {
        str(row["id"])
        for row in read_records(
            root / "data/processed/v1/evaluation/task04-v1/dev_intrinsic_rank10.ids.jsonl"
        )
    }
    used: set[str] = set()
    for prior in cast(Sequence[Mapping[str, Any]], cohort["prior_cohort_exclusions"]):
        manifest = root / str(prior["manifest"])
        if _sha256(manifest) != str(prior["manifest_sha256"]):
            raise ValueError("prior prospective cohort manifest drifted")
        eligible = [
            row
            for row in records
            if str(row["example_id"]) not in rank10 | used
            and len(cast(list[Any], row.get("hard_negatives", []))) >= 5
        ]
        seed = int(prior["selection_seed"])
        eligible.sort(
            key=lambda row: (
                hashlib.sha256(f"{seed}:{row['example_id']}".encode()).hexdigest(),
                str(row["example_id"]),
            )
        )
        ids = [str(row["example_id"]) for row in eligible[: int(prior["selected_count"])]]
        if _ids_sha256(ids) != str(prior["selected_id_list_sha256"]):
            raise ValueError("prior prospective cohort reconstruction drifted")
        used.update(ids)
    remaining = [row for row in records if str(row["example_id"]) not in rank10 | used]
    eligible = [
        row
        for row in remaining
        if len(cast(list[Any], row.get("hard_negatives", [])))
        >= int(cohort["minimum_hard_negatives"])
    ]
    eligible.sort(
        key=lambda row: (
            hashlib.sha256(f"{cohort['selection_seed']}:{row['example_id']}".encode()).hexdigest(),
            str(row["example_id"]),
        )
    )
    start = int(cohort["evaluation_offset"])
    generation = eligible[: int(cohort["selected_count"])]
    evaluation = eligible[start : start + int(cohort["evaluation_count"])]
    if len(remaining) != int(cohort["available_after_exclusions"]) or len(eligible) != int(
        cohort["eligible_count"]
    ):
        raise ValueError("scale-pilot remaining development population drifted")
    checks = {
        "generation_ids": _ids_sha256([str(row["example_id"]) for row in generation]),
        "generation_records": _records_sha256(generation),
        "evaluation_ids": _ids_sha256([str(row["example_id"]) for row in evaluation]),
        "evaluation_records": _records_sha256(evaluation),
    }
    expected = {
        "generation_ids": cohort["selected_id_list_sha256"],
        "generation_records": cohort["selected_records_sha256"],
        "evaluation_ids": cohort["evaluation_id_list_sha256"],
        "evaluation_records": cohort["evaluation_records_sha256"],
    }
    if checks != expected:
        raise ValueError("scale-pilot generation/evaluation cohort fingerprint drifted")
    generation_ids = {str(row["example_id"]) for row in generation}
    evaluation_ids = {str(row["example_id"]) for row in evaluation}
    if generation_ids & evaluation_ids or (generation_ids | evaluation_ids) & (rank10 | used):
        raise ValueError("scale-pilot cohorts overlap prior/tuned development data")
    return generation, evaluation


def prepare_scale_pilot_cohorts(config_path: Path) -> dict[str, Any]:
    """Materialize the frozen generation IDs and disjoint natural dev evaluation set."""
    config = _load(config_path)
    _assert_contract_shape(config)
    root = _root(config_path)
    generation, evaluation = _cohorts(config, root)
    cohort = cast(Mapping[str, Any], config["cohort"])
    generation_manifest = root / str(cohort["materialized_generation_manifest"])
    prepare_prospective_cohort(config_path, generation_manifest)

    evaluation_manifest = root / str(cohort["materialized_evaluation_manifest"])
    ids_path = evaluation_manifest.parent / f"{EVALUATION_SUBSET}.ids.jsonl"
    guardrails_path = evaluation_manifest.parent / "natural_guardrails.jsonl"
    sorted_ids = sorted(str(row["example_id"]) for row in evaluation)
    _atomic_jsonl(ids_path, [{"id": value} for value in sorted_ids])

    natural_pin = cast(Mapping[str, Any], config["natural_primary_scores"])
    natural_path = _assert_pin(root, natural_pin)
    natural = {str(row["query_id"]): row for row in read_records(natural_path)}
    corpus = load_corpus_index(
        root / str(cast(Mapping[str, Any], config["scoring"])["corpus_index"]["path"])
    )
    requests: list[tuple[str, Sequence[str]]] = []
    for row in evaluation:
        positive_ids = [
            str(item["doc_id"]) for item in cast(list[dict[str, Any]], row["positives"])
        ]
        requests.append((str(row["query"]), positive_ids))
    corpus_rows = evaluate_round_trip_queries(corpus, requests, workers=6, cutoffs=(20,))
    guardrails: list[dict[str, Any]] = []
    for row, corpus_row in zip(evaluation, corpus_rows, strict=True):
        example_id = str(row["example_id"])
        scores = natural.get(example_id)
        if not isinstance(scores, Mapping):
            raise ValueError("natural primary scores do not cover scale-pilot evaluation")
        positives = [float(value) for value in cast(list[Any], scores["positive_scores"])]
        negatives = [float(value) for value in cast(list[Any], scores["negative_scores"])]
        guardrails.append(
            {
                "example_id": example_id,
                "corpus_round_trip_at_20": float(corpus_row["corpus_round_trip_at_20"]),
                "sentence_level_source_hit": float(max(positives) > max(negatives)),
                "format_valid_rate": float(bool(str(row["query"]).strip())),
            }
        )
    guardrails.sort(key=lambda row: str(row["example_id"]))
    _atomic_jsonl(guardrails_path, guardrails)
    set_spec = {
        "name": EVALUATION_SUBSET,
        "source_path": str(cohort["source_records"]),
        "source_sha256": str(cohort["source_records_sha256"]),
        "id_path": str(ids_path.relative_to(root)),
        "id_field": "example_id",
        "id_count": len(evaluation),
        "id_list_sha256": _ids_sha256(sorted_ids),
        "records_sha256": _records_sha256(evaluation, sort_ids=True),
        "population_count": int(cohort["eligible_count"]),
        "excluded_count": int(cohort["eligible_count"]) - len(evaluation),
        "exclusion_reason": "prospective disjoint scale-pilot evaluation slice",
    }
    manifest = {
        "schema_version": 1,
        "version": "task05-d01b-scale-pilot-v1",
        "seed": int(cohort["selection_seed"]),
        "selection_policy": {
            "source": "unused dev after rank10 and D01b prospective v1/v2/v3",
            "offset": int(cohort["evaluation_offset"]),
            "quality_fields_used": [],
            "split_mutation": False,
        },
        "sets": {EVALUATION_SUBSET: set_spec},
        "final_tests_used": [],
    }
    _atomic_json(evaluation_manifest, manifest)
    return {
        "schema_version": 1,
        "contract": CONTRACT,
        "status": "cohorts_materialized",
        "generation_count": len(generation),
        "evaluation_count": len(evaluation),
        "generation_manifest_sha256": _sha256(generation_manifest),
        "evaluation_manifest_sha256": _sha256(evaluation_manifest),
        "evaluation_ids_sha256": _sha256(ids_path),
        "natural_guardrails_sha256": _sha256(guardrails_path),
        "final_tests_used": [],
    }


def preflight_scale_pilot(
    config_path: Path, *, require_materialized: bool = True
) -> dict[str, Any]:
    """Fail closed on cohort, selector, model, budget, metric, seed, and final-test drift."""
    config = _load(config_path)
    _assert_contract_shape(config)
    root = _root(config_path)
    _assert_pin(root, cast(Mapping[str, Any], config["adr"]))
    source_contract = cast(Mapping[str, Any], config["source_1_5b_decision"])
    source_path = root / str(source_contract["path"])
    source_result = json.loads(source_path.read_text(encoding="utf-8"))
    source_decision = cast(Mapping[str, Any], source_result.get("decision", {}))
    if (
        source_result.get("contract") != "task05-d01b-probe-dev-confirm-v1"
        or source_result.get("status") != "dev_confirm_complete"
        or source_decision.get("status") != "non_inferior_only"
        or source_decision.get("selection_claim") is not None
        or source_result.get("retained_for_finalist_freeze") is not False
        or source_result.get("four_point_five_b_authorized") is not False
        or source_result.get("final_tests_used") != []
    ):
        raise ValueError("actual completed 1.5B decision drifted")
    source_comparison = json.loads(
        (source_path.parent / "comparison_report.json").read_text(encoding="utf-8")
    )
    source_bootstrap = cast(Mapping[str, Any], source_comparison.get("paired_query_bootstrap", {}))
    source_metrics = cast(Mapping[str, Any], source_bootstrap.get("metrics", {}))
    if (
        source_bootstrap.get("rng") != "numpy.random.PCG64"
        or source_bootstrap.get("samples") != 10_000
        or source_bootstrap.get("seed") != 20_260_721
        or source_bootstrap.get("query_count") != 6598
        or source_metrics.get("corpus_ndcg_at_10")
        != {
            "ci_high": 0.015391286067850935,
            "ci_low": 0.006927431152133765,
            "difference": 0.011187611748928645,
            "variant_win_fraction": 1.0,
        }
        or source_comparison.get("final_tests_used") != []
    ):
        raise ValueError("actual completed 1.5B estimate or bootstrap drifted")
    source_criteria = cast(Mapping[str, Any], source_decision.get("criteria", {}))
    if cast(Mapping[str, Any], source_criteria.get("corpus_ndcg_at_10", {})).get(
        "passed"
    ) is not False or any(
        cast(Mapping[str, Any], source_criteria.get(metric, {})).get("passed") is not True
        for metric in (
            "corpus_round_trip_at_20",
            "sentence_level_source_hit",
            "format_valid_rate",
        )
    ):
        raise ValueError("actual completed 1.5B guardrail decision drifted")
    cohort = cast(Mapping[str, Any], config["cohort"])
    _assert_pin(
        root,
        {"path": cohort["manifest"], "sha256": cohort["manifest_sha256"]},
    )
    generation, evaluation = _cohorts(config, root)
    selector = cast(Mapping[str, Any], config["selector"])
    _assert_pin(root, cast(Mapping[str, Any], selector["implementation"]))
    _assert_pin(root, cast(Mapping[str, Any], selector["retrospective_contract"]))
    _assert_pin(root, cast(Mapping[str, Any], config["natural_primary_scores"]))

    arms = cast(Mapping[str, Any], config["arms"])
    shared = cast(Mapping[str, Any], arms["shared_model"])
    observed_arms: dict[str, Any] = {}
    for role in ("baseline", "controlled"):
        arm = cast(Mapping[str, Any], arms[role])
        generation_config = root / str(arm["generation_config"])
        if _sha256(generation_config) != str(arm["generation_config_sha256"]):
            raise ValueError(f"{role} generation config drifted")
        resolved = load_config(generation_config)
        manifest = _assert_pin(
            root,
            {"path": arm["training_manifest"], "sha256": arm["training_manifest_sha256"]},
        )
        manifest_raw = json.loads(manifest.read_text(encoding="utf-8"))
        trained_model = cast(
            Mapping[str, Any], cast(Mapping[str, Any], manifest_raw["config"])["model"]
        )
        if any(trained_model.get(key) != shared.get(key) for key in shared):
            raise ValueError(f"{role} does not share the pinned Bielik checkpoint/revision")
        if (
            resolved.model.name_or_path != shared["name_or_path"]
            or resolved.model.revision != shared["revision"]
            or resolved.model.trust_remote_code is not shared["trust_remote_code"]
            or resolved.run.seed != 42
            or resolved.generation.do_sample is not EXPECTED_DECODING["do_sample"]
            or resolved.generation.temperature != EXPECTED_DECODING["temperature"]
            or resolved.generation.top_p != EXPECTED_DECODING["top_p"]
            or resolved.generation.max_new_tokens != EXPECTED_DECODING["max_new_tokens"]
            or resolved.generation.max_attempts_per_query
            != EXPECTED_DECODING["max_attempts_per_query"]
            or resolved.generation.target_query_count
            != EXPECTED_DECODING["queries_per_arm_per_passage"]
            or resolved.generation.preserve_duplicate_slots
            is not bool(arm["preserve_duplicate_slots"])
        ):
            raise ValueError(f"{role} model/seed/K/duplicate policy drifted")
        adapter = root / str(arm["adapter"])
        if _artifact_fingerprint(adapter) != str(arm["adapter_sha256"]):
            raise ValueError(f"{role} adapter drifted")
        observed_arms[role] = {
            "generation_config_sha256": _sha256(generation_config),
            "training_manifest_sha256": _sha256(manifest),
            "adapter_sha256": _artifact_fingerprint(adapter),
        }
    for role in ("primary", "shadow"):
        pin = cast(Mapping[str, Any], cast(Mapping[str, Any], config["scoring"])[role])
        judge = _load(root / str(pin["config"]))
        if any(
            judge.get(key) != pin.get(key)
            for key in ("name_or_path", "revision", "trust_remote_code")
        ):
            raise ValueError(f"{role} judge identity drifted")
    corpus_pin = cast(Mapping[str, Any], cast(Mapping[str, Any], config["scoring"])["corpus_index"])
    corpus_manifest = json.loads((root / str(corpus_pin["path"]) / "manifest.json").read_text())
    if corpus_manifest.get("index_fingerprint") != corpus_pin["fingerprint"]:
        raise ValueError("corpus index drifted")

    probe = cast(Mapping[str, Any], config["probe"])
    recipe_path = _assert_pin(root, {"path": probe["recipe"], "sha256": probe["recipe_sha256"]})
    _assert_pin(
        root,
        {"path": probe["comparison_contract"], "sha256": probe["comparison_contract_sha256"]},
    )
    _assert_pin(root, {"path": probe["primary_judge"], "sha256": probe["primary_judge_sha256"]})
    _assert_pin(root, {"path": probe["corpus"], "sha256": probe["corpus_sha256"]})
    recipe = ProbeRecipe.from_dict(_load(recipe_path))
    runtime = ProbeRecipe.from_dict(
        asdict(recipe)
        | {
            "seed": probe["seed"],
            "max_steps": probe["max_steps"],
            "batch_size": probe["batch_size"],
        }
    )
    if (
        runtime.model_name_or_path != probe["model_name_or_path"]
        or runtime.revision != probe["model_revision"]
        or runtime.seed != probe["seed"]
        or runtime.max_steps != probe["max_steps"]
        or runtime.batch_size != probe["batch_size"]
        or runtime.max_length != probe["max_length"]
        or runtime.normalize_embeddings is not True
        or runtime.loss != "in_batch_cross_entropy_with_paired_hard_negative"
        or runtime.negatives_per_example != 1
    ):
        raise ValueError("probe model, seed, or frozen recipe drifted")
    if (
        runtime.max_steps
        * runtime.batch_size
        * runtime.max_length
        * (2 + runtime.negatives_per_example)
        != probe["token_count"]
    ):
        raise ValueError("probe token budget drifted")
    if (
        runtime.negative_recipe.strategy != "hn0_filter"
        or runtime.negative_recipe.false_negative_policy != "drop"
    ):
        raise ValueError("probe HN0+filter/drop recipe drifted")

    materialized: dict[str, Any] = {}
    if require_materialized:
        generation_manifest = root / str(cohort["materialized_generation_manifest"])
        evaluation_manifest = root / str(cohort["materialized_evaluation_manifest"])
        if not generation_manifest.is_file() or not evaluation_manifest.is_file():
            raise ValueError("prepare-cohorts must run before the real pilot preflight")
        generation_raw = json.loads(generation_manifest.read_text(encoding="utf-8"))
        if generation_raw.get("final_tests_used") != [] or generation_raw.get(
            "selected_example_ids"
        ) != [str(row["example_id"]) for row in generation]:
            raise ValueError("materialized generation cohort drifted")
        evaluation_raw = json.loads(evaluation_manifest.read_text(encoding="utf-8"))
        if evaluation_raw.get("final_tests_used") != [] or list(evaluation_raw.get("sets", {})) != [
            EVALUATION_SUBSET
        ]:
            raise ValueError("materialized evaluation manifest drifted or references final tests")
        eval_ids = {
            str(row["id"])
            for row in read_records(
                root / str(evaluation_raw["sets"][EVALUATION_SUBSET]["id_path"])
            )
        }
        expected_eval = {str(row["example_id"]) for row in evaluation}
        if eval_ids != expected_eval:
            raise ValueError("materialized evaluation IDs drifted")
        guardrails = list(read_records(evaluation_manifest.parent / "natural_guardrails.jsonl"))
        if {str(row.get("example_id")) for row in guardrails} != expected_eval:
            raise ValueError("natural guardrails do not cover the exact unused dev cohort")
        materialized = {
            "generation_manifest_sha256": _sha256(generation_manifest),
            "evaluation_manifest_sha256": _sha256(evaluation_manifest),
            "natural_guardrails_sha256": _sha256(
                evaluation_manifest.parent / "natural_guardrails.jsonl"
            ),
        }
    free = shutil.disk_usage(root).free
    minimum_free = int(cast(Mapping[str, Any], config["resources"])["minimum_free_disk_bytes"])
    if free < minimum_free:
        raise ValueError(f"insufficient free disk for crash-safe pilot: {free} < {minimum_free}")
    return {
        "schema_version": 1,
        "contract": CONTRACT,
        "status": "verified",
        "config_sha256": _sha256(config_path),
        "generation_cohort_count": len(generation),
        "evaluation_cohort_count": len(evaluation),
        "generation_evaluation_overlap": 0,
        "arms": observed_arms,
        "selector_commit": "2164822",
        "probe_recipe_fingerprint": runtime.fingerprint,
        "probe_seed": 42,
        "metrics": EXPECTED_METRICS,
        "bootstrap": {"samples": 10_000, "seed": 20_260_721},
        "free_disk_bytes": free,
        "minimum_free_disk_bytes": minimum_free,
        "materialized": materialized,
        "four_point_five_b_full_authorized": False,
        "final_tests_used": [],
    }


def compare_scale_pilot(config_path: Path) -> dict[str, Any]:
    """Apply P-04 after both one-seed probe arms complete; never promote directly."""
    preflight = preflight_scale_pilot(config_path)
    config = _load(config_path)
    root = _root(config_path)
    outputs = cast(Mapping[str, Any], config["outputs"])
    run_root = root / str(outputs["probe_runs"])
    measurement_root = root / str(outputs["measurements"])
    control_id = "D01B-SCALE-PILOT-W06-PROBE-S42"
    variant_id = "D01B-SCALE-PILOT-HYBRID-PROBE-S42"
    control_dir, variant_dir = run_root / control_id, run_root / variant_id
    control_result = json.loads((control_dir / "result.json").read_text(encoding="utf-8"))
    variant_result = json.loads((variant_dir / "result.json").read_text(encoding="utf-8"))
    probe = cast(Mapping[str, Any], config["probe"])
    contract = StatisticalContract.load(root / str(probe["comparison_contract"]))
    guardrails = (
        root / str(cast(Mapping[str, Any], config["cohort"])["materialized_evaluation_manifest"])
    ).parent / "natural_guardrails.jsonl"
    report = build_dev_screen_report(
        arm_id=variant_id,
        control_id=control_id,
        arm_result=variant_result,
        control_result=control_result,
        arm_per_query_path=variant_dir / "corpus_retrieval_per_query.jsonl",
        control_per_query_path=control_dir / "corpus_retrieval_per_query.jsonl",
        arm_guardrails_path=guardrails,
        control_guardrails_path=guardrails,
        contract=contract,
    )
    report["actual_frozen_subset"] = EVALUATION_SUBSET
    decision = evaluate_p04_comparison(report, control_manifest=control_result, contract=contract)
    intrinsic = json.loads((measurement_root / "report.json").read_text(encoding="utf-8"))
    intrinsic_passed = intrinsic.get("all_preregistered_gates_passed") is True
    eligible = decision.get("status") == "eligible" and intrinsic_passed
    summary: dict[str, Any] = {
        "schema_version": 1,
        "contract": CONTRACT,
        "status": "pilot_complete",
        "preflight": preflight,
        "decision": decision,
        "intrinsic_guardrails_passed": intrinsic_passed,
        "dev_confirm_authorized": eligible,
        "retained_for_finalist_freeze": False,
        "four_point_five_b_full_authorized": False,
        "multiplicity_note": (
            "screen_only; confirm requires prospectively adjusted primary interval"
        ),
        "final_tests_used": [],
    }
    measurement_root.mkdir(parents=True, exist_ok=True)
    write_json(measurement_root / "probe_comparison_report.json", report)
    write_json(measurement_root / "probe_decision.json", decision)
    write_json(measurement_root / "summary.json", summary)
    return summary
