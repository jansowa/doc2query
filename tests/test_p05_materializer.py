from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, TypedDict

import pytest

from doc2query.evaluation.corpus import sha256_file
from doc2query.evaluation.p05_materializer import materialize_p05_cohort
from doc2query.evaluation.statistical_contract import build_budget_manifest
from doc2query.utils.records import read_records


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


class MaterializerPaths(TypedDict):
    natural_pairs_path: Path
    w05_generations_path: Path
    natural_fingerprint_path: Path
    w05_fingerprint_path: Path
    budget_path: Path
    gold_output_path: Path
    synthetic_output_path: Path
    mixed_output_path: Path
    manifest_output_path: Path


def _fixture(tmp_path: Path, *, count: int = 8) -> MaterializerPaths:
    natural = tmp_path / "inputs" / "natural.jsonl"
    w05 = tmp_path / "inputs" / "w05.jsonl"
    natural_rows = []
    w05_rows = []
    for index in range(count):
        pair_id = f"p-{index}"
        doc_id = f"d-{index}"
        passage = f"Pasaż numer {index}."
        natural_rows.append(
            {
                "example_id": pair_id,
                "pair_id": pair_id,
                "doc_id": doc_id,
                "passage": passage,
                "query": f"naturalne pytanie {index}",
                "split": "train" if index % 2 else "dev",
                "positives": [{"doc_id": doc_id, "text": passage}],
                "hard_negatives": [{"doc_id": f"n-{index}", "text": "Negatyw."}],
            }
        )
        w05_rows.append(
            {
                "example_id": pair_id,
                "pair_id": pair_id,
                "doc_id": doc_id,
                "passage": passage,
                "generated": f"syntetyczne pytanie {index}",
                "generator_id": "W05-1.5B-50K-8GB",
                "split": "train" if index % 2 else "dev",
            }
        )
    _write_jsonl(natural, natural_rows)
    _write_jsonl(w05, w05_rows)
    natural_fingerprint = tmp_path / "inputs" / "natural.fingerprint.json"
    w05_fingerprint = tmp_path / "inputs" / "w05.fingerprint.json"
    canonical_ids = json.dumps(
        sorted(row["pair_id"] for row in natural_rows),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    common = {
        "splits": ["train", "dev"],
        "final_tests_used": [],
        "negative_recipe": {"strategy": "HN0+filter", "false_negative_policy": "drop"},
        "eligible_pair_ids_sha256": hashlib.sha256(canonical_ids.encode()).hexdigest(),
    }
    _write_json(
        natural_fingerprint,
        common | {"artifact_path": str(natural.resolve()), "sha256": sha256_file(natural)},
    )
    _write_json(
        w05_fingerprint,
        common
        | {
            "artifact_path": str(w05.resolve()),
            "sha256": sha256_file(w05),
            "source_data_sha256": sha256_file(natural),
            "generator_id": "W05-1.5B-50K-8GB",
        },
    )
    budget = tmp_path / "inputs" / "budget.json"
    _write_json(
        budget,
        build_budget_manifest(
            token_count=32,
            pair_count=count,
            unique_passage_count=count,
            queries_per_passage=1,
        ),
    )
    return {
        "natural_pairs_path": natural,
        "w05_generations_path": w05,
        "natural_fingerprint_path": natural_fingerprint,
        "w05_fingerprint_path": w05_fingerprint,
        "budget_path": budget,
        "gold_output_path": tmp_path / "out" / "gold.jsonl",
        "synthetic_output_path": tmp_path / "out" / "synthetic.jsonl",
        "mixed_output_path": tmp_path / "out" / "mixed.jsonl",
        "manifest_output_path": tmp_path / "out" / "manifest.json",
    }


def _materialize(paths: MaterializerPaths, *, seed: int = 17) -> dict[str, Any]:
    return materialize_p05_cohort(**paths, seed=seed)


def test_materialization_is_deterministic_and_uses_one_common_cohort(tmp_path: Path) -> None:
    first_paths = _fixture(tmp_path / "first")
    second_paths = _fixture(tmp_path / "second")
    first = _materialize(first_paths)
    second = _materialize(second_paths)
    assert first["cohort_fingerprint"] == second["cohort_fingerprint"]
    for key in ("gold_output_path", "synthetic_output_path", "mixed_output_path"):
        assert first_paths[key].read_bytes() == second_paths[key].read_bytes()
    gold = list(read_records(first_paths["gold_output_path"]))
    synthetic = list(read_records(first_paths["synthetic_output_path"]))
    mixed = list(read_records(first_paths["mixed_output_path"]))
    assert [row["p05_pair_id"] for row in gold] == [row["pair_id"] for row in synthetic]
    assert [row["pair_id"] for row in synthetic] == [row["pair_id"] for row in mixed]
    assert [row["p05_doc_id"] for row in gold] == [row["doc_id"] for row in mixed]


def test_mix_is_exact_50_50_in_screen_and_confirm_and_budgets_match(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    manifest = _materialize(paths)
    mixed = list(read_records(paths["mixed_output_path"]))
    assert [row["query_source"] for row in mixed[:2]].count("natural") == 1
    assert [row["query_source"] for row in mixed[:2]].count("synthetic_w05") == 1
    assert [row["query_source"] for row in mixed].count("natural") == 4
    assert [row["query_source"] for row in mixed].count("synthetic_w05") == 4
    dimensions = {
        (output["pair_count"], output["unique_passage_count"], output["queries_per_passage"])
        for output in manifest["outputs"].values()
    }
    assert dimensions == {(8, 8, 1)}
    assert manifest["comparison_budget"]["token_count"] == 32
    assert manifest["negative_recipe"] == {
        "strategy": "HN0+filter",
        "false_negative_policy": "drop",
    }
    assert manifest["final_tests_used"] == []


def test_explicit_uniform_k_is_supported_without_record_duplication(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    natural = list(read_records(paths["natural_pairs_path"]))
    w05 = list(read_records(paths["w05_generations_path"]))
    for index, (natural_row, w05_row) in enumerate(zip(natural, w05, strict=True)):
        doc_index = index // 2
        doc_id = f"shared-{doc_index}"
        passage = f"Wspólny pasaż {doc_index}."
        natural_row["doc_id"] = doc_id
        natural_row["passage"] = passage
        natural_row["positives"] = [{"doc_id": doc_id, "text": passage}]
        w05_row["doc_id"] = doc_id
        w05_row["passage"] = passage
    _write_jsonl(paths["natural_pairs_path"], natural)
    _write_jsonl(paths["w05_generations_path"], w05)
    natural_sha = sha256_file(paths["natural_pairs_path"])
    natural_fingerprint = json.loads(paths["natural_fingerprint_path"].read_text())
    natural_fingerprint["sha256"] = natural_sha
    _write_json(paths["natural_fingerprint_path"], natural_fingerprint)
    w05_fingerprint = json.loads(paths["w05_fingerprint_path"].read_text())
    w05_fingerprint["sha256"] = sha256_file(paths["w05_generations_path"])
    w05_fingerprint["source_data_sha256"] = natural_sha
    _write_json(paths["w05_fingerprint_path"], w05_fingerprint)
    _write_json(
        paths["budget_path"],
        build_budget_manifest(
            token_count=32, pair_count=8, unique_passage_count=4, queries_per_passage=2
        ),
    )
    manifest = materialize_p05_cohort(**paths, seed=17, queries_per_passage=2)
    assert manifest["comparison_budget"]["queries_per_passage"] == 2
    assert len({row["pair_id"] for row in read_records(paths["mixed_output_path"])}) == 8


def test_explicit_parquet_outputs_are_supported(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    paths["gold_output_path"] = tmp_path / "out" / "gold.parquet"
    paths["synthetic_output_path"] = tmp_path / "out" / "synthetic.parquet"
    paths["mixed_output_path"] = tmp_path / "out" / "mixed.parquet"
    manifest = _materialize(paths)
    assert len(list(read_records(paths["gold_output_path"]))) == 8
    assert manifest["outputs"]["mixed_50_50"]["sha256"] == sha256_file(paths["mixed_output_path"])


@pytest.mark.parametrize("failure", ["duplicate", "missing_query"])
def test_duplicate_or_missing_query_fails_closed(tmp_path: Path, failure: str) -> None:
    paths = _fixture(tmp_path)
    rows = list(read_records(paths["w05_generations_path"]))
    if failure == "duplicate":
        rows.append(dict(rows[0]))
    else:
        rows[0]["generated"] = ""
    _write_jsonl(paths["w05_generations_path"], rows)
    fingerprint = json.loads(paths["w05_fingerprint_path"].read_text())
    fingerprint["sha256"] = sha256_file(paths["w05_generations_path"])
    _write_json(paths["w05_fingerprint_path"], fingerprint)
    with pytest.raises(ValueError, match=r"duplicate|non-empty"):
        _materialize(paths)


def test_fingerprint_drift_fails_closed(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    paths["natural_pairs_path"].write_text(
        paths["natural_pairs_path"].read_text() + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="fingerprint drift"):
        _materialize(paths)


def test_missing_hn0_filter_eligibility_proof_fails_closed(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    fingerprint = json.loads(paths["natural_fingerprint_path"].read_text())
    fingerprint.pop("eligible_pair_ids_sha256")
    _write_json(paths["natural_fingerprint_path"], fingerprint)
    with pytest.raises(ValueError, match="eligible_pair_ids_sha256"):
        _materialize(paths)


def test_incompatible_w05_generator_id_fails_closed(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    fingerprint = json.loads(paths["w05_fingerprint_path"].read_text())
    fingerprint["generator_id"] = "W04"
    _write_json(paths["w05_fingerprint_path"], fingerprint)
    with pytest.raises(ValueError, match="generator_id"):
        _materialize(paths)


def test_indivisible_budget_fails_closed(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    _write_json(
        paths["budget_path"],
        build_budget_manifest(
            token_count=30, pair_count=6, unique_passage_count=6, queries_per_passage=1
        ),
    )
    with pytest.raises(ValueError, match="divisible"):
        _materialize(paths)


def test_final_test_path_and_manifest_are_rejected(tmp_path: Path) -> None:
    paths = _fixture(tmp_path / "path-case")
    forbidden = tmp_path / "path-case" / "inputs" / "test_native_pl.jsonl"
    paths["natural_pairs_path"].rename(forbidden)
    paths["natural_pairs_path"] = forbidden
    with pytest.raises(ValueError, match="final-test"):
        _materialize(paths)

    paths = _fixture(tmp_path / "manifest-case")
    fingerprint = json.loads(paths["natural_fingerprint_path"].read_text())
    fingerprint["final_tests_used"] = ["test_native_pl"]
    _write_json(paths["natural_fingerprint_path"], fingerprint)
    with pytest.raises(ValueError, match="final_tests_used"):
        _materialize(paths)


def test_different_passage_fails_closed(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    rows = list(read_records(paths["w05_generations_path"]))
    rows[0]["passage"] = "Inny pasaż."
    _write_jsonl(paths["w05_generations_path"], rows)
    fingerprint = json.loads(paths["w05_fingerprint_path"].read_text())
    fingerprint["sha256"] = sha256_file(paths["w05_generations_path"])
    _write_json(paths["w05_fingerprint_path"], fingerprint)
    with pytest.raises(ValueError, match="different passage"):
        _materialize(paths)
