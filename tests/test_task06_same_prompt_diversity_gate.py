from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from doc2query.preferences.diversity_gate import (
    GateFailure,
    SameGroupDiversityGatePolicy,
    apply_same_prompt_diversity_gate,
    evaluate_group,
    load_gate_policy,
)
from doc2query.training.dpo import canonical_fingerprint

POLICY = Path("configs/preferences/task06_same_prompt_diversity_gate_v1.yaml")
COHORT = "c" * 64
DECODING = (
    {"slot": 0, "temperature": 0.3, "top_p": 0.95, "seed": 6601},
    {"slot": 1, "temperature": 0.5, "top_p": 0.95, "seed": 6602},
    {"slot": 2, "temperature": 0.7, "top_p": 0.95, "seed": 6603},
    {"slot": 3, "temperature": 0.9, "top_p": 0.95, "seed": 6604},
    {"slot": 4, "temperature": 0.3, "top_p": 0.90, "seed": 6611},
    {"slot": 5, "temperature": 0.5, "top_p": 0.90, "seed": 6612},
    {"slot": 6, "temperature": 0.7, "top_p": 0.90, "seed": 6613},
    {"slot": 7, "temperature": 1.0, "top_p": 0.90, "seed": 6614},
)
DIVERSE = (
    "jak leczyć zapalenie oskrzeli u dorosłych",
    "objawy przewlekłej obturacyjnej choroby płuc",
    "ile trwa rekonwalescencja po zabiegu kolana",
    "czym różni się grypa od przeziębienia",
    "kto wynalazł pierwszą szczepionkę przeciw wściekliźnie",
    "gdzie znajduje się największy port lotniczy Norwegii",
    "definicja koronawirusa według słownika medycznego",
    "procedura zgłoszenia reklamacji sprzętu rolniczego",
)


def _policy() -> SameGroupDiversityGatePolicy:
    return load_gate_policy(POLICY)


def _group(
    queries: tuple[str, ...],
    *,
    group_id: str = "task06-preference::400673::5960803",
    prompt: str = "Pasaż: sprawdzalny fakt.\nZapytanie:\n",
    prompt_overrides: dict[int, str] | None = None,
    identity: str = "d" * 64,
    split: str = "train",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, query in enumerate(queries):
        text = (prompt_overrides or {}).get(index, prompt)
        slot = DECODING[index % len(DECODING)]
        rows.append(
            {
                "evaluation_id": f"{group_id.split('::', 1)[1]}::same-prompt::{index}",
                "evaluation_group_id": group_id,
                "example_id": group_id.split("::", 1)[1],
                "doc_id": "5960803",
                "generated": query,
                "candidate_index": index,
                "prompt": text,
                "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "generation_config": dict(slot),
                "seed": int(slot["seed"]),
                "generation_identity_sha256": identity,
                "frozen_cohort_fingerprint": COHORT,
                "metadata": {"split": split},
                "final_tests_used": [],
            }
        )
    return rows


def _write_cohort(
    tmp_path: Path, groups: list[list[dict[str, Any]]], *, identity: str = "d" * 64
) -> tuple[Path, Path, Path]:
    rows = [row for group in groups for row in group]
    tmp_path.mkdir(parents=True, exist_ok=True)
    generations = tmp_path / "generations.jsonl"
    generations.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    digest = hashlib.sha256(generations.read_bytes()).hexdigest()
    summary = tmp_path / "generations.jsonl.summary.json"
    summary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "contract": "task06-same-prompt-preference-expansion-v1",
                "status": "same_prompt_generation_complete",
                "prompt_count": len(groups),
                "generation_count": len(rows),
                "output_sha256": digest,
                "final_tests_used": [],
            }
        ),
        encoding="utf-8",
    )
    identity_path = tmp_path / "generations.jsonl.identity.json"
    identity_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "contract": "task06-same-prompt-preference-expansion-v1",
                "exact_same_prompt_required": True,
                "identity_sha256": identity,
                "final_tests_used": [],
            }
        ),
        encoding="utf-8",
    )
    return generations, summary, identity_path


def test_frozen_policy_pins_the_documented_thresholds() -> None:
    policy = _policy()
    assert policy.policy_id == "task06-same-prompt-diversity-gate-v1"
    assert policy.status == "frozen_before_pair_read"
    assert policy.group.min_effective_candidates == 3
    assert policy.group.max_duplicate_rate == 0.50
    assert policy.group.max_effective_self_bleu == 0.75
    assert policy.group.max_min_pairwise_query_jaccard == 0.85
    assert policy.normalization.near_duplicate_lemma_jaccard == 0.90
    assert policy.final_tests_used == []


def test_policy_rejects_an_unreproducible_lemma_backend(tmp_path: Path) -> None:
    raw = json.loads(json.dumps(_policy().model_dump(mode="json")))
    raw["normalization"]["lemma_backend"] = "spacy_pl:pl_core_news_lg:3.7.0:v1"
    path = tmp_path / "policy.yaml"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_gate_policy(path)


def test_policy_rejects_min_effective_above_group_size(tmp_path: Path) -> None:
    raw = _policy().model_dump(mode="json")
    raw["group"]["min_effective_candidates"] = 9
    path = tmp_path / "policy.yaml"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValidationError, match="exceeds the frozen group size"):
        load_gate_policy(path)


def test_diverse_group_is_eligible_with_visible_components() -> None:
    verdict = evaluate_group(_group(DIVERSE), _policy())
    assert verdict.eligible is True
    assert verdict.failure_reasons == []
    assert verdict.candidate_count == 8
    assert verdict.distinct_normalized_count == 8
    assert verdict.duplicate_rate == 0.0
    assert verdict.effective_candidate_count == 8
    assert verdict.effective_cluster_sizes == [1] * 8
    assert verdict.effective_self_bleu is not None
    assert verdict.min_pairwise_representative_query_jaccard is not None
    assert verdict.distinct_temperature_count == 5
    assert verdict.distinct_top_p_count == 2
    assert verdict.distinct_seed_count == 8


def test_collapsed_group_fails_effective_count_and_duplicate_rate() -> None:
    collapsed = ("czym jest koronawirus",) * 6 + (
        "czym są wirusy korona wirusowe",
        "objawy zakażenia koronawirusem u dzieci",
    )
    verdict = evaluate_group(_group(collapsed), _policy())
    assert verdict.eligible is False
    assert verdict.duplicate_rate == pytest.approx(5 / 8)
    assert verdict.distinct_normalized_count == 3
    assert verdict.effective_candidate_count == 3
    assert verdict.failure_reasons == [GateFailure.DUPLICATE_RATE_ABOVE_THRESHOLD.value]
    assert verdict.effective_cluster_sizes == [6, 1, 1]
    assert sum(verdict.effective_cluster_sizes) == 8


def test_exact_duplicates_are_collapsed_before_counting_effective_candidates() -> None:
    queries = (
        DIVERSE[0],
        f"  {DIVERSE[0].upper()}  ",
        DIVERSE[1],
        DIVERSE[2],
        DIVERSE[3],
        DIVERSE[4],
        DIVERSE[5],
        DIVERSE[6],
    )
    verdict = evaluate_group(_group(queries), _policy())
    assert verdict.distinct_normalized_count == 7
    assert verdict.effective_candidate_count == 7
    assert verdict.effective_cluster_sizes[0] == 2
    assert verdict.representative_candidate_ids[0].endswith("::same-prompt::0")


def test_lemma_reorderings_are_treated_as_one_effective_candidate() -> None:
    queries = (
        "objawy grypy u dzieci",
        "u dzieci objawy grypy",
        "grypy objawy dzieci u",
        "jak długo trwa kwarantanna po kontakcie z chorym",
        "kto zbudował most nad rzeką Wisłą w Toruniu",
        "definicja inflacji bazowej według NBP",
        "procedura wymiany oleju w kombajnie zbożowym",
        "ile kalorii ma szklanka mleka krowiego",
    )
    verdict = evaluate_group(_group(queries), _policy())
    assert verdict.distinct_normalized_count == 8
    assert verdict.effective_candidate_count == 6
    assert verdict.effective_cluster_sizes[0] == 3
    assert verdict.eligible is True


def test_group_of_near_identical_candidates_is_not_pairable() -> None:
    verdict = evaluate_group(_group((DIVERSE[0],) * 8), _policy())
    assert verdict.eligible is False
    assert GateFailure.NO_PAIRABLE_CANDIDATE_PAIR.value in verdict.failure_reasons
    assert GateFailure.SELF_BLEU_ABOVE_THRESHOLD.value in verdict.failure_reasons
    assert verdict.effective_candidate_count == 1
    assert verdict.effective_self_bleu is None


def test_mixed_prompts_within_a_group_are_rejected() -> None:
    rows = _group(DIVERSE, prompt_overrides={3: "Pasaż: inny prompt.\nZapytanie:\n"})
    verdict = evaluate_group(rows, _policy())
    assert verdict.eligible is False
    assert GateFailure.PROMPT_MISMATCH.value in verdict.failure_reasons


def test_short_group_is_rejected_on_size_even_when_diverse() -> None:
    verdict = evaluate_group(_group(DIVERSE[:4]), _policy())
    assert verdict.eligible is False
    assert verdict.failure_reasons == [GateFailure.UNEXPECTED_GROUP_SIZE.value]


def test_empty_group_and_forbidden_split_are_refused() -> None:
    policy = _policy()
    with pytest.raises(ValueError, match="cannot be empty"):
        evaluate_group([], policy)
    with pytest.raises(ValueError, match="refuses split"):
        evaluate_group(_group(DIVERSE, split="test"), policy)


def test_gate_publishes_atomic_artifacts_and_reports_rejections(tmp_path: Path) -> None:
    eligible = _group(DIVERSE)
    collapsed = _group((DIVERSE[0],) * 8, group_id="task06-preference::400674::5960804")
    generations, summary, identity = _write_cohort(tmp_path, [eligible, collapsed])
    output = tmp_path / "diversity_gate"
    manifest = apply_same_prompt_diversity_gate(
        generations_path=generations,
        generations_summary_path=summary,
        generations_identity_path=identity,
        policy_path=POLICY,
        output_dir=output,
    )
    assert manifest.status == "diversity_gate_applied_not_paired"
    assert (manifest.group_count, manifest.candidate_count) == (2, 16)
    assert (manifest.eligible_group_count, manifest.rejected_group_count) == (1, 1)
    assert manifest.pairs_built is False
    assert manifest.judge_scores_read is False
    assert manifest.candidates_ranked is False
    assert manifest.model_loading_performed is False
    assert manifest.final_tests_used == []
    payload = manifest.model_dump(mode="json")
    payload.pop("manifest_fingerprint")
    assert manifest.manifest_fingerprint == canonical_fingerprint(payload)

    verdicts = [
        json.loads(line)
        for line in (output / "group_verdicts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["group_id"] for row in verdicts] == [
        "task06-preference::400673::5960803",
        "task06-preference::400674::5960804",
    ]
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["rejected_group_rate"] == 0.5
    assert report["failure_reason_counts"] == {
        GateFailure.DUPLICATE_RATE_ABOVE_THRESHOLD.value: 1,
        GateFailure.INSUFFICIENT_EFFECTIVE_CANDIDATES.value: 1,
        GateFailure.NO_PAIRABLE_CANDIDATE_PAIR.value: 1,
        GateFailure.SELF_BLEU_ABOVE_THRESHOLD.value: 1,
    }
    assert report["eligible_groups"]["group_count"] == 1
    assert report["all_groups"]["duplicate_rate"]["count"] == 2
    assert report["policy"]["policy_id"] == "task06-same-prompt-diversity-gate-v1"
    assert "total_score" not in report
    assert json.loads(
        (output / "manifest.json").read_text(encoding="utf-8")
    ) == manifest.model_dump(mode="json")


def test_gate_accepts_the_v2_generation_contract(tmp_path: Path) -> None:
    """The gate must accept every same-prompt cohort contract, not only v1."""
    generations, summary, identity = _write_cohort(tmp_path, [_group(DIVERSE)])
    for path in (summary, identity):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["contract"] = "task06-same-prompt-preference-expansion-v2"
        path.write_text(json.dumps(payload), encoding="utf-8")
    manifest = apply_same_prompt_diversity_gate(
        generations_path=generations,
        generations_summary_path=summary,
        generations_identity_path=identity,
        policy_path=POLICY,
        output_dir=tmp_path / "gate_v2",
    )
    assert manifest.eligible_group_count == 1


def test_gate_rejects_mismatched_summary_and_identity_contracts(tmp_path: Path) -> None:
    generations, summary, identity = _write_cohort(tmp_path, [_group(DIVERSE)])
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["contract"] = "task06-same-prompt-preference-expansion-v2"
    summary.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="pin different contracts"):
        apply_same_prompt_diversity_gate(
            generations_path=generations,
            generations_summary_path=summary,
            generations_identity_path=identity,
            policy_path=POLICY,
            output_dir=tmp_path / "gate_mismatch",
        )


def test_gate_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    generations, summary, identity = _write_cohort(tmp_path, [_group(DIVERSE)])
    output = tmp_path / "diversity_gate"
    output.mkdir()
    with pytest.raises(FileExistsError):
        apply_same_prompt_diversity_gate(
            generations_path=generations,
            generations_summary_path=summary,
            generations_identity_path=identity,
            policy_path=POLICY,
            output_dir=output,
        )
    assert list(output.iterdir()) == []
    assert not list(tmp_path.glob(".diversity_gate.staging-*"))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload.__setitem__("output_sha256", "e" * 64), "SHA-256 drifted"),
        (lambda payload: payload.__setitem__("generation_count", 3), "record count drifted"),
        (
            lambda payload: payload.__setitem__("status", "same_prompt_generation_partial"),
            "not complete",
        ),
        (lambda payload: payload.__setitem__("final_tests_used", ["test_native_pl"]), "final-test"),
    ],
)
def test_gate_rejects_generation_provenance_drift(
    tmp_path: Path, mutate: Any, message: str
) -> None:
    generations, summary, identity = _write_cohort(tmp_path, [_group(DIVERSE)])
    payload = json.loads(summary.read_text(encoding="utf-8"))
    mutate(payload)
    summary.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        apply_same_prompt_diversity_gate(
            generations_path=generations,
            generations_summary_path=summary,
            generations_identity_path=identity,
            policy_path=POLICY,
            output_dir=tmp_path / "diversity_gate",
        )


def test_gate_refuses_judge_scores_and_identity_drift(tmp_path: Path) -> None:
    rows = _group(DIVERSE)
    rows[2]["primary_margin"] = 4.2
    generations, summary, identity = _write_cohort(tmp_path, [rows])
    with pytest.raises(ValueError, match="judge scores"):
        apply_same_prompt_diversity_gate(
            generations_path=generations,
            generations_summary_path=summary,
            generations_identity_path=identity,
            policy_path=POLICY,
            output_dir=tmp_path / "gate_scores",
        )
    drifted = _group(DIVERSE)
    drifted[1]["generation_identity_sha256"] = "f" * 64
    generations, summary, identity = _write_cohort(tmp_path / "drift", [drifted])
    with pytest.raises(ValueError, match="identity drift"):
        apply_same_prompt_diversity_gate(
            generations_path=generations,
            generations_summary_path=summary,
            generations_identity_path=identity,
            policy_path=POLICY,
            output_dir=tmp_path / "gate_identity",
        )


def test_gate_refuses_final_test_paths(tmp_path: Path) -> None:
    generations, summary, identity = _write_cohort(tmp_path, [_group(DIVERSE)])
    with pytest.raises(ValueError, match="final-test path is forbidden"):
        apply_same_prompt_diversity_gate(
            generations_path=generations,
            generations_summary_path=summary,
            generations_identity_path=identity,
            policy_path=POLICY,
            output_dir=tmp_path / "test_native_pl" / "gate",
        )


def test_gate_refuses_mixed_cohorts(tmp_path: Path) -> None:
    first = _group(DIVERSE)
    second = _group(DIVERSE, group_id="task06-preference::400674::5960804")
    for row in second:
        row["frozen_cohort_fingerprint"] = "a" * 64
    generations, summary, identity = _write_cohort(tmp_path, [first, second])
    with pytest.raises(ValueError, match="mix frozen cohorts"):
        apply_same_prompt_diversity_gate(
            generations_path=generations,
            generations_summary_path=summary,
            generations_identity_path=identity,
            policy_path=POLICY,
            output_dir=tmp_path / "diversity_gate",
        )
