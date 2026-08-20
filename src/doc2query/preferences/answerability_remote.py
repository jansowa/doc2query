"""Portable judging packet: run the answerability judge on a machine that has the GPU.

The base machine has 8 GB of VRAM, which is not enough to serve a 27B judge at an honest
quantization, and it cannot reach the port of the machine that can.  So the judging is
split into three steps with an explicit, hashed handover:

1. **export** (here) — freeze the exact item set into a packet: deduplicated passages,
   one row per (query, passage) item, a manifest with SHA-256 of the item file and the
   pinned prompt version.  The packet carries **no labels**: the Groq references and the
   constructed-class expectations stay on this machine, so the remote judge cannot be
   tuned against them even by accident.
2. **judge** (there) — a dependency-free script talks to a local vLLM OpenAI-compatible
   endpoint, writes one durable, resumable journal row per verdict, and records the model
   identity the server reports.
3. **import** (here) — validate the returned journal against the packet manifest (schema,
   prompt version, item coverage, no unknown items, single verdict per item), then run the
   already-existing calibration analysis.

Because the packet is label-free and the manifest is hashed on both ends, the remote hop
adds no way to leak the calibration labels and no way to silently judge a different item
set than the one that was frozen.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from doc2query.preferences.answerability_judge import (
    CONTRACT,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    JudgeItem,
    judge_item_id,
)
from doc2query.training.dpo import file_sha256
from doc2query.utils.records import read_durable_jsonl_prefix, read_records, write_json

PACKET_SCHEMA = "task06-answerability-packet-v1"
REMOTE_JOURNAL_SCHEMA = "task06-answerability-remote-verdict-v1"
VERDICTS = frozenset({"yes", "no", "uncertain"})
VERDICT_ORDER = ("yes", "no", "uncertain")
# Wersje promptu, ktore wolno zaimportowac. `-v2-batched` ocenia N zapytan jednego pasazu
# w jednym requescie; jego dopuszczenie do kalibracji wymaga przejscia bramki A/B
# (amendment do ADR V2-01), a import zawsze raportuje rozbicie po wersjach.
PROMPT_VERSION_SINGLE = PROMPT_VERSION
PROMPT_VERSION_BATCHED = "task06-answerability-pl-v2-batched"
KNOWN_PROMPT_VERSIONS = frozenset({PROMPT_VERSION_SINGLE, PROMPT_VERSION_BATCHED})


def build_packet_rows(items: Sequence[JudgeItem]) -> tuple[list[dict[str, Any]], list[str]]:
    """Deduplicate passages and emit label-free item rows referencing them by index."""
    passages: list[str] = []
    index_of: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for item in sorted(items, key=lambda entry: entry.item_id):
        passage_index = index_of.get(item.passage)
        if passage_index is None:
            passage_index = len(passages)
            index_of[item.passage] = passage_index
            passages.append(item.passage)
        rows.append({"i": item.item_id, "p": passage_index, "q": item.query})
    return rows, passages


def write_packet(items: Sequence[JudgeItem], packet_dir: Path) -> dict[str, Any]:
    """Write the label-free packet plus its hashed manifest."""
    if not items:
        raise ValueError("the judging packet needs at least one item")
    ids = [item.item_id for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("packet items must have unique IDs")
    rows, passages = build_packet_rows(items)
    packet_dir.mkdir(parents=True, exist_ok=True)
    items_path = packet_dir / "items.jsonl"
    with items_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"passages": passages}, ensure_ascii=False) + "\n")
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "schema": PACKET_SCHEMA,
        "contract": CONTRACT,
        "prompt_version": PROMPT_VERSION,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "item_count": len(rows),
        "passage_count": len(passages),
        "items_sha256": file_sha256(items_path),
        "item_ids_fingerprint": hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest(),
        "labels_included": False,
        "final_tests_used": [],
    }
    write_json(packet_dir / "manifest.json", manifest)
    return manifest


def load_remote_journal(
    journal_path: Path, manifest: Mapping[str, Any], items: Sequence[JudgeItem]
) -> dict[str, dict[str, Any]]:
    """Validate a journal that came back from the remote judge; refuse anything off-contract."""
    if str(manifest.get("prompt_version")) != PROMPT_VERSION:
        raise ValueError("packet manifest pins a different prompt version than this build")
    known = {item.item_id for item in items}
    if hashlib.sha256("\n".join(sorted(known)).encode()).hexdigest() != str(
        manifest["item_ids_fingerprint"]
    ):
        raise ValueError("local items do not match the packet the remote judge was given")
    verdicts: dict[str, dict[str, Any]] = {}
    models: set[str] = set()
    for event in read_durable_jsonl_prefix(journal_path):
        if event.get("event") != "verdict":
            continue
        if str(event.get("schema")) != REMOTE_JOURNAL_SCHEMA:
            raise ValueError(f"unexpected remote journal schema: {event.get('schema')!r}")
        version = str(event.get("prompt_version"))
        if version not in KNOWN_PROMPT_VERSIONS:
            raise ValueError(f"remote journal has an unknown prompt version: {version!r}")
        item_id = str(event["item_id"])
        if item_id not in known:
            raise ValueError(f"remote journal contains an item outside the packet: {item_id}")
        verdict = str(event["verdict"])
        if verdict not in VERDICTS:
            raise ValueError(f"invalid remote verdict: {verdict!r}")
        previous = verdicts.get(item_id)
        if previous is not None and str(previous["verdict"]) != verdict:
            raise ValueError(f"remote journal disagrees with itself on {item_id}")
        verdicts[item_id] = dict(event)
        models.add(str(event.get("model", "")))
    if len(models) > 1:
        raise ValueError(f"remote journal mixes judge models: {sorted(models)}")
    return verdicts


def journal_provenance(verdicts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Rozbicie werdyktow po wersji promptu, paczkowaniu i fallbacku.

    Kalibracja liczona na mieszance przyrzadow bylaby nieinterpretowalna, wiec te liczby
    musza byc widoczne w raporcie, a nie tylko w journalu.
    """
    versions: Counter[str] = Counter()
    batch_sizes: Counter[str] = Counter()
    fallback = 0
    for row in verdicts.values():
        versions[str(row.get("prompt_version"))] += 1
        batch_sizes[str(row.get("batch_size"))] += 1
        if bool(row.get("fallback")):
            fallback += 1
    total = sum(versions.values())
    return {
        "verdicts": total,
        "by_prompt_version": dict(sorted(versions.items())),
        "by_batch_size": dict(sorted(batch_sizes.items())),
        "fallback_verdicts": fallback,
        "fallback_share": fallback / total if total else None,
        "single_instrument": len(versions) <= 1,
    }


def remote_identity(verdicts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Whatever identity the remote server reported, plus the honest limits of that pin."""
    models = sorted({str(row.get("model", "")) for row in verdicts.values()})
    served = sorted(
        {json.dumps(row.get("served_model"), sort_keys=True) for row in verdicts.values()}
    )
    return {
        "model": models[0] if len(models) == 1 else models,
        "served_model_metadata": [json.loads(value) for value in served],
        "digest_pinning": "weaker_than_ollama_path",
        "digest_pinning_note": (
            "vLLM reports a model name and its own metadata, not a content digest of the "
            "weights, so this pin is an operator declaration plus server metadata — not a "
            "cryptographic pin. The packet item set is pinned by SHA-256 on both ends."
        ),
    }


# Progi zamrożone w reports/decisions/task06_answerability_judge_v1.md, §5.
MINIMUM_CONSENSUS_ACCURACY = 0.85
MINIMUM_CONSENSUS_BALANCED_ACCURACY = 0.75
MINIMUM_UNGROUNDED_REJECTION = 0.80
MAXIMUM_GOOD_CLASS_REJECTION = 0.20
MAXIMUM_UNCERTAIN_SHARE = 0.25
GOOD_CLASSES = ("good_alternative", "good_specific")


def apply_acceptance_criteria(
    items: Sequence[JudgeItem],
    verdicts: Mapping[str, Mapping[str, Any]],
    analysis: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate K1-K3 of the ADR mechanically; no threshold lives outside this function."""
    consensus_true: list[bool] = []
    consensus_pred: list[bool] = []
    by_class: dict[str, Counter[str]] = {}
    verdict_counts: Counter[str] = Counter()
    for item in items:
        record = verdicts.get(item.item_id)
        if record is None:
            continue
        verdict = str(record["verdict"])
        verdict_counts[verdict] += 1
        source = str(item.metadata["source"])
        if source == "groq_audit":
            references = cast(Mapping[str, bool], item.metadata["groq_answerable"])
            values = set(references.values())
            if len(references) != 2 or len(values) != 1 or verdict == "uncertain":
                continue
            consensus_true.append(next(iter(values)))
            consensus_pred.append(verdict == "yes")
        else:
            by_class.setdefault(str(item.metadata["label"]), Counter())[verdict] += 1

    matched = [truth == pred for truth, pred in zip(consensus_true, consensus_pred, strict=True)]
    accuracy = sum(matched) / len(matched) if matched else None
    recalls: dict[str, float | None] = {}
    for label, flag in (("yes", True), ("no", False)):
        subset = [
            truth == pred
            for truth, pred in zip(consensus_true, consensus_pred, strict=True)
            if truth is flag
        ]
        recalls[label] = sum(subset) / len(subset) if subset else None
    balanced = (
        (recalls["yes"] + recalls["no"]) / 2
        if recalls["yes"] is not None and recalls["no"] is not None
        else None
    )
    judged = sum(verdict_counts.values())
    uncertain_share = verdict_counts["uncertain"] / judged if judged else None

    class_rates = {
        label: {
            "count": sum(counts.values()),
            "no_share": counts["no"] / sum(counts.values()) if sum(counts.values()) else None,
            "verdicts": dict(sorted(counts.items())),
        }
        for label, counts in sorted(by_class.items())
    }
    no_shares: dict[str, float | None] = {
        label: cast("float | None", value["no_share"]) for label, value in class_rates.items()
    }
    ungrounded = no_shares.get("ungrounded")
    good_ok = all(
        (share := no_shares.get(label)) is not None and share <= MAXIMUM_GOOD_CLASS_REJECTION
        for label in GOOD_CLASSES
    )
    k1 = (
        accuracy is not None
        and balanced is not None
        and accuracy >= MINIMUM_CONSENSUS_ACCURACY
        and balanced >= MINIMUM_CONSENSUS_BALANCED_ACCURACY
    )
    k2 = ungrounded is not None and ungrounded >= MINIMUM_UNGROUNDED_REJECTION and good_ok
    k3 = uncertain_share is not None and uncertain_share <= MAXIMUM_UNCERTAIN_SHARE
    return {
        "k1_consensus": {
            "accuracy": accuracy,
            "balanced_accuracy": balanced,
            "recall_yes": recalls["yes"],
            "recall_no": recalls["no"],
            "decided_consensus_sides": len(matched),
            "minimum_accuracy": MINIMUM_CONSENSUS_ACCURACY,
            "minimum_balanced_accuracy": MINIMUM_CONSENSUS_BALANCED_ACCURACY,
            "met": k1,
        },
        "k2_constructed_classes": {
            "per_class": class_rates,
            "minimum_ungrounded_no_share": MINIMUM_UNGROUNDED_REJECTION,
            "maximum_good_class_no_share": MAXIMUM_GOOD_CLASS_REJECTION,
            "met": k2,
        },
        "k3_abstention": {
            "uncertain_share": uncertain_share,
            "verdict_counts": dict(sorted(verdict_counts.items())),
            "maximum_uncertain_share": MAXIMUM_UNCERTAIN_SHARE,
            "met": k3,
        },
        "groq_agreement_reported_not_gated": analysis.get("agreement_with_groq"),
        "accepted": bool(k1 and k2 and k3),
        "status": (
            "accepted_as_axis_a_answerability_signal"
            if (k1 and k2 and k3)
            else "rejected_axis_a_without_answerability_filter"
        ),
        "manual_review_required": True,
        "manual_review_can_only_invalidate": True,
    }


def candidate_pool_items(cohort_dirs: Sequence[Path]) -> list[JudgeItem]:
    """Every diversity-gate representative of every eligible group, as judge items.

    This is **precomputation of a per-candidate measurement**, not policy: it certifies
    answerability for the pool a pair may later be drawn from, exactly as the frozen
    scoring pass already computed ``pool_margin`` for the same candidates before any pair
    policy existed.  It builds no pair, freezes no threshold and orders nothing.

    Candidate identity stays local: the packet keys items by ``sha256(prompt, query,
    passage)``, so verdicts join back here without the packet ever carrying a candidate ID.
    """
    items: dict[str, JudgeItem] = {}
    for cohort_dir in cohort_dirs:
        gate_path = cohort_dir / "diversity_gate" / "group_verdicts.jsonl"
        scoring_path = cohort_dir / "d01_controlled" / "scoring" / "per_generation.jsonl"
        for path in (gate_path, scoring_path):
            if not path.is_file():
                raise ValueError(f"missing cohort input: {path}")
        allowed: set[str] = set()
        for verdict in read_records(gate_path):
            if bool(verdict["eligible"]):
                allowed.update(str(value) for value in verdict["representative_candidate_ids"])
        for row in read_records(scoring_path):
            candidate_id = str(row["evaluation_id"])
            if candidate_id not in allowed:
                continue
            if row.get("final_tests_used") != []:
                raise ValueError("scored candidate declares final-test usage")
            query = str(row["generated"])
            passage = str(cast(Mapping[str, Any], row["positive"])["text"])
            item_id = judge_item_id(query, passage)
            items.setdefault(
                item_id,
                JudgeItem(
                    item_id=item_id,
                    query=query,
                    passage=passage,
                    metadata={
                        "source": "candidate_pool",
                        "cohort_id": cohort_dir.name,
                        "candidate_id": candidate_id,
                        "group_id": str(row["evaluation_group_id"]),
                    },
                ),
            )
    if not items:
        raise ValueError("the candidate pool is empty")
    return [items[item_id] for item_id in sorted(items)]


def packet_items_preview(packet_dir: Path) -> list[dict[str, Any]]:
    """Read a packet back (used by tests and by an operator sanity check)."""
    rows = list(read_records(packet_dir / "items.jsonl"))
    passages = list(rows[0]["passages"])
    return [
        {"item_id": row["i"], "query": row["q"], "passage": passages[int(row["p"])]}
        for row in rows[1:]
    ]


# Progi bramki A/B zamrozone w amendmencie o paczkowaniu (reports/decisions/
# task06_answerability_judge_v1_batching_amendment_2026-08-20.md).
MINIMUM_BATCH_AGREEMENT = 0.98
DRIFT_SIGNIFICANCE = 0.05


def _binomial_two_sided_p(successes: int, trials: int) -> float:
    """Dokladny dwustronny test znakowy przy p=0.5; bez zaleznosci zewnetrznych.

    Uzywany do wykrycia **niesymetrycznych** migracji werdyktow: jesli przejscia
    yes->no i no->yes rownowaza sie, to szum przyrzadu; jesli jedna strona dominuje,
    to dryf systematyczny i bramka musi go zlapac.
    """
    if trials == 0:
        return 1.0
    coefficients = [math.comb(trials, k) for k in range(trials + 1)]
    total = float(sum(coefficients))
    observed = coefficients[successes]
    tail = sum(value for value in coefficients if value <= observed)
    return min(1.0, tail / total)


def compare_journal_verdicts(
    baseline: Mapping[str, Mapping[str, Any]], candidate: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Porownaj dwa journale per item: zgodnosc, macierz przejsc, test dryfu.

    `baseline` to przyrzad, na ktorym zamrozono kryteria (pojedyncze requesty),
    `candidate` to wariant paczkowy. Porownanie obejmuje wylacznie itemy obecne w obu.
    """
    shared = sorted(set(baseline) & set(candidate))
    matrix: Counter[tuple[str, str]] = Counter()
    for item_id in shared:
        matrix[(str(baseline[item_id]["verdict"]), str(candidate[item_id]["verdict"]))] += 1
    agreed = sum(count for (left, right), count in matrix.items() if left == right)
    agreement = agreed / len(shared) if shared else None
    drift = []
    for index, left in enumerate(VERDICT_ORDER):
        for right in VERDICT_ORDER[index + 1 :]:
            forward = matrix[(left, right)]
            backward = matrix[(right, left)]
            p_value = _binomial_two_sided_p(forward, forward + backward)
            drift.append(
                {
                    "pair": f"{left}->{right} vs {right}->{left}",
                    "forward": forward,
                    "backward": backward,
                    "p_value": p_value,
                    "significant": (forward + backward) > 0 and p_value < DRIFT_SIGNIFICANCE,
                }
            )
    shares = {
        "baseline": {
            verdict: sum(count for (left, _), count in matrix.items() if left == verdict)
            / len(shared)
            if shared
            else None
            for verdict in VERDICT_ORDER
        },
        "candidate": {
            verdict: sum(count for (_, right), count in matrix.items() if right == verdict)
            / len(shared)
            if shared
            else None
            for verdict in VERDICT_ORDER
        },
    }
    systematic_drift = any(entry["significant"] for entry in drift)
    accepted = bool(
        agreement is not None and agreement >= MINIMUM_BATCH_AGREEMENT and not systematic_drift
    )
    return {
        "schema": "task06-answerability-batching-ab-v1",
        "compared_items": len(shared),
        "baseline_only": len(set(baseline) - set(candidate)),
        "candidate_only": len(set(candidate) - set(baseline)),
        "agreement": agreement,
        "minimum_agreement": MINIMUM_BATCH_AGREEMENT,
        "transition_matrix": {
            f"{left}->{right}": count for (left, right), count in sorted(matrix.items())
        },
        "verdict_shares": shares,
        "drift_tests": drift,
        "drift_significance_level": DRIFT_SIGNIFICANCE,
        "systematic_drift": systematic_drift,
        "accepted": accepted,
        "status": "batching_accepted" if accepted else "batching_rejected_keep_single_requests",
        "final_tests_used": [],
    }


def render_ab_report(result: Mapping[str, Any], provenance: Mapping[str, Any]) -> str:
    """Raport markdown; bramka musi byc czytelna bez zagladania do JSON-a."""
    lines = [
        "# Bramka A/B: paczkowanie zapytań sędziego odpowiadalności",
        "",
        f"Porównanych itemów: **{result['compared_items']}** "
        f"(tylko baseline: {result['baseline_only']}, tylko kandydat: {result['candidate_only']}).",
        "",
        f"- zgodność werdyktów per item: **{result['agreement']:.4f}** "
        f"(próg {result['minimum_agreement']})",
        f"- dryf systematyczny: **{'TAK' if result['systematic_drift'] else 'nie'}** "
        f"(test znakowy na parach klas, poziom istotnosci "
        f"{result['drift_significance_level']})",
        f"- werdykt bramki: **{result['status']}**",
        "",
        "## Macierz przejść (baseline → kandydat)",
        "",
        "| przejście | liczba |",
        "|---|---|",
    ]
    lines += [f"| `{key}` | {value} |" for key, value in result["transition_matrix"].items()]
    lines += ["", "## Rozkład werdyktów", "", "| klasa | baseline | kandydat |", "|---|---|---|"]
    for verdict in VERDICT_ORDER:
        base = result["verdict_shares"]["baseline"][verdict]
        cand = result["verdict_shares"]["candidate"][verdict]
        lines.append(f"| {verdict} | {base:.4f} | {cand:.4f} |")
    lines += [
        "",
        "## Testy dryfu",
        "",
        "| para | w jedną stronę | w drugą | p | istotny |",
        "|---|---|---|---|---|",
    ]
    for entry in result["drift_tests"]:
        lines.append(
            f"| `{entry['pair']}` | {entry['forward']} | {entry['backward']} | "
            f"{entry['p_value']:.4f} | {'TAK' if entry['significant'] else 'nie'} |"
        )
    lines += [
        "",
        "## Proweniencja journala kandydata",
        "",
        f"- werdyktów: {provenance['verdicts']}",
        f"- wersje promptu: {provenance['by_prompt_version']}",
        f"- liczności paczek: {provenance['by_batch_size']}",
        f"- werdykty z fallbacku: {provenance['fallback_verdicts']} "
        f"({(provenance['fallback_share'] or 0):.4f})",
        "",
        "Fallback produkuje wiersze promptem pojedynczym, więc wysoki udział fallbacku",
        "oznacza, że kandydat jest w praktyce mieszanką przyrządów — to trzeba czytać razem",
        "z werdyktem bramki, a nie zamiast niego.",
        "",
        "`final_tests_used=[]`, `used_for_pair_building=false`.",
    ]
    return "\n".join(lines) + "\n"
