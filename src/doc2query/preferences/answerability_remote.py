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
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from doc2query.preferences.answerability_judge import (
    CONTRACT,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    JudgeItem,
)
from doc2query.training.dpo import file_sha256
from doc2query.utils.records import read_durable_jsonl_prefix, read_records, write_json

PACKET_SCHEMA = "task06-answerability-packet-v1"
REMOTE_JOURNAL_SCHEMA = "task06-answerability-remote-verdict-v1"
VERDICTS = frozenset({"yes", "no", "uncertain"})


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
        if str(event.get("prompt_version")) != PROMPT_VERSION:
            raise ValueError("remote journal was produced with a different prompt version")
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


def packet_items_preview(packet_dir: Path) -> list[dict[str, Any]]:
    """Read a packet back (used by tests and by an operator sanity check)."""
    rows = list(read_records(packet_dir / "items.jsonl"))
    passages = list(rows[0]["passages"])
    return [
        {"item_id": row["i"], "query": row["q"], "passage": passages[int(row["p"])]}
        for row in rows[1:]
    ]
