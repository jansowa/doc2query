"""V2-01: local, pinned answerability judge for Task 06 pair building.

The dual-LLM audit measured the gap this judge closes: ~18% of ``chosen`` queries are
judged unanswerable from their passage even though they pass the corpus round trip, and
the round trip itself does not differentiate answerability.  The pair-policy v2 axis A
therefore needs a dedicated, **frozen** answerability verdict per candidate.

Design constraints, mirrored from the specification and the Task 06 contract:

* the judge is a **local, pinned** checkpoint (spec preference: ``qwen3.6-27b`` Q4 via
  ollama) — the runner refuses to start until the exact model digest is pinned in the
  config and matches what the local server reports;
* the judge is **not** the generator family (Bielik) and not the teacher of the
  ablation cohort, so no self-preference;
* ``uncertain`` blocks a candidate from the ``chosen`` role but is **not** counted as a
  defect — abstention must never manufacture axis-A rejected;
* every judgment lands in a durable, fsynced journal keyed by a deterministic item ID,
  so the run resumes without repeating work;
* the judge's verdicts are **calibrated before any use in pairs**: against the per-side
  ``answerable_a/b`` labels already collected by the Groq audit, and against the
  constructed classes of the reward-validation corpus (``ungrounded`` must be ``no``).

Nothing here builds a pair, trains anything or touches a final test.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from doc2query.preferences.groq_pair_audit import _proportion_ci, collect_ratings
from doc2query.preferences.pair_audit_export import load_blind_export_manifest
from doc2query.utils.records import read_durable_jsonl_prefix, read_records, write_json

CONTRACT = "task06-answerability-judge-v1"
PROMPT_VERSION = "task06-answerability-pl-v1"
JOURNAL_SCHEMA = "task06-answerability-judgment-v1"
VERDICTS = frozenset({"yes", "no", "uncertain"})
MAX_INVALID_RETRIES = 3

SYSTEM_PROMPT = (
    "Jesteś rygorystycznym polskim audytorem odpowiadalności. Otrzymasz pasaż i "
    "zapytanie wyszukiwawcze. Oceń wyłącznie jedno: czy na to zapytanie można "
    "odpowiedzieć korzystając WYŁĄCZNIE z informacji zawartych w podanym pasażu. "
    "Nie oceniaj stylu, długości ani użyteczności. Jeżeli odpowiedź wymaga wiedzy "
    "spoza pasażu, werdykt brzmi no. Jeżeli pasaż odpowiada tylko częściowo albo "
    "nie masz pewności, werdykt brzmi uncertain. Zwróć wyłącznie obiekt JSON "
    'w formacie {"verdict": "yes"} z wartością ze zbioru [yes, no, uncertain]. '
    "Bez toku rozumowania i bez komentarzy."
)

Transport = Callable[[str, Mapping[str, Any], float], dict[str, Any]]


@dataclass(frozen=True)
class JudgeItem:
    item_id: str
    query: str
    passage: str
    metadata: dict[str, Any]


def load_judge_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("contract") != CONTRACT:
        raise ValueError(f"unsupported answerability judge contract: {path}")
    if value.get("final_tests_used") != []:
        raise ValueError("the answerability judge must not use final tests")
    if value.get("prompt_version") != PROMPT_VERSION:
        raise ValueError("the judge prompt version drifted from this build")
    judge = value.get("judge")
    if not isinstance(judge, Mapping) or judge.get("backend") != "ollama":
        raise ValueError("the judge contract pins the local ollama backend")
    if float(judge.get("temperature", -1.0)) != 0.0:
        raise ValueError("the answerability judge must run at temperature zero")
    generator_family = str(value.get("generator_family", ""))
    if generator_family.casefold() in str(judge.get("model", "")).casefold():
        raise ValueError("the judge must not belong to the generator family")
    decision = value.get("decision")
    if not isinstance(decision, Mapping) or (
        decision.get("uncertain_blocks_chosen") is not True
        or decision.get("uncertain_is_not_a_defect") is not True
    ):
        raise ValueError("uncertain must block chosen and must not count as a defect")
    return value


def _http_transport(url: str, payload: Mapping[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return cast(dict[str, Any], json.loads(response.read().decode()))


def verify_pinned_model(config: Mapping[str, Any], transport: Transport) -> str:
    """Refuse to judge anything until the local weights match the pinned digest."""
    judge = cast(Mapping[str, Any], config["judge"])
    pinned = judge.get("model_digest")
    if not pinned:
        raise ValueError(
            "the judge config pins no model_digest; pin the exact local weights first "
            "(see scripts/run_task06_answerability_judge.py --print-model-info)"
        )
    listing = transport(str(judge["tags_url"]), {}, float(judge["timeout_seconds"]))
    models = {
        str(row.get("name", "")): str(row.get("digest", ""))
        for row in cast(Sequence[Mapping[str, Any]], listing.get("models", []))
    }
    digest = models.get(str(judge["model"]))
    if digest is None:
        raise ValueError(f"local backend does not serve the pinned model {judge['model']!r}")
    if digest != pinned:
        raise ValueError(
            f"local model digest {digest!r} does not match the pinned {pinned!r}"
        )
    return digest


def judge_item_id(query: str, passage: str) -> str:
    return hashlib.sha256(f"{PROMPT_VERSION}\0{query}\0{passage}".encode()).hexdigest()[:24]


def _chat_payload(item: JudgeItem, config: Mapping[str, Any]) -> dict[str, Any]:
    judge = cast(Mapping[str, Any], config["judge"])
    user = json.dumps(
        {"passage": item.passage, "query": item.query}, ensure_ascii=False, sort_keys=True
    )
    return {
        "model": judge["model"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.0,
            "seed": int(judge["seed"]),
            "num_predict": int(judge["max_completion_tokens"]),
        },
    }


def parse_verdict(content: str) -> str:
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise ValueError("judge response is not a JSON object")
    verdict = str(payload.get("verdict", "")).strip().casefold()
    if verdict not in VERDICTS:
        raise ValueError(f"invalid answerability verdict: {verdict!r}")
    return verdict


def _append_event(path: Path, event: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_judgments(journal: Path) -> dict[str, dict[str, Any]]:
    verdicts: dict[str, dict[str, Any]] = {}
    for event in read_durable_jsonl_prefix(journal):
        if event.get("event") == "verdict":
            verdicts[str(event["item_id"])] = dict(event)
    return verdicts


def run_judgments(
    items: Sequence[JudgeItem],
    *,
    config: Mapping[str, Any],
    output_dir: Path,
    transport: Transport = _http_transport,
    sleep: Callable[[float], None] = time.sleep,
    max_new_judgments: int | None = None,
) -> dict[str, Any]:
    """Judge every item once, resumably; invalid outputs fail closed per item."""
    ids = [item.item_id for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("judge items must have unique IDs")
    digest = verify_pinned_model(config, transport)
    judge = cast(Mapping[str, Any], config["judge"])
    journal = output_dir / "judgments.journal.jsonl"
    done = load_judgments(journal)
    counters: Counter[str] = Counter()
    new = 0
    for item in items:
        if item.item_id in done:
            counters["already_judged"] += 1
            continue
        if max_new_judgments is not None and new >= max_new_judgments:
            counters["deferred_by_operator_cap"] += 1
            continue
        payload = _chat_payload(item, config)
        verdict: str | None = None
        for attempt in range(1, MAX_INVALID_RETRIES + 1):
            response = transport(str(judge["url"]), payload, float(judge["timeout_seconds"]))
            content = str(
                cast(Mapping[str, Any], response.get("message", {})).get("content", "")
            )
            try:
                verdict = parse_verdict(content)
                break
            except (ValueError, json.JSONDecodeError) as exc:
                _append_event(
                    journal,
                    {
                        "schema": JOURNAL_SCHEMA,
                        "event": "invalid_verdict",
                        "item_id": item.item_id,
                        "attempt": attempt,
                        "error": str(exc)[:500],
                    },
                )
                sleep(0.0)
        new += 1
        if verdict is None:
            counters["failed_closed"] += 1
            continue
        _append_event(
            journal,
            {
                "schema": JOURNAL_SCHEMA,
                "event": "verdict",
                "item_id": item.item_id,
                "verdict": verdict,
                "model_digest": digest,
                "prompt_version": PROMPT_VERSION,
                "metadata": item.metadata,
            },
        )
        counters[f"verdict_{verdict}"] += 1
    return {
        "contract": CONTRACT,
        "model_digest": digest,
        "item_count": len(items),
        "judged_count": len(load_judgments(journal)),
        "counters": dict(sorted(counters.items())),
        "final_tests_used": [],
    }


def calibration_items_from_audit(export_dir: Path) -> list[JudgeItem]:
    """One item per (pair, side) of the blind export, tagged with the Groq labels."""
    manifest = load_blind_export_manifest(export_dir / "manifest.json")
    sample = {
        str(row["pair_id"]): row
        for row in read_records(export_dir / str(manifest.sample["path"]))
    }
    ratings = collect_ratings(export_dir / "groq_dual_llm")
    items: list[JudgeItem] = []
    for key_row in read_records(export_dir / "machine_key.jsonl"):
        audit_id = str(key_row["audit_id"])
        pair = sample[str(key_row["pair_id"])]
        automatic = str(key_row["automatic_chosen_option"])
        for role in ("chosen", "rejected"):
            suffix = (
                ("a" if automatic == "A" else "b")
                if role == "chosen"
                else ("b" if automatic == "A" else "a")
            )
            references = {
                model: bool(rating[f"answerable_{suffix}"])
                for model, model_ratings in ratings.items()
                if (rating := model_ratings.get(audit_id)) is not None
            }
            if not references:
                continue
            query = str(pair[role])
            items.append(
                JudgeItem(
                    item_id=judge_item_id(query, str(pair["passage"])),
                    query=query,
                    passage=str(pair["passage"]),
                    metadata={
                        "source": "groq_audit",
                        "audit_id": audit_id,
                        "role": role,
                        "groq_answerable": references,
                    },
                )
            )
    if not items:
        raise ValueError("the audit export holds no collected answerability labels yet")
    return items


def calibration_items_from_reward_corpus(
    corpus_path: Path, cohort_records_path: Path
) -> list[JudgeItem]:
    """Constructed-class sanity items: ``ungrounded`` must be ``no``, good classes ``yes``."""
    expectations = {"good_specific": "yes", "good_alternative": "yes", "ungrounded": "no"}
    passages: dict[str, str] = {}
    for row in read_records(cohort_records_path):
        positives = row.get("positives") or []
        if positives:
            passages[str(row["example_id"])] = str(positives[0]["text"])
    items: list[JudgeItem] = []
    for row in read_records(corpus_path):
        label = str(row.get("label"))
        expected = expectations.get(label)
        if expected is None:
            continue
        passage = passages[str(row["example_id"])]
        query = str(row["query"])
        items.append(
            JudgeItem(
                item_id=judge_item_id(query, passage),
                query=query,
                passage=passage,
                metadata={"source": "reward_corpus", "label": label, "expected": expected},
            )
        )
    if not items:
        raise ValueError("the reward corpus holds no calibration classes")
    return items


def analyze_calibration(
    items: Iterable[JudgeItem], judgments: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Agreement with each Groq judge, with their consensus, and per constructed class."""
    per_groq: dict[str, list[bool]] = {}
    consensus: list[bool] = []
    per_class: dict[str, list[bool]] = {}
    uncertain: Counter[str] = Counter()
    judged = 0
    for item in items:
        record = judgments.get(item.item_id)
        if record is None:
            continue
        judged += 1
        verdict = str(record["verdict"])
        source = str(item.metadata["source"])
        if verdict == "uncertain":
            uncertain[source] += 1
            continue
        answerable = verdict == "yes"
        if source == "groq_audit":
            references = cast(Mapping[str, bool], item.metadata["groq_answerable"])
            for model, label in references.items():
                per_groq.setdefault(model, []).append(answerable == label)
            values = set(references.values())
            if len(references) == 2 and len(values) == 1:
                consensus.append(answerable == next(iter(values)))
        else:
            expected = str(item.metadata["expected"]) == "yes"
            per_class.setdefault(str(item.metadata["label"]), []).append(
                answerable == expected
            )
    return {
        "schema_version": 1,
        "contract": CONTRACT,
        "judged_items": judged,
        "agreement_with_groq": {
            model: _proportion_ci(values) for model, values in sorted(per_groq.items())
        },
        "agreement_with_groq_consensus": _proportion_ci(consensus),
        "constructed_class_accuracy": {
            label: _proportion_ci(values) for label, values in sorted(per_class.items())
        },
        "uncertain_counts": dict(sorted(uncertain.items())),
        "used_for_pair_building": False,
        "final_tests_used": [],
    }


def write_calibration_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, dict(report))
