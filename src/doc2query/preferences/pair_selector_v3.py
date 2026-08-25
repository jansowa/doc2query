"""Selektor preferencji v3: lokalny sędzia porównuje zapytania parami.

Kontrakt i granice zamraża ADR `reports/decisions/task06_judge_selected_pair_policy_v3.md`
(spisany przed pierwszym wywołaniem). Ten moduł realizuje jego §3-§6:

* **ślepość** — sędzia widzi pasaż i dwa zapytania, nigdy score'ów, round-tripu ani
  informacji o tym, co wybrał jakikolwiek automat;
* **zamiana pozycji** — każde porównanie leci w obu kolejnościach; niezgoda między
  kolejnościami to `position_flip`, czyli przypadek **nierozstrzygnięty**, nie remis;
* **trzy rubryki** z definicjami i jawną hierarchią konfliktu (§4 ADR), teksty
  zamrożone tutaj jako stałe — modyfikacja to zmiana kontraktu, nie parametr;
* **kalibracja przed progiem** — na klasach korpusu walidacyjnego nagrody, gdzie
  prawda jest znana z konstrukcji; próg agregacji zamraża osobny amendment po
  odczycie tych liczb.

Transport jest OpenAI-compatible (`{base_url}/chat/completions`), więc adres i klucz
są **parametrami wywołania**, nigdy nie trafiają do repozytorium. Journal jest trwały,
run wznawialny, a nic tutaj nie buduje par ani nie autoryzuje treningu.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from doc2query.utils.records import read_records, write_json

CONTRACT = "task06-judge-selected-pair-policy-v3"
JOURNAL_SCHEMA = "task06-v3-pairwise-judgment-v1"
PROMPT_VERSION = "task06-v3-pairwise-pl-v1"
VERDICTS = frozenset({"A", "B", "tie"})
MAX_RETRIES = 4

_HIERARCHY = (
    "Hierarchia przy konflikcie kryteriów jest wiążąca: ugruntowanie przed "
    "użytecznością wyszukiwawczą, użyteczność przed naturalnością, naturalność przed "
    "długością. Kopiowanie długiego fragmentu pasażu dyskwalifikuje zapytanie "
    "niezależnie od pozostałych kryteriów."
)
_OUTPUT = (
    'Zwróć wyłącznie obiekt JSON: {"better": "A"|"B"|"tie", "confidence": liczba w '
    "[0,1]}. Bez żadnego tekstu poza JSON."
)

# Rubryki zamrożone przez §4 ADR. Kolejność i treść są częścią kontraktu.
RUBRICS: dict[str, str] = {
    "R1_grounding": (
        "Jesteś rygorystycznym polskim audytorem zapytań wyszukiwawczych. Otrzymasz "
        "pasaż i dwa zapytania, A i B. Oceń wyłącznie UGRUNTOWANIE: czy na zapytanie "
        "można odpowiedzieć korzystając WYŁĄCZNIE z informacji zawartych w pasażu, "
        "czy nie wymaga wiedzy zewnętrznej i czy nie przekręca faktów, liczb ani nazw "
        "własnych. Zapytanie o coś, czego w pasażu nie ma, jest gorsze niezależnie od "
        "tego, jak dobrze brzmi. Wskaż lepiej ugruntowane zapytanie. " + _OUTPUT
    ),
    "R2_retrieval_usefulness": (
        "Jesteś polskim ekspertem wyszukiwania. Otrzymasz pasaż i dwa zapytania, A i "
        "B. Oceń wyłącznie UŻYTECZNOŚĆ WYSZUKIWAWCZĄ: czy zapytanie ma sens jako "
        "realne zapytanie użytkownika, czy nie jest tak ogólne, że pasowałoby do "
        "tysiąca innych pasaży, czy nie jest przepisanym zdaniem z pasażu i czy nie "
        "zdradza odpowiedzi w swojej treści. Wskaż bardziej użyteczne zapytanie. "
        + _OUTPUT
    ),
    "R3_holistic": (
        "Jesteś rygorystycznym polskim audytorem zapytań wyszukiwawczych. Otrzymasz "
        "pasaż i dwa zapytania, A i B. Oceń całościowo: ugruntowanie w pasażu, "
        "użyteczność wyszukiwawczą, naturalność sformułowania, brak nadmiernego "
        "kopiowania i brak zdradzania odpowiedzi. " + _HIERARCHY + " Wskaż lepsze "
        "zapytanie. " + _OUTPUT
    ),
}
RANKING_RUBRIC = "R3_holistic"
Transport = Callable[[Mapping[str, Any]], dict[str, Any]]
Order = Literal["ab", "ba"]


@dataclass(frozen=True)
class PairwiseItem:
    """Jedno porównanie: pasaż i dwa zapytania w kanonicznej kolejności."""

    item_id: str
    passage: str
    query_first: str
    query_second: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class JudgeEndpoint:
    """Adres i model sędziego; nigdy nie zapisywane do repozytorium."""

    base_url: str
    api_key: str
    model: str
    temperature: float = 0.0
    max_completion_tokens: int = 512
    timeout_seconds: float = 300.0
    allow_reasoning: bool = False


def http_transport(endpoint: JudgeEndpoint) -> Transport:
    """Minimalny klient OpenAI-compatible; klucz trafia wyłącznie do nagłówka."""

    def call(payload: Mapping[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            endpoint.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {endpoint.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=endpoint.timeout_seconds) as response:
            return cast(dict[str, Any], json.loads(response.read().decode("utf-8")))

    return call


def chat_payload(
    item: PairwiseItem, rubric: str, order: Order, endpoint: JudgeEndpoint
) -> dict[str, Any]:
    """Zbuduj request; przy `order="ba"` zapytania są zamienione miejscami."""
    if rubric not in RUBRICS:
        raise ValueError(f"nieznana rubryka: {rubric}")
    first, second = (
        (item.query_first, item.query_second)
        if order == "ab"
        else (item.query_second, item.query_first)
    )
    user = (
        f"Pasaż:\n{item.passage}\n\nZapytanie A:\n{first}\n\nZapytanie B:\n{second}\n\n"
        "Które zapytanie jest lepsze?"
    )
    payload: dict[str, Any] = {
        "model": endpoint.model,
        "messages": [
            {"role": "system", "content": RUBRICS[rubric]},
            {"role": "user", "content": user},
        ],
        "temperature": endpoint.temperature,
        "max_tokens": endpoint.max_completion_tokens,
        "response_format": {"type": "json_object"},
    }
    if not endpoint.allow_reasoning:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    return payload


def parse_verdict(content: str) -> tuple[str, float]:
    """Wyciągnij `better` i `confidence`; cokolwiek innego jest błędem, nie remisem."""
    start = content.find("{")
    end = content.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("odpowiedź sędziego nie zawiera obiektu JSON")
    payload = json.loads(content[start : end + 1])
    better = str(payload.get("better", "")).strip().upper()
    if better == "TIE":
        better = "tie"
    if better not in VERDICTS:
        raise ValueError(f"niedozwolony werdykt: {better!r}")
    confidence = payload.get("confidence", 0.0)
    return better, float(confidence) if isinstance(confidence, (int, float)) else 0.0


def _canonical(verdict: str, order: Order) -> str:
    """Przelicz werdykt na kolejność kanoniczną, żeby swap nie zaburzał liczenia."""
    if verdict == "tie" or order == "ab":
        return verdict
    return "B" if verdict == "A" else "A"


def journal_key(item_id: str, rubric: str, order: Order) -> str:
    return f"{item_id}|{rubric}|{order}"


def load_journal(path: Path) -> dict[str, dict[str, Any]]:
    """Trwały prefiks journala; niepełna ostatnia linia jest pomijana."""
    if not path.is_file():
        return {}
    done: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            break
        if row.get("event") == "judgment":
            done[str(row["key"])] = row
    return done


def _append(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run_pairwise(
    *,
    items: Sequence[PairwiseItem],
    endpoint: JudgeEndpoint,
    journal_path: Path,
    transport: Transport | None = None,
    rubrics: Sequence[str] | None = None,
    orders: Sequence[Order] = ("ab", "ba"),
    progress_every: int = 200,
    sleep_seconds: float = 0.0,
) -> dict[str, Any]:
    """Zbierz werdykty dla każdej pary, rubryki i kolejności, wznawialnie."""
    call = transport or http_transport(endpoint)
    selected = list(rubrics or RUBRICS)
    done = load_journal(journal_path)
    resumed = len(done)
    errors = 0
    started = time.perf_counter()
    for index, item in enumerate(items):
        for rubric in selected:
            for order in orders:
                key = journal_key(item.item_id, rubric, order)
                if key in done:
                    continue
                payload = chat_payload(item, rubric, order, endpoint)
                verdict: str | None = None
                confidence = 0.0
                failure: str | None = None
                for attempt in range(1, MAX_RETRIES + 1):
                    try:
                        response = call(payload)
                        choices = cast(list[Any], response["choices"])
                        content = str(cast(Mapping[str, Any], choices[0]["message"])["content"])
                        verdict, confidence = parse_verdict(content)
                        break
                    except (urllib.error.URLError, TimeoutError, OSError) as exc:
                        failure = f"transport:{exc}"
                        time.sleep(min(30.0, 2.0**attempt))
                    except (ValueError, KeyError, json.JSONDecodeError) as exc:
                        failure = f"schema:{exc}"
                if verdict is None:
                    errors += 1
                    _append(
                        journal_path,
                        {
                            "schema": JOURNAL_SCHEMA,
                            "event": "failure",
                            "key": key,
                            "reason": failure,
                        },
                    )
                    continue
                row = {
                    "schema": JOURNAL_SCHEMA,
                    "event": "judgment",
                    "key": key,
                    "item_id": item.item_id,
                    "rubric": rubric,
                    "order": order,
                    "verdict": verdict,
                    "canonical_verdict": _canonical(verdict, order),
                    "confidence": confidence,
                    "metadata": dict(item.metadata),
                    "prompt_version": PROMPT_VERSION,
                    "allow_reasoning": endpoint.allow_reasoning,
                }
                _append(journal_path, row)
                done[key] = row
                if sleep_seconds:
                    time.sleep(sleep_seconds)
        if progress_every and (index + 1) % progress_every == 0:
            print(f"[selector] {index + 1}/{len(items)} par", flush=True)
    return {
        "contract": CONTRACT,
        "items": len(items),
        "rubrics": selected,
        "orders": list(orders),
        "judgments": len(done),
        "resumed": resumed,
        "collected": len(done) - resumed,
        "failures": errors,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "allow_reasoning": endpoint.allow_reasoning,
        "final_tests_used": [],
    }


GOOD_CLASSES = ("good_specific", "good_alternative")
BAD_CLASSES = ("too_general", "ungrounded", "copy_verbatim", "wrong_focus", "wrong_form")


def calibration_items(corpus_path: Path, passages_path: Path) -> list[PairwiseItem]:
    """Pary o kierunku znanym z konstrukcji: klasa dobra przeciw klasie zepsutej."""
    passages = {
        str(row["example_id"]): str(row["passage"]) for row in read_records(passages_path)
    }
    by_example: dict[str, dict[str, str]] = {}
    for row in read_records(corpus_path):
        label = str(row["label"])
        if label in GOOD_CLASSES or label in BAD_CLASSES:
            by_example.setdefault(str(row["example_id"]), {})[label] = str(row["query"])
    items: list[PairwiseItem] = []
    for example_id in sorted(by_example):
        labels = by_example[example_id]
        passage = passages.get(example_id)
        if passage is None:
            continue
        for good in GOOD_CLASSES:
            for bad in BAD_CLASSES:
                if good not in labels or bad not in labels:
                    continue
                # Kanoniczna kolejność jest zawsze (dobre, zepsute); swap robi runner,
                # więc obciążenie pozycyjne jest mierzalne, a nie wmieszane w dane.
                items.append(
                    PairwiseItem(
                        item_id=f"{example_id}|{good}|{bad}",
                        passage=passage,
                        query_first=labels[good],
                        query_second=labels[bad],
                        metadata={
                            "example_id": example_id,
                            "good_label": good,
                            "bad_label": bad,
                            "expected_canonical": "A",
                        },
                    )
                )
    if not items:
        raise ValueError("korpus walidacyjny nie dał żadnej pary kalibracyjnej")
    return items


def analyze_calibration(journal_path: Path) -> dict[str, Any]:
    """Policz czystość per rubryka, obciążenie pozycyjne i krzywą czystość/wydajność."""
    rows = [row for row in load_journal(journal_path).values()]
    if not rows:
        raise ValueError("journal nie zawiera żadnego werdyktu")
    per_rubric: dict[str, Counter[str]] = {}
    by_item: dict[str, dict[tuple[str, str], str]] = {}
    for row in rows:
        rubric = str(row["rubric"])
        canonical = str(row["canonical_verdict"])
        per_rubric.setdefault(rubric, Counter())[canonical] += 1
        by_item.setdefault(str(row["item_id"]), {})[(rubric, str(row["order"]))] = canonical

    rubric_report: dict[str, Any] = {}
    for rubric, counts in sorted(per_rubric.items()):
        total = sum(counts.values())
        rubric_report[rubric] = {
            "judgments": total,
            "correct_share": counts["A"] / total,
            "wrong_share": counts["B"] / total,
            "tie_share": counts["tie"] / total,
        }

    flips: Counter[str] = Counter()
    votes_per_item: dict[str, int] = {}
    complete_items = 0
    for item_id, verdicts in by_item.items():
        rubrics = sorted({rubric for rubric, _order in verdicts})
        if len(verdicts) != len(rubrics) * 2:
            continue
        complete_items += 1
        correct = 0
        for rubric in rubrics:
            ab = verdicts[(rubric, "ab")]
            ba = verdicts[(rubric, "ba")]
            if ab != ba:
                flips[rubric] += 1
                continue
            if ab == "A":
                correct += 2
        votes_per_item[item_id] = correct
    curve = {}
    for threshold in (6, 5, 4, 3):
        surviving = [item for item, votes in votes_per_item.items() if votes >= threshold]
        curve[f"min_votes_{threshold}"] = {
            "surviving_items": len(surviving),
            "yield": len(surviving) / complete_items if complete_items else None,
        }
    return {
        "contract": CONTRACT,
        "prompt_version": PROMPT_VERSION,
        "complete_items": complete_items,
        "per_rubric": rubric_report,
        "position_flip_counts": dict(sorted(flips.items())),
        "position_flip_share": {
            rubric: count / complete_items if complete_items else None
            for rubric, count in sorted(flips.items())
        },
        "aggregation_curve": curve,
        "threshold_frozen_here": False,
        "note": (
            "Próg agregacji zamraża osobny amendment po odczycie tych liczb i przed "
            "zbudowaniem pierwszej pary v3 (§6 ADR)."
        ),
        "final_tests_used": [],
    }


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    write_json(path, dict(report))


def endpoint_from_args(
    *,
    base_url: str,
    api_key: str | None,
    api_key_env: str,
    model: str,
    allow_reasoning: bool,
    max_completion_tokens: int,
) -> JudgeEndpoint:
    """Klucz z argumentu albo ze zmiennej środowiskowej; nigdy z repozytorium."""
    key = api_key if api_key else os.environ.get(api_key_env, "")
    if not key:
        raise ValueError(
            f"brak klucza API: podaj --api-key albo ustaw zmienną {api_key_env}"
        )
    if not base_url.startswith(("http://", "https://")):
        raise ValueError("base-url musi być pełnym adresem http(s)")
    return JudgeEndpoint(
        base_url=base_url,
        api_key=key,
        model=model,
        allow_reasoning=allow_reasoning,
        max_completion_tokens=max_completion_tokens,
    )


def plan_summary(items: Iterable[PairwiseItem], rubrics: Sequence[str]) -> dict[str, Any]:
    rows = list(items)
    calls = len(rows) * len(rubrics) * 2
    return {
        "items": len(rows),
        "rubrics": list(rubrics),
        "orders": 2,
        "calls": calls,
        "estimated_minutes_at_19_1_per_second": round(calls / 19.1 / 60, 1),
    }


__all__ = [
    "CONTRACT",
    "RANKING_RUBRIC",
    "RUBRICS",
    "JudgeEndpoint",
    "PairwiseItem",
    "analyze_calibration",
    "calibration_items",
    "chat_payload",
    "endpoint_from_args",
    "http_transport",
    "journal_key",
    "load_journal",
    "parse_verdict",
    "plan_summary",
    "run_pairwise",
    "write_report",
]
