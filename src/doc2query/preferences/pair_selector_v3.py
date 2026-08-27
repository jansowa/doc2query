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
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from doc2query.utils.records import read_records, write_json

CONTRACT = "task06-judge-selected-pair-policy-v3"
JOURNAL_SCHEMA = "task06-v3-pairwise-judgment-v1"
PROMPT_VERSION = "task06-v3-pairwise-pl-v1"
VERDICTS = frozenset({"A", "B", "tie"})
MAX_RETRIES = 4
# Jeżeli tyle wywołań padnie, a żadne nie przejdzie, run przerywa się z przyczyną —
# inaczej zły model albo nieznane pole API zapisałoby tysiące cichych porażek.
FAIL_FAST_AFTER = 8

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
# Pola, które część serwerów OpenAI-compatible odrzuca; przy odmowie run próbuje raz
# bez nich i zapisuje ten fakt, zamiast padać 10 800 razy na tym samym.
OPTIONAL_PAYLOAD_FIELDS = ("chat_template_kwargs", "response_format")


class JudgeApiError(RuntimeError):
    """Błąd wywołania sędziego z zachowanym statusem i ciałem odpowiedzi.

    Ciało jest tu kluczowe: vLLM zwraca w nim konkretną przyczynę (nieznany model,
    nieobsługiwane pole, zły klucz), a bez niego diagnoza sprowadza się do „coś nie
    działa". `retryable` rozdziela błędy przejściowe od trwałych — powtarzanie
    czterech prób przy 400 tylko wydłużało ciszę.
    """

    def __init__(self, message: str, *, status: int | None, body: str, retryable: bool):
        super().__init__(message)
        self.status = status
        self.body = body
        self.retryable = retryable

    def summary(self, limit: int = 400) -> str:
        body = " ".join(self.body.split())[:limit]
        return f"HTTP {self.status}: {body}" if self.status else f"{self}: {body}"


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
    # Serwer vLLM bez uwierzytelniania jest normalny w sieci lokalnej; wtedy nagłówek
    # Authorization jest pomijany, a nie wysyłany z pustym kluczem.
    temperature: float = 0.0
    max_completion_tokens: int = 512
    timeout_seconds: float = 120.0
    allow_reasoning: bool = False


def http_transport(endpoint: JudgeEndpoint) -> Transport:
    """Minimalny klient OpenAI-compatible; klucz trafia wyłącznie do nagłówka."""

    def call(payload: Mapping[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if endpoint.api_key:
            headers["Authorization"] = f"Bearer {endpoint.api_key}"
        request = urllib.request.Request(
            endpoint.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=endpoint.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                body = "(brak ciała odpowiedzi)"
            # 429 i 5xx są przejściowe; 4xx to kontrakt API i powtarzanie nic nie da.
            retryable = exc.code == 429 or exc.code >= 500
            raise JudgeApiError(
                f"serwer sędziego odrzucił request ({exc.code})",
                status=exc.code,
                body=body,
                retryable=retryable,
            ) from exc
        except (TimeoutError, urllib.error.URLError, OSError) as exc:
            raise JudgeApiError(
                f"transport nie doszedł do serwera sędziego: {exc}",
                status=None,
                body=str(exc),
                retryable=True,
            ) from exc
        try:
            return cast(dict[str, Any], json.loads(raw))
        except json.JSONDecodeError as exc:
            raise JudgeApiError(
                "serwer sędziego zwrócił coś, co nie jest JSON-em",
                status=None,
                body=raw[:2000],
                retryable=False,
            ) from exc

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


def strip_reasoning(content: str) -> str:
    """Usuń bloki myślenia; model rozumujący potrafi w nich zawrzeć nawiasy klamrowe."""
    text = content
    for opening, closing in (("<think>", "</think>"), ("<thinking>", "</thinking>")):
        while opening in text:
            start = text.index(opening)
            stop = text.find(closing, start)
            if stop < 0:
                text = text[:start]
                break
            text = text[:start] + text[stop + len(closing) :]
    return text.strip()


def _last_json_object(text: str) -> str:
    """Zwróć ostatni domknięty obiekt JSON; werdykt jest zawsze na końcu odpowiedzi."""
    end = text.rfind("}")
    while end >= 0:
        depth = 0
        for index in range(end, -1, -1):
            if text[index] == "}":
                depth += 1
            elif text[index] == "{":
                depth -= 1
                if depth == 0:
                    return text[index : end + 1]
        end = text.rfind("}", 0, end)
    raise ValueError("odpowiedź sędziego nie zawiera obiektu JSON")


def parse_verdict(content: str) -> tuple[str, float]:
    """Wyciągnij `better` i `confidence`; cokolwiek innego jest błędem, nie remisem."""
    text = strip_reasoning(content)
    if not text:
        raise ValueError("odpowiedź sędziego jest pusta po usunięciu bloku myślenia")
    payload = json.loads(_last_json_object(text))
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
    # Dzielimy TYLKO po "\n": pasaże msmarco_pl zawierają U+0085 i U+2028, które
    # `splitlines()` traktuje jako łamanie linii, a JSON ich nie escapuje.
    for line in path.read_text(encoding="utf-8").split("\n"):
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


def _field_rejected_by_server(error: JudgeApiError, payload: Mapping[str, Any]) -> str | None:
    """Rozpoznaj odmowę konkretnego pola, żeby ponowić bez niego, a nie polec 10 800 razy."""
    if error.status is None or error.retryable:
        return None
    haystack = error.body.lower()
    for field_name in OPTIONAL_PAYLOAD_FIELDS:
        if field_name in payload and field_name.lower() in haystack:
            return field_name
    return None


def probe_endpoint(
    endpoint: JudgeEndpoint, transport: Transport | None = None
) -> dict[str, Any]:
    """Jedno wywołanie sprawdzające, zanim polecą tysiące.

    Bez tego zły model albo nieobsługiwane pole objawiają się jako minuty ciszy.
    Przy porażce wyjątek nosi ciało odpowiedzi serwera, czyli faktyczną przyczynę.
    """
    call = transport or http_transport(endpoint)
    item = PairwiseItem(
        item_id="probe",
        passage="Stolicą Polski jest Warszawa, największe miasto kraju nad Wisłą.",
        query_first="jakie miasto jest stolicą Polski",
        query_second="ile kosztuje bilet do kina",
        metadata={"source": "probe"},
    )
    started = time.perf_counter()
    payload = dict(chat_payload(item, RANKING_RUBRIC, "ab", endpoint))
    dropped: list[str] = []
    while True:
        try:
            response = call(payload)
            break
        except JudgeApiError as exc:
            field_name = _field_rejected_by_server(exc, payload)
            if field_name is None:
                raise
            payload.pop(field_name, None)
            dropped.append(field_name)
    choices = cast(list[Any], response.get("choices") or [])
    if not choices:
        raise JudgeApiError(
            "odpowiedź sędziego nie zawiera choices",
            status=None,
            body=str(response)[:2000],
            retryable=False,
        )
    content = str(cast(Mapping[str, Any], choices[0]["message"])["content"])
    verdict, confidence = parse_verdict(content)
    return {
        "ok": True,
        "verdict": verdict,
        "confidence": confidence,
        "seconds": round(time.perf_counter() - started, 2),
        "dropped_payload_fields": dropped,
        "raw_content_prefix": " ".join(content.split())[:200],
    }


def run_pairwise(
    *,
    items: Sequence[PairwiseItem],
    endpoint: JudgeEndpoint,
    journal_path: Path,
    transport: Transport | None = None,
    rubrics: Sequence[str] | None = None,
    orders: Sequence[Order] = ("ab", "ba"),
    progress_every: int = 20,
    concurrency: int = 1,
) -> dict[str, Any]:
    """Zbierz werdykty dla każdej pary, rubryki i kolejności, wznawialnie.

    Równoległość jest po stronie klienta: serwer vLLM robi continuous batching, więc
    kilkanaście jednoczesnych requestów podnosi przepustowość niemal liniowo, a
    generacja jest tu wąskim gardłem (512 tokenów na werdykt, 1024 z rozumowaniem).
    Zadania są **wyznaczone przed startem** po odjęciu tego, co już jest w journalu,
    więc żaden werdykt nie może zostać policzony dwa razy, a zapis do journala idzie
    pod jednym zamkiem, żeby linie nie przeplatały się w środku.
    """
    if concurrency < 1:
        raise ValueError("concurrency musi być dodatnie")
    call = transport or http_transport(endpoint)
    selected = list(rubrics or RUBRICS)
    done = load_journal(journal_path)
    resumed = len(done)
    tasks = [
        (item, rubric, order)
        for item in items
        for rubric in selected
        for order in orders
        if journal_key(item.item_id, rubric, order) not in done
    ]
    write_lock = threading.Lock()
    counters = {"collected": 0, "failures": 0}
    first_failure: list[str] = []
    started = time.perf_counter()

    disabled_fields: list[str] = []

    def judge(task: tuple[PairwiseItem, str, Order]) -> dict[str, Any]:
        item, rubric, order = task
        key = journal_key(item.item_id, rubric, order)
        failure: str | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            payload = dict(chat_payload(item, rubric, order, endpoint))
            for field_name in disabled_fields:
                payload.pop(field_name, None)
            try:
                response = call(payload)
                choices = cast(list[Any], response.get("choices") or [])
                if not choices:
                    raise ValueError(f"odpowiedź bez pola choices: {str(response)[:300]}")
                message = cast(Mapping[str, Any], choices[0].get("message") or {})
                content = message.get("content")
                if content is None:
                    raise ValueError(f"odpowiedź bez treści: {str(message)[:300]}")
                verdict, confidence = parse_verdict(str(content))
                return {
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
                    "dropped_payload_fields": list(disabled_fields),
                }
            except JudgeApiError as exc:
                failure = f"api:{exc.summary()}"
                dropped = _field_rejected_by_server(exc, payload)
                if dropped is not None:
                    with write_lock:
                        if dropped not in disabled_fields:
                            disabled_fields.append(dropped)
                            print(
                                f"[selector] serwer odrzucił pole {dropped!r}; "
                                "ponawiam bez niego i zapisuję ten fakt w journalu",
                                flush=True,
                            )
                    continue
                if not exc.retryable:
                    break
                time.sleep(min(30.0, 2.0**attempt))
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                failure = f"schema:{exc}"
        return {
            "schema": JOURNAL_SCHEMA,
            "event": "failure",
            "key": key,
            "reason": failure,
        }

    def record(row: Mapping[str, Any]) -> None:
        with write_lock:
            _append(journal_path, row)
            if row["event"] == "judgment":
                counters["collected"] += 1
                if counters["collected"] == 1:
                    print(
                        f"[selector] pierwsza odpowiedź OK po "
                        f"{time.perf_counter() - started:.1f} s "
                        f"(werdykt {row['verdict']}, kanonicznie {row['canonical_verdict']})",
                        flush=True,
                    )
            else:
                counters["failures"] += 1
                reason = str(row.get("reason"))
                if not first_failure:
                    first_failure.append(reason)
                    print(f"[selector] PIERWSZY BŁĄD: {reason}", flush=True)
            finished = counters["collected"] + counters["failures"]
            if (
                counters["collected"] == 0
                and counters["failures"] >= FAIL_FAST_AFTER
            ):
                raise RuntimeError(
                    f"{counters['failures']} wywołań pod rząd zawiodło i ani jedno nie "
                    f"przeszło; przerywam zamiast zapisywać porażki. Pierwsza przyczyna: "
                    f"{first_failure[0]}"
                )
            if progress_every and finished % progress_every == 0:
                rate = finished / max(1e-9, time.perf_counter() - started)
                remaining = (len(tasks) - finished) / rate / 60 if rate else 0.0
                print(
                    f"[selector] {finished}/{len(tasks)} wywołań, {rate:.2f}/s, "
                    f"zostało ~{remaining:.0f} min",
                    flush=True,
                )

    if concurrency == 1:
        for task in tasks:
            record(judge(task))
    else:
        pool = ThreadPoolExecutor(max_workers=concurrency)
        try:
            futures = [pool.submit(judge, task) for task in tasks]
            for future in as_completed(futures):
                record(future.result())
        except BaseException:
            # Bez anulowania abort czekałby na wszystkie requesty w locie, czyli do
            # `concurrency` razy timeout. Zaległe zadania nie są potrzebne — journal i tak
            # ma tylko to, co zdążyło się zapisać, a wznowienie dobierze resztę.
            for future in futures:
                future.cancel()
            pool.shutdown(wait=False)
            raise
        else:
            pool.shutdown(wait=True)

    elapsed = time.perf_counter() - started
    return {
        "contract": CONTRACT,
        "items": len(items),
        "rubrics": selected,
        "orders": list(orders),
        "concurrency": concurrency,
        "planned_calls": len(tasks),
        "judgments": resumed + counters["collected"],
        "resumed": resumed,
        "collected": counters["collected"],
        "failures": counters["failures"],
        "elapsed_seconds": round(elapsed, 1),
        "observed_calls_per_second": round(
            (counters["collected"] + counters["failures"]) / elapsed, 3
        )
        if elapsed > 0
        else None,
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
    """Policz czystość per rubryka, obciążenie pozycyjne i krzywą czystość/wydajność.

    Rozróżnienie, które trzeba trzymać osobno, bo łatwo je pomylić:

    * **wydajność** (`yield`) — jaki odsetek par filtr zatrzymuje;
    * **czystość** (`precision`) — jaki odsetek **zatrzymanych** par ma poprawny
      kierunek. To ona decyduje o jakości danych treningowych, bo w prawdziwym
      użyciu nie wiemy, która strona jest dobra: filtrujemy po **spójności** głosów
      sędziego, nie po zgodzie z prawdą.

    Głosy liczą się parami: rubryka wnosi dwa głosy tylko wtedy, gdy obie kolejności
    dają ten sam werdykt (inaczej to `position_flip` i zero głosów). Osiągalne progi
    są więc parzyste — 2, 4 i 6 — a nieparzyste dawałyby te same zbiory.
    """
    rows = list(load_journal(journal_path).values())
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
    votes: dict[str, tuple[int, int]] = {}
    complete_items = 0
    for item_id, verdicts in by_item.items():
        rubrics = sorted({rubric for rubric, _order in verdicts})
        if len(verdicts) != len(rubrics) * 2:
            continue
        complete_items += 1
        correct = 0
        wrong = 0
        for rubric in rubrics:
            forward = verdicts[(rubric, "ab")]
            reverse = verdicts[(rubric, "ba")]
            if forward != reverse:
                flips[rubric] += 1
                continue
            if forward == "A":
                correct += 2
            elif forward == "B":
                wrong += 2
        votes[item_id] = (correct, wrong)

    curve: dict[str, Any] = {}
    for threshold in (2, 4, 6):
        kept = [
            (correct, wrong)
            for correct, wrong in votes.values()
            if max(correct, wrong) >= threshold and correct != wrong
        ]
        good = sum(1 for correct, wrong in kept if correct > wrong)
        curve[f"min_votes_{threshold}"] = {
            "kept_items": len(kept),
            "yield": len(kept) / complete_items if complete_items else None,
            "precision": good / len(kept) if kept else None,
            "kept_with_wrong_direction": len(kept) - good,
        }
    unresolved = sum(1 for correct, wrong in votes.values() if correct == wrong)
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
        "unresolved_items": unresolved,
        "aggregation_curve": curve,
        "achievable_thresholds": [2, 4, 6],
        "threshold_frozen_here": False,
        "note": (
            "Próg agregacji zamraża osobny amendment po odczycie tych liczb i przed "
            "zbudowaniem pierwszej pary v3 (§6 ADR). `precision` liczy się na parach "
            "ZATRZYMANYCH przez filtr, `yield` to ich udział w całości."
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
    allow_no_auth: bool = False,
    timeout_seconds: float = 120.0,
) -> JudgeEndpoint:
    """Klucz z argumentu albo ze zmiennej środowiskowej; nigdy z repozytorium."""
    key = api_key if api_key else os.environ.get(api_key_env, "")
    if not key and not allow_no_auth:
        raise ValueError(
            f"brak klucza API: podaj --api-key, ustaw zmienną {api_key_env} "
            "albo jawnie użyj --allow-no-auth dla serwera bez uwierzytelniania"
        )
    if not base_url.startswith(("http://", "https://")):
        raise ValueError("base-url musi być pełnym adresem http(s)")
    return JudgeEndpoint(
        base_url=base_url,
        api_key=key,
        model=model,
        allow_reasoning=allow_reasoning,
        max_completion_tokens=max_completion_tokens,
        timeout_seconds=timeout_seconds,
    )


def plan_summary(items: Iterable[PairwiseItem], rubrics: Sequence[str]) -> dict[str, Any]:
    """Sam budżet wywołań; czasu NIE szacujemy z góry.

    Zmierzone 19,1 itemu/s dotyczyło sędziego odpowiadalności generującego 24 tokeny.
    Tu limit to 512 tokenów (1024 z rozumowaniem), a wąskim gardłem jest dekodowanie,
    więc przepustowość trzeba odczytać z runu (`observed_calls_per_second`), a nie
    przenosić ze starego pomiaru.
    """
    rows = list(items)
    return {
        "items": len(rows),
        "rubrics": list(rubrics),
        "orders": 2,
        "calls": len(rows) * len(rubrics) * 2,
        "throughput_note": "czas z observed_calls_per_second w trakcie runu",
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
