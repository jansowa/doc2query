#!/usr/bin/env python3
"""Runner serwerowy pipeline'u wad: klasyfikacja, mutacje, answerability, potwierdzenia.

Wykonuje część LLM zamrożonego ADR
`reports/decisions/task06_defect_pair_pipeline_v1.md` na serwerze inferencji
(OpenAI-compatible; Qwen3.8-27B u właściciela, Groq w eksploracji). Składanie par
i pomiary lematyczne dzieją się LOKALNIE po przeniesieniu verdictów — ten skrypt
tylko pyta i zapisuje.

Etapy per grupa (`groups.jsonl` z `export_defect_pipeline_input.py`):

  A. answerability `chosen` — NIE ⇒ grupa wypada w całości (bez dalszych wywołań);
  B. klasyfikacja kandydatów studenckich spoza (chosen, current_rejected);
  C. mutacje `chosen` dla klas bez organicznej podaży w grupie (jedna runda);
  D. answerability każdego kandydata na negatyw (wymóg zależny od klasy);
  E. potwierdzenie preferencji rubryką R3_holistic w OBU kolejnościach pozycji.

Filtry deterministyczne (LCS, Jaccard, długości, forma) liczą się lokalnie
przy składaniu — ale oczywiste odpady (równoważność z chosen, długości) są
odcinane już tutaj, żeby nie płacić za answerability i potwierdzenia śmieci.

Wznawialny: journal per wywołanie, klucz (group_id, stage, item, wariant).
Równoległość na poziomie grup; fail-fast po serii nieudanych grup.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from doc2query.preferences.pair_selector_v3 import (
    RUBRICS,
    JudgeApiError,
    JudgeEndpoint,
    _last_json_object,
    http_transport,
    strip_reasoning,
)

PROMPT_VERSION = "task06-defect-pipeline-pl-v1"
ADR = "reports/decisions/task06_defect_pair_pipeline_v1.md"
FAIL_FAST_AFTER = 8
LLM_CLASSES = ("ok", "too_general", "not_answerable", "answer_leak", "copy_phrasing", "off_topic")
MUTATION_CLASSES = ("too_general", "not_answerable", "answer_leak")
# §4 ADR: oczywiste odpady odcinane przed płatnymi weryfikacjami.
EQUIVALENCE_JACCARD = 0.6
LENGTH_RANGE = (2, 24)
LENGTH_RATIO = (0.4, 2.5)
QUESTION_WORDS = (
    "co",
    "czy",
    "czym",
    "gdzie",
    "ile",
    "jak",
    "jaka",
    "jaki",
    "jakie",
    "kiedy",
    "kim",
    "kto",
    "dlaczego",
    "który",
    "która",
    "które",
)

SYSTEM = (
    "Jesteś precyzyjnym asystentem do budowy danych treningowych dla polskiego "
    "systemu wyszukiwania. Piszesz WYŁĄCZNIE po polsku i zwracasz wyłącznie JSON."
)

CLASSIFY_TEMPLATE = """Pasaż:
{passage}

Zapytanie wyszukiwawcze do oceny:
{query}

Sklasyfikuj zapytanie względem pasażu do DOKŁADNIE JEDNEJ klasy:
- "ok" — ugruntowane w pasażu, pasaż zawiera odpowiedź, nie kopiuje długiego
  fragmentu, nie zdradza odpowiedzi, jest wystarczająco konkretne;
- "too_general" — w temacie pasażu, ale tak ogólne, że pasuje do tysięcy pasaży;
- "not_answerable" — w temacie i z terminami z pasażu, ale pasaż NIE zawiera
  odpowiedzi na to pytanie;
- "answer_leak" — zawiera w sobie fakt będący odpowiedzią (stwierdzenie w formie
  pytania, np. pytanie tak/nie podające wartość z pasażu);
- "copy_phrasing" — kopiuje ciągły fragment ≥5 słów z pasażu zamiast formułować
  potrzebę informacyjną własnymi słowami;
- "off_topic" — nie dotyczy treści tego pasażu albo jest niegramatycznym szumem.

Przykłady:
Pasaż o populacji Sandpoint w Idaho (7397 mieszkańców):
- "ile mieszkańców ma sandpoint idaho" → "ok"
- "jakie są dane demograficzne miast" → "too_general"
- "jaka jest mediana dochodu w sandpoint idaho" → "not_answerable"
- "czy sandpoint w idaho ma 7397 mieszkańców" → "answer_leak"
- "sandpoint z 2014 r. w idaho liczy 7397 mieszkańców" → "copy_phrasing"
- "definicja krawędzi ślusarskiej" → "off_topic"

Zwróć wyłącznie JSON: {{"class": "<klasa>", "uzasadnienie": "<jedno zdanie>"}}"""

ANSWERABLE_TEMPLATE = """Pasaż:
{passage}

Zapytanie:
{query}

Czy ten pasaż ZAWIERA odpowiedź na to zapytanie? Odpowiedz ściśle: true tylko
wtedy, gdy odpowiedź da się wskazać w treści pasażu; false, gdy pasaż jest tylko
w temacie, ale odpowiedzi nie podaje.

Zwróć wyłącznie JSON: {{"answerable": <true|false>, "dowod": "<fragment pasażu albo pusty>"}}"""

MUTATIONS: dict[str, dict[str, str]] = {
    "too_general": {
        "cel": "usuń albo uogólnij element, który czyni potrzebę informacyjną konkretną",
        "wymog": "zostań w temacie pasażu",
    },
    "not_answerable": {
        "cel": (
            "przesuń pytanie na atrybut tego samego obiektu, którego pasaż nie podaje "
            "(np. inna wielkość, data, cena, przyczyna)"
        ),
        "wymog": "obiekt i terminologia zostają z pasażu, znika wyłącznie odpowiedź",
    },
    "answer_leak": {
        "cel": "wbuduj w zapytanie fakt z pasażu będący odpowiedzią",
        "wymog": (
            "ten sam obiekt i ten sam fakt, o który pytał oryginał; zachowaj FORMĘ "
            "oryginału — fraza kluczowa pozostaje frazą kluczową (bez znaku zapytania), "
            "pytanie pełne pytaniem"
        ),
    },
}

MUTATE_TEMPLATE = """Pasaż:
{passage}

Kontrakt zapytania:
{controls}

Poprawne zapytanie wyszukiwawcze do tego pasażu:
{chosen}

Zadanie: przekształć poprawne zapytanie w gorsze przez MINIMALNĄ edycję —
zmień, usuń lub dodaj możliwie najmniej słów, zachowując styl, rejestr,
formę i przybliżoną długość oryginału. Wprowadź dokładnie jedną wadę
`{defect}`: {cel}.
Warunek zachowania: {wymog}.

Twarde wymagania:
1. edycja minimalna — jeśli wadę da się wprowadzić zmianą 1-3 słów, nie zmieniaj więcej;
2. wynik nie może być identyczny z oryginałem ani lepszy od niego;
3. dokładnie jedno zapytanie, jedna linia, bez komentarza.

Zwróć wyłącznie JSON: {{"query": "<zapytanie>", "edycja": "<co zmieniono>"}}"""


def words(text: str) -> list[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFKC", str(text)).lower())


def jaccard(a: str, b: str) -> float:
    first, second = set(words(a)), set(words(b))
    if not first or not second:
        return 0.0
    return len(first & second) / len(first | second)


def form_violation(query: str, form: str) -> bool:
    tokens = words(query)
    if not tokens:
        return True
    looks_question = tokens[0] in QUESTION_WORDS or query.strip().endswith("?")
    if form == "keyword_query":
        return looks_question
    if form == "full_question":
        return not looks_question
    return False


def cheap_reject(query: str, chosen: str, form: str) -> str | None:
    """Filtry z §4 ADR odcinane przed płatną weryfikacją; None = przechodzi."""
    count = len(words(query))
    if not (LENGTH_RANGE[0] <= count <= LENGTH_RANGE[1]):
        return "length"
    chosen_count = max(1, len(words(chosen)))
    if not (LENGTH_RATIO[0] <= count / chosen_count <= LENGTH_RATIO[1]):
        return "length_ratio"
    if jaccard(query, chosen) > EQUIVALENCE_JACCARD:
        return "equivalent_to_chosen"
    if form_violation(query, form):
        return "form_violation"
    return None


class Journal:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()
        self.done: dict[str, dict[str, Any]] = {}
        if path.is_file():
            for line in path.read_text(encoding="utf-8").split("\n"):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    break
                self.done[str(row["key"])] = row

    def get(self, key: str) -> dict[str, Any] | None:
        return self.done.get(key)

    def put(self, key: str, row: dict[str, Any]) -> None:
        record = {"key": key, **row}
        with self.lock:
            self.done[key] = record
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())


def make_ask(endpoint: JudgeEndpoint, reasoning_effort: str | None) -> Any:
    transport = http_transport(endpoint)

    def ask(system: str, user: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": endpoint.model,
            "temperature": 0.0,
            "max_completion_tokens": endpoint.max_completion_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
        }
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        else:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        last: JudgeApiError | None = None
        for attempt in range(6):
            try:
                response = payload_call(transport, payload)
                break
            except JudgeApiError as error:
                if not error.retryable:
                    raise
                last = error
                time.sleep(min(60.0, 3.0 * 2**attempt))
        else:
            raise last if last is not None else RuntimeError("transport bez odpowiedzi")
        content = str(response["choices"][0]["message"]["content"])
        parsed = json.loads(_last_json_object(strip_reasoning(content)))
        if not isinstance(parsed, dict):
            raise ValueError(f"oczekiwano obiektu JSON, dostałem {type(parsed)}")
        return parsed

    return ask


def payload_call(transport: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Wyślij payload; na 400 o nieznanym polu zrzuć pola opcjonalne raz."""
    try:
        return dict(transport(payload))
    except JudgeApiError as error:
        body = (error.body or "").lower()
        dropped = False
        for field in ("reasoning_effort", "chat_template_kwargs"):
            if field in payload and field in body:
                payload = {k: v for k, v in payload.items() if k != field}
                dropped = True
        if dropped:
            return dict(transport(payload))
        raise


def confirm_preference(ask: Any, passage: str, chosen: str, candidate: str) -> dict[str, Any]:
    """R3_holistic w obu kolejnościach; zwraca głosy i jednomyślność na chosen."""
    votes: list[str] = []
    for order in ("ab", "ba"):
        first, second = (chosen, candidate) if order == "ab" else (candidate, chosen)
        user = (
            f"Pasaż:\n{passage}\n\nZapytanie A:\n{first}\n\nZapytanie B:\n{second}\n\n"
            "Które zapytanie jest lepsze?"
        )
        verdict = ask(RUBRICS["R3_holistic"], user)
        better = str(verdict.get("better", "")).strip().upper()
        chosen_letter = "A" if order == "ab" else "B"
        votes.append("chosen" if better == chosen_letter else better.lower() or "invalid")
    return {"votes": votes, "unanimous_chosen": votes == ["chosen", "chosen"]}


def process_group(group: dict[str, Any], journal: Journal, ask: Any) -> dict[str, Any]:
    gid = str(group["group_id"])
    passage = str(group["passage"])
    chosen = str(group["chosen"]["query"])
    form = str(group["form"])
    stats = {"group_id": gid, "calls": 0, "kept": 0, "dropped_cheap": 0}

    def cached(key: str, factory: Any) -> dict[str, Any]:
        row = journal.get(key)
        if row is not None:
            return row
        result = factory()
        stats["calls"] += 1
        journal.put(key, result)
        return result

    # A. answerability chosen — fail-closed dla całej grupy.
    chosen_row = cached(
        f"{gid}::answerable::chosen",
        lambda: {"verdict": ask(SYSTEM, ANSWERABLE_TEMPLATE.format(passage=passage, query=chosen))},
    )
    if not bool(chosen_row["verdict"].get("answerable")):
        journal.put(f"{gid}::group_status", {"status": "chosen_not_answerable"})
        return stats

    # B. klasyfikacja pozostałych kandydatów studenckich.
    organic_by_class: dict[str, list[dict[str, Any]]] = {}
    for candidate in group["others"]:
        query = str(candidate["query"])
        reason = cheap_reject(query, chosen, form)
        if reason == "length" or reason == "length_ratio":
            stats["dropped_cheap"] += 1
            continue
        row = cached(
            f"{gid}::classify::{candidate['candidate_id']}",
            lambda query=query: {
                "verdict": ask(SYSTEM, CLASSIFY_TEMPLATE.format(passage=passage, query=query))
            },
        )
        label = str(row["verdict"].get("class", ""))
        if label in LLM_CLASSES:
            organic_by_class.setdefault(label, []).append({**candidate, "cheap_reject": reason})

    # C. mutacje dla klas bez podaży organicznej (kandydat bez cheap_reject).
    candidates: list[dict[str, Any]] = []
    for label in MUTATION_CLASSES:
        organic = [c for c in organic_by_class.get(label, []) if c["cheap_reject"] is None]
        if organic:
            candidates.append({**organic[0], "defect_class": label, "population": "mined_organic"})
            continue
        spec = MUTATIONS[label]
        row = cached(
            f"{gid}::mutate::{label}",
            lambda label=label, spec=spec: {
                "verdict": ask(
                    SYSTEM,
                    MUTATE_TEMPLATE.format(
                        passage=passage,
                        controls=str(group["controls"]),
                        chosen=chosen,
                        defect=label,
                        cel=spec["cel"],
                        wymog=spec["wymog"],
                    ),
                )
            },
        )
        query = str(row["verdict"].get("query", "")).strip()
        if not query or cheap_reject(query, chosen, form) is not None:
            stats["dropped_cheap"] += 1
            continue
        candidates.append(
            {
                "candidate_id": f"{gid}::mutated::{label}",
                "query": query,
                "defect_class": label,
                "population": "mutated_synthetic",
                "cheap_reject": None,
            }
        )
    # kandydaci lexical_contrast: wszyscy organiczni not_answerable i ok idą do D,
    # wybór nastąpi lokalnie na lematach.
    for label in ("not_answerable", "ok"):
        for extra in organic_by_class.get(label, []):
            if extra["cheap_reject"] is None and all(
                c["candidate_id"] != extra["candidate_id"] for c in candidates
            ):
                candidates.append({**extra, "defect_class": label, "population": "mined_organic"})

    # D + E. answerability i potwierdzenie preferencji.
    for candidate in candidates:
        query = str(candidate["query"])
        cid = str(candidate["candidate_id"])
        cached(
            f"{gid}::answerable::{cid}",
            lambda query=query: {
                "verdict": ask(SYSTEM, ANSWERABLE_TEMPLATE.format(passage=passage, query=query))
            },
        )
        if candidate["defect_class"] == "ok":
            # Kandydat na stronę chosen lexical_contrast; preferencję policzy
            # lokalne składanie, tu wystarcza answerability.
            continue
        cached(
            f"{gid}::confirm::{cid}",
            lambda query=query: confirm_preference(ask, passage, chosen, query),
        )
        stats["kept"] += 1
    journal.put(f"{gid}::group_status", {"status": "done", **stats})
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-key-env", default="JUDGE_API_KEY")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="0 = wszystkie grupy")
    parser.add_argument(
        "--reasoning-effort", default="", help="np. none dla Groq; puste = chat_template_kwargs"
    )
    parser.add_argument("--max-completion-tokens", type=int, default=400)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    key = args.api_key or os.environ.get(args.api_key_env, "")
    endpoint = JudgeEndpoint(
        base_url=args.base_url,
        api_key=key,
        model=args.model,
        temperature=0.0,
        max_completion_tokens=args.max_completion_tokens,
    )
    ask = make_ask(endpoint, args.reasoning_effort or None)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    journal = Journal(args.output_dir / "verdicts.journal.jsonl")

    groups: list[dict[str, Any]] = []
    for line in args.groups.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            groups.append(json.loads(line))
    if args.limit:
        groups = groups[: args.limit]
    pending = [g for g in groups if journal.get(f"{g['group_id']}::group_status") is None]
    print(
        f"[pipeline] grup: {len(groups)}, do zrobienia: {len(pending)}, "
        f"journal: {len(journal.done)} wpisów",
        flush=True,
    )

    started = time.time()
    failures = 0
    done_count = 0
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(process_group, g, journal, ask): g for g in pending}
        try:
            for future in as_completed(futures):
                group = futures[future]
                try:
                    future.result()
                    with lock:
                        done_count += 1
                        failures = 0
                        if done_count % 20 == 0:
                            rate = done_count / max(1.0, time.time() - started)
                            eta = (len(pending) - done_count) / max(rate, 1e-6)
                            print(
                                f"[pipeline] {done_count}/{len(pending)} grup, "
                                f"{rate * 60:.1f} grup/min, ETA {eta / 60:.0f} min",
                                flush=True,
                            )
                except Exception as error:  # raport i licznik fail-fast
                    with lock:
                        failures += 1
                        print(
                            f"[pipeline] grupa {group['group_id']} padła: "
                            f"{type(error).__name__}: {error}",
                            flush=True,
                        )
                        if failures >= FAIL_FAST_AFTER:
                            print(
                                f"[pipeline] {failures} kolejnych porażek — przerywam; "
                                "journal zachowany, restart wznowi",
                                flush=True,
                            )
                            for other in futures:
                                other.cancel()
                            raise SystemExit(3) from None
        except KeyboardInterrupt:
            for other in futures:
                other.cancel()
            print("[pipeline] przerwane; journal zachowany, restart wznowi", flush=True)
            raise SystemExit(130) from None
    print(
        f"[pipeline] koniec: {done_count}/{len(pending)} grup w "
        f"{(time.time() - started) / 60:.1f} min; verdicts: {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
