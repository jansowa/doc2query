#!/usr/bin/env python3
"""Cztery zadania nocne dla serwera inferencji, wszystkie w jednym journalu.

Każde zadanie ma jasno zapisane, po co jest i czego NIE wolno z niego wnioskować:

1. `lexical_mutation` — mutacje `not_answerable` **zachowujące keywordy pasażu**
   dla grup, w których kopalnia nie miała kandydata o pokryciu ≥0,6. Wprost
   przewidziane w ADR §6; buduje klasę `lexical_contrast`.
2. `label_purity` — niezależna weryfikacja etykiet 1 794 złożonych par innym
   promptem niż ten, który je nadał. To **pomiar czystości etykiet**, nie bramka:
   pary już przeszły zamrożone filtry, a wynik służy do raportu i ewentualnego
   amendmentu, nie do cichego przefiltrowania kohorty.
3. `answer_leak_v2` — mutacja `answer_leak` z twardym wymogiem zachowania FORMY.
   Klasa v1 oblała audyt anty-skrótowy (AUC 0,8731), bo mutacje przeskakiwały na
   pytania tak/nie. Nowe generacje przechodzą **ten sam** audyt, próg 0,80
   pozostaje bez zmian; jeśli znowu oblą, klasa zostaje odłożona.
4. `chosen_recheck` — druga opinia o answerability `chosen` dla 368 grup, które
   wypadły w całości. Grupa wraca do gry tylko przy **zgodnej** odpowiedzi TAK
   obu wywołań (pierwotnego i tego), nigdy przy samej zmianie zdania.
5. `class_backfill` — mutacja klasy wady tam, gdzie grupa jeszcze jej nie ma.
   Limit ADR §3 (≤1 para na grupę i klasę) zostaje; rośnie tylko pokrycie klas,
   bo dziś 1 035 grup wystawia jedną klasę, a mogłyby trzy.
6. `sft_data_audit` — audyt losowej próbki danych SFT na osiach, których dzisiejsze
   filtry nie sprawdzają (odpowiadalność z pasażu, jakość tłumaczenia, sensowność
   zapytania). To **pomiar wad w istniejącej puli**, nie filtr: żadna para nie
   jest tym zadaniem usuwana, a przefiltrowanie puli oznaczałoby nową kohortę
   SFT i nowy punkt startowy, czyli osobną decyzję właściciela.
7. `teacher_probe_queries` — zapytania teachera dla 1 984 pasaży kohorty probe
   (rozłącznej z obiema kohortami treningowymi). To **ramię odniesienia** dla
   probe embeddera: mierzy sufit, do którego w ogóle może dobić generator po
   DPO. Nie jest kandydatem na finalistę i nie wchodzi do żadnej kohorty par.

Wznawialne po journalu, równoległość na elementach, fail-fast po serii błędów.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from doc2query.preferences.pair_selector_v3 import (
    JudgeApiError,
    JudgeEndpoint,
    _last_json_object,
    http_transport,
    strip_reasoning,
)

PROMPT_VERSION = "task06-night-jobs-pl-v1"
FAIL_FAST_AFTER = 10

SYSTEM = (
    "Jesteś precyzyjnym asystentem do budowy danych treningowych dla polskiego "
    "systemu wyszukiwania. Piszesz WYŁĄCZNIE po polsku i zwracasz wyłącznie JSON."
)

LEXICAL_TEMPLATE = """Pasaż:
{passage}

Poprawne zapytanie do tego pasażu:
{chosen}

Zadanie: napisz zapytanie wyszukiwawcze, które jest **pułapką leksykalną**:
- ma zawierać jak najwięcej słów kluczowych, nazw własnych i liczb Z TEGO pasażu;
- ale pasaż NIE MOŻE zawierać odpowiedzi na nie — pytaj o atrybut tego samego
  obiektu, którego pasaż nie podaje (inna wielkość, data, cena, przyczyna, skutek).

Twarde wymagania:
1. maksymalnie dużo słownictwa pasażu, minimum słów spoza niego;
2. zapytanie musi brzmieć naturalnie, jak realne pytanie użytkownika;
3. jedno zapytanie, jedna linia, bez komentarza;
4. zachowaj formę: {form} (fraza kluczowa bez znaku zapytania albo pytanie pełne).

Zwróć wyłącznie JSON:
{{"query": "<zapytanie>", "uzyte_slowa_z_pasazu": ["<slowo>", "..."],
 "czego_pasaz_nie_podaje": "<jedno zdanie>"}}"""

PURITY_TEMPLATE = """Pasaż:
{passage}

Zapytanie A: {chosen}
Zapytanie B: {rejected}

Twierdzenie do sprawdzenia: zapytanie B jest gorsze od A dokładnie dlatego, że
ma wadę `{defect}`, gdzie:
- `not_answerable` = pasaż nie zawiera odpowiedzi na B, choć B jest w temacie;
- `too_general` = B jest tak ogólne, że pasuje do tysięcy innych pasaży;
- `answer_leak` = B zawiera w sobie odpowiedź, której szuka;
- `copy_phrasing` = B kopiuje dosłowny, długi fragment pasażu.

Oceń niezależnie, bez zakładania, że twierdzenie jest prawdziwe.

Zwróć wyłącznie JSON:
{{"wada_potwierdzona": <true|false>, "b_gorsze_od_a": <true|false>,
 "faktyczna_klasa": "<{defect}|inna_wada|brak_wady>", "uzasadnienie": "<jedno zdanie>"}}"""

LEAK_TEMPLATE = """Pasaż:
{passage}

Poprawne zapytanie wyszukiwawcze do tego pasażu:
{chosen}

Zadanie: przekształć je w gorsze przez MINIMALNĄ edycję tak, by **zawierało w
sobie odpowiedź**, której szuka (fakt wprost z pasażu).

Twardy wymóg formy — najważniejszy w tym zadaniu:
- jeśli oryginał jest frazą kluczową, wynik też musi być frazą kluczową: BEZ
  znaku zapytania i BEZ zaczynania od słowa pytajnego (czy, jak, co, ile...);
- jeśli oryginał jest pytaniem pełnym, wynik zostaje pytaniem pełnym;
- zachowaj przybliżoną długość oryginału (±3 słowa).
Forma oryginału: {form}

Zwróć wyłącznie JSON: {{"query": "<zapytanie>", "wbudowany_fakt": "<fakt z pasażu>"}}"""

BACKFILL_TEMPLATE = """Pasaż:
{passage}

Poprawne zapytanie wyszukiwawcze do tego pasażu:
{chosen}

Zadanie: przekształć je w gorsze przez MINIMALNĄ edycję — zmień, usuń lub dodaj
możliwie najmniej słów, zachowując styl, rejestr, formę i przybliżoną długość.
Wprowadź dokładnie jedną wadę `{defect}`: {cel}
Warunek zachowania: {wymog}
Forma oryginału: {form} — musi zostać zachowana.

Zwróć wyłącznie JSON: {{"query": "<zapytanie>", "edycja": "<co zmieniono>"}}"""

PROBE_TEMPLATE = """Pasaż:
{passage}

Kontrakt zapytania:
{controls}

Napisz jedno polskie zapytanie wyszukiwawcze, na które można odpowiedzieć
wyłącznie na podstawie tego pasażu. Nie kopiuj długich fragmentów pasażu,
zachowaj konieczne nazwy własne, liczby i terminy, nie zdradzaj odpowiedzi.

Zwróć wyłącznie JSON: {{"query": "<zapytanie>"}}"""

BACKFILL_DEFECTS = {
    "too_general": (
        "usuń albo uogólnij element, który czyni potrzebę informacyjną konkretną",
        "zostań w temacie pasażu",
    ),
    "not_answerable": (
        "przesuń pytanie na atrybut tego samego obiektu, którego pasaż nie podaje",
        "obiekt i terminologia zostają z pasażu, znika wyłącznie odpowiedź",
    ),
    "copy_phrasing": (
        "wstaw do zapytania dosłowny, ciągły fragment co najmniej pięciu słów z pasażu",
        "zapytanie ma dalej dać się odpowiedzieć na podstawie pasażu",
    ),
}

PROBE_CONTROLS = (
    "Forma: full_question\nIntencja: fact_lookup\nDocelowy fragment: beginning",
    "Forma: keyword_query\nIntencja: definition\nDocelowy fragment: middle",
    "Forma: full_question\nIntencja: procedure\nDocelowy fragment: end",
    "Forma: keyword_query\nIntencja: entity_lookup\nDocelowy fragment: middle",
)

SFT_AUDIT_TEMPLATE = """Pasaż:
{passage}

Zapytanie (z danych treningowych, tłumaczone maszynowo z angielskiego):
{query}

Oceń tę parę na czterech osiach, niezależnie od siebie:

1. `odpowiadalne` — czy na to zapytanie da się odpowiedzieć WYŁĄCZNIE na podstawie
   tego pasażu? false, gdy pasaż jest tylko w temacie albo odpowiada częściowo.
2. `polszczyzna` — czy zapytanie jest poprawnym, naturalnym polskim zdaniem lub
   frazą? false przy kalkach z angielskiego, resztkach angielskich słów,
   przekręconych nazwach własnych, bezsensownej składni.
3. `sensowne_zapytanie` — czy to wygląda na realną potrzebę informacyjną
   użytkownika wyszukiwarki? false dla fragmentów zdań, śmieci, zapytań
   nawigacyjnych bez treści.
4. `zbyt_ogolne` — czy zapytanie jest tak ogólne, że pasowałoby do tysięcy
   różnych pasaży?

Zwróć wyłącznie JSON:
{{"odpowiadalne": <true|false>, "polszczyzna": <true|false>,
 "sensowne_zapytanie": <true|false>, "zbyt_ogolne": <true|false>,
 "glowny_problem": "<brak|nieodpowiadalne|tlumaczenie|niesensowne|zbyt_ogolne>"}}"""

ANSWERABLE_TEMPLATE = """Pasaż:
{passage}

Zapytanie:
{query}

Czy ten pasaż ZAWIERA odpowiedź na to zapytanie? true tylko wtedy, gdy odpowiedź
da się wskazać w treści pasażu; false, gdy pasaż jest w temacie, ale odpowiedzi
nie podaje.

Zwróć wyłącznie JSON: {{"answerable": <true|false>, "dowod": "<fragment pasażu albo pusty>"}}"""


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
        record = {"key": key, "prompt_version": PROMPT_VERSION, **row}
        with self.lock:
            self.done[key] = record
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())


def make_ask(endpoint: JudgeEndpoint, reasoning_effort: str | None) -> Any:
    transport = http_transport(endpoint)

    def request(user: str, max_tokens: int, *, guided: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": endpoint.model,
            "temperature": 0.0,
            "max_completion_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
            ],
        }
        if guided:
            payload["response_format"] = {"type": "json_object"}
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        else:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        last: JudgeApiError | None = None
        for attempt in range(6):
            try:
                return dict(transport(payload))
            except JudgeApiError as error:
                if not error.retryable:
                    raise
                last = error
                time.sleep(min(60.0, 3.0 * 2**attempt))
        raise last if last is not None else RuntimeError("transport bez odpowiedzi")

    def ask(user: str) -> dict[str, Any]:
        errors: list[str] = []
        for attempt in range(3):
            response = request(
                user, endpoint.max_completion_tokens * (2**attempt), guided=attempt == 0
            )
            choice = (response.get("choices") or [{}])[0]
            finish = str(choice.get("finish_reason", "?"))
            try:
                content = str(choice["message"]["content"])
                parsed = json.loads(_last_json_object(strip_reasoning(content)))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                errors.append(f"{type(error).__name__} (finish={finish}): {error}")
                continue
            if isinstance(parsed, dict):
                return parsed
            errors.append(f"oczekiwano obiektu JSON, dostałem {type(parsed)}")
        raise ValueError("; ".join(errors))

    return ask


def _items(job: str, root: Path) -> list[dict[str, Any]]:
    def rows(path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            raise SystemExit(f"brak wejścia dla zadania {job}: {path}")
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").split("\n")
            if line.strip()
        ]

    if job == "lexical_mutation":
        return rows(root / "lexical_worklist.jsonl")
    if job == "label_purity":
        return rows(root / "pairs_to_verify.jsonl")
    if job == "answer_leak_v2":
        return rows(root / "answer_leak_groups.jsonl")
    if job == "chosen_recheck":
        return rows(root / "dropped_groups.jsonl")
    if job == "class_backfill":
        return rows(root / "class_backfill.jsonl")
    if job == "teacher_probe_queries":
        return rows(root / "probe_passages.jsonl")
    if job == "sft_data_audit":
        return rows(root / "sft_sample.jsonl")
    raise SystemExit(f"nieznane zadanie: {job}")


def _run_item(job: str, item: dict[str, Any], journal: Journal, ask: Any) -> None:
    key_base = f"{job}::{item['id']}"
    if journal.get(key_base) is not None:
        return
    passage = str(item["passage"])
    if job == "lexical_mutation":
        verdict = ask(
            LEXICAL_TEMPLATE.format(
                passage=passage, chosen=str(item["chosen"]), form=str(item["form"])
            )
        )
        query = str(verdict.get("query", "")).strip()
        check = ask(ANSWERABLE_TEMPLATE.format(passage=passage, query=query)) if query else {}
        journal.put(key_base, {"verdict": verdict, "answerable_check": check})
        return
    if job == "label_purity":
        verdict = ask(
            PURITY_TEMPLATE.format(
                passage=passage,
                chosen=str(item["chosen"]),
                rejected=str(item["rejected"]),
                defect=str(item["defect_class"]),
            )
        )
        journal.put(key_base, {"verdict": verdict, "defect_class": str(item["defect_class"])})
        return
    if job == "answer_leak_v2":
        verdict = ask(
            LEAK_TEMPLATE.format(
                passage=passage, chosen=str(item["chosen"]), form=str(item["form"])
            )
        )
        query = str(verdict.get("query", "")).strip()
        check = ask(ANSWERABLE_TEMPLATE.format(passage=passage, query=query)) if query else {}
        journal.put(key_base, {"verdict": verdict, "answerable_check": check})
        return
    if job == "class_backfill":
        defect = str(item["defect_class"])
        cel, wymog = BACKFILL_DEFECTS[defect]
        verdict = ask(
            BACKFILL_TEMPLATE.format(
                passage=passage,
                chosen=str(item["chosen"]),
                defect=defect,
                cel=cel,
                wymog=wymog,
                form=str(item["form"]),
            )
        )
        query = str(verdict.get("query", "")).strip()
        check = ask(ANSWERABLE_TEMPLATE.format(passage=passage, query=query)) if query else {}
        journal.put(
            key_base, {"verdict": verdict, "answerable_check": check, "defect_class": defect}
        )
        return
    if job == "sft_data_audit":
        verdict = ask(SFT_AUDIT_TEMPLATE.format(passage=passage, query=str(item["query"])))
        journal.put(key_base, {"verdict": verdict})
        return
    if job == "teacher_probe_queries":
        queries: list[dict[str, Any]] = []
        for index, controls in enumerate(PROBE_CONTROLS):
            verdict = ask(PROBE_TEMPLATE.format(passage=passage, controls=controls))
            queries.append(
                {
                    "control_index": index,
                    "controls": controls,
                    "query": str(verdict.get("query", "")),
                }
            )
        journal.put(key_base, {"queries": queries})
        return
    verdict = ask(ANSWERABLE_TEMPLATE.format(passage=passage, query=str(item["chosen"])))
    journal.put(key_base, {"verdict": verdict})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jobs",
        default="lexical_mutation,label_purity,answer_leak_v2,chosen_recheck",
        help="lista zadań po przecinku, w kolejności wykonania",
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-key-env", default="JUDGE_API_KEY")
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--reasoning-effort", default="")
    parser.add_argument("--max-completion-tokens", type=int, default=400)
    args = parser.parse_args()

    endpoint = JudgeEndpoint(
        base_url=args.base_url,
        api_key=args.api_key or os.environ.get(args.api_key_env, ""),
        model=args.model,
        temperature=0.0,
        max_completion_tokens=args.max_completion_tokens,
    )
    ask = make_ask(endpoint, args.reasoning_effort or None)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    journal = Journal(args.output_dir / "night_jobs.journal.jsonl")

    for job in [name.strip() for name in args.jobs.split(",") if name.strip()]:
        items = _items(job, args.input_dir)
        pending = [item for item in items if journal.get(f"{job}::{item['id']}") is None]
        print(
            f"[noc] {job}: {len(items)} elementów, do zrobienia {len(pending)}",
            flush=True,
        )
        if not pending:
            continue
        started = time.time()
        failures = 0
        done = 0
        lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(_run_item, job, item, journal, ask): item for item in pending}
            for future in as_completed(futures):
                item = futures[future]
                try:
                    future.result()
                    with lock:
                        done += 1
                        failures = 0
                        if done % 50 == 0:
                            rate = done / max(1.0, time.time() - started)
                            eta = (len(pending) - done) / max(rate, 1e-6) / 60
                            print(
                                f"[noc] {job}: {done}/{len(pending)}, "
                                f"{rate * 60:.1f}/min, ETA {eta:.0f} min",
                                flush=True,
                            )
                except Exception as error:  # raport i licznik fail-fast
                    with lock:
                        failures += 1
                        print(
                            f"[noc] {job}: element {item['id']} padł: "
                            f"{type(error).__name__}: {error}",
                            flush=True,
                        )
                        if failures >= FAIL_FAST_AFTER:
                            print(
                                f"[noc] {failures} kolejnych porażek — przerywam zadanie {job}; "
                                "journal zachowany",
                                flush=True,
                            )
                            for other in futures:
                                other.cancel()
                            break
        print(f"[noc] {job} zakończone w {(time.time() - started) / 60:.1f} min", flush=True)

    print(f"[noc] koniec; werdykty: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
