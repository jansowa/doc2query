#!/usr/bin/env python3
"""Eksploracja promptów: teacher pisze stronę `rejected` z **nazwaną** wadą.

Kontekst pomiarowy, bez którego ten skrypt nie ma sensu:

* pary v3 mają kontrast trywialny — w 75,3% strona odrzucona jest nie o tym pasażu
  (`reports/measurements/task07_pair_contrast_diagnostic_2026-08-28.md`);
* teacher jako autor **lepszych** zapytań został już zmierzony i **nie** bije
  lokalnego D01 na zamrożonym sygnale budującym (34,7% / 41,6%,
  `reports/measurements/task06_teacher_vs_student_v3_2026-08-16.md`).

Dlatego domyślna rola teachera jest tu odwrócona: `chosen` zostaje **studenckie**
(model, który trenujemy, ma zostać w swoim rozkładzie), a teacher pisze wyłącznie
`rejected` — zapytanie **na temat pasażu, ale z jedną konkretną wadą**. Wada jest
znana z konstrukcji, więc daje etykietę do weryfikacji, tak jak w kalibracji
selektora v3.

To jest **eksploracja promptów**, nie budowa danych treningowych: wynik idzie do
ręcznego przeglądu, żadna para nie wchodzi do żadnej kohorty i żadna polityka nie
jest tym plikiem zamrażana. Zamrożenie wymaga osobnego prospektywnego ADR.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Any

from doc2query.evaluation.groq_audits import load_api_key
from doc2query.preferences.pair_selector_v3 import (
    JudgeApiError,
    JudgeEndpoint,
    _last_json_object,
    http_transport,
    strip_reasoning,
)
from doc2query.utils.records import read_records, write_json

PROMPT_VERSION = "task06-teacher-defect-pl-v1"

# Każda klasa: (nazwa, co ma być zepsute, co MUSI zostać nietknięte).
DEFECTS: dict[str, dict[str, str]] = {
    "copy_phrasing": {
        "cel": (
            "zapytanie ma kopiować z pasażu dosłowny, ciągły fragment co najmniej "
            "pięciu słów, zamiast formułować potrzebę informacyjną własnymi słowami"
        ),
        "wymog": ("musi dalej dotyczyć tego pasażu i dać się na jego podstawie odpowiedzieć"),
    },
    "not_answerable": {
        "cel": (
            "zapytanie ma wyglądać wiarygodnie i używać nazw własnych oraz terminów z "
            "pasażu, ale pasaż NIE MOŻE zawierać odpowiedzi — pyta o rzecz pokrewną, "
            "której w pasażu nie ma"
        ),
        "wymog": (
            "nie wolno zmieniać tematu na inny obiekt: nazwy własne i terminy zostają "
            "te z pasażu, brakuje wyłącznie odpowiedzi"
        ),
    },
    "too_general": {
        "cel": (
            "zapytanie ma być tak ogólne, że pasowałoby do tysięcy różnych pasaży: "
            "usuń z niego to, co czyni potrzebę informacyjną konkretną"
        ),
        "wymog": ("musi zostać w temacie pasażu i nie wolno mu być pytaniem o inny obiekt"),
    },
    "answer_leak": {
        "cel": (
            "zapytanie ma zawierać odpowiedź, której szuka: podaje wprost fakt z "
            "pasażu, więc nie jest już potrzebą informacyjną, a stwierdzeniem w formie "
            "pytania"
        ),
        "wymog": "musi dotyczyć tego pasażu i tego samego faktu",
    },
}

SYSTEM = (
    "Jesteś precyzyjnym asystentem do budowy danych treningowych dla polskiego "
    "systemu wyszukiwania. Piszesz WYŁĄCZNIE po polsku i zwracasz wyłącznie JSON."
)

TEMPLATE = """Pasaż:
{passage}

Kontrakt zapytania, którego dotyczy zadanie:
{controls}

Poprawne zapytanie do tego pasażu (napisane przez inny model, NIE powtarzaj go):
{chosen}

Zadanie: napisz JEDNO polskie zapytanie wyszukiwawcze, które jest **gorsze** od
powyższego dokładnie w jeden, zadany sposób.

Wada, którą masz wprowadzić — `{defect}`: {cel}.
Warunek zachowania: {wymog}.

Twarde wymagania:
1. dokładnie jedno zapytanie, jedna linia, bez cudzysłowów, numeracji i komentarza;
2. zapytanie nie może być identyczne ani prawie identyczne z poprawnym zapytaniem;
3. zapytanie ma wyglądać na napisane przez człowieka szukającego informacji, a nie
   na celowo popsute — wada ma być realna, nie karykaturalna;
4. nie zdradzaj w zapytaniu, że jest gorsze.

Zwróć wyłącznie JSON o polach:
{{"query": "<zapytanie>", "defect": "{defect}", "dlaczego_gorsze": "<jedno zdanie>",
 "pasaz_zawiera_odpowiedz": <true|false>}}"""


def _passage(prompt: str) -> str:
    return prompt.split("Pasaż:\n", 1)[1].split("\n\nZapytanie:", 1)[0].strip()


def _controls(prompt: str) -> str:
    lines = [
        line
        for line in prompt.split("\n")
        if re.match(r"^(Forma|Intencja|Docelowy fragment|Długość):", line)
    ]
    return "\n".join(lines) if lines else "(brak jawnych kontrolek)"


def _ask(
    transport: Any, endpoint: JudgeEndpoint, user: str, reasoning_effort: str | None
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": endpoint.model,
        "temperature": endpoint.temperature,
        "max_completion_tokens": endpoint.max_completion_tokens,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
    }
    # Na Groq modele Qwen najpierw "myślą" i bez tego pola nie dochodzą do JSON-a
    # w limicie tokenów; lokalny vLLM steruje tym przez chat_template_kwargs.
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    last: JudgeApiError | None = None
    for attempt in range(6):
        try:
            response = transport(payload)
            break
        except JudgeApiError as error:
            if not error.retryable:
                raise
            last = error
            # Darmowy tier Groq ma niskie RPM; 429 to normalny rytm, nie awaria.
            time.sleep(min(60.0, 5.0 * 2**attempt))
    else:
        raise last if last is not None else RuntimeError("transport bez odpowiedzi")
    try:
        content = str(response["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as error:
        raise JudgeApiError(
            f"odpowiedź bez treści: {error}", status=None, body=json.dumps(response)[:400]
        ) from error
    parsed = json.loads(_last_json_object(strip_reasoning(content)))
    if not isinstance(parsed, dict):
        raise ValueError(f"oczekiwano obiektu JSON, dostałem {type(parsed)}")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs",
        type=Path,
        default=Path("artifacts/task07/handoff_v3_bottom/packaged/preference_train.jsonl"),
    )
    parser.add_argument("--base-url", default="https://api.groq.com/openai/v1")
    parser.add_argument("--model", default="qwen/qwen3.6-27b")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--defect", action="append", choices=sorted(DEFECTS), default=None)
    parser.add_argument("--passages", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--max-completion-tokens", type=int, default=400)
    parser.add_argument("--reasoning-effort", default="none", help="puste = nie wysyłaj pola")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/task06/teacher_defect_explore_v1/samples.jsonl"),
    )
    args = parser.parse_args()
    defects = args.defect or ["copy_phrasing", "not_answerable", "too_general"]

    rows = [dict(row) for row in read_records(args.pairs)]
    rows.sort(key=lambda row: str(row["preference_id"]))
    sample = random.Random(args.seed).sample(rows, args.passages)

    done: set[tuple[str, str]] = set()
    if args.output.is_file():
        for line in args.output.read_text(encoding="utf-8").split("\n"):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "teacher" in row:
                done.add((str(row["preference_id"]), str(row["defect"])))

    endpoint = JudgeEndpoint(
        base_url=args.base_url,
        api_key=load_api_key(args.env_file),
        model=args.model,
        temperature=0.0,
        max_completion_tokens=args.max_completion_tokens,
    )
    transport = http_transport(endpoint)

    results: list[dict[str, Any]] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as handle:
        for row in sample:
            prompt = str(row["prompt"])
            passage = _passage(prompt)
            for defect in defects:
                if (str(row["preference_id"]), defect) in done:
                    continue
                user = TEMPLATE.format(
                    passage=passage,
                    controls=_controls(prompt),
                    chosen=str(row["chosen"]),
                    defect=defect,
                    cel=DEFECTS[defect]["cel"],
                    wymog=DEFECTS[defect]["wymog"],
                )
                record: dict[str, Any] = {
                    "prompt_version": PROMPT_VERSION,
                    "model": args.model,
                    "preference_id": str(row["preference_id"]),
                    "defect": defect,
                    "passage": passage,
                    "controls": _controls(prompt),
                    "student_chosen": str(row["chosen"]),
                    "student_rejected": str(row["rejected"]),
                }
                try:
                    record["teacher"] = _ask(
                        transport, endpoint, user, args.reasoning_effort or None
                    )
                except (JudgeApiError, ValueError, json.JSONDecodeError) as error:
                    record["error"] = f"{type(error).__name__}: {error}"
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                results.append(record)
                teacher = record.get("teacher", {})
                print(
                    f"\n[{defect}] {record['preference_id'][:8]}\n"
                    f"  pasaż: {passage[:160]}\n"
                    f"  chosen (student):   {record['student_chosen']}\n"
                    f"  rejected (student): {record['student_rejected']}\n"
                    f"  rejected (teacher): {teacher.get('query', record.get('error'))}\n"
                    f"  uzasadnienie: {teacher.get('dlaczego_gorsze', '-')}"
                    f"  | pasaż odpowiada: {teacher.get('pasaz_zawiera_odpowiedz', '-')}",
                    flush=True,
                )

    write_json(
        args.output.with_name("summary.json"),
        {
            "schema_version": 1,
            "role": "prompt_exploration_not_training_data",
            "prompt_version": PROMPT_VERSION,
            "model": args.model,
            "passages": args.passages,
            "defects": defects,
            "calls": len(results),
            "errors": sum(1 for row in results if "error" in row),
            "pairs_built": 0,
            "final_tests_used": [],
        },
    )
    print(f"\nzapisane: {args.output} ({len(results)} próbek)")


if __name__ == "__main__":
    main()
