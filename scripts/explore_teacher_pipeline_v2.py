#!/usr/bin/env python3
"""Eksploracja v2: dwa mocniejsze tryby pozyskiwania negatywów na temat.

Adresuje wprost dwa najgroźniejsze ograniczenia projektu v1
(`reports/plans/task07_defect_pair_pipeline_design_2026-08-29.md` §6.5):

* **`classify`** — teacher jako klasyfikator wad istniejących kandydatów
  studenckich. Zmierzona podaż: 82,8% grup ma ≥1 ugruntowanego kandydata poza
  `chosen` i obecnym `rejected`. Negatyw pozostaje wtedy **organiczny** (rozkład
  studenta, zero luki stylu), a teacher tylko nadaje etykietę klasy — słabszą niż
  etykieta z konstrukcji, więc wymaga weryfikacji answerability judge'em tam,
  gdzie klasa tego dotyczy.
* **`mutate`** — synteza przez **minimalną edycję** `chosen` zamiast wolnej
  generacji: wstrzyknięcie wady ma zmienić możliwie mało tokenów, więc para
  różni się wadą, a nie stylem autora. To bezpośrednia obrona przed skrótem
  „unikaj stylu teachera".

Eksploracja, nie budowa danych: wyniki idą do ręcznego przeglądu.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
import unicodedata
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

PROMPT_VERSION = "task06-teacher-pipeline-pl-v2"
CLASSES = ("ok", "too_general", "not_answerable", "answer_leak", "copy_phrasing", "off_topic")

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

MUTATE_TEMPLATE = """Pasaż:
{passage}

Poprawne zapytanie wyszukiwawcze do tego pasażu:
{chosen}

Zadanie: przekształć poprawne zapytanie w gorsze przez MINIMALNĄ edycję —
zmień, usuń lub dodaj możliwie najmniej słów, zachowując styl, rejestr i
przybliżoną długość oryginału. Wprowadź dokładnie jedną wadę `{defect}`: {cel}.
Warunek zachowania: {wymog}.

Twarde wymagania:
1. wynik ma wyglądać jak zapytanie tego samego autora — nie zmieniaj formy
   (pytanie pełne zostaje pytaniem pełnym, fraza kluczowa frazą kluczową);
2. edycja minimalna: jeśli wadę da się wprowadzić zmianą 1-3 słów, nie zmieniaj
   więcej;
3. wynik nie może być identyczny z oryginałem ani lepszy od niego;
4. dokładnie jedno zapytanie, jedna linia, bez komentarza.

Zwróć wyłącznie JSON:
{{"query": "<zapytanie>", "defect": "{defect}", "edycja": "<co zmieniono>"}}"""

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
        "cel": "wbuduj w zapytanie fakt z pasażu będący odpowiedzią, np. jako pytanie tak/nie",
        "wymog": "ten sam obiekt i ten sam fakt, o który pytał oryginał",
    },
}


def content_words(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", str(text)).lower()
    return [word for word in re.findall(r"\w+", normalized) if len(word) >= 4]


def coverage(query: str, passage_words: set[str]) -> float:
    words = content_words(query)
    if not words:
        return 0.0
    return sum(word in passage_words for word in words) / len(words)


def ask(transport: Any, endpoint: JudgeEndpoint, user: str, reasoning_effort: str | None) -> Any:
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
            time.sleep(min(60.0, 5.0 * 2**attempt))
    else:
        raise last if last is not None else RuntimeError("transport bez odpowiedzi")
    content = str(response["choices"][0]["message"]["content"])
    return json.loads(_last_json_object(strip_reasoning(content)))


def _groups(pairs_path: Path, bundle_path: Path, count: int, seed: int) -> list[dict[str, Any]]:
    pairs = [dict(row) for row in read_records(pairs_path)]
    pairs.sort(key=lambda row: str(row["preference_id"]))
    sample = random.Random(seed).sample(pairs, count)
    wanted = {str(row["chosen_candidate_id"]): row for row in sample}
    groups: list[dict[str, Any]] = []
    for group in read_records(bundle_path):
        hits = [c for c in group["candidates"] if str(c["candidate_id"]) in wanted]
        if hits:
            pair = wanted[str(hits[0]["candidate_id"])]
            groups.append({"group": dict(group), "pair": pair})
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("classify", "mutate"))
    parser.add_argument(
        "--pairs",
        type=Path,
        default=Path("artifacts/task07/handoff_v3_bottom/packaged/preference_train.jsonl"),
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path("artifacts/task06/v3_tournament_bundle_v1/tournament_bundle.jsonl"),
    )
    parser.add_argument("--base-url", default="https://api.groq.com/openai/v1")
    parser.add_argument("--model", default="qwen/qwen3.6-27b")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--groups", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--reasoning-effort", default="none")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/task06/teacher_pipeline_explore_v2")
    )
    args = parser.parse_args()

    endpoint = JudgeEndpoint(
        base_url=args.base_url,
        api_key=load_api_key(args.env_file),
        model=args.model,
        temperature=0.0,
        max_completion_tokens=400,
    )
    transport = http_transport(endpoint)
    effort = args.reasoning_effort or None
    groups = _groups(args.pairs, args.bundle, args.groups, args.seed)
    print(f"grup dopasowanych do bundla: {len(groups)}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.mode}_samples.jsonl"
    results: list[dict[str, Any]] = []
    with output.open("a", encoding="utf-8") as handle:
        for entry in groups:
            group, pair = entry["group"], entry["pair"]
            passage_words = set(content_words(group["passage"]))
            chosen_id = str(pair["chosen_candidate_id"])
            if args.mode == "classify":
                for candidate in group["candidates"]:
                    query = str(candidate["query"])
                    record: dict[str, Any] = {
                        "prompt_version": PROMPT_VERSION,
                        "mode": "classify",
                        "group_id": str(group["group_id"]),
                        "candidate_id": str(candidate["candidate_id"]),
                        "is_chosen": str(candidate["candidate_id"]) == chosen_id,
                        "query": query,
                        "coverage": round(coverage(query, passage_words), 3),
                    }
                    try:
                        verdict = ask(
                            transport,
                            endpoint,
                            CLASSIFY_TEMPLATE.format(passage=group["passage"], query=query),
                            effort,
                        )
                        record["teacher"] = verdict
                    except (JudgeApiError, ValueError, json.JSONDecodeError) as error:
                        record["error"] = f"{type(error).__name__}: {error}"
                    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    handle.flush()
                    results.append(record)
                    verdict = record.get("teacher", {})
                    marker = "★chosen" if record["is_chosen"] else "       "
                    print(
                        f"[{verdict.get('class', 'ERR'):>14}] {marker} "
                        f"cov={record['coverage']:.2f}  {query}",
                        flush=True,
                    )
            else:
                for defect, spec in MUTATIONS.items():
                    record = {
                        "prompt_version": PROMPT_VERSION,
                        "mode": "mutate",
                        "group_id": str(group["group_id"]),
                        "preference_id": str(pair["preference_id"]),
                        "defect": defect,
                        "chosen": str(pair["chosen"]),
                    }
                    try:
                        record["teacher"] = ask(
                            transport,
                            endpoint,
                            MUTATE_TEMPLATE.format(
                                passage=group["passage"],
                                chosen=str(pair["chosen"]),
                                defect=defect,
                                cel=spec["cel"],
                                wymog=spec["wymog"],
                            ),
                            effort,
                        )
                    except (JudgeApiError, ValueError, json.JSONDecodeError) as error:
                        record["error"] = f"{type(error).__name__}: {error}"
                    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    handle.flush()
                    results.append(record)
                    teacher = record.get("teacher", {})
                    print(
                        f"[{defect:>14}] {record['chosen']}\n"
                        f"{'':>17}→ {teacher.get('query', record.get('error'))}"
                        f"   (edycja: {teacher.get('edycja', '-')})",
                        flush=True,
                    )

    write_json(
        args.output_dir / f"{args.mode}_summary.json",
        {
            "schema_version": 1,
            "role": "prompt_exploration_not_training_data",
            "prompt_version": PROMPT_VERSION,
            "mode": args.mode,
            "model": args.model,
            "groups": len(groups),
            "calls": len(results),
            "errors": sum(1 for row in results if "error" in row),
            "pairs_built": 0,
            "final_tests_used": [],
        },
    )


if __name__ == "__main__":
    main()
