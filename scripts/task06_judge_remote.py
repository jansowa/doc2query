#!/usr/bin/env python3
"""Sędzia odpowiadalności na maszynie z GPU (vLLM, endpoint zgodny z OpenAI).

Skrypt jest **samodzielny**: tylko biblioteka standardowa, żadnych importów z repo, więc
wystarczy skopiować jego i `items.jsonl`. Nie zawiera żadnych etykiet — pakiet jest
label-free z założenia, więc nic tu nie da się dostroić pod wynik.

Właściwości, na które zwracam uwagę, bo od nich zależy wiarygodność runu:

* `temperature=0` i przypięty `seed`; werdykt to jedno słowo z [yes, no, uncertain];
* trwały journal (jedna linia na werdykt, `flush` + `fsync`) — przerwanie w dowolnym
  momencie nie traci pracy, a ponowne uruchomienie **wznawia** i nie powtarza itemów;
* itemy tego samego pasażu idą po sobie **w jednym pasie**, żeby serwer mógł ponownie
  użyć prefiksu KV zamiast liczyć prefill od zera dla każdego kandydata;
* równoległość konfigurowalna (`--parallel`, domyślnie 8), a pasaż nigdy nie jest
  dzielony między pasy;
* thinking wyłączony (zamrożony prompt wymaga „bez toku rozumowania”), a odpowiedź
  wymuszona jako obiekt JSON.

Użycie:

    python3 task06_judge_remote.py --base-url http://127.0.0.1:8000/v1 \
        --model Qwen/Qwen3.8-27B-FP8 --items items.jsonl --journal verdicts.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PROMPT_VERSION = "task06-answerability-pl-v1"
JOURNAL_SCHEMA = "task06-answerability-remote-verdict-v1"
VERDICTS = ("yes", "no", "uncertain")
MAX_INVALID_RETRIES = 3
MAX_TRANSPORT_RETRIES = 5

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

EXPECTED_SYSTEM_PROMPT_SHA256 = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()

# Ctrl-C musi zatrzymywac run szybko: watki robocze sa non-daemon, a ThreadPoolExecutor
# czeka na nie przy wyjsciu z bloku `with`, wiec sam KeyboardInterrupt w watku glownym
# NIE przerwalby liczenia. Flaga jest sprawdzana przed kazdym itemem.
STOP = threading.Event()


def _request_stop(signum, frame):  # sygnatura wymuszona przez modul signal
    if STOP.is_set():
        print("\ndrugie przerwanie: wychodze natychmiast", file=sys.stderr)
        os._exit(130)
    STOP.set()
    print(
        "\nprzerwanie: koncze rozpoczete requesty i zatrzymuje sie "
        "(journal jest kompletny, wznowisz ta sama komenda)",
        file=sys.stderr,
        flush=True,
    )


def load_items(path):
    """Wczytaj pakiet: pierwsza linia to pasaże, dalej itemy odwołujące się do nich indeksem."""
    with open(path, encoding="utf-8") as handle:
        header = json.loads(handle.readline())
        passages = list(header["passages"])
        items = []
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            items.append(
                {
                    "item_id": str(row["i"]),
                    "query": str(row["q"]),
                    "passage": passages[int(row["p"])],
                }
            )
    if not items:
        raise SystemExit("pakiet nie zawiera żadnych itemów")
    ids = [item["item_id"] for item in items]
    if len(ids) != len(set(ids)):
        raise SystemExit("pakiet zawiera zduplikowane item_id")
    return items


def lanes_by_passage(items, lanes):
    """Itemy jednego pasażu trafiają obok siebie do jednego pasa (prefix cache)."""
    grouped = {}
    for item in items:
        grouped.setdefault(item["passage"], []).append(item)
    ordered = sorted(grouped, key=lambda passage: hashlib.sha256(passage.encode()).hexdigest())
    buckets = [[] for _ in range(lanes)]
    for index, passage in enumerate(ordered):
        buckets[index % lanes].extend(sorted(grouped[passage], key=lambda e: e["item_id"]))
    return buckets


def load_done(journal_path):
    done = {}
    if not os.path.exists(journal_path):
        return done
    with open(journal_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # Ostatnia linia mogła zostać ucięta przy zabiciu procesu; pomijamy ją.
                continue
            if row.get("event") == "verdict":
                done[str(row["item_id"])] = row
    return done


def post_json(url, payload, timeout, api_key=None):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    request = urllib.request.Request(
        url, data=json.dumps(payload, ensure_ascii=False).encode(), headers=headers, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def get_json(url, timeout, api_key=None):
    headers = {}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    request = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def format_duration(seconds):
    seconds = int(max(0.0, seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


class Progress:
    """Jednolinijkowy pasek postępu na stderr; stdout zostaje czysty dla logu.

    Liczy postęp względem CAŁEGO pakietu (itemy z wcześniejszych uruchomień wchodzą do
    licznika `done`), ale tempo i ETA liczy z bieżącego uruchomienia — po wznowieniu ETA
    jest wtedy uczciwe, a nie zaniżone przez pracę zrobioną wcześniej.
    """

    BAR_WIDTH = 28

    def __init__(self, total, already_done, stream=sys.stderr, min_interval=0.5):
        self.total = total
        self.already_done = already_done
        self.stream = stream
        self.min_interval = min_interval
        self.counts = {"yes": 0, "no": 0, "uncertain": 0, "failed": 0}
        self.started = time.time()
        self.last_render = 0.0
        self.enabled = stream is not None and stream.isatty()

    def bump(self, key):
        self.counts[key] = self.counts.get(key, 0) + 1
        self.render()

    @property
    def judged_now(self):
        return sum(self.counts.values())

    def render(self, force=False):
        now = time.time()
        if not force and now - self.last_render < self.min_interval:
            return
        self.last_render = now
        done = self.already_done + self.judged_now
        share = done / self.total if self.total else 1.0
        filled = round(share * self.BAR_WIDTH)
        bar = "#" * filled + "-" * (self.BAR_WIDTH - filled)
        elapsed = now - self.started
        rate = self.judged_now / elapsed if elapsed > 0 and self.judged_now else 0.0
        remaining = self.total - done
        eta = format_duration(remaining / rate) if rate > 0 else "?"
        line = (
            f"[{bar}] {share * 100:5.1f}%  {done}/{self.total}  "
            f"{rate:5.1f} it/s  minelo {format_duration(elapsed)}  ETA {eta}  "
            f"yes {self.counts['yes']} no {self.counts['no']} "
            f"unc {self.counts['uncertain']} fail {self.counts['failed']}"
        )
        if self.enabled:
            self.stream.write("\r" + line + " " * 4)
            self.stream.flush()
        elif force or self.judged_now % 200 == 0:
            # Bez terminala (nohup, tee do pliku) drukujemy pelne linie, nie \r.
            print(line, file=self.stream, flush=True)

    def finish(self):
        self.render(force=True)
        if self.enabled:
            self.stream.write("\n")
            self.stream.flush()


def parse_verdict(content):
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise ValueError("odpowiedź nie jest obiektem JSON")
    verdict = str(payload.get("verdict", "")).strip().casefold()
    if verdict not in VERDICTS:
        raise ValueError("werdykt poza schematem: " + repr(verdict))
    return verdict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # Adres endpointu jest parametrem operatorskim i NIE należy do repozytorium.
    parser.add_argument(
        "--base-url", default=os.environ.get("JUDGE_BASE_URL", "http://127.0.0.1:8000/v1")
    )
    parser.add_argument(
        "--model",
        help="Nazwa modelu; pominięta, jeśli /v1/models serwuje dokładnie jeden model.",
    )
    parser.add_argument("--items", default="items.jsonl")
    parser.add_argument("--journal", default="verdicts.jsonl")
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "dummy"))
    args = parser.parse_args()

    if EXPECTED_SYSTEM_PROMPT_SHA256 != hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest():
        raise SystemExit("prompt systemowy został zmodyfikowany")

    items = load_items(args.items)
    served = {}
    try:
        served = get_json(args.base_url.rstrip("/") + "/models", 30.0, args.api_key)
    except Exception as exc:  # metadane są pomocnicze, nie blokujące
        print("uwaga: nie udało się odczytać /models: " + str(exc), file=sys.stderr)

    if not args.model:
        names = [str(row.get("id")) for row in (served.get("data") or []) if row.get("id")]
        if len(names) != 1:
            raise SystemExit(
                "podaj --model: /v1/models zwrocil " + str(len(names)) + " modeli: " + str(names)
            )
        args.model = names[0]
    print("model: " + str(args.model))

    done = load_done(args.journal)
    # Wznowienie innym modelem uniewaznia caly journal przy imporcie, wiec lepiej
    # zatrzymac sie teraz niz po kilku godzinach liczenia.
    previous_models = {str(row.get("model")) for row in done.values()} - {"None"}
    if previous_models and previous_models != {str(args.model)}:
        raise SystemExit(
            "journal "
            + args.journal
            + " zawiera werdykty modelu "
            + str(sorted(previous_models))
            + ", a teraz uruchamiasz "
            + str(args.model)
            + ". Import odrzuca journal mieszajacy modele. Uzyj tego samego modelu albo "
            "zacznij nowy journal (--journal INNA_NAZWA.jsonl)."
        )
    if done:
        print(f"WZNAWIANIE: {len(done)} werdyktow z poprzednich uruchomien zostaje bez zmian")
    print(
        f"itemow: {len(items)}, juz ocenionych: {len(done)}, "
        f"do zrobienia: {len(items) - len(done)}, pasow: {args.parallel}"
    )
    guard = threading.Lock()
    counters = {"yes": 0, "no": 0, "uncertain": 0, "failed": 0, "skipped": len(done)}
    progress = Progress(len(items), len(done))
    url = args.base_url.rstrip("/") + "/chat/completions"

    def judge(item):
        if STOP.is_set():
            return
        payload = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {"passage": item["passage"], "query": item["query"]},
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                },
            ],
            "temperature": 0.0,
            "seed": args.seed,
            "max_tokens": args.max_tokens,
            "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        verdict = None
        usage = None
        for attempt in range(1, MAX_INVALID_RETRIES + 1):
            body = None
            for transport_attempt in range(1, MAX_TRANSPORT_RETRIES + 1):
                try:
                    body = post_json(url, payload, args.timeout, args.api_key)
                    break
                except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
                    if transport_attempt == MAX_TRANSPORT_RETRIES:
                        print(f"blad transportu, item {item['item_id']}: {exc}", file=sys.stderr)
                    else:
                        time.sleep(min(30.0, 2.0**transport_attempt))
            if body is None:
                break
            usage = body.get("usage")
            content = ""
            choices = body.get("choices") or []
            if choices:
                content = str((choices[0].get("message") or {}).get("content") or "")
            try:
                verdict = parse_verdict(content)
                break
            except (ValueError, json.JSONDecodeError) as exc:
                print(
                    f"niepoprawna odpowiedz (proba {attempt}) dla "
                    f"{item['item_id']}: {str(exc)[:120]}",
                    file=sys.stderr,
                )
        with guard:
            if verdict is None:
                counters["failed"] += 1
                progress.bump("failed")
                return
            row = {
                "schema": JOURNAL_SCHEMA,
                "event": "verdict",
                "item_id": item["item_id"],
                "verdict": verdict,
                "prompt_version": PROMPT_VERSION,
                "model": args.model,
                "served_model": served,
                "seed": args.seed,
                "thinking": False,
                "usage": usage,
            }
            with open(args.journal, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            counters[verdict] += 1
            progress.bump(verdict)

    pending = [item for item in items if item["item_id"] not in done]
    buckets = lanes_by_passage(pending, max(1, args.parallel))

    def walk(bucket):
        for item in bucket:
            if STOP.is_set():
                return
            judge(item)

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        for future in [pool.submit(walk, bucket) for bucket in buckets]:
            future.result()

    progress.finish()
    remaining = len(items) - len(done) - progress.judged_now
    status = "PRZERWANE" if STOP.is_set() else "gotowe"
    print(f"{status}: " + json.dumps(counters, ensure_ascii=False))
    print("journal: " + os.path.abspath(args.journal))
    if STOP.is_set() and remaining > 0:
        print(
            f"zostalo {remaining} itemow - uruchom te sama komende, zeby wznowic",
            flush=True,
        )
    if counters["failed"]:
        print(f"UWAGA: {counters['failed']} itemow bez werdyktu (fail-closed)", file=sys.stderr)


if __name__ == "__main__":
    main()
