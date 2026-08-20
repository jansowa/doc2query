#!/usr/bin/env python3
"""Sędzia odpowiadalności na maszynie z GPU (vLLM, endpoint zgodny z OpenAI).

Skrypt jest **samodzielny**: tylko biblioteka standardowa, żadnych importów z repo, więc
wystarczy skopiować jego i `items.jsonl`. Pakiet jest **label-free**, więc nic tu nie da
się dostroić pod wynik.

Dwa tryby, oba z tym samym kontraktem decyzyjnym `[yes, no, uncertain]`:

* `--batch-size 1` — jedno zapytanie na request, prompt `task06-answerability-pl-v1`.
  To jest przyrząd, na którym zamrożono kryteria K1-K3 w ADR V2-01;
* `--batch-size N>1` — N zapytań **tego samego pasażu** w jednym requeście, prompt
  `task06-answerability-pl-v2-batched`. Pasaż i system prompt lecą raz na paczkę zamiast
  raz na zapytanie, co przy 6,5 zapytania na pasaż zbija prefill kilkukrotnie. Wymaga
  przejścia bramki A/B (zgodność werdyktów per item) — patrz amendment do ADR V2-01.

Właściwości, na których stoi wiarygodność runu:

* `temperature=0`, przypięty `seed`, thinking wyłączony, wartość werdyktu domknięta
  schematem (enum), więc model nie może odpowiedzieć poza przestrzenią decyzyjną;
* trwały journal (`flush` + `fsync` po każdym zapisie) — przerwanie nie traci pracy,
  a ponowne uruchomienie **wznawia** i nie powtarza itemów;
* runner odmawia dopisywania do journala wyprodukowanego innym modelem, innym
  dekodowaniem albo inną wersją promptu (poza jawnym fallbackiem paczki);
* adres endpointu wybierany raz, z jawną rodziną adresów i krótkim connectem, bo
  `http.client` nie robi Happy Eyeballs i martwa trasa IPv6 potrafi zabić przepustowość;
* itemy jednego pasażu nigdy nie są dzielone między pasy, a paczka to zawsze jeden pasaż.

Użycie:

    python3 task06_judge_remote.py --items items.jsonl --journal verdicts.jsonl \
        --base-url http://127.0.0.1:8000/v1 --parallel 8 --batch-size 4
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import signal
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

PROMPT_VERSION_SINGLE = "task06-answerability-pl-v1"
PROMPT_VERSION_BATCHED = "task06-answerability-pl-v2-batched"
JOURNAL_SCHEMA = "task06-answerability-remote-verdict-v1"
VERDICTS = ("yes", "no", "uncertain")
# Ile razy powtorzyc request, ktory wrocil poza schematem. Serwer NIE jest deterministyczny
# (continuous batching zmienia sklad batcha, a wiec numeryke), wiec ponowienie realnie
# ratuje czesc przypadkow - zmierzone: proba 1 i 2 poza schematem, proba 3 poprawna.
DEFAULT_INVALID_RETRIES = 3
MAX_TRANSPORT_RETRIES = 5
# Budzet wyjscia paczki: ~12 tokenow na werdykt + narzut struktury JSON.
TOKENS_PER_VERDICT = 12
BATCH_TOKEN_OVERHEAD = 20

SYSTEM_PROMPT_SINGLE = (
    "Jesteś rygorystycznym polskim audytorem odpowiadalności. Otrzymasz pasaż i "
    "zapytanie wyszukiwawcze. Oceń wyłącznie jedno: czy na to zapytanie można "
    "odpowiedzieć korzystając WYŁĄCZNIE z informacji zawartych w podanym pasażu. "
    "Nie oceniaj stylu, długości ani użyteczności. Jeżeli odpowiedź wymaga wiedzy "
    "spoza pasażu, werdykt brzmi no. Jeżeli pasaż odpowiada tylko częściowo albo "
    "nie masz pewności, werdykt brzmi uncertain. Zwróć wyłącznie obiekt JSON "
    'w formacie {"verdict": "yes"} z wartością ze zbioru [yes, no, uncertain]. '
    "Bez toku rozumowania i bez komentarzy."
)

# Wariant wielowerdyktowy. Kryterium merytoryczne jest przepisane BEZ ZMIAN z wersji
# pojedynczej (to warunek porownywalnosci w bramce A/B); dodane jest tylko wymuszenie
# niezaleznej oceny kazdego zapytania i format odpowiedzi z identyfikatorami.
SYSTEM_PROMPT_BATCHED = (
    "Jesteś rygorystycznym polskim audytorem odpowiadalności. Otrzymasz jeden pasaż i "
    "listę zapytań wyszukiwawczych. Dla KAŻDEGO zapytania oceń NIEZALEŻNIE od "
    "pozostałych wyłącznie jedno: czy na to zapytanie można odpowiedzieć korzystając "
    "WYŁĄCZNIE z informacji zawartych w podanym pasażu. Nie oceniaj stylu, długości "
    "ani użyteczności. Nie porównuj zapytań między sobą i nie zakładaj, że mają różne "
    "werdykty. Jeżeli odpowiedź wymaga wiedzy spoza pasażu, werdykt brzmi no. Jeżeli "
    "pasaż odpowiada tylko częściowo albo nie masz pewności, werdykt brzmi uncertain. "
    'Zwróć wyłącznie obiekt JSON w formacie {"verdicts": [{"id": 1, "verdict": "yes"}]} '
    "— dokładnie jeden element na każde otrzymane zapytanie, z tym samym id, z wartością "
    "ze zbioru [yes, no, uncertain]. Bez toku rozumowania i bez komentarzy."
)

PROMPT_SHA256 = {
    PROMPT_VERSION_SINGLE: hashlib.sha256(SYSTEM_PROMPT_SINGLE.encode("utf-8")).hexdigest(),
    PROMPT_VERSION_BATCHED: hashlib.sha256(SYSTEM_PROMPT_BATCHED.encode("utf-8")).hexdigest(),
}
# Zgodnosc wstecz: sonda diagnostyczna pyta o te nazwe.
EXPECTED_SYSTEM_PROMPT_SHA256 = PROMPT_SHA256[PROMPT_VERSION_SINGLE]

# Ctrl-C musi zatrzymywac run szybko: watki robocze sa non-daemon, a ThreadPoolExecutor
# czeka na nie przy wyjsciu z bloku `with`, wiec sam KeyboardInterrupt w watku glownym
# NIE przerwalby liczenia. Flaga jest sprawdzana przed kazda paczka.
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


class BatchError(Exception):
    """Odpowiedz paczki nie spelnia kontraktu; nosi tresc do zapisu w journalu."""

    def __init__(self, reason, content=""):
        super().__init__(reason)
        self.reason = reason
        self.content = content


class HttpStatusError(Exception):
    def __init__(self, code, body):
        super().__init__(f"HTTP {code}")
        self.code = code
        self.body = body


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
    """Itemy jednego pasażu trafiają obok siebie do jednego pasa (nigdy nie są dzielone)."""
    grouped = {}
    for item in items:
        grouped.setdefault(item["passage"], []).append(item)
    ordered = sorted(grouped, key=lambda passage: hashlib.sha256(passage.encode()).hexdigest())
    buckets = [[] for _ in range(lanes)]
    for index, passage in enumerate(ordered):
        buckets[index % lanes].extend(sorted(grouped[passage], key=lambda e: e["item_id"]))
    return buckets


def batches_in_lane(lane_items, batch_size):
    """Podziel pas na paczki: wyłącznie w obrębie jednego pasażu, maksymalnie batch_size.

    Nigdy nie miesza pasaży - paczka jest zawsze jednym pasażem, bo cały sens paczkowania
    polega na tym, że pasaż jest wysyłany raz.
    """
    if batch_size < 1:
        raise SystemExit("batch-size musi byc >= 1")
    batches = []
    current_passage = None
    current = []
    for item in lane_items:
        if item["passage"] != current_passage or len(current) >= batch_size:
            if current:
                batches.append(current)
            current = []
            current_passage = item["passage"]
        current.append(item)
    if current:
        batches.append(current)
    return batches


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


class Endpoint:
    """Rozwiazany raz adres endpointu, z jawnie wybrana rodzina adresow.

    Po co to istnieje: `curl` robi Happy Eyeballs (probuje IPv4 i IPv6 rownolegle i cicho
    bierze to, co dziala), a `http.client` bierze adresy po kolei. Jesli host ma rekord
    AAAA, a trasa IPv6 jest czarna dziura, KAZDY request placi pelny timeout - objawia sie
    to jako run "liczacy" jeden batch na kilka minut przy 0% GPU.
    """

    def __init__(self, base_url, prefer, connect_timeout):
        parsed = urllib.parse.urlsplit(base_url)
        if parsed.scheme != "http":
            raise SystemExit("obslugiwany jest tylko http (bez TLS): " + base_url)
        self.host = parsed.hostname
        self.port = parsed.port or 80
        self.path = parsed.path.rstrip("/") + "/chat/completions"
        self.models_path = parsed.path.rstrip("/") + "/models"
        self.host_header = parsed.netloc
        self.family, self.address = self._pick(prefer, connect_timeout)

    def _candidates(self, prefer):
        infos = socket.getaddrinfo(self.host, self.port, type=socket.SOCK_STREAM)
        order = {
            "ipv4": (socket.AF_INET, socket.AF_INET6),
            "ipv6": (socket.AF_INET6, socket.AF_INET),
        }
        preferred = order.get(prefer, (socket.AF_INET, socket.AF_INET6))
        ranked = sorted(
            infos, key=lambda info: preferred.index(info[0]) if info[0] in preferred else 9
        )
        seen = []
        for family, _, _, _, sockaddr in ranked:
            entry = (family, sockaddr[0])
            if entry not in seen:
                seen.append(entry)
        return seen

    def _pick(self, prefer, connect_timeout):
        candidates = self._candidates(prefer)
        if not candidates:
            raise SystemExit("nie udalo sie rozwiazac hosta " + str(self.host))
        errors = []
        for family, address in candidates:
            probe = socket.socket(family, socket.SOCK_STREAM)
            probe.settimeout(connect_timeout)
            try:
                probe.connect((address, self.port))
                label = "IPv6" if family == socket.AF_INET6 else "IPv4"
                print(f"adres: {address} ({label}, connect ok)", flush=True)
                return family, address
            except OSError as exc:
                errors.append(f"{address}: {exc}")
            finally:
                probe.close()
        raise SystemExit(
            "zaden adres hosta nie przyjmuje polaczenia na porcie "
            + str(self.port)
            + ":\n  "
            + "\n  ".join(errors)
        )

    def request(self, path, payload, timeout, api_key):
        host = "[" + self.address + "]" if self.family == socket.AF_INET6 else self.address
        connection = http.client.HTTPConnection(host, self.port, timeout=timeout)
        headers = {"Host": self.host_header}
        if api_key:
            headers["Authorization"] = "Bearer " + api_key
        try:
            if payload is None:
                connection.request("GET", path, headers=headers)
            else:
                headers["Content-Type"] = "application/json"
                body = json.dumps(payload, ensure_ascii=False).encode()
                connection.request("POST", path, body=body, headers=headers)
            response = connection.getresponse()
            raw = response.read()
            if response.status >= 400:
                raise HttpStatusError(response.status, raw.decode(errors="replace")[:600])
            return json.loads(raw.decode())
        finally:
            connection.close()


def single_schema():
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "answerability_verdict",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"verdict": {"type": "string", "enum": list(VERDICTS)}},
                "required": ["verdict"],
                "additionalProperties": False,
            },
        },
    }


def batched_schema(count):
    """Schemat generowany per request: minItems == maxItems == liczność paczki."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "answerability_verdicts",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "verdicts": {
                        "type": "array",
                        "minItems": count,
                        "maxItems": count,
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "integer"},
                                "verdict": {"type": "string", "enum": list(VERDICTS)},
                            },
                            "required": ["id", "verdict"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["verdicts"],
                "additionalProperties": False,
            },
        },
    }


def response_format(decoding, count):
    """Wariant dekodowania; zapisywany w kazdym wierszu journala.

    `json_object` wymusza tylko poprawna skladnie JSON, wiec model moze zwrocic wartosc
    poza przestrzenia decyzyjna (zmierzone: {"verdict": "verdict"}) i wtedy item traci
    werdykt zamiast dostac `uncertain`. Schemat z enumem domyka wartosc do zbioru
    [yes, no, uncertain], czyli wymusza to, co kontrakt i tak zaklada.
    """
    if decoding == "json_object":
        return {"type": "json_object"}
    if decoding == "json_schema_enum":
        return single_schema() if count == 1 else batched_schema(count)
    raise SystemExit("nieznany wariant dekodowania: " + str(decoding))


def build_payload(item, args):
    """Payload pojedynczy (prompt v1). Preflight, tryb batch-size 1 i fallback paczki."""
    return {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_SINGLE},
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
        "response_format": response_format(args.decoding, 1),
        "chat_template_kwargs": {"enable_thinking": False},
    }


def build_batch_payload(batch, args):
    """Payload paczkowy (prompt v2): jeden pasaz, N zapytan z lokalnymi id 1..N.

    `sort_keys` trzyma `passage` przed `queries`, wiec paczki tego samego pasazu maja
    wspolny prefiks promptu tak samo, jak requesty pojedyncze.
    """
    queries = [{"id": index, "query": item["query"]} for index, item in enumerate(batch, start=1)]
    return {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_BATCHED},
            {
                "role": "user",
                "content": json.dumps(
                    {"passage": batch[0]["passage"], "queries": queries},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            },
        ],
        "temperature": 0.0,
        "seed": args.seed,
        "max_tokens": TOKENS_PER_VERDICT * len(batch) + BATCH_TOKEN_OVERHEAD,
        "response_format": response_format(args.decoding, len(batch)),
        "chat_template_kwargs": {"enable_thinking": False},
    }


def parse_verdict(content):
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise ValueError("odpowiedź nie jest obiektem JSON")
    verdict = str(payload.get("verdict", "")).strip().casefold()
    if verdict not in VERDICTS:
        raise ValueError("werdykt poza schematem: " + repr(verdict))
    return verdict


def parse_batch_verdicts(content, expected_ids):
    """Zwroc {id: werdykt}; kolejnosc elementow w odpowiedzi jest IGNOROWANA.

    Naruszenie ktoregokolwiek wymogu to blad calej paczki: poprawny JSON, zbior id
    dokladnie rowny wyslanemu, brak duplikatow id, kazdy werdykt w enumie.
    """
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise BatchError("odpowiedz nie jest obiektem JSON", content)
    rows = payload.get("verdicts")
    if not isinstance(rows, list):
        raise BatchError("brak tablicy 'verdicts'", content)
    seen = {}
    for row in rows:
        if not isinstance(row, dict):
            raise BatchError("element 'verdicts' nie jest obiektem", content)
        try:
            identifier = int(row["id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BatchError("element bez poprawnego id", content) from exc
        verdict = str(row.get("verdict", "")).strip().casefold()
        if verdict not in VERDICTS:
            raise BatchError(f"werdykt poza schematem dla id={identifier}: {verdict!r}", content)
        if identifier in seen:
            raise BatchError(f"zduplikowane id w odpowiedzi: {identifier}", content)
        seen[identifier] = verdict
    if set(seen) != set(expected_ids):
        raise BatchError(
            f"zbior id nie zgadza sie z wyslanym: dostano {sorted(seen)}, "
            f"oczekiwano {sorted(expected_ids)}",
            content,
        )
    return seen


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
    licznika `already_done`), ale tempo i ETA z bieżącego uruchomienia - po wznowieniu ETA
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
        self.enabled = stream is not None and hasattr(stream, "isatty") and stream.isatty()

    def bump(self, key, count=1):
        self.counts[key] = self.counts.get(key, 0) + count
        self.render()

    @property
    def judged_now(self):
        return sum(self.counts.values())

    def render(self, force=False):
        if self.stream is None:
            return
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
        elif force or self.judged_now <= 3 or self.judged_now % 25 == 0:
            print(line, file=self.stream, flush=True)

    def finish(self):
        self.render(force=True)
        if self.enabled:
            self.stream.write("\n")
            self.stream.flush()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    # Adres endpointu jest parametrem operatorskim i NIE nalezy do repozytorium.
    parser.add_argument(
        "--base-url", default=os.environ.get("JUDGE_BASE_URL", "http://127.0.0.1:8000/v1")
    )
    parser.add_argument(
        "--model",
        help="Nazwa modelu; pominieta, jesli /v1/models serwuje dokladnie jeden model.",
    )
    parser.add_argument("--items", default="items.jsonl")
    parser.add_argument("--journal", default="verdicts.jsonl")
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Ile zapytan tego samego pasazu w jednym requescie. 1 = przyrzad z ADR V2-01.",
    )
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=24,
        help="Budzet wyjscia w trybie pojedynczym; w paczkowym liczony z licznosci paczki.",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "dummy"))
    parser.add_argument(
        "--address-family",
        choices=("ipv4", "ipv6"),
        default="ipv4",
        help="Ktora rodzine adresow probowac pierwsza (domyslnie IPv4).",
    )
    parser.add_argument("--connect-timeout", type=float, default=8.0)
    parser.add_argument(
        "--decoding",
        choices=("json_schema_enum", "json_object"),
        default="json_schema_enum",
        help="Ograniczenie dekodowania; enum domyka wartosc werdyktu do zbioru z kontraktu.",
    )
    parser.add_argument(
        "--invalid-retries",
        type=int,
        default=DEFAULT_INVALID_RETRIES,
        help="Ile prob na request, ktory wraca poza schematem.",
    )
    return parser


def main() -> None:
    # Pod pipem (np. `| tee log.txt`) stdout jest buforowany blokowo, wiec bez tego
    # pierwsze komunikaty siedza w buforze i run wyglada na zawieszony.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except AttributeError:  # Python < 3.7
            pass

    args = build_parser().parse_args()
    batched = args.batch_size > 1
    primary_version = PROMPT_VERSION_BATCHED if batched else PROMPT_VERSION_SINGLE
    items = load_items(args.items)
    endpoint = Endpoint(args.base_url, args.address_family, args.connect_timeout)
    served = {}
    try:
        served = endpoint.request(endpoint.models_path, None, 30.0, args.api_key)
    except Exception as exc:  # metadane sa pomocnicze, nie blokujace
        print("uwaga: nie udalo sie odczytac /models: " + str(exc), file=sys.stderr)

    if not args.model:
        names = [str(row.get("id")) for row in (served.get("data") or []) if row.get("id")]
        if len(names) != 1:
            raise SystemExit(
                "podaj --model: /v1/models zwrocil " + str(len(names)) + " modeli: " + str(names)
            )
        args.model = names[0]
    print("model: " + str(args.model), flush=True)
    print("endpoint: " + endpoint.host_header + endpoint.path, flush=True)
    print(
        f"prompt: {primary_version} (sha256 {PROMPT_SHA256[primary_version][:16]}), "
        f"batch-size {args.batch_size}, dekodowanie {args.decoding}",
        flush=True,
    )

    # Preflight: jeden request PRZED wejsciem w petle. Bez tego odrzucenie pola przez
    # serwer objawia sie jako wielogodzinne "zawieszenie".
    # Sonda paczkowa musi byc reprezentatywna: itemy JEDNEGO pasazu, tak jak w runie.
    probe_batch = None
    if batched:
        first_passage = items[0]["passage"]
        probe_batch = [item for item in items if item["passage"] == first_passage][
            : min(args.batch_size, 2)
        ]
        if len(probe_batch) < 2:
            probe_batch = probe_batch * 2  # gwarantuj wieloelementowa odpowiedz w sondzie
    probe = build_batch_payload(probe_batch, args) if batched else build_payload(items[0], args)
    try:
        probe_body = endpoint.request(endpoint.path, probe, min(args.timeout, 120.0), args.api_key)
    except HttpStatusError as exc:
        raise SystemExit(
            f"PREFLIGHT: serwer odrzucil payload (HTTP {exc.code}).\n"
            f"Odpowiedz serwera: {exc.body}\n"
            "Nie zmieniam payloadu po cichu: zamrozony protokol wymaga wylaczonego "
            "thinkingu i domknietej wartosci werdyktu. Przeslij ten komunikat."
        ) from exc
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        raise SystemExit(
            f"PREFLIGHT: brak odpowiedzi z {endpoint.host_header}{endpoint.path} ({exc}). "
            "Sprawdz, czy endpoint i port sa poprawne oraz czy serwer odpowiada."
        ) from exc
    probe_choices = probe_body.get("choices") or []
    probe_content = (
        str((probe_choices[0].get("message") or {}).get("content") or "") if probe_choices else ""
    )
    try:
        if batched:
            parse_batch_verdicts(probe_content, list(range(1, len(probe_batch) + 1)))
        else:
            parse_verdict(probe_content)
    except (BatchError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"PREFLIGHT: serwer odpowiedzial, ale odpowiedz jest poza kontraktem ({exc}).\n"
            f"Tresc: {probe_content[:400]!r}\n"
            "Jesli widac tok rozumowania, thinking nie zostal wylaczony - przeslij ten "
            "komunikat, zamiast podnosic budzet tokenow."
        ) from exc
    print("preflight ok: serwer zwraca poprawna odpowiedz", flush=True)

    done = load_done(args.journal)
    # Wiersze zapisane przed amendmentem nie maja pola "decoding" - powstaly przy
    # json_object, wiec traktujemy je jawnie, zeby wznowienie ich nie zmieszalo z enumem.
    previous_decodings = {
        str(row.get("decoding", "json_object_przed_amendmentem")) for row in done.values()
    }
    if previous_decodings and previous_decodings != {str(args.decoding)}:
        raise SystemExit(
            f"journal {args.journal} zawiera werdykty z dekodowaniem "
            f"{sorted(previous_decodings)}, a teraz uruchamiasz {args.decoding}. Mieszanie "
            "ograniczen dekodowania w jednej kalibracji jest niedopuszczalne - zacznij nowy "
            "journal (--journal INNA_NAZWA.jsonl)."
        )
    # Fallback paczki celowo zapisuje wiersze promptem pojedynczym, wiec ta wersja jest
    # dopuszczalna obok wersji glownej; kazda inna kombinacja to pomieszane przyrzady.
    allowed_versions = {primary_version, PROMPT_VERSION_SINGLE}
    previous_versions = {str(row.get("prompt_version")) for row in done.values()}
    if previous_versions - allowed_versions:
        raise SystemExit(
            f"journal {args.journal} zawiera werdykty promptem "
            f"{sorted(previous_versions - allowed_versions)}, a teraz uruchamiasz "
            f"{primary_version}. Zacznij nowy journal (--journal INNA_NAZWA.jsonl)."
        )
    previous_models = {str(row.get("model")) for row in done.values()} - {"None"}
    if previous_models and previous_models != {str(args.model)}:
        raise SystemExit(
            f"journal {args.journal} zawiera werdykty modelu {sorted(previous_models)}, a teraz "
            f"uruchamiasz {args.model}. Import odrzuca journal mieszajacy modele. Uzyj tego "
            "samego modelu albo zacznij nowy journal (--journal INNA_NAZWA.jsonl)."
        )
    if done:
        print(f"WZNAWIANIE: {len(done)} werdyktow z poprzednich uruchomien zostaje bez zmian")

    pending = [item for item in items if item["item_id"] not in done]
    buckets = lanes_by_passage(pending, max(1, args.parallel))
    lane_batches = [batches_in_lane(bucket, args.batch_size) for bucket in buckets]
    planned = sum(len(batches) for batches in lane_batches)
    print(
        f"itemow: {len(items)}, juz ocenionych: {len(done)}, do zrobienia: {len(pending)}, "
        f"pasow: {args.parallel}, requestow do wyslania: {planned}",
        flush=True,
    )

    guard = threading.Lock()
    counters = {"yes": 0, "no": 0, "uncertain": 0, "failed": 0, "skipped": len(done)}
    stats = {"batches_ok": 0, "batches_retried": 0, "batches_fallback": 0}
    progress = Progress(len(items), len(done))

    def append_events(rows):
        with open(args.journal, "a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def post(payload, label):
        """Wyslij request z ponowieniami transportu; 4xx (poza 429) nie jest ponawiany."""
        for transport_attempt in range(1, MAX_TRANSPORT_RETRIES + 1):
            try:
                return endpoint.request(endpoint.path, payload, args.timeout, args.api_key)
            except HttpStatusError as exc:
                if 400 <= exc.code < 500 and exc.code != 429:
                    print(
                        f"HTTP {exc.code} od serwera ({label}): {exc.body}",
                        file=sys.stderr,
                        flush=True,
                    )
                    return None
                if transport_attempt == MAX_TRANSPORT_RETRIES:
                    print(
                        f"blad HTTP {exc.code} ({label}): {exc.body}", file=sys.stderr, flush=True
                    )
                else:
                    time.sleep(min(30.0, 2.0**transport_attempt))
            except (OSError, urllib.error.URLError, TimeoutError) as exc:
                if transport_attempt == MAX_TRANSPORT_RETRIES:
                    print(f"blad transportu ({label}): {exc}", file=sys.stderr, flush=True)
                else:
                    time.sleep(min(30.0, 2.0**transport_attempt))
        return None

    def response_text(body):
        choices = body.get("choices") or []
        if not choices:
            return "", None
        choice = choices[0]
        return str((choice.get("message") or {}).get("content") or ""), choice.get("finish_reason")

    def judge_single(item, batch_id, position, batch_size, fallback):
        """Jeden item, prompt pojedynczy. Tryb batch-size 1 oraz fallback paczki."""
        payload = build_payload(item, args)
        verdict = None
        last_error = None
        last_content = ""
        usage = None
        attempt = 0
        for attempt_index in range(1, args.invalid_retries + 1):
            body = post(payload, "item " + item["item_id"])
            if body is None:
                break
            usage = body.get("usage")
            content, finish = response_text(body)
            if finish == "length":
                last_error = "finish_reason=length"
                last_content = content[:300]
                continue
            try:
                verdict = parse_verdict(content)
                attempt = attempt_index  # ktora proba dala werdykt (serwer nie jest determ.)
                break
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = str(exc)[:200]
                last_content = content[:300]
        with guard:
            if verdict is None:
                counters["failed"] += 1
                if last_error is not None:
                    append_events(
                        [
                            {
                                "schema": JOURNAL_SCHEMA,
                                "event": "out_of_schema",
                                "item_id": item["item_id"],
                                "prompt_version": PROMPT_VERSION_SINGLE,
                                "model": args.model,
                                "decoding": args.decoding,
                                "attempts": args.invalid_retries,
                                "error": last_error,
                                "content": last_content,
                                "batch_id": batch_id,
                                "fallback": fallback,
                            }
                        ]
                    )
                progress.bump("failed")
                return
            append_events(
                [
                    {
                        "schema": JOURNAL_SCHEMA,
                        "event": "verdict",
                        "item_id": item["item_id"],
                        "verdict": verdict,
                        "attempt": attempt,
                        "prompt_version": PROMPT_VERSION_SINGLE,
                        "model": args.model,
                        "served_model": served,
                        "seed": args.seed,
                        "thinking": False,
                        "decoding": args.decoding,
                        "batch_id": batch_id,
                        "batch_size": batch_size,
                        "position_in_batch": position,
                        "fallback": fallback,
                        "usage": usage,
                    }
                ]
            )
            counters[verdict] += 1
            progress.bump(verdict)

    def judge_batch(batch):
        batch_id = hashlib.sha256(
            "\0".join(item["item_id"] for item in batch).encode()
        ).hexdigest()[:16]
        expected = list(range(1, len(batch) + 1))
        payload = build_batch_payload(batch, args)
        verdicts = None
        last_error = None
        last_content = ""
        usage = None
        # Kontrakt: 1 ponowienie calej paczki, potem rozbicie na pojedyncze requesty.
        for attempt in (1, 2):
            body = post(payload, "paczka " + batch_id)
            if body is None:
                last_error = "brak odpowiedzi serwera"
                break
            usage = body.get("usage")
            content, finish = response_text(body)
            if finish == "length":
                last_error = "finish_reason=length (budzet max_tokens za maly)"
                last_content = content[:300]
            else:
                try:
                    verdicts = parse_batch_verdicts(content, expected)
                    break
                except (BatchError, json.JSONDecodeError) as exc:
                    last_error = str(exc)[:200]
                    last_content = str(getattr(exc, "content", content))[:300]
            if attempt == 1:
                with guard:
                    stats["batches_retried"] += 1
        if verdicts is None:
            with guard:
                stats["batches_fallback"] += 1
                append_events(
                    [
                        {
                            "schema": JOURNAL_SCHEMA,
                            "event": "batch_failed",
                            "batch_id": batch_id,
                            "batch_size": len(batch),
                            "item_ids": [item["item_id"] for item in batch],
                            "prompt_version": PROMPT_VERSION_BATCHED,
                            "model": args.model,
                            "decoding": args.decoding,
                            "error": last_error,
                            "content": last_content,
                        }
                    ]
                )
                print(
                    f"paczka {batch_id} ({len(batch)} itemow) poza kontraktem: {last_error} "
                    "-> rozbijam na pojedyncze requesty (fallback)",
                    file=sys.stderr,
                    flush=True,
                )
            for position, item in enumerate(batch, start=1):
                if STOP.is_set():
                    return
                judge_single(item, batch_id, position, len(batch), True)
            return
        with guard:
            stats["batches_ok"] += 1
            rows = [
                {
                    "schema": JOURNAL_SCHEMA,
                    "event": "batch_usage",
                    "batch_id": batch_id,
                    "batch_size": len(batch),
                    "prompt_version": PROMPT_VERSION_BATCHED,
                    "model": args.model,
                    "decoding": args.decoding,
                    "usage": usage,
                }
            ]
            for position, item in enumerate(batch, start=1):
                verdict = verdicts[position]
                rows.append(
                    {
                        "schema": JOURNAL_SCHEMA,
                        "event": "verdict",
                        "item_id": item["item_id"],
                        "verdict": verdict,
                        "attempt": 1,
                        "prompt_version": PROMPT_VERSION_BATCHED,
                        "model": args.model,
                        "served_model": served,
                        "seed": args.seed,
                        "thinking": False,
                        "decoding": args.decoding,
                        "batch_id": batch_id,
                        "batch_size": len(batch),
                        "position_in_batch": position,
                        "fallback": False,
                    }
                )
                counters[verdict] += 1
                progress.counts[verdict] = progress.counts.get(verdict, 0) + 1
            append_events(rows)
            progress.render()

    def walk(batches):
        for batch in batches:
            if STOP.is_set():
                return
            if batched:
                judge_batch(batch)
            else:
                judge_single(batch[0], None, 1, 1, False)

    def heartbeat():
        # Pierwszy request na swiezym serwerze moze trwac dlugo (rozgrzewka, kolejka).
        while not STOP.is_set() and not finished.wait(15.0):
            progress.render(force=True)

    finished = threading.Event()
    threading.Thread(target=heartbeat, daemon=True).start()
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        for future in [pool.submit(walk, batches) for batches in lane_batches]:
            future.result()

    finished.set()
    progress.finish()
    remaining = len(items) - len(done) - progress.judged_now
    status = "PRZERWANE" if STOP.is_set() else "gotowe"
    print(f"{status}: " + json.dumps(counters, ensure_ascii=False))
    if batched:
        print(
            f"paczki: {stats['batches_ok']} poprawnych, {stats['batches_retried']} ponowionych, "
            f"{stats['batches_fallback']} rozbitych na pojedyncze requesty (fallback)"
        )
    print("journal: " + os.path.abspath(args.journal))
    if STOP.is_set() and remaining > 0:
        print(f"zostalo {remaining} itemow - uruchom te sama komende, zeby wznowic", flush=True)
    if counters["failed"]:
        print(
            f"UWAGA: {counters['failed']} itemow bez werdyktu (fail-closed). Ich odpowiedzi sa "
            "zapisane w journalu jako zdarzenia out_of_schema - import je pomija, ale zostaja "
            "jako dowod. Ponowne uruchomienie sprobuje ich jeszcze raz.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
