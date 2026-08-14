# Task 06 — amendment v2: polityka niepoprawnych completions (2026-08-14)

## Zdarzenie

Run `same_prompt_expansion_v2` przerwał się 2026-08-13 o 22:22:54 po 3559/4000
wygenerowanych kandydatach:

```
File "src/doc2query/models/templates.py", line 41, in normalize_completion
    raise ValueError("query completion must be a single line")
```

Wiersz o indeksie 3559 to slot 7 (temperatura 1.2, top_p 0.97), rekord 59 —
model zwrócił completion zawierający znak nowej linii. `normalize_completion`
słusznie tego nie akceptuje, ale `generate_same_prompt_expansion` wywoływał ją
bez obsługi wyjątku, więc pojedynczy zdegenerowany sample zabijał cały run.
Ryzyko ujawniło się dopiero w v2, bo v1 nie miał slotów cieplejszych niż 1.0.

Uruchomienie było bezobsługowe, z `; systemctl poweroff`, więc komputer wyłączył
się o 22:23:01 zgodnie z projektem — także po błędzie. Journal zachował 3559
wierszy, nic poza przerwanym batchem nie przepadło.

## Decyzja

Etap generacji same-prompt przyjmuje **tę samą politykę, którą ma zamrożony
pipeline D01 z Task 05** (`d01_pipeline`): niepoprawny completion jest liczony
jako `invalid` i **resamplowany na nowym seedzie**, a nie przycinany od razu i
nie przerywa runu.

- `SAME_PROMPT_MAX_ATTEMPTS = 4` — tyle samo, co `max_attempts_per_slot: 4` w
  designie Task 06;
- seed próby *n* to `seed_bazowy + n * 7_000_000`, co jest deterministyczne i
  rozłączne z przestrzenią seedów slotów (stride 1000 na rekord, offsety 1–14);
- pierwsza próba zachowuje **dokładnie** dotychczasowy seed, więc żaden już
  wygenerowany wiersz nie zmienia się i wznowienie nie traci pracy;
- po wyczerpaniu czterech prób wiersz zachowuje **pierwszą niepustą linię**
  ostatniej próby (`format_repair="first_line"`); gdy nie ma żadnej niepustej
  linii, zapisywany jest pusty tekst (`format_repair="empty"`), który w scoringu
  wypada na wszystkich metrykach formatu;
- każdy wiersz zapisuje `attempt`, `invalid_attempts` i `format_repair`;
  summary runu podaje `max_attempts_per_slot`, `retried_row_count`,
  `invalid_completion_count` i `format_repair_counts`.

## Dlaczego to nie narusza zamrożonych kontraktów

- `identity_sha256` generacji **nie zmienia się**: polityka prób jest w kodzie,
  a tożsamość obejmuje kontrakt, config, kohortę, adapter, kontrolki, decoding i
  batch. Dzięki temu 3559 wierszy wznawia się bez utraty ~23 min GPU.
- Kohorta, prompty, kontrolki i decoding pozostają nietknięte; zmienia się
  wyłącznie zachowanie wobec zdegenerowanego sampla.
- Pierwsze 3559 wierszy powstało przed poprawką, ale wszystkie przeszły
  `normalize_completion` bez naprawy, więc mają `attempt=1` i brak flag; zbiór
  pozostaje jednorodny, a każda ewentualna naprawa w pozostałych 441 wierszach
  jest jawnie oznaczona i raportowana.
- Bramka różnorodności i jej progi pozostają bez zmian. Naprawiony wiersz nie
  jest w żaden sposób uprzywilejowany: dla bramki jest zwykłym kandydatem, a dla
  scoringu kandydatem o gorszym formacie.
- Żadna decyzja o parach nie jest tym amendmentem podejmowana;
  `tentative_pair_build_authorized=false`, `final_tests_used=[]`.

## Walidacja

Dwa nowe testy CPU: resample malformed completion na nowym seedzie (z kontrolą
`attempt`, `invalid_attempts`, `seed`) oraz naprawa dopiero po wyczerpaniu
czterech prób (z kontrolą, że urwany ogon nie wchodzi do tekstu). Ruff,
`mypy src`, pełny pytest `516 passed`.
