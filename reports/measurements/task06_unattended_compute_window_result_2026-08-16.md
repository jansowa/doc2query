# Wynik okna obliczeń bezobsługowych (2026-08-14 → 2026-08-16)

Realizacja ADR
[`task06_unattended_compute_window_2026-08-14.md`](../decisions/task06_unattended_compute_window_2026-08-14.md).
Kolejka wyprodukowała wyłącznie artefakty pomiarowe; żadna decyzja, promocja ani
agregacja nie została podjęta automatycznie. `final_tests_used=[]`.

## Przebieg

| pozycja | wartość |
|---|---|
| start / koniec | 2026-08-14 17:54:49 → 2026-08-16 19:05:25 |
| zadania | **25/25 ukończonych, 0 awarii** (25 prób na 25 zadań) |
| czas GPU | 49,18 h |
| restarty nadzorcy | brak — jeden ciągły przebieg; 7 późniejszych wpisów `queue start` to no-opy crona po wyczerpaniu kolejki |
| limit poboru | 160 W (ustawiony przez właściciela przed wyjściem) |
| dysk | 83 GB → 58 GB wolnego (zużyto ~21 GB, bramka 20 GB nietknięta) |
| wyłączenie | strażnik wyłączył maszynę 2026-08-16 o 20:45, po 4799 s stabilnego stanu wyczerpania |

Watchdog crona nie musiał niczego ratować, ale zadziałał zgodnie z projektem:
po restarcie maszyny o 21:31 wznowił kolejkę o 21:34, ta pominęła wszystkie
ukończone zadania i zakończyła się czysto.

## Kohorty same-prompt v3–v11 (9 × 3000 pasaży)

Wszystkie dziewięć kohort ma komplet: 24000 wygenerowanych i 24000 ocenionych
kandydatów (`status: measured`), po czym bramkę o **niezmienionych** progach.

| kohorta | grupy | eligible | % | resample | naprawy przez ucięcie |
|---|---|---|---|---|---|
| v3 | 3000 | 2791 | 93,0% | 12 | 0 |
| v4 | 3000 | 2817 | 93,9% | 18 | 0 |
| v5 | 3000 | 2823 | 94,1% | 17 | 0 |
| v6 | 3000 | 2803 | 93,4% | 18 | 0 |
| v7 | 3000 | 2768 | 92,3% | 9 | 0 |
| v8 | 3000 | 2804 | 93,5% | 15 | 0 |
| v9 | 3000 | 2786 | 92,9% | 18 | 0 |
| v10 | 3000 | 2780 | 92,7% | 22 | 0 |
| v11 | 3000 | 2792 | 93,1% | 24 | 0 |
| **v3–v11** | **27000** | **25164** | **93,2%** | 153 | **0** |

Z wcześniejszymi kohortami: v1 362/500 (72,4%, wąski decoding), v2 466/500
(93,2%), co daje łącznie **25992 grup** kwalifikujących się do budowy par —
o dwa rzędy wielkości powyżej rozwojowego progu 500 par i powyżej progu 1000 par
przed finalnym DPO.

Dwie obserwacje metodologiczne:

- odsetek przejść bramki jest bardzo stabilny między niezależnymi kohortami
  (92,3–94,1%, rozstęp 1,8 pp na 9 kohortach), co potwierdza, że różnica wobec
  v1 (72,4%) wynika z szerszego rozkładu decodingu, a nie z losowości kohorty;
- polityka resamplingu niepoprawnych completions sprawdziła się dziewięć razy:
  153 completions przesamplowano, a **żaden** wiersz nie wymagał naprawy przez
  ucięcie do pierwszej linii. Awaria, która zabiła pierwszy run v2, nie
  powtórzyła się ani razu.

## M-03: seedy 45 i 46 confirmu TriviaQA

Cztery treningi probe ukończone (oba ramiona × dwa seedy). Surowe
`corpus_ndcg_at_10` z podsumowań runów:

| ramię | seed 45 | seed 46 |
|---|---|---|
| W06 | 0,04976 | 0,06954 |
| Hybrid | 0,06846 | 0,06662 |

**To nie jest agregat i nie jest reinterpretacją confirmu.** Zamrożony
`compare` ma na twardo przypięte seedy `[42, 43, 44]` i liczy metrykę decyzyjną
inaczej (agregacja per-query przed bootstrapem zapytań), więc pięcioseedowa
agregacja wymaga osobnego configu i decyzji właściciela.

## Sweep budżetu probe (diagnostyka M-01/M-03) — najważniejszy wynik

Dwanaście treningów: oba ramiona × budżety 1024 i 2048 par × seedy 42/43/44.

| run | first_loss | last_loss | corpus_ndcg_at_10 |
|---|---|---|---|
| 1024 Hybrid S42 / S43 / S44 | 2,51 / 2,30 / 1,98 | 0,0001 / 0,071 / 0,082 | 0,0445 / 0,0622 / **0,0099** |
| 1024 W06 S42 / S43 / S44 | 4,01 / 1,22 / 2,10 | 0,677 / 0,421 / 0,000 | **0,0001** / 0,0246 / 0,0284 |
| 2048 Hybrid S42 / S43 / S44 | 4,65 / 0,93 / 3,11 | 0,253 / 0,397 / 0,411 | 0,0821 / **0,0011** / 0,0826 |
| 2048 W06 S42 / S43 / S44 | 2,86 / 1,34 / 1,68 | 0,047 / 0,169 / 0,700 | 0,0778 / 0,0389 / 0,0558 |

Dwa wnioski, oba istotne dla dalszych decyzji:

1. **Wariancja między seedami jest ogromna i bywa większa niż mierzone
   efekty.** Przy stałym ramieniu i stałym budżecie wyniki rozjeżdżają się od
   0,0011 do 0,0826 (2048 Hybrid). To rząd wielkości większy rozrzut niż
   przewaga `+0,0479`, którą raportuje zamknięty confirm.
2. **Strata treningowa nie jest guardrailem zbieżności.** Korelacja
   `last_loss` z `corpus_ndcg_at_10` wynosi `r = −0,199` (n=12); runy z
   `last_loss ≈ 0,0001` dają zarówno 0,0445, jak i 0,0284, a run z
   `last_loss = 0,397` daje 0,0011. Wykrywanie „seeda niezbiegniętego” musi
   opierać się na sygnale retrievalowym, nie na stracie.

Konsekwencja dla programu: CI zamkniętego confirmu jest bootstrapem **zapytań**
przy `resample_training_seeds: false`, więc konstrukcyjnie nie wyraża
niestabilności seedów, którą właśnie zmierzono w sąsiednim reżimie budżetowym.
Zanim probe zostanie użyty jako instrument selekcji finalistów (M-01), trzeba
zdefiniować guardrail zbieżności oparty na retrievalu i rozstrzygnąć, czy
niepewność seedowa ma wejść do przedziału decyzyjnego. Ten raport tego **nie**
rozstrzyga i niczego nie promuje ani nie unieważnia.

## Co pozostaje niewykonane

Budowa par `chosen/rejected` (wymaga zamrożenia polityki wag, progów i
kalibracji), audyt dual-LLM, Task 07 oraz scoring GPU kohorty teachera z
równoległego okna tokenowego. `task07_training_authorized=false`,
`tentative_pair_build_authorized=false`.
