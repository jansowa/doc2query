# Pomiar: detekcja zapadnięcia probe w trakcie runu i automatyczny reseed (2026-08-21/22)

ADR (zamrożony **przed** pierwszym runem):
[`task04_m03_in_run_collapse_detection_v1.md`](../decisions/task04_m03_in_run_collapse_detection_v1.md).
Kontrakt progów: `configs/evaluation/task04_m03_in_run_collapse_detection_v1.yaml`.
Implementacja: `src/doc2query/evaluation/probe_in_run_collapse.py`,
haki w `src/doc2query/evaluation/embedder_probe.py`, flaga
`scripts/train_probe_embedder.py --collapse-detection-config`.
Kolejka: `configs/probe_inrun_collapse_queue_2026-08-21.tsv`.
Artefakty: `runs/task04_probe_inrun_collapse_v1/*`,
`reports/measurements/task04/in_run_collapse_detection_v1/summary.json`.
Skrypt pomiarowy: `scripts/measure_probe_inrun_collapse_detection.py`.

Pomiar dotyczy **przyrządu**, nie ramion. Pięć runów użyło identycznego wejścia
(baseline W06) i identycznych hiperparametrów co seria wariancji S47–S51; różni
je wyłącznie seed oraz obecność detekcji. Nie powstała żadna różnica
Hybrid−W06, więc nic nie mogło zostać wypromowane ani zdegradowane. Katalogi
`runs/task04_probe_variance_v1/` pozostały nietknięte, progów nie zmieniano,
`final_tests_used=[]`, `task07_training_authorized=false`.

## Wynik: wszystkie prerejestrowane kryteria przeszły

| Kryterium | Treść (skrót) | Wynik |
|---|---|---|
| A1 | flaga wyłączona nie zmienia artefaktów | **pass** |
| A2 | znane zapadnięcie wykryte ≤ krok 768, reseed, run przyjęty zbieżny | **pass** |
| A3 | zero fałszywych alarmów na trzech zdrowych seedach | **pass** |
| A4 | trajektoria z detekcją identyczna jak bez | **pass** |
| A5 | liczba prób i seedy jawne w provenance | **pass** |

| run | seed → efektywny | próby | wykrycia | recall pośredni (256/512/768) | `corpus_recall_at_100` | `corpus_ndcg_at_10` |
|---|---|---|---|---|---|---|
| `INRUN-CONTROL-S47-OFF` | 47 → 47 | 1 | — (detekcja wyłączona) | — | 0,114288 | 0,047456 |
| `INRUN-S47-ON` | 47 → 47 | 1 | 0 | 0,844 / 0,945 / 0,922 | 0,114288 | 0,047456 |
| `INRUN-S48-ON` | 48 → 48 | 1 | 0 | 0,891 / 0,977 / 0,945 | 0,124700 | 0,049891 |
| `INRUN-S50-ON` | 50 → **2050** | **3** | **2** | 0,922 / 0,961 / 0,969 | 0,161955 | 0,083220 |
| `INRUN-S51-ON` | 51 → 51 | 1 | 0 | 0,945 / 0,953 / 0,969 | 0,143264 | 0,058183 |

Zamrożony guardrail M-03, zastosowany post hoc do tej serii (podłoga 0,062350),
uznał **wszystkie pięć** runów za zbieżne.

## A1 i A4: zmiana narzędzia nie ruszyła istniejącej ścieżki

Run kontrolny z wyłączoną flagą odtworzył zamrożony `PROBE-VAR-W06-4.5B-S47`
**bit w bit**: `first_loss` 3,8139052391052246, `last_loss` 1,09434974193573,
`corpus_recall_at_100` 0,11428770414385407, `corpus_ndcg_at_10`
0,04745608941152794 — te same cyfry co w artefakcie z 2026-08-17. Zestaw plików
i zestaw kluczy `train_summary.json` oraz `result.json` są identyczne; nie
powstał żaden plik detekcji.

Ponieważ niedeterminizm GPU okazał się **zerowy** na tej ścieżce, kryterium A4
stało się najostrzejsze z możliwych: dopuszczalna różnica wynosiła 0. Run
`INRUN-S47-ON` z włączoną detekcją dał wszystkie cztery liczby **dokładnie
równe** runowi kontrolnemu. Zapisywanie i przywracanie stanu RNG wokół kontroli
pośrednich rzeczywiście nie zaburza trajektorii treningu.

Potwierdzenie uboczne, mocniejsze niż wymagało kryterium: `INRUN-S48-ON`
(0,049891 / 0,124700) i `INRUN-S51-ON` (0,058183 / 0,143264) odtworzyły co do
cyfry wartości zamrożonych S48 i S51 z serii wariancji, mimo włączonej detekcji.

## A2: detekcja zadziałała, i to dwa razy pod rząd

Seed 50 to znane zapadnięcie z serii wariancji (`corpus_recall_at_100` 0,0024,
praktycznie poziom losowy). Przebieg runu:

| próba | seed | wynik | krok wykrycia | recall pośredni (256 / 512) | reguła |
|---|---|---|---|---|---|
| 0 | 50 | zapadnięcie | 512 | 0,336 / 0,422 | podłoga retrievalowa |
| 1 | 1050 | zapadnięcie | 512 | 0,289 / 0,141 | podłoga retrievalowa |
| 2 | 2050 | przyjęta | — | 0,922 / 0,961 | — |

Reseed wykorzystał dwie z trzech dozwolonych prób — zapasu było dokładnie tyle,
ile trzeba, co jest samo w sobie sygnałem: przy 18,5% ryzyka na run szansa dwóch
zapadnięć z rzędu to ~3,4%, więc albo trafiliśmy na ogon, albo ryzyko przy tym
budżecie jest wyższe od zmierzonego wcześniej. Trzech prób nie zwiększam — to
wymagałoby nowego ADR i kalibracji, której nie ma.

**Separacja sygnału jest czysta.** Kontrole pośrednie zapadniętych przebiegów
dały 0,141–0,422, zdrowych 0,844–0,977, przy podłodze 0,521. Między
najwyższym zapadnięciem a najniższym zdrowym jest przerwa **2×**, bez ani
jednego przypadku pośredniego na 15 zmierzonych kontrolach.

### Podłoga wyszła 0,521, nie 0,195 — i to jest zgodne z kontraktem

ADR zakładał korpus pośredni 2048 dokumentów. Faktyczna pula ma **768**
dokumentów, bo 3072 pary treningowe mają K=4 zapytania na pasaż, czyli tylko 768
unikalnych pozytywów. Zamrożona reguła mówi „2048 albo wszystkie, jeśli jest ich
mniej", więc zadziałała dokładnie jak zapisano, a podłoga policzyła się z
faktycznego rozmiaru puli: `4,0 × 100/768 = 0,5208`. Progu nie zmieniano ani
przed, ani po odczycie. Skutek uboczny jest wart odnotowania: mniejsza pula daje
**wyższą** i przez to bardziej agresywną podłogę, więc kryterium A3 było
trudniejsze do zdania, niż zakładał ADR — i mimo to przeszło.

### Oszczędność czasu

Zapadnięty przebieg kosztował **99 s** treningu zamiast pełnego runu (~1500 s):
detekcja odcina kodowanie korpusu (~735 s) i skan retrievalowy. Dwie przerwane
próby kosztowały łącznie 198 s zamiast ~3000 s. Całe zadanie `inrun_s50_on`
zamknęło się w **1750 s** wobec ~4500 s, jakie kosztowałyby trzy pełne runy —
oszczędność ~46 minut na jednym zadaniu.

Koszt własny mechanizmu jest niski: trzy kontrole pośrednie po ~3,4 s, łącznie
**10,1–10,8 s** na run, czyli ~5% czasu treningu i ~0,7% czasu całego runu.
ADR szacował ~35 s (~19% treningu); realny narzut jest trzykrotnie mniejszy, bo
pula pośrednia ma 768, a nie 2048 dokumentów.

## A3: zero fałszywych alarmów, ale próba jest mała

Na trzech zdrowych seedach (47, 48, 51) wykonano 9 kontroli pośrednich i żadna
nie trafiła — ani regułą retrievalową, ani kierunkiem straty. To zero na 9
kontrolach i zero na 3 runach; **nie jest to dowód swoistości**, tylko brak
sygnału przeciw niej na próbie tej wielkości. Górna granica 95% CI dla odsetka
fałszywych alarmów przy 0/3 runach wynosi ~0,63, więc liczba mówi bardzo mało i
tak ma być cytowana.

### Kierunek straty wypadł słabiej niż reguła retrievalowa

Na dwóch zapadnięciach tej serii `loss_non_decreasing` trafił tylko raz:

- próba seed 50: strata niemalejąca na obu kontrolach (reguła by zadziałała);
- próba seed 1050: strata **malejąca** na obu kontrolach, mimo recallu 0,141 —
  reguła straty **przeoczyła** to zapadnięcie w całości.

Jest to zgodne z wcześniejszym ustaleniem (czułość 2/5 dla reguły końcowej) i
potwierdza rozdział ról zapisany w ADR: kierunek straty jest tanim fail-fastem,
a nie guardrailem zbieżności. Gdyby detekcja opierała się tylko na stracie,
połowa zapadnięć tej serii przeszłaby dalej. `loss_based_guardrail_permitted`
pozostaje `false`.

## Czego ten pomiar nie pokazuje

- **Reseed jest selekcją i już widać jej cenę.** Przyjęty przebieg seeda 2050
  dał `corpus_ndcg_at_10` **0,083220**, czyli wyraźnie powyżej całego zakresu
  serii zamrożonej (0,0475–0,0582). To nie jest poprawa ramienia — to jeden
  przebieg wylosowany warunkowo („nie wykryto zapadnięcia") i akurat z dobrego
  ogona. Skutek widać w agregacie: średnia `corpus_ndcg_at_10` tej serii to
  0,0572 przy sd **0,0152**, wobec sd 0,0046 czterech zbieżnych runów serii
  wariancji. Średnich tej serii nie wolno używać jako oszacowania ramienia W06.
- Detekcja **nie zastępuje** guardraila M-03. Zapadnięcia, które przeżyją
  kontrole pośrednie, nadal musi wyłapać filtr na artefaktach końcowych ze
  sparowanym odrzucaniem seedów.
- Pomiar nie tworzy porównania ramion, nie zmienia statusu żadnego ramienia i
  nie autoryzuje treningu Task 07.
- Zamknięte pomiary (kalibracja M-03, confirm TriviaQA, sweep budżetu, seria
  wariancji) **nie zostały przeliczone ani unieważnione**; ich artefakty są
  nietknięte. Runów z detekcją nie wolno mieszać w jednym porównaniu z runami
  bez detekcji.

## Uwagi wykonawcze

- Zużycie GPU: 6292 s (~1,75 h) na pięć zadań, poniżej limitu ośmiu na partię.
- **Wznawialność potwierdzona w praktyce, nie tylko w testach.** Maszyna została
  wyłączona w trakcie zadania `inrun_s47_on`; po ponownym uruchomieniu kolejka
  pominęła dwa ukończone zadania, a przerwany run dokończył się w **270 s**,
  odtwarzając zbuforowane embeddingi korpusu i pomijając trening. Dziennik prób
  przetrwał wyłączenie i przerwana próba nie została powtórzona.
- Doprecyzowanie dopisane do ADR **przed** pierwszym runem z włączoną detekcją
  (biegł wtedy wyłącznie run kontrolny): remisy w pośrednim Recall@100 są
  rozstrzygane pesymistycznie. Bez tego całkowicie zapadnięty enkoder, który
  odwzorowuje wszystkie teksty na jeden wektor, dostałby recall 1,0 i detektor
  byłby ślepy na awarię, dla której powstał.
- 16 testów CPU (`tests/test_task04_in_run_collapse.py`), pełny pytest 751
  przechodzi, ruff czysty, mypy bez nowych błędów, CI bez GPU i bez pobierania
  modeli.

`final_tests_used=[]`. `task07_training_authorized=false`.
