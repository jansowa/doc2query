# Pomiar: metrologia przyrządu probe — wariancja w obrębie jednego ramienia (2026-08-17)

Kontrakt kolejki: `configs/probe_variance_queue_2026-08-17.tsv`.
Artefakty: `runs/task04_probe_variance_v1/PROBE-VAR-W06-4.5B-S47…S51`,
`reports/measurements/task04/probe_within_arm_variance_v1.json`.
Skrypt: `scripts/measure_probe_within_arm_variance.py`.
Guardrail zbieżności: zamrożony M-03
(`configs/evaluation/task04_m03_probe_convergence_guardrail_v1.yaml`), użyty bez
zmian przez `src/doc2query/evaluation/probe_convergence.py`.

Ten pomiar dotyczy **przyrządu, nie ramion**. Pięć runów użyło identycznego
wejścia (baseline W06) i identycznych hiperparametrów co zadania M-03 z kolejki
2026-08-14; różni je wyłącznie seed (47–51). Nie powstała żadna różnica
Hybrid−W06, więc nic nie mogło zostać wypromowane ani zdegradowane. Nie dotknięto
żadnego zamrożonego artefaktu Tasków 04–05. `final_tests_used=[]`.

## Wynik

| run | `corpus_recall_at_100` | `corpus_ndcg_at_10` | strata (pierwsza → ostatnia) | zbieżny |
|---|---|---|---|---|
| S47 | 0,114288 | 0,047456 | 3,814 → 1,094 | tak |
| S48 | 0,124700 | 0,049891 | 2,375 → 0,105 | tak |
| S49 | 0,136552 | 0,052183 | 2,788 → 0,422 | tak |
| **S50** | **0,002388** | **0,000652** | **1,036 → 2,121** | **nie** |
| S51 | 0,143264 | 0,058183 | 1,741 → 0,997 | tak |

Podłoga zbieżności zastosowana przez guardrail: 0,062350 (poziom losowy
0,0028616). S50 leży poniżej podłogi i praktycznie na poziomie losowym.

### Zapadnięcia są prawidłowością, nie pechem

- w tej serii: **1/5 = 20%**;
- łącznie z 22 runami kalibracji M-03: **5/27 = 18,5%**.

Replikaty różnią się **tylko seedem**, więc zapadnięcie nie jest własnością
ramienia, danych ani budżetu — jest własnością procedury treningu probe. Przy
18,5% ryzyka na run, porównanie dwóch ramion na trzech seedach (6 runów) ma
około **71%** szans, że co najmniej jeden run w nim się zapadnie
(`1 − 0,8148⁶ = 0,707`). To wyjaśnia niestabilność wyników Tasków 04–05 lepiej niż
jakakolwiek hipoteza o generatorze.

### Wariancja wewnątrz ramienia jest znośna — korekta wcześniejszej oceny

Dla czterech zbieżnych runów: średnia `corpus_ndcg_at_10` = **0,051928**,
sd = **0,004595**, współczynnik zmienności **8,8%**. Wynikające półszerokości
95% CI: 0,0052 (n=3), 0,0040 (n=5), 0,0029 (n=10).

To istotnie mniej niż sd **0,0126** par Hybrid−W06 z kalibracji M-03, na której
opierało się wcześniejsze podejrzenie, że przyrząd zasadniczo nie ma
rozdzielczości na próg `+0,01`. Wniosek trzeba skorygować: przy tej wariancji
próg **jest** w zasięgu już przy 3–5 seedach. Zastrzeżenie: obie estymacje
pochodzą z n=4 i są bardzo szumne, więc różnicy 2,5× nie należy traktować jako
ustalonej. Ostrożna wersja wniosku brzmi: **problemem nie jest szum wokół
średniej, a jednostronne odchyłki wnoszone przez zapadnięcia.**

Ilustracja wagi tego rozróżnienia: zapadnięty run kontrolny W06-S43 wytworzył
różnicę per-seed **+0,1024**, czyli o rząd wielkości większą od mierzonego efektu,
i to on wyniósł nagłówek TriviaQA confirm na +0,0479.

### Kierunek zmiany straty jako darmowy detektor: swoisty, ale niedoczuły

We wszystkich czterech zbieżnych runach strata zmalała, a w zapadniętym wzrosła.
Reguła `last_loss >= first_loss` sprawdzona na **wszystkich 27** runach
(22 kalibracji M-03 + 5 tutaj):

| miara | wynik |
|---|---|
| czułość (wykryte zapadnięcia) | **2/5 = 0,40** |
| fałszywe alarmy | **0/22 = 0,00** |

Wykryte: `PROBE-VAR-W06-4.5B-S50` oraz — co istotne —
`D01B-TRIVIA-CONFIRM-W06-4.5B-S43`, czyli dokładnie ten run, który zniekształcił
nagłówek confirmu. Przeoczone trzy zapadnięcia miały malejącą stratę
(`PROBE-BUDGET-1024-HYBRID-S44` 1,976 → 0,082; `PROBE-BUDGET-1024-W06-S42`
4,013 → 0,677; `PROBE-BUDGET-2048-HYBRID-S43` 0,927 → 0,397).

Jest to zgodne z ustaleniem M-03, że **poziom** straty jest słabym sygnałem
(`r = −0,199`), i uzupełnia je: **kierunek** zmiany straty jest sygnałem
swoistym, ale wychwytuje tylko część zapadnięć. Nadaje się więc jako tani
fail-fast, nie jako zamiennik guardraila retrievalowego.

## Rekomendacja: detekcja i auto-reseed zamiast dokładania seedów

Wnioskiem operacyjnym **nie** jest „uruchamiać więcej seedów”, bo to skaluje
koszt liniowo, nie usuwając przyczyny, i nadal wpuszcza zapadnięte runy do
agregatów. Kolejność powinna być:

1. **Wykrywać zapadnięcie w trakcie runu, nie po nim.** Dziś guardrail działa na
   artefaktach końcowych, bo `embedder_probe.py` utrwala tylko `losses[0]`
   i `losses[-1]`. Utrwalenie krzywej straty i pośredniej ewaluacji retrievalowej
   na checkpoincie (np. co 256 kroków) pozwoliłoby przerwać run po kilku minutach
   zamiast po ~22.
2. **Automatyczny reseed po wykryciu**, z jawnym zapisem w provenance ile razy i
   z jakim seedem — tak, aby liczba prób nigdy nie była ukryta.
3. **Dopiero potem zwiększać liczbę seedów**, gdy odsetek zapadnięć jest znany
   i kontrolowany.

Żaden z tych kroków nie jest tym raportem autoryzowany: zmiana
`embedder_probe.py` dotyka narzędzia użytego w zamkniętych pomiarach i wymaga
prospektywnego ADR z rachunkiem, które wyniki przestają być porównywalne.

## Granice

- Pięć runów jednego ramienia; wszystkie statystyki oparte na n=4 zbieżnych
  runach są szumne i nie zastępują kalibracji M-03.
- Pomiar nie tworzy porównania ramion i nie zmienia statusu żadnego ramienia.
- Progów guardraila M-03 nie kalibrowano ani nie zmieniano; użyto ich w wersji
  zamrożonej.
- `final_tests_used=[]`.
