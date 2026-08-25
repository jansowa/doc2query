# Pomiar: bramka V2.1-05 na ukończonym audycie par v2.1 (2026-08-25)

Polityka i predykcje zamrożone **przed zbudowaniem pierwszej pary v2.1**:
[`task06_defect_pair_policy_v2_1.md`](../decisions/task06_defect_pair_policy_v2_1.md).
Moduł liczący: `src/doc2query/preferences/defect_pair_gate_v2_1.py` (napisany
**przed** odczytem, odmawia niedokończonego audytu). Artefakty:
`artifacts/task06/preference_audit_v4_defect_pairs_v2_1/groq_dual_llm/{analysis.json,pair_verdicts.jsonl,gate_v2_1_05.json}`.

Audyt: **`complete`**, 400/400 requestów u obu sędziów, 800 par, 1 600 ocen, zero
par bez werdyktu, trzy okna dziennych budżetów Groq (24.08 ×2, 25.08 domknięcie 23
requestami), bez zmiany promptu, rubryki, modeli i limitów.

## Werdykt: bramka **niezdana** — blokuje P3 ze statusem `inconclusive`

| # | predykcja | próg | wynik | CP 95% | werdykt |
|---|---|---|---|---|---|
| P1 | nieodpowiadalne `chosen`, guardrail | ≤ 5% | 41/800 = **5,12%** (`gpt-oss`), 46/800 = **5,75%** (`qwen`) | [3,91%; 6,60%] / [4,46%; 7,29%] | **guardrail nie zapalił** |
| P2 | `consensus_supports_automatic` | ≥ 30% | **45,12%** | [42,19%; 48,09%] | **PASS** |
| P3 | `consensus_contradicts_automatic` | ≤ 3,1% | 17/800 = **2,12%** | [1,36%; **3,17%**] | **INCONCLUSIVE** |
| P4' | kontrast w parze (`rejected` − `chosen`) | ≥ +20 pp | **+38,2 pp** / **+55,5 pp** | [34,6; 41,9] / [51,5; 59,4] | **PASS** |
| P5 | remisy (bez progu) | — | 47,5% / 21,1% | — | n/d |

Konsensus: wspiera 361, przeczy 17, rozbieżność 21, abstencja 401.

Zgodnie z §4.4 ADR `INCONCLUSIVE` jest **fail-closed**: pary v2.1 **nie idą do
żadnego treningu**, `task07_training_authorized=false` bez zmian, a jedyną
dopuszczalną reakcją jest nowy prospektywny ADR. Progów nie zmieniono, audytu nie
powtórzono, próbki nie rozszerzono.

## Dlaczego to nie jest ta sama porażka co w v2.0

**Brakuje 0,0704 pp na granicy przedziału** — górna granica CP dla 17/800 wynosi
3,1704% wobec progu 3,1%. Przy 16 parach byłoby 3,0219%, czyli PASS. Znowu **o jedną
parę** — ale różnica wobec v2.0 jest zasadnicza i po to reguła była przepisana:

- v2.0 orzekło **FAIL** na podstawie punktu (3,20% wobec 3,1%), czyli twierdziło, że
  polityka jest zła, choć próba tego nie rozstrzygała;
- v2.1 orzeka **INCONCLUSIVE**, czyli mówi prawdę: *ta próba nie rozstrzyga tego
  progu*. Polityka nie jest sfalsyfikowana, a bramka nie jest zdana.

## To był przewidziany scenariusz, z liczbą zapisaną z góry

ADR §4.6 zapisał analizę wrażliwości **przed odczytem**: „P3 jest jedyną predykcją,
która może zakończyć się `INCONCLUSIVE` z realnym prawdopodobieństwem, i to wyłącznie
wtedy, gdy prawdziwy udział sprzeczności jest istotnie wyższy niż w osi A audytu v2.
Taki wynik będzie sygnałem, że wartości osi A były optymistyczne przez selekcję
podpróby — i tak zostanie opisany, bez zmiany progu."

Dokładnie to się stało:

| | założenie planistyczne (oś A audytu v2) | zmierzona prawda |
|---|---|---|
| P3 sprzeczności | 1,30% → moc **0,964** | **2,12%** → moc **0,467** |
| P1 nieodpowiadalne `chosen` | 3,90% / 3,25% | **5,12% / 5,75%** |

Podpróba osi A z przegranej bramki v2.0 była więc **optymistyczna na obu wymiarach**.
Przy n=800 i prawdzie 2,12% moc dla P3 to 0,467, czyli rzut monetą — i wypadł po
stronie nierozstrzygnięcia. Nie jest to niespodzianka ani pech: to zapisany z góry
tryb awarii, który się zmaterializował.

## Co jednak zostało potwierdzone

- **Mechanizm osi defektowych działa, trzeci raz z rzędu i na największej próbce**:
  kontrast w parze to +38,2 pp i +55,5 pp wobec progu +20 pp, u obu sędziów, z CI
  daleko od progu.
- **Zgodność z automatem wzrosła**: konsensus wspiera automat w **45,12%** par wobec
  30,80% w v2.0 i 24,40% w v1 — czyli jednoosiowa polityka na osi A jest wyraźnie
  bardziej zgodna z niezależnymi sędziami niż obie poprzednie.
- **Sprzeczności spadły**: 2,12% wobec 3,20% (v2.0) i 6,20% (v1).
- **Filtr odpowiadalności trzyma poziom**: 5,12% / 5,75% wobec 16,6% / 18,8% w v1,
  czyli nadal ~3× lepiej, choć gorzej niż 4,80% / 5,20% w v2.0 i gorzej niż zakładała
  oś A. Guardrail nie zapalił się, ale **nie wolno tego raportować jako „P1
  przeszła"**: przy n=800 moc konfirmacyjna P1 to 0,39/0,76, a punktowe udziały leżą
  **powyżej** progu 5%.

## Wejście projektowe do decyzji właściciela (bez rekomendacji progu)

Liczby potrzebne do ewentualnego v2.2, policzone na zmierzonej prawdzie 2,125%:

| cel | wymagane n | dopuszczalne sprzeczności | koszt Groq |
|---|---|---|---|
| moc 0,80 dla P3 | **1 760 par** | ≤ 42 | 4–5 okien |
| moc 0,90 dla P3 | **2 360 par** | ≤ 59 | 6–7 okien |

Twarde ograniczenie podaży: populacja to **2 253 pary**, z czego **800 już
oglądanych**, więc nieoglądany zapas to **1 453 pary**. Próba 1 760 par nie da się
złożyć z samego zapasu, a §4.1 ADR **zakazuje eskalacji** (dolewania par do już
odczytanej próbki), więc v2.2 musiałoby albo:

1. zmierzyć P3 na całym zapasie 1 453 par (moc ~0,74 przy prawdzie 2,125%), albo
2. rozszerzyć kohorty v4–v11 — co jest zamknięte do **pozytywnej** bramki, czyli
   wymaga świadomego zdjęcia tego warunku, albo
3. uznać, że progu 3,1% nie da się rozstrzygnąć tą populacją, i przeprojektować samą
   predykcję P3 — z progiem wyprowadzonym z czegoś innego niż „połowa spadku wobec
   v1".

**P1 jest przy tym już nierozstrzygalny definitywnie**: przy prawdzie 5,12% i progu
5% żadne n nie da przejścia konfirmacyjnego, bo prawda leży po niewłaściwej stronie
progu. Guardrail pozostaje jedyną sensowną formą tej predykcji.

Wybór należy do właściciela i wymaga nowego prospektywnego ADR **przed** kolejnym
odczytem. Ten raport niczego nie autoryzuje i nie jest human evidence.

`task07_training_authorized=false`, `final_tests_used=[]`.
