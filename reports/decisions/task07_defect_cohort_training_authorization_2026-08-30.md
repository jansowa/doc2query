# Autoryzacja treningu na kohorcie par z wadami (2026-08-30)

## Status

**Decyzja właściciela** z 2026-08-30, wymagana przez ADR
[`task06_defect_pair_pipeline_v1.md`](task06_defect_pair_pipeline_v1.md) §7.4:
autoryzacja z 2026-08-28 obejmowała wyłącznie kohorty v3 (`bottom`, `near_miss`),
więc nowa kohorta wymagała osobnej zgody. Właściciel autoryzował trening i
przyjął rekomendację pominięcia intrinsics generatora na rzecz probe embeddera.
`final_tests_used=[]`.

## 1. Co jest autoryzowane

Trzy ramiona (`dpo`, `continued_sft`, `score_weighted_continued_sft`) na kohorcie
`defect_v1` — plan `task07-dpo-plan-defect-v1-s42`, 1 632 pary treningowe
(1 794 pary w kohorcie, podział train/dev po klastrach), logproby referencji
policzone i zwalidowane. Wybory na dev, zapis adapterów i manifestów.

Poza zakresem, bez zmian: zbiory testowe (`final_tests_used=[]`), pule v4–v11,
klasa `answer_leak` odłożona przez audyt anty-skrótowy (AUC 0,8731 > 0,80) i
niewchodząca do treningu bez amendmentu, reinterpretacja bramek v2/v2.1.

## 2. Stan bramek ADR §7 w chwili autoryzacji

| bramka | stan |
|---|---|
| raport pass-rate per klasa | ✅ `artifacts/task06/defect_pairs_v1/summary.json` |
| audyt anty-skrótowy | ✅ wykonany; zadziałał i **zablokował klasę** |
| ślepy spot-check ≥30 par | ❌ **niewykonany** dla tej kohorty |
| osobna autoryzacja właściciela | ✅ ten dokument |

Spot-check pozostaje niewykonany, tak samo jak przy kohorcie v3: właściciel
autoryzował trening bez niego. Zapisuję to wprost, żeby nie wyglądało później na
spełnioną bramkę. Jeśli spot-check wypadnie źle, jest argumentem przeciw parom,
a nie przeciw wynikowi treningu, i wymaga osobnego rozpatrzenia.

## 3. Decyzja o intrinsics generatora

Intrinsics na pełnym subsecie `dev_intrinsic_rank10` **wypadają z planu**.
Powód jest zmierzony, nie uznaniowy: scoring idzie 0,43 wiersza/s, czyli ~21 h na
punkt i ~150 h na siedem punktów. To metryka pomocnicza (AGENTS.md §9.2), więc
taki koszt jest nieproporcjonalny. Generacje dla wszystkich siedmiu punktów
(32 990 wierszy każda) zostają zapisane i mogą posłużyć później, gdyby ktoś chciał
policzyć scoring na próbce.

Kryterium rozstrzygającym Task 07 pozostaje **probe embedder na naturalnych
zamrożonych zapytaniach** i to on jest następnym krokiem po treningu.

## 4. Awaria, która to poprzedziła (zapisana, bo dotyczy wiarygodności okna)

W oknie bezobsługowym 2026-08-29/30 kolejka intrinsics nie wyprodukowała ani
jednego wyniku: `evaluate generator` był wołany bez configów zamrożonych sędziów,
więc padał na scoringu, a strażnik wskrzeszał go w pętli. Naprawione
(commit `a4eb625`): sędziowie podawani jawnie i sprawdzani na starcie, strażnik
ma licznik wskrzeszeń. Pipeline wad w tym samym oknie przeszedł w całości.
