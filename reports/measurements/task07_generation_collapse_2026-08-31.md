# Kolaps różnorodności generacji po DPO (2026-08-31)

## Status

Pomiar **uboczny, ale nieplanowany**: powstał przy generowaniu wejść probe, nie
jako osobny eksperyment. Liczby pochodzą z podsumowań generacji
(`runs/task07_probe_gen_v1/*/generated.summary.json`), czyli z artefaktów, nie z
przebiegu ręcznego. Kohorta probe (496 pasaży × 4 kontrolki = 1 984 sloty) jest
rozłączna z kohortami treningowymi. `final_tests_used=[]`.

Nie jest to kryterium rozstrzygające Task 07 — tym pozostaje probe embedder na
naturalnych zamrożonych zapytaniach. Jest to natomiast **pomiar zachowania
generatora**, którego dotąd nie mieliśmy, i który tłumaczy mechanizm stojący za
metrykami dev.

## 1. Wynik

Każdy punkt dostał ten sam budżet: 1 984 sloty, ten sam decoding, ten sam seed,
maksymalnie 3 próby na slot. Slot „wyczerpany" to taki, w którym trzy próby dały
wyłącznie duplikaty.

| punkt | zapytań | prób | duplikaty | wyczerpane sloty | czas |
|---|---|---|---|---|---|
| start (SFT) | **1 953** | 2 046 | 93 (4,5%) | 31 | 51 min |
| bottom continued SFT | 1 935 | 2 082 | 147 (7,1%) | 49 | 52 min |
| defect continued SFT | 1 924 | 2 104 | 180 (8,6%) | 60 | 52 min |
| **defect DPO** | 1 812 | 2 328 | 516 (22,2%) | 143 | 73 min |
| **near_miss DPO** | 1 689 | 2 574 | 885 (34,4%) | 239 | 90 min |
| **bottom DPO** | **1 615** | 2 722 | **1 107 (40,7%)** | 278 | 85 min |

Ramiona continued SFT są nieodróżnialne od punktu startowego. Wszystkie trzy
ramiona DPO tracą różnorodność, a **kolejność strat jest dokładnie tą samą
kolejnością, co koszt NLL zmierzony na dev**:

| kohorta | wzrost NLL na dev | duplikaty w generacji | zapytań z 1 984 |
|---|---|---|---|
| bottom | ×2,43 | 40,7% | 1 615 (81%) |
| near_miss | ×2,17 | 34,4% | 1 689 (85%) |
| defect | ×1,35 | 22,2% | 1 812 (91%) |

## 2. Co to znaczy

Wzrost NLL na `chosen` nie był artefaktem metryki — **przekłada się na
obserwowalne zachowanie**. Model po DPO generuje ten sam tekst dla różnych
kontrolek, aż wyczerpie limit prób. To klasyczny kolaps trybu przy stracie
optymalizującej różnicę, a nie poziom prawdopodobieństwa: spychanie `rejected`
zabiera masę prawdopodobieństwa całemu rozkładowi wyjścia, nie tylko stronie
odrzuconej.

Praktyczny skutek: DPO na parach o trywialnym kontraście kupuje margines
**kosztem zdolności do produkowania różnych zapytań dla tego samego pasażu** —
czyli dokładnie tej zdolności, po którą buduje się doc2query.

## 3. Konsekwencja dla porównania probe

Ramiona wchodzą do probe z **różną liczbą par treningowych** (1 615 do 1 953).
Gdyby zostawić to tak, porównanie mieszałoby dwa efekty: jakość zapytań i ich
liczbę. Materializacja wejść probe musi więc zrównać budżet — przez przecięcie
slotów wypełnionych przez **wszystkie** ramiona, zgodnie z precedensem Task 05
(`dual_arm_group_intersection`). Kolaps zostaje wtedy raportowany osobno, tutaj,
zamiast wyciekać do wyniku probe jako różnica objętości danych.

## 4. Czego ten pomiar nie rozstrzyga

- Nie mówi, czy zapytania po DPO są **lepsze** dla wyszukiwania — o tym orzeka
  probe.
- Nie odróżnia wpływu danych od wpływu liczby kroków (defect trenował 102 kroki,
  bottom 154); kontrola przy zrównanych krokach jest zaplanowana.
- Nie dotyczy trybu produkcyjnego z jednym zapytaniem na pasaż; mierzy zdolność
  do wypełnienia czterech różnych kontrolek.
