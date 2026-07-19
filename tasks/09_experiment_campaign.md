# Task 09 — Kampania eksperymentalna i wybór strategii

> [Centralny rejestr zadań i statusów](README.md). Każda zmiana statusu lub zakresu tego zadania musi aktualizować rejestr w tym samym commicie.

## Status

`BLOCKED`

Zadanie oczekuje na implementację wcześniejszych etapów w zakresie
dopuszczonym przez bramki. W szczególności blokują je Harness v1.1 P-01…P-04,
baseline'y P-05/P-06 oraz pełna ablacja polityki hard negative'ów z Task 04.

## Cel

Przeprowadzić eksperymenty w kolejności minimalizującej koszt i wybrać procedurę finalną na podstawie dowodów.

## Zależności

Taski 03–08 w zakresie dopuszczonym przez bramki.

## Zasada sekwencyjności

Nie uruchamiaj pełnej macierzy kartezjańskiej. Stosuj successive halving:

1. 10k przykładów / 1 seed;
2. odrzuć warianty wyraźnie słabe;
3. 50k / 2–3 seedy;
4. probe embedder;
5. 100k–500k tylko finalistom.

Budżet porównuj w tokenach i krokach, nie tylko liczbie przykładów.
Kontrakt budżetowy z Task 04 porównuje jednocześnie tokeny, pary, unikalne
pasaże i K query/pasaż oraz wersję recepty probe.

## Minimalna kolejność

### Etap 1 — pipeline

- E00 prompting;
- E01 1.5B smoke;
- E02 1.5B 10k.

### Etap 2 — model i SFT

- 1.5B vs 4.5B;
- 4.5B base vs instruct;
- ordinary vs balanced vs weighted;
- wybór max length i LoRA target modules.

### Etap 3 — kontrolki

- style only;
- focus only;
- style + focus;
- K independent vs multi-query JSON;
- coverage-aware selection.

### Etap 4 — preference

- best-of-N offline;
- continued SFT;
- DPO;
- różne typy rejected.

### Etap 5 — RL opcjonalny

- tylko po formalnej bramce.

### Etap 6 — skala

- pełne 500k na najlepszym 4.5B;
- 7B standard vs 7B PL na identycznym subset;
- finalny 7B na większych zasobach tylko, gdy przewaga uzasadnia koszt.

## Metryka wyboru

Utwórz ranking wielokryterialny, ale nie ukrywaj Pareto frontu. Priorytety:

1. probe embedder nDCG@10/MRR/Recall z CI;
2. ugruntowanie, możliwość odpowiedzi z pasażu i source retrieval;
3. diversity/focus coverage;
4. kopiowanie względem naturalnego rozkładu;
5. human preference;
6. koszt queries/s i VRAM.

Nie sumuj bezrefleksyjnie wszystkiego do jednego score. Użyj score tylko do wstępnej selekcji, a finalną decyzję opisz w ADR.
Główny wynik musi pochodzić z natywnego polskiego holdoutu; poprawa wyłącznie
na tłumaczonym teście nie wystarcza bez jawnego ADR.

## Eksperymenty opcjonalne po bramkach

- MIX0–MIX4 (100/75/50/25/0% natural) tylko dla 1–2 finalistów i przy
  dopasowanym budżecie;
- probe recipe v2 z GPL/MarginMSE tylko jako osobna, pełna replikacja
  porównań dla 2–3 finalistów;
- kontrfaktyczne negatywy dopiero po stabilnym corpus-mined HN;
- noisy self-training wyłącznie po osobnej bramce;
- drugi backbone probe jako potwierdzenie finalistów.

## Kryteria eliminacji

Odrzuć wariant, gdy:

- invalid rate jest wysoki;
- source Recall@1 istotnie spada;
- overlap poprawia się tylko kosztem ugruntowania lub możliwości odpowiedzi z pasażu;
- diversity wynika z halucynacji;
- focus controls są ignorowane;
- probe embedder przegrywa z prostszym wariantem;
- koszt wzrasta bez korzyści;
- wynik zależy od jednego seeda;
- automatyczny reward nie zgadza się z human panel.

## Raporty

Po każdym etapie twórz:

- `reports/stage_<n>_summary.md`;
- tabelę runów;
- decyzję `continue/stop`;
- listę hipotez potwierdzonych/obalonych;
- rekomendację kolejnego eksperymentu;
- szacowany koszt następnej fazy.

## ADR finalny

Utwórz `docs/adr/00xx-final-training-strategy.md` zawierający:

- wybrany model;
- SFT/DPO/GRPO;
- dane i filtering;
- kontrolki;
- liczbę query na passage;
- selektor kandydatów;
- parametry generacji;
- dowody intrinsic, extrinsic i human;
- odrzucone alternatywy;
- znane ograniczenia.

## Kryteria akceptacji

- każdy finalista ma probe embedder evaluation;
- porównania używają identycznych testów i fingerprintów;
- co najmniej kluczowe porównania mają CI i wiele seedów;
- istnieje jawna decyzja, czy DPO i RL były warte kosztu;
- wybór 4.5B vs 7B jest oparty na wyniku, nie założeniu.
