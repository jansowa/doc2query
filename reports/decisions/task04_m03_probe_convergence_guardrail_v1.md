# Task 04 / M-03 — guardrail zbieżności probe i statystyka per-seed (ADR v1, 2026-08-16)

## Kontekst

Rozszerzenie specyfikacji z 2026-08-13 dodało M-03 (stabilność seedów) do zakresu
Task 04. Sweep budżetu probe z okna bezobsługowego zmierzył dwie rzeczy, które
wymuszają ten ADR:

1. **Wariancja seedowa bywa większa niż mierzone efekty.** Przy stałym ramieniu i
   stałym budżecie `corpus_ndcg_at_10` rozjeżdża się od 0,0011 do 0,0826, czyli
   rząd wielkości powyżej przewagi `+0,0479` raportowanej przez zamknięty
   confirm.
2. **Strata treningowa nie jest guardrailem zbieżności.** Korelacja `last_loss` z
   `corpus_ndcg_at_10` wynosi `r = −0,199` (n=12). Run z `last_loss ≈ 0,0001`
   daje zarówno 0,0445, jak i 0,0284, a run z `last_loss = 0,397` daje 0,0011.

Dodatkowo zamrożony `compare` liczy 95% CI bootstrapem **zapytań** przy
`resample_training_seeds: false`, więc konstrukcyjnie nie wyraża niepewności
seedowej. Zanim probe zostanie użyty jako instrument selekcji finalistów (M-01),
potrzebna jest reguła: kiedy run jest niezbieżny i jak agregować seedy.

Właściciel polecił skalibrować progi na istniejących runach i zamrozić je dla
przyszłych porównań, bez unieważniania i bez powtarzania zamkniętego confirmu.

## Decyzja

Zamraża się kontrakt `task04-m03-probe-convergence-guardrail-v1`, zaimplementowany
w `src/doc2query/evaluation/probe_convergence.py`, uruchamiany przez
`scripts/apply_task04_m03_probe_convergence_guardrail.py`, z progami przypiętymi w
`configs/evaluation/task04_m03_probe_convergence_guardrail_v1.yaml`.

### 1. Sygnał zbieżności jest retrievalowy, nie treningowy

Guardrail używa `corpus_recall_at_100` z `corpus_retrieval_summary.json`.
`loss_based_guardrail_permitted: false` jest wpisane w kontrakt wprost, bo
korelacja `r = −0,199` wyklucza stratę jako detektor.

Wybór `recall@100`, a nie metryki decyzyjnej `ndcg@10`, jest celowy: `recall@100`
jest najbardziej nasyconym dostępnym sygnałem i odpowiada na pytanie „czy ten
embedder odzyskuje cokolwiek”, a nie „czy jest lepszy”. Użycie metryki decyzyjnej
jako filtru zbieżności byłoby selekcją na zmiennej wynikowej.

Powód techniczny, dlaczego nie ma nic lepszego: w żadnym z 22 istniejących runów
**nie ma** pośredniej ewaluacji retrievalowej ani utrwalonej krzywej straty.
`embedder_probe.py` zbiera stratę co krok, ale zapisuje wyłącznie `losses[0]` i
`losses[-1]`, a `checkpoint.pt` z pełną listą jest usuwany po sukcesie. Guardrail
musi więc działać na końcowych artefaktach. Żaden run nie ma też flagi kolapsu:
`status="measured"` i `incomplete_reasons=[]` mają wszystkie 22 runy, w tym
ewidentnie zdegenerowane.

### 2. Podłoga jest niezależna od ramienia

Run jest `non_converged`, gdy jego `corpus_recall_at_100` jest poniżej

```
applied_floor = max(
    min_chance_multiple * retrieval_depth / rozmiar_korpusu,   # 4.0 * 100/139782
    min_fraction_of_pooled_median * median(sygnał w całym porównaniu)  # 0.5 * mediana
)
```

- **podłoga losowa** (4× poziom losowy) jest zabezpieczeniem na wypadek, gdyby
  całe porównanie się zapadło i mediana przestała być wiarygodna;
- **podłoga medianowa** (połowa mediany) jest progiem głównym: run, który
  odzyskuje własny dokument mniej niż o połowę rzadziej niż typowy run tego
  samego porównania, jest zdegenerowany, nie tylko słaby.

Mediana liczona jest **wspólnie z obu ramion**, więc próg nie może zależeć od
tego, które ramię jest wariantem.

### 3. Odrzucanie seedów jest sparowane i raportowane obustronnie

- seed odrzucany jest **jako para** (oba ramiona), nawet jeśli tylko jeden run
  jest niezbieżny — filtr nie może preferencyjnie usuwać słabych runów jednego
  ramienia (`drop_policy: drop_whole_seed_pair`);
- wynik **bez filtra** jest liczony i raportowany zawsze
  (`report_unfiltered_result: true`). Jeżeli filtr zmienia wniosek, widać to
  wprost;
- lista odrzuconych seedów i ich wartości sygnału są w artefakcie.

### 4. Statystyka decyzyjna na sparowanych różnicach per-seed

Dla każdego seeda wspólnego obu ramion liczona jest różnica
`d_s = wariant(s) − anchor(s)` metryki `corpus_ndcg_at_10`. Decyzja wymaga
**łącznie**:

1. co najmniej **5 zbieżnych par seedów** (`min_converged_seed_pairs: 5`);
2. dolnej granicy 95% CI sparowanego bootstrapu **po seedach** (`paired_bootstrap`
   z Task 04, 2000 próbek, ziarno 42) nie niżej niż niezmieniony próg wyższości
   `+0.01`;
3. jednostronnego **dokładnego testu znakowego** (`exact_sign_flip_one_sided`,
   pełna enumeracja 2^n) przeciw hipotezie zerowej „`d_s − 0.01` symetryczne
   wokół zera” z `p ≤ 0.05`.

#### Dlaczego minimum to dokładnie 5 seedów

Nie jest to liczba okrągła z wygody. Dokładny jednostronny test znakowy na `n`
parach ma najmniejsze osiągalne `p` równe `1/2^n`. Przy `n = 4` daje to
`1/16 = 0.0625 > 0.05` — test **nie może** osiągnąć poziomu 0,05 przy żadnych
danych. Przy `n = 5` daje `1/32 = 0.03125 ≤ 0.05`. Pięć par seedów jest więc
najmniejszą liczbą, przy której prerejestrowana reguła decyzyjna jest w ogóle
rozstrzygalna.

### 5. Czego ten ADR nie robi

- **nie unieważnia** zamkniętego TriviaQA `dev_confirm` i **nie zleca** jego
  powtórzenia. Zamrożony `compare` pinuje seedy `[42, 43, 44]` i liczy metrykę
  decyzyjną inaczej (agregacja per-query przed bootstrapem zapytań); ten
  guardrail jest osobnym, prospektywnym kontraktem dla **przyszłych** porównań;
- **nie promuje** i **nie degraduje** żadnego ramienia: artefakt zawsze zapisuje
  `promotion_authorized=false`;
- **nie zmienia** progu wyższości `+0.01`, definicji budżetu ani żadnego
  zamrożonego configu Tasków 04–05;
- **nie otwiera** testów finalnych; `final_tests_used=[]`.

## Kalibracja i jej granice

Progi (`min_chance_multiple = 4.0`, `min_fraction_of_pooled_median = 0.5`)
skalibrowano na 22 zakończonych runach wskazanych przez właściciela:
`runs/task05_d01b_scale_interaction_4_5b_trivia_dev_confirm_v1` (10 runów,
seedy 42–46, oba ramiona) i `runs/task05_probe_budget_sensitivity_v1` (12 runów,
budżety 1024/2048, seedy 42–44, oba ramiona). Kryterium kalibracji było jawne i
jedno: **oddzielić runy widocznie zapadnięte od jedynie słabych, z zapasem**.

To jest kalibracja retrospektywna i tak jest oznaczona. Zastosowanie guardraila do
tych samych 22 runów jest **diagnostyką**, nie nową decyzją. Progi są zamrożone dla
przyszłych porównań; zmiana wymaga nowego, prospektywnego ADR. Wynik kalibracji i
diagnostyki:
[`task04_m03_probe_convergence_calibration_2026-08-16.md`](../measurements/task04_m03_probe_convergence_calibration_2026-08-16.md).

Znane ograniczenia, zapisane świadomie:

- podłoga medianowa zależy od składu porównania; przy 6 runach mediana jest
  szumna. Dlatego reguła wymaga co najmniej 5 par seedów, czyli minimum 10 runów;
- każdy filtr zbieżności oparty na tym samym zbiorze ewaluacyjnym, na którym
  podejmowana jest decyzja, może wprowadzać obciążenie. Trzy zabezpieczenia
  powyżej (sparowane odrzucanie, podłoga niezależna od ramienia, obowiązkowy
  wynik bez filtra) ograniczają je, ale nie usuwają całkowicie;
- bootstrap po 5 seedach jest zgrubny. Jest raportowany razem z dokładnym testem
  znakowym właśnie dlatego, że żadne z tych narzędzi samo nie wystarcza.

`final_tests_used=[]`. Etap jest w całości CPU i nie uruchamia treningu.
