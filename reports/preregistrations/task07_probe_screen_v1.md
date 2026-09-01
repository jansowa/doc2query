# Preregistracja: screen probe embedderów dla ramion Task 07 (v1)

## Status

Preregistracja spisana PRZED uruchomieniem jakiegokolwiek treningu probe dla
Task 07 (2026-09-01). Realizuje zapowiedzianą w
`reports/measurements/task07_generation_collapse_2026-08-31.md` §3 konsekwencję
kolapsu generacji. `final_tests_used=[]` — całość na dev; żaden finalista nie
jest tu zamrażany.

## Cel i rola wyniku

Ranking przesiewowy 13 ramion Task 07 na **prymarnej** metryce kampanii
(probe embedder, naturalne zamrożone zapytania dev), po screen-budżecie
generacji (496 pasaży). To screen: służy wyborowi ramion do pełnego pomiaru
(pełna kohorta probe + protokół confirm z wieloma seedami), NIE do ogłoszenia
zwycięzcy Task 07 ani decyzji o finalistach.

## Ramiona (13)

`start` (adapter SFT bez DPO — punkt odniesienia), kohorta bottom-v3
(`bottom_csft`, `bottom_wsft`, `bottom_dpo`), kohorta near-miss
(`nearmiss_csft`, `nearmiss_wsft`, `nearmiss_dpo`), kohorta defect
(`defect_csft`, `defect_wsft`, `defect_dpo`) oraz ramiona antykolapsowe ADR
`task07_anti_collapse_v1` (`beta02`, `rpo`, `divch`). Zapytania: gotowe
`runs/task07_probe_gen_v1/<ramię>/generated.jsonl` (nic nie jest dogenerowywane).

## Protokół (zamrożony przed startem)

1. **Budżet równany przecięciem slotów** (slot = `doc_id × form × intent`):
   do treningu wchodzą wyłącznie sloty wypełnione przez WSZYSTKIE ramiona
   i mające ≥1 negatyw po filtrze HN0 w KAŻDYM ramieniu — uogólnienie polityki
   Task 05 `dual_arm_group_intersection_hn0_filter_drop` na 13 ramion.
   Duplikaty tekstu zapytania NIE wykluczają slotu (są prawdziwym wyjściem
   ramienia); kolaps raportujemy osobno metrykami różnorodności.
   Implementacja: `scripts/build_task07_probe_inputs.py` →
   `artifacts/task07/probe_inputs_v1/` (manifest z licznościami i sha256).

   **Poprawka przed pierwszym wynikiem (2026-09-01):** kontrakt P-04 w
   `train_probe` wymaga jednolitego K zapytań na pasaż, a surowe przecięcie
   slotów daje zmienne K (rozkład 1:13, 2:137, 3:177, 4:169 na 496 pasażach).
   Dokładamy deterministyczny, ślepy na ramiona krok
   (`scripts/uniformize_task07_probe_inputs.py`): K wybrane maksymalizacją
   budżetu par → **K=3, 346 pasaży, 1 038 par na ramię**; pasaże z 4 slotami
   tracą ostatni slot w porządku leksykalnym identyfikatorów. Poprawka spisana
   po awarii kontraktowej treningu, PRZED policzeniem jakiejkolwiek ewaluacji
   (żadne ramię nie doszło do metryki).
2. **Negatywy**: surowe `hard_negatives` dziedziczone z zamrożonego dev
   (`dev_intrinsic`, manifest `data/processed/v1/evaluation/task04-v1`),
   filtr HN0 prymarnym sędzią (`sdadas/polish-reranker-roberta-v3`) z pinowaną
   kalibracją `artifacts/task02/pfn_dev_v1/calibration.json`, polityka `drop` —
   przepis identyczny z Task 05, bez żadnej nowej decyzji.
3. **Trening probe**: protokół screen Task 05 bez modyfikacji —
   `configs/evaluation/probe_v1.yaml` (`sdadas/polish-reranker-base-ranknet`),
   `--max-steps 500 --batch-size 4 --seed 42 --train-prefix-limit <N_wspólne>`,
   `--query-source synthetic`. Jeden seed na ramię (to screen; wariancja
   within-arm zmierzona w Task 05).
4. **Ewaluacja**: corpus retrieval na pełnym korpusie (2 404 263 dokumentów),
   zapytania `dev_intrinsic_rank10` (6 598 naturalnych zapytań dev).
   **Metryka pierwotna: `corpus_recall_at_10`**; `corpus_ndcg_at_10` i
   `corpus_mrr_at_10` raportowane pomocniczo. Zbiory testowe nietykane.
5. **Wykonanie**: `scripts/queue_task07_probe_embedders.sh` →
   `runs/task07_probe/<ramię>/` (kolejność ramion wg wartości informacyjnej,
   wznawialna po `result.json`).

## Reguły interpretacji (spisane przed wynikiem)

- Porównania czytamy WZGLĘDEM `start`: ramię „pomaga", jeśli przewyższa start
  na metryce pierwotnej; ranking między ramionami traktujemy jako przesiewowy
  (jeden seed, screen-budżet) — różnice rzędu wariancji within-arm z Task 05
  nie są rozstrzygające.
- Wynik probe NIE jest unieważniany ani „korygowany" metrykami powierzchniowymi
  (score sędziego, overlap, duplikaty). Metryki różnorodności raportujemy obok,
  jako kontekst kolapsu.
- Decyzje otwierane tym pomiarem: wybór ramion do pełnego pomiaru probe
  (decyzja właściciela). Pomiar NIE otwiera treningu nowych kohort ani zmian
  w danych.

## Poprawka 2 (2026-09-01, przed pierwszym wynikiem probe): analiza parowana

Spisana, zanim jakiekolwiek ramię doszło do metryki retrieval (w chwili zapisu
ramię `start` jest w trakcie kodowania korpusu). Powód: rzadkie etykiety
MS MARCO (jeden oznaczony pozytyw na zapytanie przy 2,4 mln pasaży z
niemal-duplikatami) karzą embedder za znalezienie relewantnego, ale
nieoznaczonego dokumentu. Kara jest wspólna dla ramion, więc parowanie per
zapytanie w dużej mierze ją kasuje.

1. Obok średnich raportujemy **parowane porównanie per zapytanie** każdego
   ramienia względem `start`: win/tie/loss oraz parowany bootstrap CI 95%
   różnicy `corpus_recall_at_10` (10 000 resamplingów, seed 42);
   implementacja `scripts/compare_task07_probe_paired.py` (czysty
   postprocessing istniejących `corpus_retrieval_per_query.jsonl`, bez GPU).
2. Reguła interpretacji: ramię „pomaga", gdy CI 95% różnicy parowanej vs
   `start` leży w całości powyżej zera; średnie bez CI pozostają opisowe.
3. Sanity-check metody na zamkniętych danych Task 05 (hybrid vs W05):
   delta +0,0259, CI95 [0,0191; 0,0330] — metoda odtwarza znaną decyzję.
4. Dla finalistów (etap confirm) dokładamy drugą oś: natywny holdout P-02
   (`test_native_pl`), odporny na zarzut premiowania translationese —
   wymaga GPU i osobnej preregistracji przy confirm.
5. Bez zmian pozostaje zakaz „naprawiania" etykiet lokalnymi rerankerami
   (żadnego doznaczania pozytywów bez nowego prospektywnego ADR).

`final_tests_used=[]`
