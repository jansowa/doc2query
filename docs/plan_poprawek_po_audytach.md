# Plan poprawek po przeglądach koncepcyjnych — zadania dla systemu agentowego

**Data:** 18 lipca 2026
**Wejście:** `docs/doc2query_conceptual_review.md` (audyt zewnętrzny), wcześniejszy
przegląd `przeglad_koncepcyjny.md`, stan repo po ukończeniu implementacji
harnessu ewaluacyjnego (Task 04).
**Charakter dokumentu:** backlog wykonawczy. Każde zadanie P-xx ma zostać
przeniesione do właściwych plików `tasks/*.md` (z aktualizacją
`tasks/README.md` w tym samym commicie) albo jawnie odrzucone z uzasadnieniem.

---

## 0. Decyzja nadrzędna o sekwencjonowaniu

Harness w obecnym kształcie został zbudowany według kontraktu Task 04 sprzed
audytów. Kontrakt ten ma trzy wady, które unieważniłyby porównania wykonane na
jego podstawie:

1. brak rozdzielenia rankingu w puli kandydatów od retrievalu po pełnym
   korpusie (Recall@100 w puli ~11 dokumentów jest bez znaczenia);
2. brak natywnego polskiego holdoutu — cała pętla ocen żyje w rozkładzie
   tłumaczonego MS MARCO;
3. brak polityki fałszywych negatywów dla syntetycznych query w zamrożonej
   recepcie probe embeddera.

Recepta probe embeddera i zbiory testowe muszą być **zamrożone zanim powstanie
pierwsze porównanie traktowane jako dowód**. Każda późniejsza zmiana recepty
unieważnia wcześniejsze porównania i wymusza ponowne runy. Dlatego:

- **BLOKUJĄCE** (pakiet „Harness v1.1", zadania P-01…P-04): wykonać przed
  pierwszą ewaluacją porównawczą W03/W05, przed S00 traktowanym jako baseline,
  przed eksperymentami D z Task 05 i przed całym Task 06.
- **BRAMKA PRZED SKALĄ** (P-05, P-06): wykonać przed decyzją o kampanii 4.5B
  i przed Task 09; nie blokują implementacji kodu Task 05.
- **ROZSZERZENIA TASK 05/06** (P-07, P-08): wdrażać w ramach normalnej
  realizacji tych tasków.
- **OPCJONALNE** (P-09): nie blokują niczego.
- **PORZĄDKOWE** (P-10): przy najbliższej edycji plików.

Implementacja kodu Task 05 (kontrolki, selektory) może iść równolegle z
pakietem P-01…P-04, ale **eksperymenty D00–D12 wolno uruchamiać wyłącznie na
harnessie v1.1**.

---

## P-01 — Rozdzielenie protokołów ewaluacji retrieval [BLOKUJĄCE]

**Modyfikuje:** `tasks/04_evaluation_harness.md`, moduły
`evaluation/retrieval.py`, `evaluation/report.py`, configi ewaluacji.

**Zakres:**

1. Zdefiniować dwa jawnie nazwane protokoły i nie mieszać ich metryk:
   - `candidate_pool_ranking` — pula: pozytyw(y) + odziedziczone hard
     negatives; metryki: `pool_rank`, `pool_recall@1/5`, `pool_mrr`,
     `pool_ndcg@10`, margines rerankera; zastosowanie: diagnostyka generatora,
     grounding, rewardy.
   - `corpus_retrieval` — pula: pełny zamrożony `documents.parquet` danego
     splitu; metryki: `corpus_recall@1/5/10/100`, `corpus_mrr@10`,
     `corpus_ndcg@10`, `corpus_map`; zastosowanie: **główna ocena probe i
     docelowego embeddera oraz round-trip consistency generatora**.
2. Zbudować infrastrukturę indeksu korpusowego:
   - BM25 na lematyzowanych tekstach (cache lematów z Task 02 już istnieje);
   - FAISS/brute-force na zamrożonym, przypiętym embedderze pomocniczym
     (rewizja i licencja w configu; nie jest to probe embedder).
3. Dodać metryki specyficzności query liczone na indeksie korpusowym:
   `effective_candidate_count` (liczba dokumentów powyżej progu relevance),
   margines do najlepszego dokumentu spoza znanych pozytywów, flaga
   `possibly_ambiguous_query`.
4. Raport musi podawać rozmiar przeszukiwanej puli przy każdej metryce;
   zakaz raportowania `recall@K` dla puli mniejszej niż K (test jednostkowy).
5. Rekordy dev/test z <10 negatywami po cleanupie (otwarta decyzja z Task 01):
   rozwiązać przez `corpus_retrieval` jako protokół główny (problem znika),
   a w `candidate_pool_ranking` deterministycznie douzupełnić negatywy z tego
   samego splitu i oznaczyć je `backfilled=true`. Zapisać decyzję w Task 01.

**Kryteria akceptacji:** metryki obu protokołów mają rozłączne nazwy; istnieje
test odrzucający `recall@K` przy zbyt małej puli; round-trip po korpusie
(`corpus_round_trip@{1,5,20,100}`) jest liczony dla wygenerowanych query i
raportowany obok marginesu rerankera wraz z korelacją między nimi; wynik
`corpus_retrieval` jest zadeklarowany jako podstawa porównań generatorów.

---

## P-02 — Natywny polski holdout [BLOKUJĄCE]

**Modyfikuje:** `tasks/04_evaluation_harness.md`, `tasks/09`, `tasks/10`,
`configs/data/`, `docs/datasets/`.

**Zakres:**

1. Audyt kandydatów: wybrane zadania retrieval z PIRB
   ([arXiv:2402.13350](https://arxiv.org/abs/2402.13350)), PolQA
   ([arXiv:2212.08897](https://arxiv.org/abs/2212.08897)), ewentualnie
   MAUPQA; dla każdego: licencja, sposób powstania (natywne vs tłumaczone vs
   syntetyczne!), overlap dokumentów z `msmarco_pl`, ryzyko kontaminacji
   pretrainingu Bielika i embedderów. Wynik audytu w
   `docs/datasets/native_pl_holdout.md`.
2. Zamrozić trzy zbiory: `test_native_pl` (natywne zapytania),
   `test_translated_msmarco_pl` (dotychczasowy test), opcjonalnie
   `test_transfer_ood` (np. podzbiór BEIR-PL — wyłącznie jako sygnał
   transferu, nie zamiennik natywnego). Fingerprinty i hash list ID jak dla
   pozostałych zbiorów.
3. Włączyć oba obowiązkowe zbiory do `evaluate embedder`; raport zawsze
   pokazuje wyniki translated i native obok siebie.
4. Dodać do Task 09 kryterium: finalista musi mieć wynik natywny; wariant
   poprawiający wyłącznie test tłumaczony nie może wygrać bez jawnej decyzji
   ADR. Dodać do Task 10: model card raportuje oba wyniki osobno.
5. Tani proxy „translationese": panel ludzki lub reguły oceniające
   naturalność syntetycznych query, raportowane per generator.

**Kryteria akceptacji:** `test_native_pl` jest zamrożony przed pierwszym
porównawczym probe; nie jest używany do żadnego strojenia; raport porównawczy
bez wyniku natywnego jest oznaczany jako niekompletny.

---

## P-03 — Polityka negatywów dla syntetycznych query [BLOKUJĄCE]

**Modyfikuje:** `tasks/04_evaluation_harness.md` (recepta probe),
`tasks/06`, `tasks/10`.

**Zakres:**

1. **Recepta probe v1 (zamrożona teraz):** dla każdej syntetycznej pary
   `(query_syn, positive)` przeliczyć zamrożonym primary rerankerem score
   `(query_syn, negative_j)` dla odziedziczonych negatywów. Negatyw z score
   powyżej progu kalibracyjnego (percentyl z Task 02, nie ad hoc) otrzymuje
   flagę `possible_false_negative` i zgodnie z configiem: `drop` | `demote`
   (do in-batch) | `keep+log`. Domyślnie `drop`. Identyczna procedura dla
   wszystkich wariantów, w tym baseline'u naturalnego.
2. Raportować odsetek wykrytych fałszywych negatywów per wariant generatora —
   traktować jako metrykę diagnostyczną różnorodności.
3. **Sensitivity check (jednorazowy, tani):** dla jednego generatora (W05)
   wytrenować probe na wariancie HN0 (odziedziczone, bez filtra) vs HN0+filter
   vs HN1 (BM25 re-mining z korpusu train). Jeśli różnice mieszczą się w CI —
   pozostać przy recepcie v1 i odłożyć pełną ablację. Jeśli nie — podnieść
   decyzję do ADR przed dalszymi porównaniami.
4. **Pełna ablacja HN (bramka przed Task 09, nie przed pierwszymi probe):**
   HN0 / HN0+filter / HN1 BM25 / HN2 zamrożony bi-encoder / HN3 union +
   positive-aware filtering w stylu NV-Retriever
   ([arXiv:2407.15831](https://arxiv.org/abs/2407.15831)). Provenance każdego
   negatywu (miner, score, flagi) w artefaktach. Bez założenia, że
   „najtrudniejsze = najlepsze" (ryzyko false negatives —
   [arXiv:2209.05072](https://arxiv.org/abs/2209.05072)).
5. W Task 06: te same flagi `possible_false_negative` i
   `effective_candidate_count` (z P-01) wchodzą do composite score kandydatów
   oraz do reguł odrzucania par preferencyjnych.

**Kryteria akceptacji:** recepta probe v1 z polityką negatywów jest zapisana i
zamrożona przed pierwszym porównaniem; zmiana polityki po tym punkcie wymaga
podbicia wersji recepty i ponownego uruchomienia wszystkich porównywanych
wariantów (test w pipeline porównań).

---

## P-04 — Kontrakt statystyczny i budżetowy kampanii [BLOKUJĄCE]

**Modyfikuje:** `tasks/04`, `tasks/09`; nowy plik
`docs/adr/000x-statistical-contract.md`.

**Zakres — zamrozić przed pierwszym porównaniem:**

1. Metryka główna: `corpus_ndcg@10` probe embeddera na `test_native_pl`
   (propozycja — do zatwierdzenia w ADR), z `test_translated` jako metryką
   wtórną.
2. Metryki non-inferiority i tolerancje: grounding/source retrieval,
   answerability proxy, format validity — z jawnie zapisanymi marginesami.
3. Minimalny praktycznie istotny efekt; liczba seedów per etap successive
   halving; sposób raportowania wariancji **między seedami treningu** obok
   bootstrapu po query (bootstrap nie zastępuje wariancji między runami).
4. Definicja budżetu porównań: jednocześnie liczba tokenów, liczba par,
   liczba unikalnych pasaży i K query/pasaż. Reguła: K query z jednego pasażu
   nie może niejawnie zwiększać wagi pasażu.
5. Zasady użycia dev i jednorazowego otwarcia final test (translated i
   native).

**Kryteria akceptacji:** ADR istnieje przed startem porównań; raporty
porównawcze cytują wersję kontraktu; pipeline odmawia porównania runów o
niezgodnych definicjach budżetu.

---

## P-05 — Baseline'y przed decyzją o 4.5B [BRAMKA PRZED SKALĄ]

**Modyfikuje:** `tasks/03_sft_qlora_baselines.md`, `tasks/09`.

**Zakres:**

1. **S00 rozszerzone:** prompting Bielika zero-shot **i few-shot** (3–8
   naturalnych przykładów z dev, dobieranych per styl; wnioski Promptagator —
   struktura promptu > temperatura). Ocena pełnym harnessem v1.1.
2. **S07 — baseline seq2seq:** plT5-base/large lub mT5 dostrojony na
   identycznych parach, splicie i budżecie danych co Bielik 1.5B. Wyrównać:
   liczbę próbek per passage, filtering/selection, budżet danych probe,
   koszt generacji. **Nie** używać angielskiego checkpointu docT5query jako
   baseline'u dla polskiego korpusu (generuje zapytania w złym języku /
   rozkładzie — wynik byłby pozorny); odnotować go tylko jako kontekst
   historyczny.
3. **Ocena W03/W05 na harnessie v1.1:** candidate-pool + corpus + translated
   + native.
4. **Pierwszy porównawczy probe (mała macierz):** natural-only (gold-data
   control — nie nazywać „upper bound"), W05 synthetic-only, jedna mieszanka
   natural+synthetic (np. 50/50, budget-matched wg P-04).
5. Decyzja o kampanii 4.5B wyłącznie na podstawie retrieval (nie eval loss),
   z checklistą z audytu (`doc2query_conceptual_review.md §11`).

**Kryteria akceptacji:** istnieje tabela SFT-1.5B vs prompting zero/few-shot
vs plT5 vs natural-only na identycznym harnessie; decyzja 4.5B zapisana w
raporcie etapu z odwołaniem do wyników.

---

## P-06 — S06: czyszczenie naturalnych par SFT polskim rerankerem [BRAMKA PRZED SKALĄ]

**Modyfikuje:** `tasks/03`, wykorzystuje istniejący `WeightedSFTTrainer`.

**Zakres:** filtr `pos_score >= 23.50` działa na angielskich score'ach
źródłowych; część tłumaczonych par jest uszkodzona, a SFT uczy się na nich
1:1. Offline policzyć primary rerankerem margines naturalnej pary względem jej
negatywów (artefakt i tak powstaje w benchmarku Task 02 — zachować wyniki dla
train). Warianty na 1.5B/50k: (a) drop dolnych ~5–10% wg progu
kalibracyjnego, (b) waga = funkcja marginesu, (c) kontrola bez zmian. Ocena
standardowa harnessem v1.1. Jeśli poprawa jest istotna — wariant staje się
domyślnym przygotowaniem danych dla 4.5B (DPO dziedziczy checkpoint SFT, więc
jakość targetów ogranicza wszystko dalej).

---

## P-07 — Rozszerzenia Task 05 [W RAMACH TASK 05]

**Modyfikuje:** `tasks/05_controlled_diversity_and_multiquery.md`,
`schemas.py`.

**Zakres:**

1. **Taksonomia:** rozdzielić `form` (full_question / keyword_query) od
   `intent` (fact_lookup, definition, entity_lookup, procedure, comparison,
   …) — por. EGG ([arXiv:2409.16570](https://arxiv.org/abs/2409.16570)).
   Rozkład docelowy kalibrowany rozkładem naturalnych query per domena, nie
   ustalany ręcznie. Poziom `retrieval_task` odłożyć do czasu korpusu
   wielodomenowego (obecnie jedna domena — nie budować martwej osi).
2. **Schemat evidence:** już teraz dodać do schematu opcjonalne pola
   `evidence_sentence_ids`, `evidence_type`, `evidence_confidence`
   (single-sentence focus pozostaje baseline'em; chodzi o to, by późniejsze
   rozszerzenie nie łamało schematu i cache'ów).
3. **Concept coverage (CCQGen,
   [arXiv:2502.11181](https://arxiv.org/abs/2502.11181)):**
   - D08: ekstrakcja koncepcji z lematów treściowych/encji/liczb (zasoby z
     Task 02, bez nowego modelu) + ręczny audyt jakości ekstraktora na
     ~200 pasażach;
   - D09: stateful generation — kolejny prompt dostaje listę niepokrytych
     koncepcji i skróty poprzednich query; porównać z D05/D07 przy tym samym
     budżecie; raportować narzut tokenów;
   - D10: consistency filtering wsparty pokryciem koncepcji.
4. **D11 — krzywa K:** marginal gain probe embeddera dla K = 1/2/4/8(/16),
   osobno przy stałej liczbie pasaży i stałej liczbie par (rozdzielenie
   efektu K od wielkości zbioru).
5. **D12 — selekcja:** top-N vs MMR vs coverage-aware (submodular greedy) na
   tych samych kandydatach.
6. **Mała ablacja dekodowania (raz, przed zamrożeniem
   `configs/generation/diverse.yaml`):** stałe top-p vs min-p (jeśli backend
   wspiera) vs miks 2–3 temperatur vs stateful coverage, przy stałych
   pozostałych parametrach, na metrykach intrinsic. Nie testować top-k jako
   osi różnorodności.

---

## P-08 — Rozszerzenia Task 06 (i minimalna separacja sędziów) [W RAMACH TASK 06]

**Modyfikuje:** `tasks/06_candidate_scoring_and_preference_data.md`,
`tasks/02` (drobny zapis ról), `tasks/07`.

**Zakres:**

1. **Role sędziów od razu, nie w Task 11:** primary = builder judge (focus,
   kalibracja, filtracja, preferencje); shadow = confirmatory judge (metryka
   potwierdzająca / veto przy dużej niezgodności); corpus retrieval (P-01) =
   sygnał niezależny; human panel = kalibracja. Ewaluacja intrinsic w
   raportach porównawczych zawsze raportuje shadow obok primary. Zapisać
   podział ról w Task 02/04/06; pełny audyt hackingu zostaje w Task 11.
2. **Composite score:** dodać `corpus_round_trip`, `effective_candidate_count`
   i `possible_false_negative` (z P-01/P-03) jako osobne pola; query o
   wysokim score rerankera przegrywające round-trip = źródło rejected typu
   „zbyt ogólne".
3. **Re-mining negatywów dla zaakceptowanych kandydatów** zgodnie z polityką
   z P-03 (provenance minera w rekordzie).
4. **Kandydaci teachera (jawna ablacja, nie domyślna ścieżka):** zamrożony
   większy model (np. Bielik-Minitron-7B inference-only, kwantyzowany) jako
   dodatkowe źródło kandydatów w danych preferencyjnych (linia InPars+,
   [arXiv:2508.13930](https://arxiv.org/abs/2508.13930)); koszt, licencja i
   provenance jawnie raportowane.
5. **Opcjonalny answerability judge:** zamrożony model QA/instruct zwracający
   `answerable: yes/no` (+ zdanie dowodowe) jako (a) trzeci głos przy
   disagreement primary/shadow, (b) flaga w score, (c) wsparcie testu
   adwersarialnego klasy „temat zgodny, fakt nieobecny". Te same zakazy co
   dla rerankerów (bez treningu, bez uczenia na outputach generatora).
6. W Task 07 dodać do obowiązkowych kontroli `score-weighted continued SFT`
   obok zwykłego continued SFT; jedną metodę listwise (LiPO
   [arXiv:2402.01878](https://arxiv.org/abs/2402.01878) lub PRO
   [arXiv:2306.17492](https://arxiv.org/abs/2306.17492)) dopuścić dopiero po
   stabilnym DPO i tylko jeśli ranking kandydatów jest wiarygodny.

---

## P-09 — Eksperymenty opcjonalne [OPTIONAL]

1. **MIX0–MIX4** (100/75/50/25/0% natural, budget-matched wg P-04) dla 1–2
   finalistów w Task 09; P-05 uruchamia tylko jedną mieszankę jako smoke.
2. **Wariant recepty probe z destylacją GPL/MarginMSE**
   ([arXiv:2112.07577](https://arxiv.org/abs/2112.07577)): miękkie etykiety z
   zamrożonego cross-encodera zamiast twardych par MNRL; naturalnie tłumi
   szkody z fałszywych negatywów; testować jako recepta v2 dla 2–3 finalistów
   (pełny rerun porównań w ramach v2, zgodnie z P-03 pkt „wersjonowanie").
3. **Kontrfaktyczne negatywy** (podmiana encji/liczby/relacji) — dopiero po
   stabilnym baseline'ie corpus-mined HN; ryzyko artefaktów syntetyczności.
4. **Noisy self-training loop** — późny eksperyment, wysokie ryzyko
   niestacjonarnej pętli; wymaga osobnej bramki jak GRPO.
5. Potwierdzenie wyniku finalistów na drugim backbone probe embeddera.

---

## P-10 — Porządkowe [PRZY NAJBLIŻSZEJ EDYCJI]

1. Zmienić nazwę `tasks/11_optional_alternating_cotraining.md` na
   `tasks/11_judge_robustness_audit.md` (treść zabrania cotrainingu — nazwa
   myli); zaktualizować linki.
2. W pliku researchu (`doc2query_research.md` / `doc2query_wnioski_sesja.md`)
   oznaczyć sekcje o ekspansji indeksu BM25, dual-index i keywordach
   reklamowych jako „kontekst, poza zakresem projektu"; fragment o „treningu
   rerankera na syntetycznych parach" oznaczyć jako niedozwolony w bieżącym
   zakresie (dotyczyłby osobnego projektu produktowego, nie sędziego).
3. Zaktualizować `tasks/README.md`: status Task 04 po dokończeniu harnessu
   oraz nowy stan „v1.1 wymagane przed porównaniami" do czasu wykonania
   P-01…P-04.
4. W Task 04 zmienić nazwę baseline'u natural-only z „upper/control" na
   „gold-data control".

---

## Zależności między pakietami

```text
P-01 ─┬─> P-03 ─┬─> pierwsze porównawcze probe (P-05.4)
P-02 ─┤         │
P-04 ─┴─────────┴─> eksperymenty D (Task 05), Task 06, Task 09

P-05, P-06 ──> decyzja o 4.5B ──> kampania Task 09
P-07 (kod równolegle; eksperymenty po v1.1)
P-08 ──> Task 07 (DPO)
```

## Jawnie odrzucone / zmodyfikowane punkty z audytów

1. **Angielski docT5query jako baseline** — odrzucony (zły język generacji;
   wynik pozorny). Zostaje plT5/mT5 fine-tuned (P-05).
2. **Pełna ablacja HN0–HN4 jako bloker pierwszych probe** — zmodyfikowana:
   recepta v1 + tani sensitivity check blokują; pełna ablacja jest bramką
   przed Task 09 (P-03).
3. **MIX0–MIX4 jako P0** — zmodyfikowane: definicja budżetu MIX wchodzi do
   kontraktu (P-04) od razu, pełna macierz dopiero dla finalistów (P-09);
   jedna mieszanka w pierwszym probe (P-05).
4. **`retrieval_task` jako trzecia oś taksonomii** — odroczone do czasu
   korpusu wielodomenowego (P-07.1).
5. **HN4 curriculum easy→hard** — przeniesione do opcjonalnych; nie należy do
   minimalnej ablacji.
