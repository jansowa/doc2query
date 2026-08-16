# Task 06/08 — korpus walidacyjny komponentów nagrody (ADR v1, 2026-08-14)

## Kontekst

Komponenty nagrody (`src/doc2query/rewards/lexical.py`,
`rewards/grounding.py`, `rewards/calibration.py`), etykiety focus
(`src/doc2query/data/focus_labels.py`), checker formatu
(`src/doc2query/evaluation/format.py`) oraz bramka różnorodności same-prompt
(`scripts/apply_task06_same_prompt_diversity_gate.py`) są dziś weryfikowane
wyłącznie testami jednostkowymi na fixture'ach i pośrednio przez zgodność
z primary rerankerem. Nie istnieje zbiór, w którym **z konstrukcji** wiadomo,
jaki powinien być porządek jakości kandydatów. Oznacza to, że każdy pomiar
„reward działa” jest dziś argumentem z korelacji z sędzią, a nie testem
komponentu.

Task 08 (GRPO, planowany etap po Task 07 — `AGENTS.md` §2, Faza E) wymaga
wielokryterialnej nagrody, której składowe są rozdzielne i sprawdzalne. Bez
takiego zbioru projekt rewardu do GRPO byłby oparty na niesprawdzonych
założeniach o zachowaniu składowych.

Właściciel autoryzował ten zakres komendą 2026-08-14 (okno bezobsługowe,
wybór „T-C2, potem T-C1”).

## Decyzja

Powstaje **prerejestrowany korpus diagnostyczny** `reward_validation_corpus_v1`:
180 pasaży × 8 zapytań o etykietach klas błędu **nadanych przez konstrukcję**,
napisanych bezpośrednio przez model asystujący (Claude Opus 5, `claude-opus-5[1m]`,
2026-08-14) w tej sesji i w podsesjach agentowych. Etykieta nie pochodzi od
żadnego sędziego ani od żadnego score'u — pochodzi z intencji autora zapytania,
zapisanej przed jakimkolwiek pomiarem.

To **nie jest** dowód ludzki, nie jest panelem kalibracyjnym Task 02 i nie
zmienia żadnego zamrożonego progu.

### Źródło pasaży

- `artifacts/task06/candidate_pilot_v1/cohort.records.jsonl`, split `train`,
  pierwsze 180 klastrów w porządku `sha256(cluster_id)` rosnąco;
- materializacja przez `scripts/slim_task06_cohort_passages.py` do
  `artifacts/task06/reward_validation_corpus_v1/passages.slim.jsonl`
  (`record_count=180`, `natural_query_included=false`,
  `hard_negatives_included=false`, `quality_fields_included=[]`);
- kohorta pilota jest już w całości wykluczona z kohort `same_prompt_expansion`
  v2–v5 (`exclude_prior_cohort_ids`), więc korpus nie wchodzi w żadną trwającą
  generację ani w selekcję 50k SFT.

Uzasadnienie wyboru pilota: jego 4096 kandydatów jest już przescorowanych
zamrożonym primary/shadow/corpus, więc po zwolnieniu GPU ten korpus da się
przescorować **identycznym** kontraktem i porównać rozkłady bez nowej generacji
studenta.

Ograniczenie chroniące prospektywność: autor zapytań (koordynator i wszystkie
podsesje) **nie czyta** `artifacts/task06/candidate_pilot_v1/**/scoring/**`,
`selection/**` ani `natural_primary.jsonl` przed zamknięciem korpusu. Ścieżki te
są w instrukcji podsesji jawnie zabronione. Naturalne `query` z kohorty również
nie są widoczne — chudy plik ich nie zawiera.

### Klasy (8 slotów na pasaż, dokładnie po jednym)

| slot | klasa | konstrukcja |
|---|---|---|
| 0 | `good_specific` | poprawne, konkretne zapytanie odpowiadalne z pasażu, bez kopiowania długich fragmentów |
| 1 | `good_alternative` | drugie poprawne zapytanie celujące w **inny** fragment/aspekt pasażu |
| 2 | `near_duplicate_of_good` | parafraza slotu 0 o docelowej lemma Jaccard ≥ 0.90 |
| 3 | `too_general` | językowo poprawne, tematycznie zgodne, ale odpowiadalne z wielu tysięcy pasaży |
| 4 | `ungrounded` | wymaga informacji **nieobecnej** w pasażu; podtyp `hallucinated_fact` dla `order_index` parzystych, `unanswerable` dla nieparzystych |
| 5 | `copy_verbatim` | dosłowny, długi fragment pasażu podany jako zapytanie |
| 6 | `wrong_focus` | poprawne i gruntowane zapytanie, ale celujące w bucket **inny** niż zadeklarowany `declared_focus_bucket` (deklaracja: pierwszy bucket; cel: zdanie środkowe, a przy < 3 zdaniach ostatnie) |
| 7 | `wrong_form` | treściowo sensowne, ale łamiące formę: prefiks („Zapytanie:”), metakomentarz, wiele zapytań lub jawna niezgodność `full_question`/`keyword_query` z deklaracją |

Każdy rekord zapisuje: `cluster_id`, `slot`, `label`, `sublabel`,
`declared_form`, `declared_focus_bucket`, `query`, `construction_note`
(jednozdaniowe uzasadnienie autora) oraz `author_model`.

### Prerejestrowane predykcje

Predykcje są **porządkowe w obrębie grupy jednego pasażu**, więc nie zależą od
zewnętrznej kalibracji pasm naturalnych. Sprawdzane wyłącznie na CPU:

- **P1 (kopiowanie).** `copy_verbatim` ma najwyższy `copy_density`
  i `longest_copied_ngram` w swojej grupie — w ≥ 90% grup.
- **P2 (specyficzność).** `too_general` ma niższy `content_jaccard`
  **i** niższy `entity_preservation` niż `good_specific` — w ≥ 80% grup.
- **P3 (focus).** Dla `wrong_focus`: `assign_focus(query, passage).bucket`
  różni się od `declared_focus_bucket` — w ≥ 70% rekordów. Dla `good_specific`
  z zadeklarowanym bucketem: zgadza się — w ≥ 70% rekordów.
- **P4 (format).** `format_metrics(query)["format_valid"]` jest `False` dla
  ≥ 80% rekordów `wrong_form` i `True` dla 100% rekordów `good_specific`,
  `good_alternative`, `too_general`.
- **P5 (forma powierzchniowa).** `query_style(query)` zwraca etykietę zgodną
  z `declared_form` dla ≥ 80% rekordów slotów 0, 1 i 3.
- **P6 (near-duplicate).** Normalizacja bramki różnorodności
  (`near_duplicate_lemma_jaccard = 0.90`) skleja slot 2 ze slotem 0 —
  w ≥ 70% grup.
- **P7 (bramka nie karze legalnej różnorodności).** Grupa 8 rekordów o różnych
  klasach przechodzi bramkę o **niezmienionych** progach — w ≥ 95% grup.
- **P8 (odroczone do GPU).** `ungrounded` ma niższy primary score niż
  `good_specific` w ≥ 85% grup; `too_general` ma słabszy corpus round-trip niż
  `good_specific`. Ta predykcja jest zapisana teraz, ale mierzona dopiero po
  zwolnieniu GPU przez kolejkę bezobsługową.

Wynik każdej predykcji jest raportowany liczbowo, także gdy nie przechodzi.
Nieprzejście predykcji **nie** jest podstawą do zmiany progów: znaczy albo że
komponent jest słabszy niż założono, albo że konstrukcja autora jest
niedoskonała, a rozstrzygnięcie tych dwóch wymaga osobnej pracy.

### Granice

- Korpus **nie może** rekalibrować bramki różnorodności
  (`recalibration_forbidden: true` w
  `task06_same_prompt_diversity_gate_v1.yaml`) ani żadnego zamrożonego progu,
  w szczególności `source_en_score >= 23.50`.
- Korpus **nie wchodzi** do frozen train, do żadnej kohorty preferencyjnej,
  do par DPO ani do danych treningowych jakiegokolwiek modelu. Jest zbiorem
  diagnostycznym.
- Etykiety pochodzą od modelu, nie od człowieka; nie wolno ich nazywać
  human evidence ani panelem kalibracyjnym.
- Autor etykiet nie może być sędzią mierzącym te etykiety w audycie dual-LLM
  (self-preference bias). Sędziami pozostają `gpt-oss-120b` i `qwen3.6-27b`.
- `final_tests_used=[]`; `task07_training_authorized=false`;
  `task09_authorized=false`. Task 08 pozostaje `BLOCKED` do własnej decyzji
  `reports/decisions/enable_grpo.md` — ten korpus jest materiałem
  przygotowawczym, nie otwiera GRPO.

## Kolejność wykonania

1. **CPU, wykonane:** materializacja 180 chudych pasaży + plan 6 shardów.
2. **Tokeny, to okno:** generacja 6 × 30 × 8 = 1440 rekordów przez podsesje,
   każda do własnego pliku `shards/shard_NNN.jsonl` (brak współdzielonych
   zapisów, przerwanie kosztuje najwyżej jeden shard).
3. **CPU:** walidacja schematu i kompletności, scalenie do `corpus.jsonl`
   z manifestem i hashami.
4. **CPU:** pomiar predykcji P1–P7 i raport w
   `reports/measurements/task06_reward_validation_corpus_v1.md`.
5. **GPU, po kolejce, nieautoryzowane tym ADR:** P8.
