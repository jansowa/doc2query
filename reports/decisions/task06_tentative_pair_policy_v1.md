# Task 06 — polityka par `chosen`/`rejected` (ADR v1, 2026-08-16)

## Kontekst i autoryzacja

Właściciel autoryzował 2026-08-16 trzy rzeczy naraz: zamrożenie polityki
`chosen`/`rejected`, zbudowanie tentative par i ślepy audyt dual-LLM. Do tej
pory `dpo_pair_selector` miał status `not_frozen_not_authorized`, a
`tentative_pair_build_authorized=false`.

Materiał wejściowy jest kompletny i zamknięty: kohorty same-prompt v1–v11 są
wygenerowane i ocenione (primary/shadow/corpus), a bramka różnorodności o
zamrożonych progach ([`task06_same_prompt_diversity_gate_v1.md`](task06_same_prompt_diversity_gate_v1.md))
przepuściła **25992 grupy** (v1 362/500, v2 466/500, v3–v11 25164/27000).

Ten ADR zamraża politykę **przed** odczytem jakiejkolwiek pary. Jest
prospektywny: opisuje regułę, nie wynik.

## Co było widoczne przed zamrożeniem progów

Uczciwe wyliczenie, w duchu sekcji o tym samym tytule w ADR bramki:

- **schemat** `scoring/per_generation.jsonl` i `generations.jsonl`: nazwy pól,
  typy, jeden przykładowy rekord odczytany w całości przy ustalaniu kontraktu
  wejściowego (jego `pool_margin` = 1.313126564025879);
- **brzegowy rozkład** `judges.primary_margin` z `scoring/summary.json` kohorty
  v1 (mean 3.405, p25 1.308, p50 3.415, p75 5.470, min −9.213, max 16.144) —
  odczytany przy sprawdzaniu, jakie pola summary nadają się do przypięcia
  artefaktu. Jest to rozkład **pojedynczych kandydatów w całej kohorcie**, a nie
  rozkład różnic wewnątrz grup;
- grupowe rozkłady różnorodności (te same, które uzasadniły bramkę);
- zamrożony artefakt kalibracji Task 02
  `artifacts/task02/pfn_dev_v1/calibration.json` (`score_kind: raw_pair_logit`,
  próg 8.617486953735352, query-macro TPR 0.901 / FPR 0.067);
- zamrożony kontrakt `copy_risk` z Task 05.

**Nie** odczytano: żadnej różnicy marginesów wewnątrz grupy, żadnego rankingu
kandydatów w grupie, żadnej pary, żadnej liczby grup przechodzących politykę
przy jakimkolwiek kandydacie progu. Próg `min_margin_gap` został wyprowadzony z
argumentu o skali log-odds (niżej), nie z powyższego rozkładu.

## Decyzja

Zamraża się politykę `task06-tentative-pair-policy-v1`, zaimplementowaną w
`src/doc2query/preferences/pair_policy.py`, uruchamianą przez
`scripts/build_task06_tentative_pairs.py`, z progami przypiętymi w
`configs/preferences/task06_tentative_pair_policy_v1.yaml`.

### 1. Zakres: gdzie w ogóle wolno budować pary

- wyłącznie **wewnątrz grupy same-prompt** — identyczny `prompt` i identyczny
  `prompt_sha256` dla obu kandydatów. Zestawianie W06 z D01 albo różnych
  kontrolek D01 pozostaje zakazane (kontrolka jest częścią promptu);
- wyłącznie z grup, którym bramka różnorodności nadała `eligible=true`;
- wyłącznie spośród **reprezentantów klastrów near-duplicate** wyznaczonych
  przez bramkę (`representative_candidate_ids`). Kandydaci wchłonięci do klastra
  nie mogą trafić do pary — to konstrukcyjnie wyklucza pary „ten sam tekst
  dwa razy”;
- **maksymalnie jedna para na grupę**, czyli jedna na prompt, jeden passage i
  jeden klaster near-duplicate;
- wyłącznie split `train`; `final_tests_used=[]`.

### 2. Primary jako jedyny sygnał budujący

Sygnałem porządkującym jest **wyłącznie** `pool_margin` z zamrożonego primary
(`sdadas/polish-reranker-roberta-v3`), czyli różnica surowego pair-logitu
pozytywu i najlepszego twardego negatywu tego rekordu. Nie powstaje żaden
`total_score`, żadna suma ważona i żadna kalibracja komponentów.

- `chosen` = kandydat o największym `pool_margin` wśród dopuszczalnych
  reprezentantów; remisy rozstrzyga `candidate_index`, potem `evaluation_id`;
- `chosen` musi mieć `pool_margin > 0` (primary stawia źródłowy passage nad
  wszystkimi twardymi negatywami);
- `rejected` = kandydat o **największym** `pool_margin` spośród tych, które są
  co najmniej `min_margin_gap` poniżej `chosen` (strategia `top_vs_near_miss`).

#### Dlaczego `min_margin_gap = 1.0`

`pool_margin` jest różnicą dwóch surowych pair-logitów tego samego sędziego
(`score_kind: raw_pair_logit` w zamrożonym artefakcie kalibracji Task 02).
Różnica marginesów dwóch kandydatów o wartości Δ oznacza, że jeden z nich
oddziela pozytyw od najtrudniejszego negatywu z szansami `e^Δ` razy lepszymi.
Zamrażamy Δ = **1.0**, czyli jedną naturalną jednostkę log-odds ≈ **2.72×**
lepsze szanse. Uzasadnienie jest skalowe, nie wydajnościowe:

1. jedna jednostka nat jest kanoniczną jednostką na skali logitowej i nie
   pochodzi z żadnego rozkładu tej kohorty;
2. jest rzędy wielkości powyżej szumu numerycznego zapisu score'ów, więc pary
   nie mogą powstać z zaokrągleń;
3. jest **sześciokrotnie mniejsza** niż minimalny margines pozytyw-vs-najtrudniejszy-negatyw
   wyegzekwowany we frozen train (6.0, Task 03/P-06). Różnica dwóch kandydatów
   dla tego samego passage'u ma być wyraźna, ale nie tak duża jak różnica między
   pozytywem a twardym negatywem.

Nie deklaruję, jaki odsetek grup ten próg przepuści; nie było to liczone przed
zamrożeniem i nie wolno tego użyć do zmiany progu po fakcie.

### 3. Shadow wyłącznie jako veto

`BAAI/bge-reranker-v2-m3` **nigdy** nie wybiera kandydata i nie wchodzi do
porządkowania. Para jest unieważniana (`shadow_veto`), gdy shadow zaprzecza
primary w którymkolwiek z dwóch sposobów:

- `shadow_pool_margin(chosen) < shadow_pool_margin(rejected)`;
- `shadow_pool_rank(chosen) > shadow_pool_rank(rejected)`.

Veto jest bezwzględne i fail-closed: przy sprzeczności sędziów para znika,
a nie „wygrywa większością”. Odsetek weta jest raportowany.

### 4. Corpus round-trip jako niezależny filtr

Sygnał niezależny od obu sędziów (BM25 round-trip w pełnym korpusie 2.4 mln
dokumentów):

- `chosen` wymaga `corpus_round_trip_at_20 == 1.0` — zapytanie faktycznie
  odzyskuje własny passage w top-20 całego korpusu;
- `rejected` wymaga `corpus_round_trip_at_100 == 1.0` — „gorszy, ale nadal
  minimalnie relewantny”, zgodnie z wymogiem specyfikacji, żeby rejected nie
  były wyłącznie nonsensowne i żeby para nie uczyła samej tematyczności.

Kandydat z wysokim primary, ale bez round-tripu jest wartościowym rejected typu
„zbyt ogólne” — polityka nie usuwa go, tylko nie pozwala mu być `chosen`.

### 5. Format, kopiowanie, focus

- oba role wymagają `format_valid == true`, `has_prefix == false`,
  `has_metacomment == false`, `multiple_query == false`, `empty == false`;
- dodatkowo **guard wtrącenia** `task06_lead_in_guard_v1`: kandydat, którego
  tekst po normalizacji zaczyna się od `oto` lub `otóż` (granica słowa), jest
  niedopuszczalny w obu rolach. To jest wprost domknięcie zmierzonej ślepej
  plamki `format_valid` z korpusu walidacyjnego nagrody (predykcja P4:
  45/45 rekordów wariantu `prefix_oto` przechodziło jako poprawne, bo `_PREFIX`
  wymaga dwukropka). Guard obejmuje **dokładnie zmierzony wariant** i jego
  najbliższą formę morfologiczną, nie jest ogólnym detektorem meta-komentarza,
  i **nie zmienia** `src/doc2query/evaluation/format.py`; wszystkie zamrożone
  pomiary `format_valid_rate` z Tasków 04–05 pozostają nietknięte;
- `chosen` nie może być `copy_risk` według **odziedziczonego bez zmian**
  kontraktu Task 05 (`copy_density > 0.6 i normalized_lcs > 0.8`, albo
  `longest_copied_ngram > 3`, albo `query/passage word ratio > 0.3838827838827845`,
  przy `word_length >= 4`). `rejected` **może** być copy_risk — kopiowanie jest
  jawnie wymienionym w specyfikacji źródłem rejected;
- `focus_accuracy` jest **słabym filtrem**: `chosen` z `focus_accuracy == 0.0`
  jest niedopuszczalny, ale `focus_accuracy is None` (abstencja `assign_focus`)
  **nigdy** nie karze kandydata. Podstawa: zmierzone 46/180 nierozstrzygniętych
  focusów i mediana `confidence` 0.4286 w korpusie walidacyjnym.

### 6. Komponenty jawnie wykluczone

- **`entity_preservation`** nie wchodzi do polityki w żadnej roli. Pomiar
  korpusu walidacyjnego jest wiążący: to detektor halucynowanych encji, a nie
  sygnał specyficzności (konwencja `empty=1.0` daje zapytaniu zbyt ogólnemu
  wynik doskonały; remis 1.0 w 180/180 grup);
- **`total_score`** nie istnieje i nie jest liczony;
- shadow nie jest kluczem sortowania.

### 7. Etykiety typu rejected (raportowane, nieselekcyjne)

Każda para dostaje listę etykiet wyprowadzonych z zamrożonych pól, wyłącznie do
raportu rozkładu i do analizy audytu według źródła rejected:
`lower_primary_margin` (zawsze), `weak_corpus_round_trip`,
`possible_ambiguous_query`, `copy_risk`, `lower_content_jaccard_than_chosen`
(sygnał ogólności — jedyny komponent, który w korpusie walidacyjnym rozdzielił
klasy `too_general`/`good_specific`, w 85.6% grup; użyty **relatywnie**, bez
wymyślania progu), `wrong_focus`, `shadow_agrees`, `judge_rank_disagreement`.
Etykiety nie wpływają na wybór pary.

## Kolejność wykonania (fail-closed)

1. Pary buduje się **najpierw wyłącznie** z kohort `same_prompt_expansion_v1` i
   `same_prompt_expansion_v2` (828 grup `eligible`). Tylko te dwie kohorty są
   wymienione w `authorized_cohorts` zamrożonego configu; builder odmawia
   uruchomienia na innej kohorcie.
2. Z uzyskanych par losuje się deterministycznie stratyfikowaną próbkę 500 par
   do audytu (ziarno 20260816, strata: kohorta × `requested_form` × pasmo
   `primary_margin_gap` z granicami [1,2), [2,4), [4,∞), alokacja proporcjonalna
   metodą największych reszt, porządek po `pair_id`).
3. Jeżeli kohorty v1+v2 dadzą **mniej niż 500 par**, audyt obejmuje wszystkie
   uzyskane pary, a niedobór jest raportowany. **Nie wolno** wtedy poluzować
   żadnego progu; rozszerzenie na kolejne kohorty wymaga osobnej decyzji
   właściciela zapisanej jako amendment do tego ADR.
4. Kohorty v3–v11 dostają pary **tą samą, niezmienioną polityką dopiero po
   pozytywnym audycie dual-LLM**. Do tego czasu `pair_build_authorized_cohorts`
   pozostaje dwuelementowe, a builder jest fail-closed.
5. Ten ADR **nie** autoryzuje treningu DPO. `task07_training_authorized=false`
   pozostaje bez zmian.

## Konsekwencje

- Progów nie wolno zmieniać po zobaczeniu liczby zbudowanych par ani po
  zobaczeniu wyniku audytu. Zmiana wymaga nowego, prospektywnego ADR.
- Odsetek grup, które nie dały pary, i histogram przyczyn są raportowane jawnie.
- Bramka różnorodności i `format.py` pozostają nietknięte.
- Artefakty par nie zawierają pól finalnych testów; `final_tests_used=[]`
  w każdym manifeście.
- Audyt dual-LLM jest evidence kalibracyjnym, nie sygnałem selekcji, i nie jest
  human evidence.

`final_tests_used=[]`. Etap jest w całości CPU; polityka nie ładuje modeli i nie
uruchamia GPU.
