# Task 05 — Kontrolowany styl, focus i generowanie wielu query

> [Centralny rejestr zadań i statusów](README.md). Każda zmiana statusu lub zakresu tego zadania musi aktualizować rejestr w tym samym commicie.

## Status

`IMPLEMENTED`

Kod, kontrakty, preset i tanie testy CPU są gotowe. Zaimplementowano rozdzielną
taksonomię `form`/`intent` z abstention i `intent_applicable`, kompatybilny
schemat evidence, focus F0–F3, kontrolowane SFT i inference, retry po
deduplikacji, ścisły multi-query JSON z jawną drobną naprawą, model-free
concept coverage oraz selektory top-N/MMR/coverage-aware. Przygotowano fail-closed nocny runner
D01 dla style/intent-only 1.5B/50k wraz z 3-step smoke i diagnostyczną
generacją na dev. Runner zakończył się kodem 0: oba treningi osiągnęły krok
3125, a pełne generacje objęły 6598 passage. D01 1.5B zapisał 26386 query i 6
exhausted groups, a D01 4.5B 26384 query i 8 exhausted groups. Jest to
wyłącznie wynik techniczny, nie porównanie kwalifikujące modele bez guardraili
i probe.
Plan i empiryczny budżet kolejki:
[`task05_d01_overnight_2026-07-26.md`](../reports/plans/task05_d01_overnight_2026-07-26.md).

Zaimplementowano również kompletny post-D01 pipeline dla dopasowanych runów
1.5B/W05 i 4.5B/W06: frozen-dev generation-only z passage-level journalem i
pełną identity, atomowy artefakt, progress/ETA/VRAM, wznawialny primary/shadow
scoring, kontrolne metryki formy/intencji z abstention, lexical/copy/retrieval,
disagreement i slice'y, matched-budget bootstrap report oraz fail-closed
materializację późniejszych probe inputs z przypiętym HN0+filter/drop. Dodano
fail-closed audyt ukończonych artefaktów, właściwe provenance W06 BS8,
quality-blind wspólną kohortę exact-K wszystkich czterech ramion, pełną
walidację corpus index oraz fazowy runner z lockiem i trwałym statusem. Nocny
runner nie czyta spłaszczonego `doc2query_dev.parquet` i nie uruchamia scoringu
automatycznie. Preflight post-D01 przeszedł. Pierwotną ścieżkę batch-1
przerwano po 220 passage i zachowano jej journal bez zmian. Zaimplementowano
osobną, crash-safe `batched-v2`: niezależne
strumienie RNG promptów, stały batch w identity, atomowy journal porcji i
odrębne nazwy artefaktów. Microbenchmark wybrał batch 16 (1.5B: 1.775
passage/s, 1.54 GB; 4.5B: 0.856 passage/s, 3.47 GB); nie jest to pomiar jakości
ani dowód bitowej równoważności różnych batchy. Pełne matched baseline v2
ukończyły po 6598 passage. Recovery akceptuje historyczne summary W05/W06 bez
pola `architecture`, ale nadal fail-closed sprawdza model ID, revision i
`trust_remote_code` oraz jawnie oznacza legacy provenance. Quality-blind
recovery ukończyło się z `rc=0`: wspólna kohorta zachowuje 5321
z 6598 grup, a każde z czterech ramion ma 21284 query. Nie wykonano
primary/shadow/corpus scoringu, comparison, materializacji probe
inputs ani probe training. Plan i komendy:
[`task05_d01_post_evaluation_2026-07-26.md`](../reports/plans/task05_d01_post_evaluation_2026-07-26.md).
Recovery ADR i bieżący runner:
[`task05_d01_post_campaign_2026-07-30.md`](../reports/plans/task05_d01_post_campaign_2026-07-30.md).

Niezależnie od D01 zaimplementowano i uruchomiono CPU-only pakiet kalibracji
naturalnych query na pełnym zamrożonym `dev_intrinsic_rank10` (6598 rekordów).
Prospektywny kontrakt przypina kohortę, seed, reguły `form`/`intent`, jawne
`unknown`/abstention, `intent_applicable`, stratyfikację i
`final_tests_used=[]`; wyniki D01 i finalne testy są zakazane. Zmaterializowano
ślepy formularz 500 etykiet, osobny machine key, formularz adjudykacji oraz
analogiczny pakiet dla 200 unikalnych pasaży i ekstrakcji koncepcji. Journale
są crash-safe i identity-bound, a agregatory liczą confusion/PRF, coverage,
accuracy po abstention, slice'y domenowe, reliability bins i Cohen/Fleiss
kappa oraz odmawiają statusu `complete` przy brakach lub nierozstrzygniętej
adjudykacji. Kontrakt dopuszcza jednego właściciela-oceniającego; wtedy kappa
pozostaje jawnie `NOT MEASURED`. Oba formularze nadal są puste: audyty 500/200 pozostają
`NOT MEASURED`, nie wyznaczono progu style accuracy. Powstał wznawialny,
quota-safe anotator Groq dla obu formularzy; jego docelowy run pozostaje do
zakończenia w osobnym procesie właściciela, a wyniki będą automatycznym proxy,
nie oceną człowieka. Post-D01 nie agreguje ani nie interpretuje jego aktywnych
journali. Plan, hashe i komendy:
[`task05_natural_audits_2026-07-26.md`](../reports/plans/task05_natural_audits_2026-07-26.md).

Do statusu `DONE` pozostają: eksperymenty D00–D12 na wspólnych kandydatach i
budżetach, rzeczywiste ręczne oceny i adjudykacja zmaterializowanych audytów
500 etykiet i 200 ekstrakcji koncepcji, human check oraz porównawcze probe
embeddera z CI. Opisowa kalibracja naturalnych query jest zmaterializowana,
ale nie zastępuje ręcznego pomiaru accuracy.

## Cel

Zwiększyć różnorodność zapytań i pokrycie całego pasażu bez utraty ugruntowania i możliwości odpowiedzi z pasażu.

## Zależności

Taski 02–04. Implementacja może powstawać równolegle z naprawami Task 04, ale
żaden eksperyment D00–D12 nie może wystartować przed ukończeniem Harness v1.1.

## Taksonomia formy i intencji

Nie mieszaj formy wypowiedzi z intencją. Zaimplementuj początkowo:

- `form`: `full_question`, `keyword_query`;
- `intent`: `fact_lookup`, `definition`, `entity_lookup`, `procedure`,
  `comparison` i rozszerzalne wartości domenowe.

Nie każda intencja pasuje do każdego pasażu. Dodaj `intent_applicable` i nie
wymuszaj niemożliwej intencji. Rozkład docelowy kalibruj na naturalnych query
per domena. Oś `retrieval_task` pozostaje poza zakresem do czasu korpusu
wielodomenowego.

Dodaj do schematu opcjonalne `evidence_sentence_ids`, `evidence_type`
i `evidence_confidence`, aby przyszłe rozszerzenie nie łamało cache'ów.

## Automatyczne etykietowanie naturalnych query

Pipeline:

1. reguły wysokiej precyzji;
2. opcjonalny mały klasyfikator;
3. `unknown` dla niepewnych;
4. ręczny audyt co najmniej 500 przykładów;
5. macierz pomyłek i confidence threshold.

## Focus controls

Porównaj:

### F0 — brak kontroli

Standardowy SFT.

### F1 — bucket

Prompt zawiera `beginning`, `middle` lub `end`.

### F2 — oznaczone zdanie

Pełny pasaż pozostaje w kontekście, ale zdanie docelowe jest oznaczone neutralnymi tokenami tekstowymi, np. `<FOCUS>...</FOCUS>`.

### F3 — sentence ID

Prompt podaje numer zdania i listę ponumerowanych zdań. Sprawdź, czy narzut tokenów jest akceptowalny.

Do treningu F2/F3 używaj tylko przykładów z pewnym focus assignment.

## Single-query generation

Podstawowa ścieżka produkcyjna:

```python
generate_one(passage, style, focus, length, seed) -> query
```

Dla K query uruchom macierz kontrolek, np.:

1. full_question + beginning;
2. keyword_query + middle;
3. fact_lookup + end;
4. styl najbardziej adekwatny + sentence o najwyższej niepokrytej informacji.

Następnie deduplikuj i, jeśli potrzeba, generuj brakujące query ponownie.

## Multi-query JSON

Zaimplementuj osobny eksperyment, gdzie completion ma format:

```json
{
  "queries": [
    {"text": "...", "style": "full_question", "focus_sentence_id": 1},
    {"text": "...", "style": "keyword_query", "focus_sentence_id": 3}
  ]
}
```

Wymagaj walidacji schematu i naprawy tylko drobnych błędów JSON. Nie ukrywaj invalid rate.

Porównaj multi-query JSON z K niezależnymi generacjami pod względem:

- jakości;
- pokrycia focus;
- duplikacji;
- kosztu tokenów;
- przepustowości;
- łatwości DPO/GRPO.

## Coverage-aware selection

Zaimplementuj selektor kandydatów maksymalizujący funkcję:

```text
sum quality(query_i)
+ alpha * semantic_diversity(set)
+ beta * focus_coverage(set)
+ gamma * style_coverage(set)
- duplicate_penalties
```

Użyj greedy submodular-like selection lub małego beam search. Nie wybieraj K najwyższych indywidualnych score, bo będą podobne.

## Eksperymenty

- D00: bez kontrolek;
- D01: style only;
- D02: focus bucket;
- D03: marked focus sentence;
- D04: style + focus;
- D05: K independent generations;
- D06: multi-query JSON;
- D07: K-independent + coverage-aware selection.
- D08: ekstrakcja koncepcji z lematów/encji/liczb i audyt około 200 pasaży;
- D09: stateful generation z niepokrytymi koncepcjami i poprzednimi query;
- D10: consistency filtering wsparty pokryciem koncepcji;
- D11: krzywa K=1/2/4/8(/16) osobno przy stałej liczbie pasaży i par;
- D12: top-N vs MMR vs coverage-aware na tych samych kandydatach.

Każdy na tym samym subset, seedach i budżecie generacyjnym.
Przed zamrożeniem configu diverse wykonaj jedną małą ablację: stałe top-p,
min-p (jeśli wspierane), miks 2–3 temperatur i stateful coverage. Top-k nie
jest osobną osią różnorodności.

## Kryteria sukcesu

Wariant przechodzi dalej, gdy:

- first-sentence concentration istotnie spada;
- focus entropy rośnie;
- style accuracy jest akceptowalna;
- grounding/source Recall@1 nie spada ponad ustaloną tolerancję;
- probe embedder nie pogarsza się statystycznie;
- invalid/duplicate rate pozostaje kontrolowany;
- człowiek potwierdza, że różnorodność nie jest sztuczna.

## Testy

- style parser/classifier;
- kontrolka focus trafia do promptu;
- deduplikacja diakrytyki/case/lematy;
- selector wybiera zróżnicowany set na toy example;
- JSON schema rejects invalid output;
- generator uzupełnia brakujące query po deduplikacji z limitem prób.
