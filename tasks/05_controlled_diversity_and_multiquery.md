# Task 05 — Kontrolowany styl, focus i generowanie wielu query

> [Centralny rejestr zadań i statusów](README.md). Każda zmiana statusu lub zakresu tego zadania musi aktualizować rejestr w tym samym commicie.

## Status

`IN PROGRESS`

Aktualizacja 2026-08-02 (prospektywna prerejestracja): metadata-only audyt
`dev_intrinsic` poza wykorzystanym `dev_intrinsic_rank10` znalazł 9591 z 9674
rekordów z co najmniej 5 hard negative'ami. Przed generacją zamrożono
deterministyczną kohortę 2000 rekordów (SHA-256, seed 20260802), selector z
commitu `2164822`, oba adaptery 1.5B, decoding, primary/shadow/PolDense/corpus,
paired bootstrap i wszystkie guardraile. Przecięcie z wcześniejszą kohortą
wynosi zero, a `final_tests_used=[]`. ADR:
[`task05_d01b_prospective_validation_1_5b_v1.md`](../reports/decisions/task05_d01b_prospective_validation_1_5b_v1.md).
Trwa implementacja crash-safe runnera; nie wygenerowano ani nie obejrzano
prospektywnych query. Probe, 4.5B i finalne testy pozostają niedozwolone.

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
z 6598 grup, a każde z czterech ramion ma 21284 query. Na tym etapie nie były
jeszcze wykonane primary/shadow/corpus scoring, comparison, materializacja
probe inputs ani probe training. Plan i komendy:
[`task05_d01_post_evaluation_2026-07-26.md`](../reports/plans/task05_d01_post_evaluation_2026-07-26.md).
Recovery ADR i bieżący runner:
[`task05_d01_post_campaign_2026-07-30.md`](../reports/plans/task05_d01_post_campaign_2026-07-30.md).

Pierwsza próba scoringu zatrzymała się fail-closed przed pierwszym wierszem,
ponieważ runner wskazywał train-only `bm25_train_v1`, w którym zgodnie z
kontraktem brakowało pozytywów dev. Pin poprawiono na istniejący dev-inclusive
`data/processed/v1/evaluation/corpus-bm25-v1`; ma on wszystkie 6250 wymaganych
doc ID wspólnej kohorty. Nie zinterpretowano żadnych metryk jakościowych.

Po korekcie indeksu pierwsze ramię scoringu zapisało trwale komplet 21284
wierszy, lecz końcowa bramka zatrzymała pipeline z powodu pominiętego pola
`primary_status` w agregatorze podsumowania. Pole i test regresyjny dodano;
wyniki sędziów były kompletne i nie wymagają ponownego liczenia. Scoring całej
pary nie był wtedy ukończony i nie zinterpretowano metryk jakościowych.

Przed pierwszym matched `compare` rozszerzono bramkę po wykryciu luki
metodologicznej: same lexical diversity i retrieval mogły premiować cztery
różne kopie fragmentów pasażu. Prospektywny kontrakt
`d01_copy_semantic_quality_v1` kalibruje ryzyko kopiowania wyłącznie na
naturalnych referencjach dokładnie tej samej frozen-dev kohorty, wymaga
absolutnej zgodności z naturalnym ogonem i względnej non-inferiority wobec
baseline'u, a semantic diversity liczy tylko na wspólnych grupach, w których
żadne z ośmiu query obu ramion nie ma flagi copy-risk. Przypięto
`OPI-PIB/PolDense-150M` revision
`b94ea7f951cc480369a85fa9021694eef80c3a00`, `trust_remote_code=false`.
Symetryczne query-query używa zgodnie z kartą modelu prefiksu `[sts]: `;
`[query]: ` jest przypięty dla asymetrycznego query-passage. Embeddingi są
normalizowane, identity-bound i atomowo cachowane. Bramka raportuje też
pairwise cosine, klastry semantyczne, passage-lemma-removed Jaccard oraz
materializuje ślepy audyt 100 high-retrieval/copy-risk przypadków. Nieudana
bramka blokuje probe inputs. CPU smoke prawdziwego modelu potwierdził output
`3x768`, skończone wektory o normie 1 oraz wyższe podobieństwo parafraz niż
niezwiązanego pytania; nie jest to wynik kampanii ani porównanie ramion.
Pomiar smoke:
[`task05_d01_poldense_smoke_2026-07-31.json`](../reports/measurements/task05_d01_poldense_smoke_2026-07-31.json).

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
pozostaje jawnie `NOT MEASURED`. Ręczne formularze nadal są puste i nie
wyznaczono progu style accuracy. Quota-safe anotator Groq ukończył automatyczne
proxy 500/500 etykiet oraz 200/200 audytów koncepcji; agregacje mają status
`complete`, ale nie zastępują oceny człowieka. Plan, wyniki, hashe i komendy:
[`task05_natural_audits_2026-07-26.md`](../reports/plans/task05_natural_audits_2026-07-26.md).

Do statusu `DONE` pozostają: niewidziana walidacja D01b, eksperymenty D00–D12
poza D01 na wspólnych kandydatach i budżetach, ewentualne rzeczywiste ręczne
oceny zmaterializowanych audytów 500 etykiet i 200 ekstrakcji koncepcji, human
check oraz porównawcze probe embeddera z CI. Opisowa kalibracja naturalnych
query jest zmaterializowana, ale nie zastępuje ręcznego pomiaru accuracy.

Aktualizacja 2026-08-02: pełny scoring i oba matched `compare` zakończyły się
`rc=0`. Kontrolowane ramiona 1.5B i 4.5B zwiększyły różnorodność i przeszły
anti-copy, lecz dostały `stop` z powodu istotnego spadku corpus round-trip@20
oraz sentence-level source hit. Nie zmaterializowano probe inputs.

Po zgłoszeniu hipotezy, że wysoki retrieval baseline'u może oznaczać zbyt łatwe
query, dodano retrospektywną diagnostykę D01b. Wykorzystuje ona zamrożone
naturalne marginesy Task 02 dla tego samego query/pozytywu/negatywów oraz
safe-anchor best-of-eight: cztery query baseline są kotwicą, a stały cel z PolDense
wybiera hybrydę tylko spośród kombinacji niepogarszających grupowych primary,
corpus, answerability, formatu i copy-risk. Shadow jest wyłączony z selekcji.
Na istniejącym dev hybryda wybrała 42.66%/42.25% query kontrolowanych i
poprawiła zarezerwowany shadow Recall@1 o 3.58/3.00 pp, jednocześnie zwiększając
różnorodność i zbliżając margin do naturalnych query. Ponieważ kontrakt powstał
po obejrzeniu D01 dev, raport pozostaje `promotion_eligible=false` i
`probe_materialization_authorized=false`. Wyniki:
[`task05_d01b_usefulness_2026-08-02.md`](../reports/measurements/task05_d01b_usefulness_2026-08-02.md).

Następna bramka to niezmieniony selektor na niewidzianej kohorcie rozwojowej,
wykluczającej 50k użyte do SFT i wszystkie finalne testy. Dopiero przejście
ordinary intrinsic oraz reserved-shadow non-inferiority może dopuścić
equal-budget probe hybrydy przeciw baseline'owi.

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
