# Task 05 — Kontrolowany styl, focus i generowanie wielu query

> [Centralny rejestr zadań i statusów](README.md). Każda zmiana statusu lub zakresu tego zadania musi aktualizować rejestr w tym samym commicie.

## Status

`TODO`

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
