# Pomiar: inwentarz podaży defektów (V2-00) i walidacja focus_v2 (V2-02), 2026-08-17

Realizacja pierwszych dwóch zadań specyfikacji
[`task06_defect_anchored_pairs_v2_spec_2026-08-17.md`](../plans/task06_defect_anchored_pairs_v2_spec_2026-08-17.md).
Oba pomiary są **wejściem projektowym** przyszłego ADR V2-03: czytają pola
jakości jawnie, nie budują żadnej pary, nie zamrażają żadnego progu i nie
zmieniają żadnego zamrożonego artefaktu. Artefakty:
[`task06/defect_inventory_v1/summary.json`](task06/defect_inventory_v1/summary.json),
[`task06/focus_v2_validation/summary.json`](task06/focus_v2_validation/summary.json).
`final_tests_used=[]`.

## V2-00: podaż par defektowych w kohortach v1–v11

Populacja: wyłącznie reprezentanci bramki różnorodności w 25992 grupach
`eligible`, z pinowaniem SHA-256 scoringu i manifestów bramki (loadery v1).
„Para osiągalna” = istnieje (czysty `chosen`, defektowy `rejected`) spełniająca
zamrożony kontrakt parowania (Jaccard ≤ 0,85, różne ID).

| wielkość | grupy | % z 25992 |
|---|---|---|
| grupy z czystym `chosen` (format+guard, rt@20, margines>0, bez copy-risk) | 21102 | 81,2% |
| **oś A osiągalna** (defekt: brak rt@100) | **17669** | **68,0%** |
| grupy z kandydatem bez rt@100 | 22404 | 86,2% |
| **oś B osiągalna** przy cięciu p75 (`content_jaccard ≥ 0,0857`) | **4857** | 18,7% |
| oś B osiągalna przy cięciu p90 (`≥ 0,1212`) | 1900 | 7,3% |
| **oś C osiągalna** (wstępnie, stare etykiety focus) | **0** | 0% |
| grupy z halucynacją encji (`entity_preservation < 1`) | **0** | 0% |

Punkty cięcia osi B to **kandydaci** (p50 = 0,0556, p75 = 0,0857, p90 = 0,1212
na 101146 kandydatach odpowiadalnych-przez-proxy); jeden z nich zamrozi dopiero
ADR V2-03. Rozkład `content_jaccard` jest niski w wartościach bezwzględnych, bo
mianownikiem jest unia lematów treściowych całego pasażu — cięcia względne
pozostają dobrze określone.

Wnioski dla ADR V2-03:

1. **Oś A ma ogromną podaż naturalną** (17669 grup) — konstruowane rejected nie
   będą potrzebne do jej kwoty; sygnałem defektu jest brak round-tripu @100,
   a certyfikację `chosen` domknie sędzia odpowiadalności (V2-01).
2. **Oś B ma wystarczającą podaż** przy cięciu p75; cięcie p90 daje pary
   „czystsze”, ale czterokrotnie rzadsze.
3. **Oś C ma podaż zero na starych etykietach** — zgodnie z przewidywaniem
   specyfikacji jest w całości zablokowana na jakości etykiet focus (niżej).

### Znalezisko: `entity_preservation` jest na tych danych stałą z konstrukcji

Zero grup z halucynacją encji nie jest własnością generatora, tylko backendu:
scoring kohort używał `SimplePolishNormalizer`, którego `analyze()` zwraca
**zawsze `entities=()`**, więc `_preservation(query.entities, …)` z konwencją
`empty=1.0` daje 1,0 każdemu kandydatowi (potwierdzone: 4000/4000 wartości
dokładnie 1,0 w kohorcie v1). To **zaostrza** diagnozę z korpusu walidacyjnego
nagrody: `entity_preservation` nie tylko nie mierzy specyficzności — w tym
pipeline nie jest nawet detektorem halucynowanych encji, bo nie widzi żadnych
encji. Rola „detektora halucynacji” wymagałaby relabelingu backendem spaCy
(`SpacyPolishNormalizer` ekstrahuje `doc.ents`) jako nowego, wersjonowanego
artefaktu — do decyzji przy V2-03; oś A tymczasem opiera się na round-tripie
i sędzim odpowiadalności, których podaż i tak wystarcza.

## V2-02: focus_v2 — czystsza segmentacja, ale **kryterium akceptacji niedowiezione**

Zaimplementowano `src/doc2query/data/focus_labels_v2.py` (11 testów CPU): nowy,
wersjonowany splitter (`focus-v2:pl-abbrev-v1`) z wetem granic po polskich
skrótach, inicjałach/pojedynczych literach, numeracji 1–3-cyfrowej i przed
kontynuacją małą literą, plus scalanie fragmentów bez liter. Formuła scoringu i
semantyka abstencji `assign_focus_v2` są **bajtowo zgodne z v1**, więc różnice
etykiet izolują wpływ segmentacji. Nic zamrożonego nie zostało zmienione.

Segmentacja na 180 pasażach korpusu walidacyjnego poprawiła się obiektywnie:

| metryka | v1 | v2 |
|---|---|---|
| zdań łącznie | 716 | 659 |
| pseudo-zdania bez liter („.”, „26.”) | 8 | **0** |
| pasaże 1-zdaniowe | 9 | 12 |

Ale etykiety focus prawie się nie ruszyły — zmieniło się **13/351** przypisań
(9 rekordów `degenerate_single_sentence` wyłączonych z mianownika, jak w P3):

| klasa (definicja sukcesu) | v1 sukces | v2 sukces | v1 abstencja | v2 abstencja |
|---|---|---|---|---|
| `good_specific` (zgodność z deklaracją) | 0,6222 | 0,6000 | 0,2556 | 0,2500 |
| `wrong_focus` (wykrycie naruszenia) | 0,6491 | 0,6433 | 0,3275 | 0,3158 |

**Kryterium akceptacji V2-02 („abstencja istotnie niższa niż 26%”) nie zostało
spełnione**: abstencja spadła o 0,6–1,2 pp, a sukces minimalnie spadł (scalone
zdania zgrubiają kubełki — pasaży 2-zdaniowych przybyło z 30 do 39, a przy dwóch
zdaniach `beginning`/`end` sklejają się z całością). Raportujemy to wprost,
zamiast dostrajać `minimum_confidence` pod ten pomiar.

Diagnoza: wąskim gardłem etykiet focus **nie jest segmentacja, tylko słaby
scorer leksykalny** (mediana confidence 0,43 zmierzona już w korpusie
walidacyjnym). Konsekwencja dla specyfikacji v2:

- oś C pozostaje **zablokowana**; jej odblokowanie wymaga mocniejszego
  przypisywacza focusa (kandydat: wariant rerankerowy `reranker/focus.py`
  liczący score zapytania względem każdego zdania — koszt GPU, osobna
  prospektywna decyzja), albo świadomej rezygnacji z osi C w pierwszym wydaniu
  polityki v2 — wybór należy do właściciela przy ADR V2-03;
- `split_sentences_v2` pozostaje wartościowy niezależnie (eliminuje
  pseudo-zdania i zawyżone `sentence_count`) i będzie właściwym splitterem dla
  ewentualnego przypisywacza rerankerowego.

## Stan po tym pomiarze

- V2-00: **wykonane** — podaż osi A/B potwierdzona liczbowo, oś C zero,
  `entity_preservation` zdiagnozowane jako stała.
- V2-02: **zaimplementowane i zmierzone; akceptacja niedowieziona** — moduł
  zostaje (lepsza segmentacja), oś C czeka na decyzję przy V2-03.
- Krok 0 (dokończenie audytu v1: 128 + 43 requesty) czeka na odnowienie
  dziennych budżetów Groq o 00:00 UTC; komenda wznowienia bez zmian.
- Nic nie zbudowano, nic nie wypromowano, `task07_training_authorized=false`,
  `final_tests_used=[]`.
