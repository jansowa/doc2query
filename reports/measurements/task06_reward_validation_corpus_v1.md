# Pomiar: korpus walidacyjny komponentów nagrody v1 (2026-08-14)

ADR (prerejestrowany przed generacją i pomiarem):
[`task06_reward_validation_corpus_v1.md`](../decisions/task06_reward_validation_corpus_v1.md).
Kontrakt: `configs/rewards/reward_validation_corpus_v1.yaml`.
Artefakty: `artifacts/task06/reward_validation_corpus_v1/corpus.jsonl`
(`sha256=5b92a8d4c595d786…`), `measurement.json`.

## Co zmierzono

1440 zapytań (180 pasaży × 8 klas), po 180 rekordów na każdą z ośmiu klas,
napisanych przez `claude-opus-5[1m]` w sesji 2026-08-14 z etykietami nadanymi
z konstrukcji. Pasaże: pierwsze 180 klastrów kohorty `candidate_pilot_v1`
w porządku `sha256(cluster_id)`, split `train`. Autor nie czytał scoringu ani
selekcji pilota, nie widział naturalnych `query` ani hard negatywów. Pomiar
w całości CPU-only, `final_tests_used=[]`.

## Wyniki prerejestrowanych predykcji

| predykcja | wynik | próg | mianownik | ocena |
|---|---|---|---|---|
| P1 kopiowanie | **0.9611** | ≥ 0.90 | 180 grup | PASS |
| P2 specyficzność | **0.0000** | ≥ 0.80 | 180 grup | **FAIL** |
| P3 focus — `wrong_focus` | **0.9766** | ≥ 0.70 | 171 rekordów | PASS |
| P3 focus — `good_specific` | **0.6222** | ≥ 0.70 | 180 rekordów | **FAIL** |
| P4 format — `wrong_form` | **0.7500** | ≥ 0.80 | 180 rekordów | **FAIL** |
| P4 format — klasy dobre | **1.0000** | = 1.00 | 540 rekordów | PASS |
| P5 forma powierzchniowa | **1.0000** | ≥ 0.80 | 540 rekordów | PASS |
| P6 near-duplicate | **1.0000** | ≥ 0.70 | 180 grup | PASS |
| P7 bramka nie karze różnorodności | **1.0000** | ≥ 0.95 | 180 grup | PASS |

Bramka różnorodności o **niezmienionych** progach przepuściła wszystkie 180 grup;
zbiór przyczyn odrzuceń jest pusty. Progów nie kalibrowano ani przed, ani po
pomiarze.

## Trzy nieprzejścia i ich diagnoza

Diagnozy poniżej są **post-hoc** i nie zmieniają werdyktów z tabeli.

### P2 — `entity_preservation` nie mierzy specyficzności (wada predykcji, nie komponentu)

Predykcja wymagała koniunkcji: `too_general` ma jednocześnie niższy
`content_jaccard` **i** niższy `entity_preservation` niż `good_specific`.
Rozkład składowych:

- `content_jaccard` niższy w **154/180** grup (0.856);
- `entity_preservation` niższy w **0/180** grup; **remis w 180/180**.

Przyczyna jest w definicji metryki: `_preservation` liczy, jaki odsetek encji
*zapytania* występuje w pasażu, z konwencją `empty=1.0`. Zapytanie zbyt ogólne
z definicji nie zawiera encji, więc dostaje wynik doskonały 1.0 — dokładnie tyle
samo, ile poprawne zapytanie z encjami obecnymi w pasażu.

Wniosek dla projektu nagrody: `entity_preservation` jest **detektorem
halucynowanych encji**, a nie sygnałem specyficzności, i nie wolno go używać do
karania ogólności. Sygnał ogólności trzeba budować osobno (kandydat: samo
`content_jaccard`, które rozdzieliło klasy w 85.6% grup, albo IDF/statystyka
korpusowa). To jest znalezisko wprost dla wielokryterialnej nagrody Task 08:
naiwne dodanie `entity_preservation` do sumy nie poprawi specyficzności,
a przy zapytaniach bez encji jest wręcz nagrodą darmową.

### P3 (`good_specific`) — dominuje focus nierozstrzygalny, nie błędny

Rozkład `assign_focus` dla klasy `good_specific` (deklarowany → wykryty):

| deklarowany | wykryty | liczba |
|---|---|---|
| beginning | beginning | 104 |
| beginning | **brak (None)** | 45 |
| beginning | middle | 16 |
| beginning | end | 6 |
| middle | middle | 8 |
| middle | brak (None) | 1 |

Zgodność 112/180 = 0.6222. Najliczniejszy tryb porażki to **46 rekordów bez
przypisania** — `assign_focus` zwraca `None`, gdy brak unikalnego zwycięzcy albo
gdy najlepszy wynik jest poniżej `minimum_confidence`. Mediana `confidence`
wynosi 0.4286, czyli sygnał jest z natury słaby.

Współprzyczyna jest w danych: rozkład `sentence_count` w kohorcie to 9 pasaży
jednozdaniowych i 30 dwuzdaniowych na 180. Przy jednym zdaniu `focus_bucket`
zwraca stale `middle`, więc etykieta focus nie ma treści; przy dwóch zdaniach
`beginning` i `end` sklejają się z całym pasażem.

### P4 (`wrong_form`) — `_PREFIX` wymaga dwukropka, więc „Oto …” przechodzi jako poprawne

Wszystkie 45 rekordów `wrong_form`, które `format_metrics` uznał za
`format_valid=True`, należą do **jednego** wariantu: `prefix_oto`. Pozostałe trzy
warianty (`prefix_zapytanie`, `meta_na_podstawie_pasazu`, `meta_wygenerowalem`)
zostały wykryte w 135/135 przypadków.

Przyczyna: `_PREFIX` w `src/doc2query/evaluation/format.py` wymaga po słowie
kluczowym dwukropka lub myślnika (`(?:zapytanie|…|oto|wygenerowane)\s*[:\-]`).
Wtrącenie „Oto …” bez dwukropka nie jest ani prefiksem, ani żadną z czterech
zaszytych fraz `_META`, więc przechodzi jako format poprawny. Wynik 0.75 to nie
szum — to dokładnie 3 z 4 wykrywanych wariantów.

Wniosek: `format_valid` jest twardy na formy, które faktycznie widziano
w outputach modeli lokalnych, ale nie jest odporny na parafrazę wtrącenia. Jako
składnik nagrody w GRPO byłby podatny na obejście przez lead-in bez interpunkcji.
Nie zmieniam teraz `format.py`: zmiana detektora dotknęłaby interpretacji
zamrożonych pomiarów `format_valid_rate` w Taskach 04 i 05 i wymaga własnego,
prospektywnego ADR.

## Znalezisko poboczne: segmentacja zdań psuje etykiety focus

Wszystkie sześć niezależnych podsesji autorskich zgłosiło ten sam problem, więc
raportuję go jako obserwację jakościową (nie jest to prerejestrowany pomiar):
`split_sentences` rozcina zdania na skrótach (`np.`, `r.`, `łac.`, `dr.`, `Inc.`,
`p.n.e.`) i na numeracji, produkując pseudo-zdania w rodzaju `.`, `1.`, `26.`,
`Zawartość.`, `Terminy i opieka.`. Skutki:

- `sentence_count` jest systematycznie zawyżony, więc reguła progowa
  „`sentence_count < 3` oznacza pasaż zdegenerowany” nie chroni pasaży, które
  faktycznie mają jedną myśl;
- `focus_buckets[0]` i `focus_buckets[-1]` często wskazują nagłówek, podpis
  zdjęcia, pasek nawigacyjny albo osieroconą kropkę, a nie treść — stąd
  46 nierozstrzygniętych focusów w P3;
- część pasaży ma dosłownie zduplikowane zdania, przez co `beginning` i `end`
  wskazują ten sam tekst.

Dotyczy to nie tylko tego korpusu, ale wszystkich etykiet focus w Taskach 05–06.
Nie „naprawiam” splittera: zmiana przeliczyłaby zamrożone artefakty i kohorty.
Jeśli właściciel zdecyduje inaczej, wymaga to prospektywnego ADR i osobnego
przeliczenia, nie cichej poprawki.

## Granice tego pomiaru

- Etykiety pochodzą od modelu, nie od człowieka; to **nie** jest human evidence
  ani panel kalibracyjny Task 02.
- Autor korpusu nie może być sędzią w audycie dual-LLM par zawierających te dane.
- Korpus jest diagnostyczny: nie wchodzi do frozen train, do kohort
  preferencyjnych ani do par DPO; nie trenowano na nim niczego.
- Nieprzejście predykcji **nie** upoważnia do zmiany żadnego zamrożonego progu.
- P8 (primary/shadow/corpus round-trip dla klas `ungrounded` i `too_general`)
  pozostaje niezmierzone: GPU jest zajęte kolejką bezobsługową
  `unattended_queue_2026-08-14`. Predykcja jest zapisana w ADR i mierzalna później
  bez zmiany korpusu.
