# Task 05 — natural-query calibration i prospektywne audyty

## Stan

Pakiet CPU-only został zaimplementowany i uruchomiony niezależnie od D01.
Źródłem jest wyłącznie zamrożony `dev_intrinsic_rank10`; nie odczytano wyników
D01 ani testów finalnych. Kontrakt
`configs/evaluation/task05_natural_audits_v1.json` prospektywnie przypina seed
`20260726`, rozmiary 500/200, osie stratyfikacji, obsługę małych domen, wersje
reguł i ekstraktora, jawne `unknown`/abstention, `intent_applicable` oraz
`final_tests_used=[]`. Nie definiuje arbitralnej bramki style accuracy.

Pełna materializacja objęła 6598 naturalnych query. Powstały dokładnie:

- 500 rekordów ślepego audytu `form`/`intent` i oddzielny machine key;
- 200 unikalnych pasaży ślepego audytu koncepcji i oddzielne propozycje;
- dwa formularze adjudykacji i instrukcja kodowania;
- per-record JSONL, opisowa kalibracja JSON/Markdown, identity, fingerprinty,
  journale i atomowy manifest końcowy.

Artefakty są lokalne w `artifacts/task05/natural_audits_v1/` i celowo nie są
commitowane, ponieważ zawierają tekst danych. Manifest ma status
`materialized_unreviewed`; audyty etykiet i koncepcji są `NOT MEASURED`.

## Wynik opisowej kalibracji (nie accuracy)

Na 6598 rekordach reguły przewidziały 4453 `full_question`, 1720
`keyword_query` i 425 `unknown` dla formy. Dla intencji: 3888 `fact_lookup`,
1165 `definition`, 500 `entity_lookup`, 597 `procedure`, 36 `comparison` i 412
`unknown`. Łączne abstention wynosi 425. `intent_applicable=true` wystąpiło
dla 5270 rekordów, a 1328 pozostało nierozstrzygniętych (`null`). Frozen dev
ma tylko jedno źródło/domenę `speakleash/msmarco_pl`; raport nie udaje więc
kalibracji między domenami. Te liczby opisują automatyczne predykcje, nie ich
poprawność.

Kluczowe fingerprinty materializacji:

- frozen cohort: `235d9b81e04ddc5e74bd2bbe884055dd74f03b6706e6030e88a4f918ac2ffab6`;
- identity: `ad7906ac4782320597dba8afc2a73d44d9797fedf0f4abfde75a4adf07a20553`;
- label blind CSV: `9dececc677350e5f66be0b633de411fb6c9d18ae7152dd6bbd47df806a8783b2`;
- label machine key: `f8fe3e03fb55502e7ce79545788f169b7fe9582874280875279729fb00ab1204`;
- concept blind CSV: `a6f46ecce4f706ec7284c0553d63772e4983002214714cffc8c65953bebfe273`;
- concept proposals: `b5729433ae88d0a95bf424e2d342d2056a91b395d0b7aba9c37d30dd785c862f`.

## Wznawialność i fail-closed

Kalibracja i ekstrakcja mają osobne trwałe journale JSONL. Czytnik naprawia
wyłącznie crash-truncated ostatnią linię, a ukończony prefiks nie jest liczony
ponownie. Identity obejmuje kontrakt, frozen manifest/cohort, kolejność ID,
seed i `final_tests_used=[]`. Drift odmawia wznowienia; jawna opcja
`--archive-incompatible` przenosi częściowy stan do odzyskiwalnego archiwum.
Finały JSONL oraz CSV są zapisywane atomowo. Progress pokazuje licznik,
remaining, throughput i ETA w sekundach.

Agregatory nie zwracają `complete`, dopóki każdy przypadek nie ma co najmniej
dwóch niezależnych ocen i wszystkie rozbieżności nie mają adjudykacji. Raport
etykiet zawiera confusion matrix, precision/recall/F1 per klasa, coverage,
accuracy na nie-abstention, wyniki per domena, reliability bins i Cohen/Fleiss
kappa. Raport koncepcji obejmuje correct/spurious/missing, liczby/jednostki,
fragmentację, duplikaty i przydatność coverage-aware wraz ze zgodnością.

## Komendy

Pełna materializacja (już wykonana; komenda jest bezpiecznie wznawialna):

```bash
.venv/bin/python scripts/task05_natural_audits.py materialize \
  --contract configs/evaluation/task05_natural_audits_v1.json \
  --output-dir artifacts/task05/natural_audits_v1
```

Oceniający mają pracować na osobnych kopiach `label_audit_blind.csv` i
`concept_audit_blind.csv`, nie otwierając machine key/proposals. Po zebraniu
dwóch plików ocen i uzupełnieniu adjudykacji:

```bash
.venv/bin/python scripts/task05_natural_audits.py aggregate-labels \
  --machine-key artifacts/task05/natural_audits_v1/label_audit_machine_key.jsonl \
  --ratings path/to/labels_reviewer_1.csv path/to/labels_reviewer_2.csv \
  --adjudication artifacts/task05/natural_audits_v1/label_adjudication.csv \
  --output-dir reports/measurements/task05_natural_label_audit_v1

.venv/bin/python scripts/task05_natural_audits.py aggregate-concepts \
  --machine-proposals artifacts/task05/natural_audits_v1/concept_audit_machine_proposals.jsonl \
  --ratings path/to/concepts_reviewer_1.csv path/to/concepts_reviewer_2.csv \
  --adjudication artifacts/task05/natural_audits_v1/concept_adjudication.csv \
  --output-dir reports/measurements/task05_concept_audit_v1
```

Do kolejnej sesji należy wrócić z dwoma kompletnymi plikami ocen etykiet,
dwoma kompletnymi plikami ocen koncepcji oraz uzupełnionymi formularzami
adjudykacji. Dopiero wtedy wolno raportować ręczne accuracy/agreement i wynik
audytu ekstrakcji.
