# Task 01 — Kontrakt danych, audyt, deduplikacja i splity

> [Centralny rejestr zadań i statusów](README.md). Każda zmiana statusu lub zakresu tego zadania musi aktualizować rejestr w tym samym commicie.

## Status

`IMPLEMENTED`

End-to-end smoke test i pełne przetworzenie `msmarco_pl` są zweryfikowane.
Z 388 699 rekordów źródłowych adapter po jawnym filtrze
`pos_score >= 23.50` zapisał 325 811 rekordów, a walidacja zaakceptowała
325 451. Zamrożone splity v1 zawierają 292 907/16 272/16 272 rekordów
train/dev/test, mają zero leakage pozytywnych dokumentów i zostały odwrócone
do 384 576/21 241/21 276 par doc2query.

Zadanie pozostaje `IMPLEMENTED`, a nie `DONE`, ponieważ nie wykonano jeszcze
raportu `data_audit.json/html` z percentylami trzech tokenizerów. Cleanup
cross-split negative'ów pozostawił mniej niż dziesięć negatywów w
29 048/9 674/9 640 rekordach train/dev/test; przed benchmarkami wymagającymi
10+ negatywów trzeba je odfiltrować lub jawnie uzupełnić. Artefakty danych są
lokalne i nie należą do Git.

## Cel

Zbudować niezawodny pipeline, który przyjmuje 500 tys. przykładów query–positive–hard-negatives, waliduje je, wykrywa duplikaty, tworzy splity bez leakage i odwraca dane do formatu passage→query.

## Zależności

Task 00.

## Wymagane skrypty

- `scripts/validate_dataset.py`
- `scripts/build_document_index.py`
- `scripts/deduplicate_documents.py`
- `scripts/build_splits.py`
- `scripts/invert_doc2query_pairs.py`
- `scripts/build_data_report.py`

Każdy skrypt musi mieć odpowiednik w bibliotece `src/doc2query/data/` i być cienkim wrapperem CLI.

## Walidacja wejścia

Sprawdź:

- obecność ID;
- typy pól;
- pusty query/pasaż;
- liczbę pozytywów i negatywów;
- powtarzające się doc_id;
- ten sam dokument jako pozytyw i negatyw;
- identyczny tekst pod różnymi ID;
- bardzo krótki lub bardzo długi tekst;
- niepolski język, jeżeli korpus ma być polski;
- znaki kontrolne, HTML, boilerplate i artefakty OCR;
- query będące niemal pełnym fragmentem pasażu;
- rozkład domen i źródeł.

Nie odrzucaj automatycznie bez raportu. Każda reguła ma tryb `warn`, `drop` lub `error` w configu.

## Deduplikacja

Zaimplementuj dwa poziomy:

1. exact dedup po znormalizowanym tekście i hashach;
2. near-duplicate clustering przez MinHash/LSH, SimHash lub równoważny skalowalny algorytm.

Nie wykonuj porównań O(n²). Zapisz:

- mapę `doc_id -> canonical_doc_id`;
- ID klastra;
- podobieństwo/rodzaj dopasowania;
- statystyki wielkości klastrów;
- próbkę ręczną największych klastrów.

## Splity bez leakage

Zbuduj graf, w którym query jest połączone z pozytywnymi dokumentami. Near-duplicate dokumenty należą do jednego komponentu. Komponent jest niepodzielną jednostką splitu.

Po przypisaniu splitu:

- usuń z train hard negative’y, które są pozytywami dev/test;
- usuń lub przepnij hard negative’y z niedozwolonego splitu zgodnie z configiem;
- sprawdź, że canonical_doc_id nie występuje w wielu splitach;
- sprawdź, że query ID nie występuje w wielu splitach;
- zachowaj rozkład domen przez group-stratified assignment, gdy to możliwe.

Domyślny rozkład: 90/5/5 albo 94/3/3, ale ma być konfigurowalny. Test i dev muszą pozostać zamrożone po utworzeniu wersji `v1`.

## Odwrócenie danych do doc2query

Dla każdego query i każdego pozytywu utwórz parę passage→query. Jeżeli query ma wiele pozytywów:

- zachowaj wszystkie pary;
- dodaj `positive_count`;
- zapewnij opcję capped sampling, aby jedno query nie dominowało;
- zachowaj listę jego hard negative’ów.

## Etykiety pomocnicze

Policz i zapisz:

- długości znakowe, słowne i tokenowe;
- query style według wstępnych reguł;
- podstawowy overlap słów i n-gramów;
- language confidence;
- liczbę zdań w pasażu;
- placeholdery na focus sentence i lematy.

Pełne przypisywanie focusu i lematów może nastąpić w Task 02, ale schemat musi je przewidywać.

## Raport danych

Wygeneruj `reports/data_audit.html` i `reports/data_audit.json` zawierające:

- liczbę rekordów przed/po każdej transformacji;
- rozkłady długości;
- percentyle tokenów dla tokenizerów 1.5B, 4.5B i 7B;
- udział query wysokiego overlapu;
- udział near-duplicate’ów;
- liczbę konfliktów splitu;
- rozkład stylów i domen;
- 50 przykładów typowych i 50 podejrzanych.

## Testy

- malformed records;
- duplicate positive/negative;
- connected component crossing split;
- test positive pojawiający się jako train negative;
- near-duplicate docs w różnych splitach;
- deterministyczny split dla tego samego seed/fingerprint;
- zmiana danych powoduje zmianę fingerprintu;
- odwrócenie multi-positive nie gubi par.

## Kryteria akceptacji

- zero canonical documents między splitami;
- zero pozytywów testowych użytych jako train negatives;
- identyczny wynik splitu przy ponownym uruchomieniu;
- raport percentyli tokenów pozwala wybrać `max_length` na podstawie danych;
- pipeline obsługuje pełny plik strumieniowo lub w shardach bez potrzeby ładowania wszystkich tekstów do RAM.

## Artefakty

- `data/processed/<version>/train.parquet`
- `data/processed/<version>/dev.parquet`
- `data/processed/<version>/test.parquet`
- `data/processed/<version>/documents.parquet`
- `data/processed/<version>/split_manifest.json`
- `data/processed/<version>/dedup_map.parquet`
- raport audytu.

Danych nie commituj do Git.
