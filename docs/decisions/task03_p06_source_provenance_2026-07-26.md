# ADR: P-06 — autorytet source score i audyt integralności tłumaczeń

Data: 2026-07-26

Status: `ACCEPTED — SUPERSEDES P-06 MASS RESCORING`

Zakres: frozen train v1 i przyszły audyt wyłącznie train/dev;
`final_tests_used=[]`

## Ustalenia provenance

`speakleash/msmarco_pl` powstał z użyciem silnego rerankera na angielskich
danych: pozytyw został oceniony przed kopaniem negatywów, a negatywy musiały
mieć score niższy o wymagany margin. Repozytorium zachowuje te wartości jako
`source_en_score` i `source_en_difference_between_max_scores`.

Audyt `data/processed/v1/train.parquet` potwierdził:

- 292 907 query i 384 576 query–positive pairs;
- 2 896 363 odziedziczone hard negative'y;
- brak brakujących source score'ów;
- 0 pozytywów poniżej `23.50`; minimum `23.50`, mediana `27.0`;
- źródłowy margin max-positive minus max-negative: minimum `6.0`, p05
  `6.375`, mediana `8.75`, p95 `11.0`;
- 0 query z niedodatnim źródłowym marginem.

Adapter już egzekwuje inkluzywny próg `source_en_score >= 23.50`. Założenie,
że pełny train wymaga ponownego automatycznego czyszczenia słabszym lokalnym
rerankerem, było błędne.

## Decyzja

1. P-06 `ordinary/drop/weighted` według lokalnego primary marginu jest
   `SUPERSEDED` i nie jest bramką przed skalą.
2. Nie kończyć pełnego scoringu `artifacts/task03/p06/train_margins_v1`.
   Niekompletny journal jest wyłącznie przerwanym artefaktem diagnostycznym;
   nie wolno użyć go do selekcji, wag SFT ani kalibracji progu.
3. Source score i source margin są autorytatywne dla etykiet dokumentów.
   Lokalny słabszy sędzia nie może ich nadpisywać.
4. Lokalny primary/shadow pozostaje właściwy do oceny nowych syntetycznych
   query, dla których source score nie istnieje.
5. Ryzyko uszkodzenia relacji przez tłumaczenie badamy osobno jako `P06-T`:
   mały, ślepy audyt integralności tłumaczeń, nie masowy filtr.

## Prospektywny kontrakt P06-T

Zamrozić przed lokalnym scoringiem próbkę 300 rekordów train, seed 42, bez
testów finalnych, po 75 rozłącznych rekordów:

1. najniższy decyl `source_en_score` pozytywu;
2. najniższy decyl źródłowego marginu, po wyłączeniu stratum 1;
3. rekordy z `text_quality_flags`, a przy niedoborze deterministyczny fill z
   najwyższego surface translation-risk, po wyłączeniu wcześniejszych;
4. losowa kontrola z pozostałych rekordów.

Artefakt musi zachować ID, stratum, source provenance i fingerprint, ale
formularz oceny ma być ślepy na source score i lokalne score'y. Pola oceny:

- czy polskie query zachowuje spójną intencję;
- czy można odpowiedzieć z polskiego pozytywnego pasażu;
- czy tłumaczenie query/pasażu jest semantycznie uszkodzone;
- czy występuje błąd kodowania lub tekstu;
- opcjonalna klasa powtarzalnego błędu i krótka uwaga.

Na zamrożonej próbce można policzyć primary/shadow disagreement i istniejące
heurystyki wyłącznie do kolejności ręcznego przeglądu. Nie ustalać progu drop,
nie trenować `weighted`, nie zmieniać frozen train i nie używać finalnego testu.

Jeżeli ręczny audyt nie wykaże powtarzalnej, automatycznie wykrywalnej klasy
błędu, dane pozostają bez zmian. Jeżeli ją wykaże, osobny prospektywny ADR ma
zdefiniować precyzyjny filtr i porównanie z ordinary control na dev. Sam niski
lokalny margin nigdy nie wystarcza.

## Wpływ na bramki

P-06 mass rescoring jest zamknięty bez eksperymentu SFT. P06-T zastępuje go
jako audyt danych przed nową kampanią 4.5B, ale nie otwiera `dev_confirm` ani
testów finalnych. Task 09 nadal czeka również na pełną bramkę hard negative'ów
i pozostałe zależności wskazane w centralnym rejestrze.
