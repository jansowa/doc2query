# Task 02 — Reranker, lematyzacja, focus i proxy ugruntowania

## Status

`TODO`

## Cel

Zbudować niezależne, skalibrowane komponenty oceniające, które będą używane do analizy generatora, budowy preferencji i ewentualnego RL.

## Zależności

Task 01.

## Część A — baseline rerankerów

Porównaj na zamrożonym dev/test co najmniej:

1. `sdadas/polish-reranker-base-ranknet`;
2. `BAAI/bge-reranker-v2-m3` lub inny silny model wielojęzyczny o zgodnej licencji;
3. własny lekki cross-encoder dostrojony na danych projektu.

Jeżeli model wymaga `trust_remote_code`, odnotuj ryzyko, przypnij revision i zapewnij opcję wyłączenia.

## Część B — trening własnego rerankera

Przygotuj:

- pozytywy `(query, positive_passage, 1)`;
- hard negative’y `(query, hard_negative, 0)`;
- opcjonalnie pary listwise/pairwise;
- sampling zapobiegający dominacji łatwych negatywów;
- osobny test niewidzianych query i dokumentów.

Minimalny wariant ma używać lekkiego modelu, który może później działać na CPU. Silniejszy wariant jest przeznaczony do offline scoringu.

Porównaj loss:

- binary cross entropy;
- RankNet/pairwise margin;
- opcjonalnie listwise loss, jeżeli implementacja jest stabilna.

Raportuj MRR, nDCG@10, Recall@1/5, AUC i kalibrację. Ranking jest ważniejszy niż accuracy na zbalansowanych parach.

## Część C — scoring zdaniowy i focus

Podziel pasaż na zdania. Dla naturalnego query oblicz score query–sentence i przypisz:

- `focus_sentence_id`;
- `focus_score`;
- `focus_margin` między pierwszym i drugim zdaniem;
- `focus_bucket`: beginning/middle/end;
- `focus_is_ambiguous`.

Nie używaj niepewnej etykiety jako twardej prawdy. Dla niskiego marginesu ustaw `ambiguous` i pomiń przykład w eksperymentach wymagających precyzyjnego focusu.

## Część D — lematyzacja i overlap

Zaimplementuj interfejs:

```python
class TextNormalizer(Protocol):
    def analyze(self, text: str) -> AnalyzedText: ...
```

Backendy:

- `simple`: Unicode, lower-case, tokenizacja, polskie stopwordy;
- `spacy_pl`: polski pipeline spaCy na CPU;
- opcjonalnie `stanza_pl` jako ablacją jakości.

Cache:

- pasaże: offline Parquet/SQLite/LMDB;
- query: batched `nlp.pipe`;
- klucz cache zawiera wersję modelu i config normalizacji.

Oblicz:

- content lemma sets i multisets;
- entity/number/unit tokens;
- Jaccard, overlap coefficient, precision, recall;
- longest copied n-gram;
- normalized LCS;
- copy density.

## Część E — kalibracja rewardów

Na naturalnym dev:

1. wyznacz rozkłady wszystkich komponentów;
2. wyznacz robust z-score lub percentylową normalizację;
3. zdefiniuj overlap reward jako funkcję pasmową opartą na percentylach naturalnych query;
4. sprawdź korelacje między komponentami;
5. nie dopuszczaj, aby całkowity score był praktycznie kopią jednego komponentu.

## Część F — test adwersarialny

Utwórz ręczny zestaw co najmniej 100 przypadków:

- query prawidłowe;
- query będące kopią zdania;
- query tematycznie podobne, lecz takie, na które nie można odpowiedzieć z pasażu;
- query z błędną liczbą lub encją;
- query zbyt ogólne;
- query dotyczące innego zdania;
- query zdradzające odpowiedź.

Każdy reward ma przewidywalnie reagować. Dodaj te przykłady jako testy regresyjne.

## Wymagane skrypty

- `scripts/benchmark_rerankers.py`
- `scripts/train_reranker.py`
- `scripts/calibrate_reranker.py`
- `scripts/assign_focus_labels.py`
- `scripts/precompute_text_analysis.py`
- `scripts/calibrate_rewards.py`

## Kryteria akceptacji

- własny reranker co najmniej nie pogarsza istotnie polskiego baseline’u na zamrożonym teście albo jest znacznie szybszy przy akceptowalnej stracie;
- scoring źródłowego pozytywu jest raportowany względem 10+ hard negative’ów;
- overlap reward nie nagradza ani pełnej kopii, ani całkowicie niezwiązanego query;
- lematyzacja działa na CPU i ma cache;
- wszystkie rewardy mają unit testy i zakresy liczbowe;
- istnieje raport korelacji automatów z ręcznymi etykietami.
