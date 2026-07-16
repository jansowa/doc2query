# Task 02 — Gotowy reranker, lematyzacja, focus i proxy ugruntowania

> [Centralny rejestr zadań i statusów](README.md). Każda zmiana statusu lub zakresu tego zadania musi aktualizować rejestr w tym samym commicie.

## Status

`IMPLEMENTED`

Benchmark na projektowych splitach dev/test oczekuje na artefakty Task 01 i
rzeczywiste inference modeli.

## Cel

Zintegrować i zwalidować gotowe, zamrożone modele oceniające. Mają one służyć do analizy generatora, budowy preferencji i ewentualnego RL. **Trening własnego rerankera nie należy do zakresu tego projektu.**

## Zależności

Task 01.

## Zasada nadrzędna

Nie implementuj `train_reranker.py`, nie aktualizuj wag rerankera i nie wykorzystuj danych projektu do uczenia cross-encodera. Dane z pozytywami i hard negative’ami służą tutaj do:

- wyboru gotowego modelu;
- pomiaru jakości i zgodności domenowej;
- kalibracji skali score’ów i progów;
- budowy marginesów rankingowych;
- testów adwersarialnych i wykrywania reward hackingu.

Ewentualne dostrajanie rerankera może powstać wyłącznie jako osobny projekt po udokumentowaniu, że wszystkie rozsądne gotowe modele zawodzą na ręcznie ocenionym holdoucie. Nie jest to automatyczny kolejny krok.

## Część A — wybór gotowych rerankerów

Jako domyślnego głównego sędziego przetestuj:

1. `sdadas/polish-reranker-roberta-v3` — preferowany polski baseline, długi kontekst, bez `trust_remote_code`;
2. jeden niezależny model kontrolny, np. `BAAI/bge-reranker-v2-m3` albo inny silny wielojęzyczny reranker o zgodnej licencji;
3. opcjonalnie drugi polski model od sdadas tylko jako dodatkowy punkt odniesienia, nie jako jedyne potwierdzenie wyników.

Dla każdego modelu:

- przypnij dokładny revision/commit;
- zapisz licencję i warunki użycia;
- odnotuj wymaganie `trust_remote_code` i domyślnie je wyłączaj;
- zmierz RAM, VRAM, throughput, maksymalną długość i zachowanie przy truncation;
- nie zakładaj, że surowe logity różnych modeli są porównywalne.

## Część B — benchmark na danych projektu

Na zamrożonym dev/test policz dla każdej grupy `query + positive + 10+ hard negatives`:

- Recall@1 i Recall@5 źródłowego pozytywu;
- MRR i nDCG@10;
- margines `score(positive) - max score(hard_negative)`;
- odsetek przypadków z marginesem ujemnym lub bliskim zera;
- wyniki według domeny, długości pasażu, typu query i poziomu trudności;
- korelację kolejności modeli na poziomie przykładów.

Reranker nie musi być perfekcyjny. Musi jednak wystarczająco często odróżniać źródłowy pasaż od hard negative’ów i mieć przewidywalne błędy. Jeśli żaden gotowy model nie spełnia tej bramki, nie przechodź automatycznie do treningu własnego modelu — najpierw sprawdź ensemble, dodatkowy answerability checker lub ograniczenie roli rerankera w rewardzie.

## Część C — kalibracja bez uczenia rerankera

Zaimplementuj kalibrację wyłącznie nad wyjściami zamrożonego modelu:

- robust z-score;
- mapowanie percentylowe;
- opcjonalnie Platt scaling lub isotonic regression na małym, odseparowanym zbiorze ręcznych etykiet;
- osobną kalibrację score’u absolutnego i marginesu rankingowego.

Kalibrator jest małym modułem statystycznym, nie nowym rerankerem. Nie używaj testu do wyboru progów.

## Część D — scoring zdaniowy i focus

Podziel pasaż na zdania. Dla naturalnego query oblicz score query–sentence i przypisz:

- `focus_sentence_id`;
- `focus_score`;
- `focus_margin` między pierwszym i drugim zdaniem;
- `focus_bucket`: `beginning/middle/end`;
- `focus_is_ambiguous`.

Nie używaj niepewnej etykiety jako twardej prawdy. Dla niskiego marginesu ustaw `ambiguous` i pomiń przykład w eksperymentach wymagających precyzyjnego focusu. Dla długich pasaży batchuj zdania i cache’uj wyniki.

## Część E — lematyzacja i overlap

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

## Część F — kalibracja rewardów

Na naturalnym dev:

1. wyznacz rozkłady wszystkich komponentów;
2. wyznacz robust z-score lub percentylową normalizację;
3. zdefiniuj overlap reward jako funkcję pasmową opartą na percentylach naturalnych query;
4. sprawdź korelacje między komponentami;
5. nie dopuszczaj, aby całkowity score był praktycznie kopią jednego komponentu;
6. raportuj osobno wynik głównego i kontrolnego rerankera.

## Część G — test adwersarialny i ręczny holdout

Utwórz ręczny zestaw co najmniej 150 przypadków, w tym:

- query prawidłowe;
- query będące kopią zdania;
- query tematycznie podobne, lecz takie, na które nie można odpowiedzieć z pasażu;
- query z błędną liczbą lub encją;
- query zbyt ogólne;
- query dotyczące innego zdania;
- query zdradzające odpowiedź;
- query, dla których główny i kontrolny reranker się nie zgadzają.

Każdy reward ma przewidywalnie reagować. Dodaj te przypadki jako testy regresyjne. Zmierz korelację score’ów z ręczną oceną answerability i trafności.

## Wymagane skrypty

- `scripts/benchmark_rerankers.py`
- `scripts/calibrate_reranker.py`
- `scripts/score_query_passages.py`
- `scripts/assign_focus_labels.py`
- `scripts/precompute_text_analysis.py`
- `scripts/calibrate_rewards.py`

## Kryteria akceptacji

- żaden skrypt nie trenuje ani nie modyfikuje wag rerankera;
- wybrany primary reranker ma przypięty revision i udokumentowaną licencję;
- wyniki źródłowego pozytywu są raportowane względem 10+ hard negative’ów;
- istnieje co najmniej jeden niezależny shadow judge albo jawne uzasadnienie jego braku;
- disagreement między sędziami jest raportowany i nie jest ukrywany przez uśrednienie;
- overlap reward nie nagradza ani pełnej kopii, ani całkowicie niezwiązanego query;
- lematyzacja działa na CPU i ma cache;
- wszystkie rewardy mają unit testy i zakresy liczbowe;
- istnieje raport korelacji automatów z ręcznymi etykietami.
