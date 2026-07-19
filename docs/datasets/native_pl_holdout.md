# Natywny polski holdout — audyt i kontrakt v1

## Decyzja

Obowiązkowym kandydatem na `test_native_pl` jest **test PolQA**, a
`test_translated_msmarco_pl` jest aliasem dotychczasowego zamrożonego testu
MS MARCO-PL. Nie traktujemy całego PIRB ani całego MAUPQA jako danych
natywnych. `test_transfer_ood` nie jest częścią P-02 v1.

Na dzień 2026-07-19 oba zbiory są materializowane i zweryfikowane. Po usunięciu
problemu sieciowego pobrano z przypiętego revision oficjalny test PolQA oraz
pełny korpus. Test dał 956 pytań z co najmniej jednym pozytywem obecnym w
korpusie; 44 z 1 000 pytań odrzucono jawnie z powodu braku takiego pozytywu.
Pełny korpus ma 7 097 288 unikalnych dokumentów. Hashe poniżej pochodzą z
rzeczywistych plików i ponownie przeszły `--verify`; nie są wynikami
eksperymentalnymi.

## Źródła pierwotne i audyt kandydatów

### PIRB

Źródła:

- publikacja: <https://aclanthology.org/2024.lrec-main.1117/>;
- oficjalny kod: <https://github.com/sdadas/pirb>;
- stan kodu audytowany 2026-07-19; przed materializacją należy przypiąć commit,
  a nie ruchomy `master`.

PIRB obejmuje 41 zadań, ale nie ma jednego pochodzenia ani jednej licencji
danych. Publikacja wyróżnia PolEval-2022, 11 tłumaczonych przez Google Translate
zbiorów BEIR-PL, 12 podzbiorów MAUPQA, MFAQ, GPT-exams i dziewięć Web Datasets.
Kod benchmarku jest Apache-2.0, lecz ta licencja **nie zastępuje licencji
składowych danych**. README PIRB wymaga osobnej prośby o Web Datasets, ogranicza
je do badań i zakazuje redystrybucji. Nie kopiujemy ich do repozytorium.

Wnioski:

- PIRB jest dobrym katalogiem i opcjonalnym przyszłym benchmarkiem OOD, ale
  nazwa „PIRB” nie może być etykietą `native`;
- BEIR-PL jest tłumaczony i może być tylko `test_transfer_ood`;
- GPT-exams jest syntetyczny, a duża część MAUPQA jest tłumaczona lub
  generowana;
- Web Datasets zawierają realne pytania polskich użytkowników, lecz ich
  ograniczenia dostępu i redystrybucji wymagają osobnej decyzji;
- PolEval ma natywne domeny, ale PolQA/PolEval częściowo się pokrywają. Nie wolno
  sumować ich jak niezależnych testów.

PIRB stosuje ograniczanie liczby zapytań, lecz jego pełny run nadal obejmuje
wiele dużych korpusów. Nasz wrapper nie uruchamia PIRB w CI ani podczas
zamrażania P-02.

### PolQA

Źródła:

- publikacja: <https://aclanthology.org/2024.lrec-main.1125/>;
- oficjalna karta i repozytorium danych:
  <https://huggingface.co/datasets/ipipan/polqa>;
- przypięty revision repozytorium:
  `d78d036ef08ab3b9f4d85a2893f4d3a0c95a6f37`;
- przypięty automatyczny eksport Parquet:
  `288bec95fd4c29c70d90b075a81cf53090abe9ce`;
- licencja zadeklarowana przez wydawcę: CC BY-SA 4.0.

PolQA ma 7 000 pytań, z czego po 1 000 należy do validation i test. Pytania
pochodzą z polskich teleturniejów, quizów i konkursów, a pasaże z polskiej
Wikipedii (snapshot z marca 2022). Pary question–passage były ręcznie
oznaczane; test zawiera pozytywy i ocenione negatywy. To spełnia wymaganie
natywnego pochodzenia lepiej niż zbiory tłumaczone lub generowane.

Ograniczenia:

- domena to głównie wiedza ogólna/Wikipedia, więc wynik nie reprezentuje całego
  polskiego wyszukiwania;
- pytania są publiczne od co najmniej 2022 r.; nie można wykluczyć kontaminacji
  pretrainingu Bielika ani publicznych baz embeddera. Raport ma podawać tę
  niepewność, a nie przedstawiać PolQA jako „sekretny” test;
- publikacja PIRB opisuje znaczny overlap PolQA z PolEval-2022. Nie używamy obu
  jako niezależnych składników jednego wyniku;
- pełny korpus ma ponad 7 mln pasaży. Jest zamrożony na partycji projektu, ale
  jego indeksowanie pozostaje świadomie odłożone; nie należy zapisywać cache
  na małej partycji `/`.

### MAUPQA

Źródła:

- publikacja: <https://aclanthology.org/2023.bsnlp-1.2/>;
- oficjalna karta danych: <https://huggingface.co/datasets/ipipan/maupqa>;
- licencja zadeklarowana przez wydawcę: CC BY-SA 4.0.

Karta MAUPQA mówi wprost, że 14 podzbiorów jest głównie generowanych maszynowo
lub tłumaczonych, wszystkie przykłady są w splicie `train`, a zalecanym zbiorem
ewaluacyjnym jest PolQA. Podzbiór `msmarco` jest tłumaczeniem MS MARCO,
`nq` tłumaczeniem NQ, a warianty GPT są generowane. `1z10` i `czy-wiesz-v2`
mają polskie źródła, ale dobór pasaży jest automatyczny, a `1z10` korzysta z
Whispera i GPT-3.5.

Decyzja: MAUPQA nie jest holdoutem. Może w przyszłości służyć do treningu lub
diagnostyki po audycie overlapu, ale użycie części `msmarco` w ewaluacji
naruszałoby niezależność od obecnych danych MS MARCO-PL.

## Overlap i kontaminacja

Importer zapisał audyt exact overlap:

1. zapytania: `casefold` i normalizacja białych znaków;
2. dokumenty: SHA-256 tekstu po tej samej normalizacji;
3. near-duplicate pozostaje `NOT MEASURED`, dopóki nie zostanie uruchomiona
   jawna procedura MinHash/embeddingowa.

W zamrożonych rekordach nie znaleziono żadnego identycznego po normalizacji
zapytania ani dokumentu: odpowiednio `0/956` unikalnych native query względem
16 175 translated query oraz `0/10 467` dokumentów native względem 157 621
dokumentów translated. To wynik wyłącznie exact-match. Near-duplicate nadal
ma jawny stan `NOT MEASURED`; nie wolno interpretować zer exact-match jako
dowodu braku podobnych semantycznie przykładów.

Ryzyko kontaminacji modeli jest osobne od overlapu bieżącego treningu:

- Bielik i publiczne embeddery mogły widzieć polską Wikipedię i publiczne
  pytania quizowe w pretrainingu;
- retrievery PIRB były trenowane między innymi na polskim MS MARCO;
- modele Silver Retriever były trenowane na MAUPQA i nie mogą być rzetelnie
  oceniane na składowych MAUPQA, co zaznacza także publikacja PIRB.

## Zamrożony kontrakt i profile kosztu

Konfiguracja: `configs/evaluation/native_pl_holdout_v1.yaml`.

| Profil | Zapytania na zbiór | Korpus native | Rola |
|---|---:|---|---|
| `quick` | 100 | tylko ocenione pasaże PolQA | diagnostyka, cel 5–10 min |
| `medium` | 500 | tylko ocenione pasaże PolQA | szersza diagnostyka |
| `full` | wszystkie | pełny korpus PolQA | jedyny tryb porównawczy |

„5–10 min” jest celem operacyjnym, nie gwarancją dla każdego modelu i sprzętu.
Korpus diagnostyczny zawiera wyłącznie deduplikowaną unię pozytywów i
ocenionych negatywów wybranych zapytań. Dla native ma odpowiednio 1 114 i
5 558 dokumentów, a dla translated 996 i 5 069 dokumentów; ich hashe są w
pliku fingerprintów.
Nie jest to corpus-wide retrieval i raport oznacza te profile jako
nieporównawcze.
Wybór ID to rosnące
`sha256(task04-native-pl-v1:set_name:example_id)`. Każdy profil ma osobny hash
listy ID. Nie wolno porównywać wyników z różnych profili ani nazywać
`quick`/`medium` substytutem pełnego retrieval.

Manifest przechowuje osobno:

- SHA-256 źródła i kanonicznych rekordów;
- SHA-256 pełnej i profilowej listy ID;
- revision, licencję, origin języka i zakaz strojenia na native;
- stan brakującego źródła zamiast pustego zbioru lub wymyślonego hasha.

Zamrożone wartości są w
`configs/evaluation/native_pl_holdout_v1_fingerprints.json`. Najważniejsze:

- test PolQA: SHA-256 źródła
  `d0c990692ecf73234d898ea21142113beb8410ba67c53376082e6604d5c0464c`,
  956 rekordów, hash ID
  `31946d1d79ebe04e9b865d7fb5c149fe6b87030f73a0316f4327c1663c12de1d`;
- pełny korpus źródłowy: SHA-256
  `902fef5a8710d41a5603e8b067baaef20eb3b2b9181e639e60834b6ca2cd0a66`;
- kanoniczny pełny korpus: SHA-256
  `8dc01f02efcd8e1f179e22ce6e5548cd9e4a1c28cbc1c20f3e2068358a4d823f`;
- translated MS MARCO-PL: 16 272 rekordy, hash ID
  `42a4d429a99d8cc7c024fcc22febf8f4a5f9cb3bfa4afd5346b1aa9baef5633f`.

Komenda niczego nie pobiera:

```bash
uv run python scripts/freeze_native_pl_holdout.py \
  --translated-manifest data/processed/v1/evaluation/task04-v1/manifest.json \
  --polqa-test data/raw/native_pl/polqa/<revision>/test.csv \
  --polqa-passages data/raw/native_pl/polqa/<revision>/passages.jsonl \
  --output-dir data/processed/v1/evaluation/task04-native-pl-v1
```

`--polqa-passages` jest opcjonalne i potrzebne tylko dla `full`. Adapter
strumieniuje korpus, wykrywa konflikty ID przez dyskowy SQLite obok artefaktu
i usuwa staging po błędzie; nie utrzymuje 7 mln ID w RAM.

Cache i źródła trzeba kierować na partycję projektu, np.:

```bash
export HF_HOME="$PWD/data/cache/huggingface"
export HF_DATASETS_CACHE="$PWD/data/cache/huggingface/datasets"
```

Nie używać `/tmp`, `~/.cache/huggingface` ani pełnego PIRB jako domyślnej
ścieżki. Istniejący manifest jest immutable; ponowne zamrażanie kończy się
błędem.

## Osobne raportowanie i kompletność

`evaluate embedder` zawsze tworzy dwa osobne wpisy:

- `evaluation_sets.test_native_pl`;
- `evaluation_sets.test_translated_msmarco_pl`.

Brak zmierzonego native ustawia `report_status: incomplete`; wartości brakujące
pozostają `null`/`NOT MEASURED`, nigdy zero. Native jest final-test-only: nie
wolno nim dobierać promptów, progów, hiperparametrów ani wariantu generatora.
Wyniki native i translated nie są uśredniane.

## Jawny sygnał translationese

`translationese-surface-v1` raportuje:

- pozostałości częstych angielskich tokenów;
- trzy jawne wzorce potencjalnych kalk;
- podejrzane odstępy/interpunkcję;
- słaby sygnał długiego zapytania ASCII-only bez polskich znaków.

Każda flaga jest widoczna, a ASCII-only ma małą wagę. Wynik jest tani,
deterministyczny i nie używa modelu, ale **nie dowodzi**, że tekst jest
tłumaczeniem ani że jest nienaturalny. Służy do wykrywania zmian rozkładu,
które potem wymagają panelu ludzkiego.

## Stan artefaktów

Nie ma już brakującego artefaktu P-02. Wcześniejszy timeout HF CDN był
przejściowym problemem sieciowym związanym ze środowiskiem, a nie blockerem
licencyjnym. Po jego usunięciu pobrano i zweryfikowano oba przypięte pliki.
Źródła, dane przetworzone i cache pozostają poza Git na pojemnej partycji
projektu. Repozytorium przechowuje jedynie kontrakt i rzeczywiste fingerprinty.
Nie zbudowano pełnoskalowego indeksu i nie uruchomiono żadnego probe.
