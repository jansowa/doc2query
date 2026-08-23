# Pomiar: benchmark obu zamrożonych sędziów na frozen dev (2026-08-23)

Runner: `scripts/run_task02_full_dev_judge_benchmark.py` (wznawialny, shardowany),
agregacja: `scripts/aggregate_task02_full_dev_judge_benchmark.py`.
Artefakty: `artifacts/task02/full_dev_judge_benchmark_v1/`
(`results/<shard>/{scores.jsonl,benchmark.json}`, `journal.jsonl`, `aggregate.json`).
Środowisko: `.venv-gpu` (torch 2.6.0+cu124), RTX 3060 Ti, peak 4,5 GB.

Run **nie zmienia** żadnego progu — w szczególności przypiętego query-macro progu
Youdena `possible_false_negative` — ani wag sędziów, ani rubryki. **Testy finalne
pozostają zamknięte**: wejściem jest wyłącznie `data/processed/v1/dev.parquet`, a
runner odrzuca ścieżkę o nazwie zawierającej `test`. `final_tests_used=[]`.

## 1. Co zostało zmierzone

Oba sędziowie na **6 598 query** frozen dev (8 096 grup query–pozytyw), czyli na
całej populacji dev, którą przyrząd potrafi ocenić. Poprzednio shadow miał
zmierzone **775** wspólnych query bramki HN, więc pokrycie rośnie **8,5×**.

14 shardów po 500 rekordów, 177 s na shard przy obu sędziach — **~41 min GPU**.

| miara (query-macro) | primary `sdadas/polish-reranker-roberta-v3` | shadow `BAAI/bge-reranker-v2-m3` |
|---|---|---|
| Recall@1 | **0,9457** | 0,9218 |
| MRR | **0,9656** | 0,9463 |
| nDCG@10 | **0,9736** | 0,9584 |
| udział ujemnego marginesu | **5,43%** | 7,82% |

Primary jest lepszy na każdej z czterech miar, co jest zgodne z jego rolą
domyślnego sędziego — i jest to pierwszy pomiar tej przewagi na populacji
większej niż 775 query.

**Rozbieżność sędziów** (8 096 wspólnych grup): rank 9,56%, zwycięzca 7,36%,
korelacja Pearsona marginesów **0,6547**. Bramka HN raportowała 9,81%
disagreement na 775 query — nowa liczba na 8,5× większej próbce jest praktycznie
ta sama, więc tamten wynik nie był artefaktem małej próbki. Surowe logity nadal
nie są uśredniane między sędziami.

### Slice'y, które faktycznie różnicują

| slice | n (query) | primary R@1 | shadow R@1 |
|---|---|---|---|
| `synthetic_positive=false` | 6 555 | 0,9426 | 0,9199 |
| `synthetic_positive=true` | 881 | 0,9728 | 0,9466 |
| `text_quality=clean` | 6 525 | 0,9457 | 0,9216 |
| `text_quality=flagged` | 114 | 0,9298 | 0,9386 |

Kierunek `text_quality` rozchodzi się między sędziami (primary gorzej na
flagowanych, shadow lepiej) przy n=114, więc to przesłanka, nie rozstrzygnięcie.

## 2. Czego ten run **nie** domknął — trzy jawne luki

1. **Rekordy z mniej niż 10 negatywami są poza kontraktem przyrządu.** Frozen dev
   ma 16 272 rekordy, z czego pasmo `neg10` to 6 598, `neg07_09` **8 959** i
   `neg03_06` **715**. `build_scoring_groups` odmawia rekordu z mniej niż 10
   negatywami (`benchmark requires at least 10 hard negatives`) i run zatrzymał
   się na pierwszym takim shardzie — **to poprawne zachowanie kontraktu
   „1 pozytyw + 10 hard negatywów", nie usterka**. Zmierzenie pozostałych
   **9 674 rekordów (59,5% dev)** wymagałoby redefinicji populacji rankingowej
   (inna liczba kandydatów zmienia Recall@k i nDCG@10 nieporównywalnie), czyli
   osobnej decyzji — tego runu do tego **nie** rozszerzyłem i niczego nie
   dopasowałem, żeby „przeszło". Niepełny shard po tym zatrzymaniu usunięto,
   żeby nie leżał jako częściowy artefakt.
2. **Wymagane slice'y pozostają otwarte z powodu metadanych, nie obliczeń.**
   Cztery z siedmiu wymiarów są w frozen dev **zdegenerowane**: `domain`,
   `query_type` i `difficulty` mają jedną wartość `unknown` na wszystkich 6 598
   query, a `source_en_difficulty` jedną wartość `easy`. `passage_length` ma 6 582
   `short` wobec 25 `medium`. Realnie różnicują tylko `synthetic_positive` i
   `text_quality`. Bez uzupełnienia metadanych w kontrakcie danych wymóg
   „wymaganych slice'ów" z §9.1 AGENTS.md jest nieosiągalny — i to jest luka
   kontraktu danych (Task 01), nie tego pomiaru.
3. **Część testowa benchmarku pozostaje zamknięta** i nie wolno jej otworzyć
   przed zamrożeniem faktycznych finalistów.

## 3. Konsekwencje

- Status Task 02 pozostaje `IMPLEMENTED`. Pełne pokrycie dev **w granicach
  kontraktu przyrządu** jest dowiezione; dwie pozostałe luki są nazwane wyżej i
  żadna nie jest zamykana milcząco.
- Żaden próg nie został ustalony ani przeliczony na podstawie tych liczb.
  Przypięty próg Youdena pozostaje bez zmian; ten run jest benchmarkiem, nie
  kalibracją.
- Wynik nie autoryzuje treningu ani nie otwiera żadnej bramki:
  `task07_training_authorized=false`, `final_tests_used=[]`.
