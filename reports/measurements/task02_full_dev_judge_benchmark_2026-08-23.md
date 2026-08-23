# Pomiar: benchmark obu zamrożonych sędziów na frozen dev (2026-08-23)

Runner: `scripts/run_task02_full_dev_judge_benchmark.py` (wznawialny, shardowany),
agregacja: `scripts/aggregate_task02_full_dev_judge_benchmark.py`.
Artefakty: `artifacts/task02/full_dev_judge_benchmark_v1/`
(`results/<shard>/{scores.jsonl,benchmark.json}`, `journal.jsonl`, `aggregate.json`).
Środowisko: `.venv-gpu` (torch 2.6.0+cu124), RTX 3060 Ti, peak 4,5 GB, ~100 min GPU.

Run **nie zmienia** żadnego progu — w szczególności przypiętego query-macro progu
Youdena `possible_false_negative` — ani wag sędziów, ani rubryki. **Testy finalne
pozostają zamknięte**: wejściem jest wyłącznie `data/processed/v1/dev.parquet`, a
runner odrzuca ścieżkę o nazwie zawierającej `test`. `final_tests_used=[]`.

## 1. Pokrycie: 16 258 z 16 272 rekordów frozen dev

Poprzednio shadow miał zmierzone **775** wspólnych query bramki HN. Teraz oba
sędziowie mają **16 258 query**, czyli **21× więcej** i praktycznie cały frozen dev.

| pasmo | liczba negatywów | pula kandydatów | query | shardy |
|---|---|---|---|---|
| `neg10` | 10 | 11 | 6 598 | 14 |
| `neg07_09` | 7–9 | 8–10 | 8 959 | 18 |
| `neg04_06` | 4–6 | 5–7 | 701 | 2 |
| `neg03` | ≤3 | ≤4 | **14 — wyłączone** | 1 |

Wyłączone są **14 rekordów (0,086%)**: przy puli poniżej 5 kandydatów Recall@5
jest matematycznie nieokreślony, a policzenie go innym cutoffem dałoby inną
metrykę pod tą samą nazwą. To jedyna nieoglądana część dev.

**Pasm nie wolno sumować ani uśredniać.** Recall@k i nDCG@10 zależą od liczby
kandydatów, więc liczby z różnych pasm nie są porównywalne — każdy raport shardowy
nosi `bands_must_not_be_pooled: true`, a runner przekazuje minimum negatywów jawnie
(`--min-hard-negatives`, domyślnie **10**, czyli poprzedni kontrakt bez zmian).
Porównanie **sędziów wewnątrz pasma** jest natomiast w pełni legalne: oba widzą
identyczne pule.

## 2. Wynik per pasmo (query-macro)

| pasmo | miara | primary `sdadas/polish-reranker-roberta-v3` | shadow `BAAI/bge-reranker-v2-m3` |
|---|---|---|---|
| `neg10` (n=6 598) | Recall@1 | **0,9457** | 0,9218 |
| | MRR | **0,9656** | 0,9463 |
| | nDCG@10 | **0,9736** | 0,9584 |
| | ujemny margines | **5,43%** | 7,82% |
| `neg07_09` (n=8 959) | Recall@1 | **0,9327** | 0,9228 |
| | MRR | **0,9573** | 0,9481 |
| | nDCG@10 | **0,9678** | 0,9606 |
| | ujemny margines | **6,73%** | 7,72% |
| `neg04_06` (n=701) | Recall@1 | 0,9489 | **0,9584** |
| | MRR | 0,9684 | **0,9728** |
| | nDCG@10 | 0,9763 | **0,9795** |
| | ujemny margines | 5,11% | **4,16%** |

### Znalezisko: przewaga primary maleje z rozmiarem puli i w najmniejszym pasmie się odwraca

Różnica Recall@1 primary − shadow wynosi **+2,39 pp** w `neg10`, **+0,99 pp** w
`neg07_09` i **−0,95 pp** w `neg04_06`. Kierunek jest spójny na wszystkich czterech
miarach w każdym pasmie, więc nie jest to szum jednej metryki — choć `neg04_06` ma
n=701, więc jest to przesłanka, nie rozstrzygnięcie. Interpretacja wymaga
ostrożności: mniejsza pula jest **łatwiejsza** (widać to w absolutnych poziomach,
`neg04_06` ma najwyższy Recall@1 u obu sędziów), a rekordy z mniejszą liczbą
negatywów pochodzą z uboższego wyniku kopania negatywów, więc różnią się od
`neg10` nie tylko rozmiarem puli. Wniosek operacyjny jest wąski i mocny:
**domyślny wybór primary jest uzasadniony tam, gdzie faktycznie działa reward
proxy (pełne pule 10 negatywów), a nie jako teza o globalnej wyższości.**

### Rozbieżność sędziów

| pasmo | rank | zwycięzca | Pearson marginesów | n grup |
|---|---|---|---|---|
| `neg10` | 9,56% | 7,36% | 0,6547 | 8 096 |
| `neg07_09` | 9,91% | 7,67% | 0,6239 | 12 000 |
| `neg04_06` | 6,60% | 5,80% | 0,5670 | 1 121 |

Bramka HN raportowała 9,81% disagreement na 775 query — nowa liczba na pasmie
`neg10` (9,56% przy 8 096 grupach) jest praktycznie ta sama, więc tamten wynik
**nie był artefaktem małej próbki**. Surowe logity nadal nie są uśredniane między
sędziami.

### Slice'y

Realnie różnicują tylko dwa wymiary; `synthetic_positive=true` (n=881 w `neg10`)
wychodzi **łatwiejszy** dla obu sędziów (primary R@1 0,9728 wobec 0,9426), a
`text_quality=flagged` (n=114) rozchodzi kierunki między sędziami (primary 0,9298,
shadow 0,9386) przy próbce za małej na wniosek.

## 3. Czego ten run **nie** domknął

1. **Wymagane slice'y pozostają nieosiągalne z powodu metadanych, nie obliczeń.**
   Cztery z siedmiu wymiarów są w frozen dev zdegenerowane: `domain`, `query_type`
   i `difficulty` mają jedną wartość `unknown`, `source_en_difficulty` jedną
   wartość `easy`, a `passage_length` 6 582 `short` wobec 25 `medium` (w `neg10`).
   Bez uzupełnienia metadanych w kontrakcie danych wymóg „wymaganych slice'ów" z
   §9.1 AGENTS.md jest nieosiągalny — to luka Task 01, nie tego pomiaru.
2. **14 rekordów `neg03`** pozostaje niezmierzonych z podanego wyżej powodu
   metrologicznego.
3. **Część testowa benchmarku pozostaje zamknięta** i nie wolno jej otworzyć
   przed zamrożeniem faktycznych finalistów.

## 4. Konsekwencje

- Status Task 02 pozostaje `IMPLEMENTED`: pokrycie dev jest domknięte poza 14
  rekordami, ale wymagane slice'y i część testowa nie.
- Żaden próg nie został ustalony ani przeliczony na podstawie tych liczb.
  Przypięty próg Youdena pozostaje bez zmian; ten run jest benchmarkiem, nie
  kalibracją.
- Wynik nie autoryzuje treningu i nie otwiera żadnej bramki:
  `task07_training_authorized=false`, `final_tests_used=[]`.
