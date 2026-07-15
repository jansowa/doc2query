# Task 11 — Opcjonalny naprzemienny rozwój rerankera i generatora

## Status

`OPTIONAL / RESEARCH`

## Cel

Zbadać, czy iteracyjne ulepszanie rerankera na adwersarialnych query generatora i ponowne ulepszanie generatora daje korzyść ponad zamrożonego sędziego.

## Zależności

Wszystkie wcześniejsze taski i wyraźna decyzja badawcza.

## Zakaz

Nie implementuj jednoczesnego end-to-end update’u rerankera i generatora w jednym kroku. Reward staje się wtedy niestacjonarny, a interpretacja wyników bardzo trudna.

## Procedura iteracyjna

### Iteracja 0

- zamrożony reranker R0;
- generator G0 po najlepszym SFT/DPO;
- pełny baseline na naturalnym teście.

### Iteracja 1

1. G0 generuje kandydaty na train passages;
2. wybierz failure cases:
   - wysoki R0 score, niska ocena człowieka;
   - query pasujące do wielu hard negative’ów;
   - halucynowane fakty;
   - kopiowanie;
3. zbuduj dane adwersarialne dla rerankera;
4. trenuj R1 na mieszaninie danych realnych i syntetycznych;
5. wybierz R1 na osobnym naturalnym dev, nie na syntetykach;
6. zamroź R1;
7. przelicz preferencje/rewardy;
8. trenuj G1 krótkim DPO lub GRPO;
9. oceń G1 drugim niezależnym rerankerem i człowiekiem.

Maksymalnie 2–3 iteracje.

## Replay i zapobieganie zapominaniu

Reranker zawsze trenuj z dużym udziałem oryginalnych naturalnych par. Monitoruj wyniki na stałych benchmarkach z Iteracji 0.

Generator zachowuje KL/odniesienie do stabilnego SFT albo używa DPO z referencją. Nie pozwól, aby cały model optymalizował się wyłącznie pod najnowszego rerankera.

## Sędziowie niezależni

Utrzymuj:

- primary reward reranker;
- shadow reranker innej architektury;
- human panel;
- source retrieval przez docelowy/probe embedder.

Poprawa tylko na primary rerankerze jest sygnałem overfittingu.

## Eksperyment kontrolny

Porównaj:

- frozen R0 + G update;
- R1 update, ale bez G update;
- alternating R1/G1;
- większy statyczny reranker bez iteracji.

Może się okazać, że prostszy silniejszy zamrożony reranker jest wystarczający.

## Kryteria akceptacji

Kontynuuj iteracje wyłącznie, gdy:

- R1 lepiej koreluje z człowiekiem na naturalnym holdoucie;
- G1 poprawia probe embedder lub human preference;
- shadow reranker potwierdza kierunek;
- nie rośnie rozjazd domenowy;
- koszt procesu jest uzasadniony.

W przeciwnym razie zakończ i zachowaj prostszy pipeline.
