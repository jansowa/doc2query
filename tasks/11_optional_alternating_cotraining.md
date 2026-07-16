# Task 11 — Audyt odporności sędziego i ścieżki awaryjne

> [Centralny rejestr zadań i statusów](README.md). Każda zmiana statusu lub zakresu tego zadania musi aktualizować rejestr w tym samym commicie.

## Status

`OPTIONAL`

Jest to późny eksperyment badawczy, a nie element domyślnej ścieżki.

## Cel

Sprawdzić, czy generator nie optymalizuje się pod przypadkowe słabości jednego gotowego rerankera. Ten task nie obejmuje treningu ani dostrajania rerankera.

## Zależności

Wszystkie wcześniejsze taski i wyraźna decyzja badawcza.

## Zakaz

- nie aktualizuj wag rerankera;
- nie twórz własnego cross-encodera;
- nie używaj syntetycznych query generatora do uczenia modelu-sędziego;
- nie uznawaj poprawy na primary rerankerze za wystarczający dowód poprawy jakości.

## Procedura

### Krok 1 — zamrożeni sędziowie

Utrzymuj:

- primary: preferowany gotowy polski reranker;
- shadow: model innej rodziny lub wielojęzyczny reranker;
- source retrieval przez docelowy albo probe embedder;
- mały panel ręczny z answerability, trafnością, kopiowaniem i stylem.

Wszystkie modele mają przypięte revision i pozostają zamrożone przez cały eksperyment.

### Krok 2 — zbiór disagreement i failure cases

Zbierz przypadki:

- wysoki primary score, niska ocena człowieka;
- wysoki primary score, niski shadow score;
- query pasujące również do wielu hard negative’ów;
- halucynowane fakty lub błędne liczby;
- zbyt ogólne query;
- kopiowanie zdania;
- identyczny szablon z podmienioną encją;
- preferowanie pierwszego zdania mimo kontrolki focus.

Nie używaj tych przypadków do uczenia rerankera. Służą do zmiany rewardu, progów, filtrów, promptów albo strategii generacji.

### Krok 3 — kontrolowane warianty sędziego

Porównaj:

1. tylko primary reranker;
2. primary + shadow jako veto przy dużej niezgodności;
3. primary + reguły answerability/overlap;
4. ensemble znormalizowanych score’ów;
5. bez rerankera w online RL, z rerankerem tylko do offline best-of-N;
6. wybór kandydatów przez probe embedder + ręczne filtry.

Celem nie jest maksymalizacja jednego automatycznego score’u, lecz stabilność decyzji na różnych sędziach i zgodność z ręcznym panelem.

## Bramka „gotowy reranker nie wystarcza”

Dopiero gdy wszystkie poniższe warunki są spełnione, przygotuj ADR opisujący osobny przyszły projekt adaptacji rerankera:

- co najmniej dwa silne gotowe modele istotnie zawodzą na naturalnym holdoucie domenowym;
- problem nie wynika z truncation, formatu wejścia, kalibracji ani błędnych hard negative’ów;
- ensemble i dodatkowe checkery nie rozwiązują problemu;
- ręczny holdout jest wystarczająco duży, by wykazać powtarzalny błąd;
- koszt adaptacji jest uzasadniony wpływem na końcowy probe embedder.

Sam ADR nie uruchamia treningu. Wymaga osobnej decyzji użytkownika i osobnego zakresu prac.

## Kryteria akceptacji

- primary i shadow pozostają bitowo niezmienione;
- raport zawiera agreement/disagreement oraz przykłady jakościowe;
- poprawa generatora utrzymuje się dla probe embeddera i panelu ręcznego, nie tylko primary score’u;
- istnieje decyzja: `single_judge`, `ensemble`, `offline_only` albo `disable_reranker_reward`;
- jeśli gotowe rerankery są wystarczające, raport wprost zamyka temat ich treningu.
