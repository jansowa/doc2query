# Task 05 — kontrolowana generacja

Implementacja rozdziela formę (`full_question`, `keyword_query`) od intencji
retrieval i abstenuje do `unknown`, gdy reguły nie są pewne. Opcjonalny kontrakt
evidence używa zerowych indeksów zdań i nie zmienia starych cache'y: nowe pola
są dodawane przy ponownej inwersji, a konsumenci muszą tolerować ich brak.

Publiczne prymitywy znajdują się w `doc2query.generation`: kontrolowany prompt
F0–F3, bounded retry po deduplikacji, ścisły multi-query JSON, ekstrakcja
koncepcji oraz selekcja `top_n`, `mmr` i `coverage_aware`. Parser JSON naprawia
wyłącznie ogrodzenie Markdown i pojedynczy przecinek przed końcowym `}`;
każda naprawa pozostaje jawna w polu `repaired`, a pozostałe błędy zwiększają
invalid rate.

`configs/generation/controlled_diverse.yaml` jest wyłącznie presetem
implementacyjnym. Eksperymenty D00–D12, kalibracja rozkładów per domena, audyt
500 etykiet i audyt około 200 ekstrakcji koncepcji czekają na Harness v1.1 oraz
ADR P-04. Nie uzyskano jeszcze żadnego wyniku jakościowego.
