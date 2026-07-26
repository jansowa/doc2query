# Task 03 — prospektywny kontrakt S07 plT5

Data: 2026-07-25

Status kontraktu: `ACCEPTED / FROZEN BEFORE MODEL DOWNLOAD`

Zakres danych: wyłącznie train i frozen dev; `final_tests_used=[]`.
`dev_confirm` i P-06 pozostają zamknięte.

## Decyzja modelowa

Wybrano `allegro/plt5-base` z revision
`56379680948ce8b42d3d48df86569cfc210d3060`, bez `trust_remote_code`.
Jest to encoder–decoder T5 wytrenowany na polskich korpusach, z polskim
tokenizerem SentencePiece 50k i licencją CC BY 4.0. Model będzie dostrajany
w całości, a nie przez LoRA; jest to właściwy kontrolny baseline architektury
seq2seq i mieści się w klasie kosztu pozwalającej na pełny fine-tuning.

Odrzucono `google/mt5-base`, ale przypięto jego porównywalny snapshot
`2eb15465c5dd7f72a8f7984306ad05ebc3dd1e1f`. Wielojęzyczny tokenizer i
pretraining zwiększają koszt dla korpusu wyłącznie polskiego, a kontrakt nie
zawiera hipotezy uzasadniającej post-hoc zmianę modelu po wyniku plT5.

Maszynowym źródłem prawdy jest
[`configs/evaluation/s07_contract_v1.yaml`](../../configs/evaluation/s07_contract_v1.yaml).

## Równoważność z W05

S07 używa dokładnie ścieżek train/dev, deterministycznego capowania i B1 z
W05: 50 000 par train, 1 000 dev, seed 42, jeden epoch, efektywny batch 16,
3 125 kroków optymalizatora, 512 tokenów źródła i 64 tokeny targetu.
Fingerprint referencyjnej kohorty W05 jest przypięty jako
`017a26eb...c06b73`; pełny S07 musi po treningu odtworzyć tę wartość albo run
jest nieważny. Koszt jest ograniczony do jednego runu na jednej GPU, 8 godzin
i 90% peak reserved VRAM. Przekroczenie limitu pamięci, non-finite loss lub
zmiana resume identity zatrzymują wykonanie.

## Kolejność i bramki

Wznawialny runner `scripts/run_s07_plt5_baseline.sh` wymusza kolejność:

1. walidacja kontraktu bez ładowania wag;
2. `ruff`, strict `mypy` i pełny `pytest`;
3. 20-krokowy smoke na lokalnym mini-T5, zapis pełnego modelu, checkpointów
   i generacji;
4. dwukrokowy memory probe rzeczywistego plT5 przy długości 512;
5. dopiero po peak reserved `<=90%` pełny fine-tuning 50k;
6. Harness v1.1 na pierwszych 5 000 rekordów `dev_intrinsic_rank10`, greedy
   plus cztery sample, primary, shadow i pełny BM25 corpus;
7. dokładnie 2 486 deterministycznych query wspólnej kohorty P-05 oraz
   downstream probe `dev_screen`, seed 42, 250 kroków i HN0+filter.

Runner nie zawiera komend dla `dev_confirm`, żadnego finalnego subsetu ani
P-06. Generacja P-05 zapisuje append-only journal i wznawia wyłącznie dokładny
prefiks wejścia. Trening korzysta z atomowych checkpointów oraz istniejącego
kontraktu resume identity obejmującego model, revision, kohortę, seed i
parametry trajektorii.

## Faktycznie wykonane tanie bramki

- preflight: `pass`, wszystkie 14 kontroli kontraktu przeszły;
- `ruff`: pass;
- strict `mypy`: 119 plików bez błędów;
- `pytest`: 184 passed;
- smoke: 20/20 kroków, 128 train i 32 dev, dwa kompletne checkpointy, pełny
  model zapisany; eval loss 1,665 po kroku 10 i 1,643 po kroku 20;
- generacja smoke: technicznie zapisano 100/100 rekordów; losowy mini-model
  zwrócił puste teksty, więc artefakt potwierdza ścieżkę encoder–decoder, ale
  nie jest pomiarem jakości ani bramką formatu rzeczywistego plT5;
- test wznowienia: ponowne wywołanie zwróciło `already_complete` dla
  checkpointu 20.

Pierwszy preflight w sandboxie nie widział CUDA, ale kontrolowane uruchomienie
poza sandboxem potwierdziło RTX 3060 Ti i wykonało wyłącznie dwukrokowy memory
probe. Probe odtworzył fingerprint W05 `017a26eb...c06b73`, wykonał
forward/backward, zapisał checkpoint oraz pełny model i zmierzył peak VRAM
allocated/reserved `2,683/2,717 GiB` przy throughput `0,500 przykładu/s`.
Bramka pamięci przeszła. Pełnego treningu, Harness v1.1 ani downstream probe
nie uruchomiono, więc nadal nie ma wyniku jakościowego S07 ani porównania
kosztu generacji z W05.

Probe ujawnił brak jawnych zależności tokenizera plT5. Do grupy training i
bootstrapu GPU dodano przypięte `sentencepiece==0.2.2` oraz
`protobuf==7.35.1`; memory probe zwraca teraz kod niezerowy, gdy subprocess
zawiedzie, i strumieniuje jego log. Runner obsługuje
`S07_STOP_AFTER_MEMORY_PROBE=1` oraz konfigurowalny
`S07_TRAIN_TIMEOUT` (domyślnie `8h`).

## Aneks techniczny: wybór microbatcha po memory probe

Po przerwaniu pierwszego, nieukończonego startu wykonano prospektywny sweep
microbatch `1/2/4/8/16`, zawsze przy effective batch 16, oraz 30-krokowe
potwierdzenie BS8/BS16. BS16 był o 74,2% szybszy od BS8, przeszedł bez OOM i
osiągnął peak reserved 3,256 GiB. Kontrakt przypina zatem microbatch 16 i
gradient accumulation 1. Nie zmienia to 50 tys. par, 3125 optimizer steps,
seedów, schedulerów ani porównywalnego budżetu W05. Wyniki i reguła wyboru:
[`s07_batch_probe_2026-07-25.md`](../measurements/task03_s07/s07_batch_probe_2026-07-25.md).

Stary checkpoint BS1 zachowano jako artefakt przerwanego runu; nie jest
wznawiany z nową trajectory identity. Aneks nie otwiera P-06, `dev_confirm`
ani finalnych testów.

Po zgłoszonej niestabilności wcześniejszego scoringu wykonawcze limity S07
przypięto konserwatywnie: scoring batch 16, primary reranker batch 16 i dwa
workery BM25; shadow pozostaje przy batch 8. Są to ustawienia throughput-only,
które nie zmieniają kohorty, modeli sędziów ani metryk. Runner pozwala je
jawnie ograniczyć przez `S07_SCORING_BATCH_SIZE` i `S07_BM25_WORKERS`.

Wznowienie jest etapowe. Pełny SFT używa atomowych checkpointów co 50 kroków
i po przerwaniu wybiera najnowszy kompletny stan wraz z optimizerem,
schedulerem i RNG. Generacja Harness zapisuje zweryfikowany prefiks do
`.partial`, scoring ma własny journal, a generacja probe jest append-only.
Jeśli krótki downstream probe zostanie przerwany przed ukończeniem treningu,
runner archiwizuje jego niekompletny katalog i restartuje tylko ten etap;
ukończony trening probe pozwala wznowić samą ewaluację.
