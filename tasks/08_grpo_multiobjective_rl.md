# Task 08 — Wielokryterialny GRPO/RL

## Status

`OPTIONAL / BLOCKED UNTIL GATE`

## Cel

Sprawdzić, czy online RL daje dodatkową korzyść, której nie osiągnięto przez SFT, kontrolki i DPO. Ten task jest opcjonalny i zaczyna się od Bielika 1.5B.

## Zależności

Taski 02, 04, 05 i 07. Wymagana decyzja w `reports/decisions/enable_grpo.md` z uzasadnieniem.

## Warunki rozpoczęcia

- rewardy mają testy adwersarialne;
- composite score koreluje z oceną człowieka;
- DPO zostało porównane z continued SFT;
- znana jest konkretna wada, którą RL ma poprawić;
- memory probe potwierdza wykonalność.

## Implementacja

Użyj TRL `GRPOTrainer` lub równoważnego online policy optimization. Reward functions mają być osobnymi callable’ami i zwracać pełny breakdown do logów.

Minimalne rewardy:

- grounding score;
- margin do hard negative’ów;
- overlap band;
- format;
- style-control compliance;
- focus-control compliance;
- group diversity;
- duplicate penalty;
- length band.

## Normalizacja

Każdy reward:

- kalibruj na naturalnym dev;
- clampuj do jawnego zakresu;
- loguj mean/std/percentyle;
- monitoruj udział w total reward;
- nie pozwól na NaN/Inf;
- ma fallback przy awarii rerankera/lemmatyzera, ale awarie są raportowane i nie mogą być nagradzane.

## Startowy config 1.5B

```yaml
num_generations: 4
max_completion_length: 64
per_device_train_batch_size: 1
gradient_accumulation_steps: 4_or_multiple_compatible_with_num_generations
temperature: 0.8
top_p: 0.95
beta: 0.0
gradient_checkpointing: true
qlora: true
```

Efektywny batch musi spełniać wymagania implementacji GRPO względem `num_generations`.

## Generacja

Na 16 GB:

- domyślnie bez colocated vLLM;
- benchmark zwykłego `generate()` i wspieranego continuous batching;
- vLLM server mode tylko na osobnym GPU;
- colocated vLLM tylko po memory probe i z bezpiecznym marginesem;
- ogranicz długość promptów przez smart truncation, nie usuwając focus sentence.

## CPU reward services

Lemmatyzer i lekki reranker mogą działać na CPU:

- batchuj requesty;
- użyj kolejki i timeoutów;
- cachuj passage features;
- licz query features online;
- mierz czas rewardu osobno;
- opcjonalnie eksportuj reranker do ONNX/OpenVINO INT8.

Nie ukrywaj, jeśli CPU reward staje się wąskim gardłem.

## Harmonogram eksperymentów

### R00 — reward dry run

Bez aktualizacji modelu. Generuj i licz rewardy, sprawdzając rozkłady oraz przykłady top/bottom.

### R01 — 1.5B, grounding + format

Najprostszy stabilny reward.

### R02 — dodaj overlap band

Sprawdź, czy nie powstają ogólne lub halucynowane query.

### R03 — dodaj focus/style

Tylko z kontrolowanymi promptami.

### R04 — group diversity

Wymaga grupy K completions tego samego promptu.

### R05 — leave-one-reward-out

Dla finalnego składu usuń kolejno każdy komponent.

### R06 — 4.5B

Tylko gdy 1.5B pokazuje stabilny, zewnętrznie potwierdzony efekt.

## Reward hacking monitors

Automatycznie wykrywaj:

- spadek entropii i collapse do jednego szablonu;
- ekstremalnie krótkie query;
- wzrost uniwersalnych pytań pasujących do wielu passage;
- kopiowanie encji bez relacji;
- manipulację znakami/interpunkcją pod tokenizer;
- wzrost score rewardu bez wzrostu source retrieval;
- rozjazd rerankera online i drugiego niezależnego rerankera;
- wzrost total reward przy pogorszeniu human panel.

Zdefiniuj stop conditions.

## Porównania obowiązkowe

- SFT;
- continued SFT;
- DPO;
- best-of-N offline selection bez aktualizacji modelu;
- GRPO.

Best-of-N jest ważną kontrolą: może dać większość korzyści RL bez ryzyka niestabilności.

## Kryteria akceptacji

GRPO jest kandydatem finalnym tylko, gdy:

- poprawia probe embedder lub wyraźnie poprawia ważną wadę bez utraty probe score;
- efekt nie znika przy użyciu niezależnego rerankera;
- ręczna ocena potwierdza poprawę;
- nie ma wyraźnego reward hackingu;
- koszt generacji/treningu jest uzasadniony względem DPO/best-of-N.
