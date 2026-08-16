# Task 08 — Wielokryterialny GRPO/RL

> [Centralny rejestr zadań i statusów](README.md). Każda zmiana statusu lub zakresu tego zadania musi aktualizować rejestr w tym samym commicie.

## Status

`BLOCKED`

Zadanie jest planowanym etapem programu, zablokowanym wyłącznie
zależnościami (Task 07 i warunki startowe poniżej), a nie decyzją o
opcjonalności.

Aktualizacja 2026-08-14 (materiał przygotowawczy do projektu nagrody, bez
zmiany statusu): w Task 06 zamrożono i zmierzono korpus walidacyjny komponentów
nagrody (ADR
[`task06_reward_validation_corpus_v1.md`](../reports/decisions/task06_reward_validation_corpus_v1.md),
pomiar
[`task06_reward_validation_corpus_v1.md`](../reports/measurements/task06_reward_validation_corpus_v1.md)).
Jest to pierwszy w programie zbiór, w którym porządek jakości kandydatów jest
znany **z konstrukcji**, a nie z korelacji z sędzią, więc bezpośrednio dotyczy
projektu wielokryterialnej nagrody tego zadania. Dwa wyniki są istotne dla
składu nagrody: `entity_preservation` nie jest sygnałem specyficzności (przy
zapytaniach bez encji daje maksimum przez konwencję `empty=1.0`, remis w 180/180
grup), a `format_valid` jest podatny na obejście przez wtrącenie bez
interpunkcji („Oto …”), czyli jako składnik nagrody premiowałby formę, której
nie wykrywa. Karanie ogólności trzeba oprzeć na osobnym sygnale
(`content_jaccard` rozdzielił klasy w 85.6% grup) i zaprojektować prospektywnie.
Task 08 pozostaje `BLOCKED`: ten korpus jest materiałem przygotowawczym i **nie**
zastępuje wymaganej decyzji `reports/decisions/enable_grpo.md`.

Aktualizacja 2026-08-13 (decyzja właściciela): projekt ma jawny cel
edukacyjny — praktyczne opanowanie DPO i GRPO (AGENTS.md §2). GRPO przestaje
być ścieżką warunkową „tylko gdy DPO zawiedzie": ścieżka 1.5B (R00–R05) jest
domyślnie planowana po Task 07. Zmierzone porównanie DPO vs continued SFT vs
offline best-of-N pozostaje obowiązkowym kontekstem interpretacyjnym, a
twarde przesłanki (zysk offline selekcji nieinternalizowany przez DPO albo
zmierzona wada odporna na SFT/DPO) są odtąd bramką awansu wyników GRPO do
selekcji finalistów, nie warunkiem uruchomienia. Decyzja
`reports/decisions/enable_grpo.md` nadal jest wymagana i może powoływać się
na cel edukacyjny plus gotowość techniczną. Sprzęt: R00–R05 na 1.5B / 8 GB
(RTX 3060 Ti); R06 4.5B preferencyjnie na maszynie 16 GB (RTX 5070 Ti);
serwisy reward na mocnym CPU / 64 GB RAM tej maszyny.

## Cel

Sprawdzić, czy online RL daje dodatkową korzyść, której nie osiągnięto przez
SFT, kontrolki i DPO, oraz zrealizować edukacyjny cel właściciela: praktyczne
opanowanie wielokryterialnego GRPO na lokalnym sprzęcie. Task jest planowanym
etapem programu i zaczyna się od Bielika 1.5B; awans jego wyników do selekcji
finalistów podlega osobnej bramce opisanej niżej.

## Zależności

Taski 02, 04, 05 i 07. Wymagana decyzja w `reports/decisions/enable_grpo.md` z uzasadnieniem.

## Warunki rozpoczęcia

- rewardy mają testy adwersarialne;
- composite score koreluje z oceną człowieka albo — przy obowiązującym
  waiverze — z owner-approved dual-LLM audytem (nie nazywać go human
  evidence) oraz z wynikiem probe;
- DPO zostało porównane z continued SFT i offline best-of-N na tych samych
  kandydatach (obowiązkowy kontekst interpretacyjny, nie wymóg „porażki" DPO);
- memory probe potwierdza wykonalność na docelowej karcie (1.5B: 8 GB;
  4.5B: 16 GB).

## Bramka awansu do selekcji finalistów

Uruchomienie GRPO nie wymaga wykazania niewystarczalności DPO (cel
edukacyjny, AGENTS.md §2). Awans wyników GRPO do macierzy finalistów wymaga
natomiast co najmniej jednego z warunków Fazy E: offline selekcja daje zysk
probe, którego DPO nie internalizuje (sygnały grupowe, np. diversity zbioru
K query, nie wyrażają się w parach preferencji), albo GRPO poprawia
konkretną, zmierzoną wadę odporną na SFT/DPO bez utraty probe score. Runy,
które tej bramki nie przechodzą, są raportowane jako
`educational/feasibility` — z pełnym rejestrem eksperymentu, ale bez claimu
selekcyjnego.

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

Na 8–16 GB:

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

Tylko gdy 1.5B pokazuje stabilny, zewnętrznie potwierdzony efekt;
preferencyjnie na maszynie 16 GB.

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
