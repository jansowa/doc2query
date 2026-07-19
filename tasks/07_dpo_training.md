# Task 07 — DPO i kontrola continued SFT

> [Centralny rejestr zadań i statusów](README.md). Każda zmiana statusu lub zakresu tego zadania musi aktualizować rejestr w tym samym commicie.

## Status

`TODO`

## Cel

Sprawdzić, czy preference optimization poprawia konkretne wady generatora ponad to, co daje zwykłe dalsze SFT na wybranych dobrych przykładach.

## Zależności

Task 06.

## Implementacja

Użyj TRL `DPOTrainer` lub równoważnej, dobrze przetestowanej implementacji. Obsłuż QLoRA i start z adaptera SFT.

Wymagania:

- prompt/chosen/rejected;
- adapter SFT jako punkt startowy;
- completion truncation bez utraty ważnych tokenów;
- `precompute_ref_log_probs` jako opcja oszczędzająca pamięć;
- walidacja, że model referencyjny odpowiada dokładnie punktowi startowemu;
- logowanie chosen/rejected rewards, margins, accuracies i length statistics;
- zapis adaptera i manifestu.

## Eksperyment kontrolny

Dla każdej konfiguracji DPO uruchom `continued SFT`:

- te same prompty;
- tylko `chosen` jako completion;
- ten sam przybliżony budżet tokenów i kroków;
- analogiczny LR search.

Bez tej kontroli nie wolno przypisać poprawy samemu DPO.
Dodatkową obowiązkową kontrolą jest `score-weighted continued SFT` przy tym
samym przybliżonym budżecie. Co najwyżej jedną metodę listwise (LiPO albo PRO)
wolno dopuścić dopiero po stabilnym DPO i tylko przy potwierdzonej jakości
rankingu kandydatów.

## Konfiguracja startowa

```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
gradient_checkpointing: true
max_length: 768
max_prompt_length: derived_from_data
beta: 0.1
loss_type: sigmoid
learning_rate: 1.0e-5
precompute_ref_log_probs: true
```

Parametry zależne od wersji TRL mają być sprawdzone z aktualną dokumentacją i smoke testem.

## Ablacje

- beta `0.05, 0.1, 0.2`;
- LR `5e-6, 1e-5, 2e-5`;
- top-vs-bottom vs top-vs-near-miss preferences;
- bez overlap component w scorerze;
- bez focus component;
- bez diversity component;
- DPO na 20k vs 50k/100k par;
- classical sigmoid vs co najwyżej jeden alternatywny stabilny loss po baseline.

Nie wykonuj pełnej siatki. Użyj sekwencyjnego wyboru i przerwij słabe runy.

## Ewaluacja

Po każdym runie:

1. intrinsic generator test;
2. preference test accuracy;
3. stały panel przykładów;
4. probe embedder;
5. bootstrap względem SFT i continued SFT;
6. failure slices.

Monitoruj szczególnie:

- czy DPO nie skraca nadmiernie query;
- czy poprawa overlapu nie pogarsza grounding;
- czy model nie uczy się artefaktów scorerów;
- czy diversity wynika z jakości, a nie losowości;
- czy preferencje na naturalnych style’ach się utrzymują.

## Memory strategy 16 GB

Kolejność oszczędzania pamięci:

1. precompute ref logprobs;
2. batch 1;
3. krótszy max length;
4. gradient checkpointing;
5. QLoRA;
6. niższy rank LoRA;
7. activation offloading tylko po benchmarku kosztu.

Nie duplikuj niepotrzebnie pełnych wag base modelu.

## Testy

- toy DPO loss: chosen wyżej niż rejected zmniejsza loss;
- ref logprobs mają właściwy shape i dataset order;
- restart nie miesza precomputed logprobs;
- PEFT save/load;
- DPO dataset nie zawiera par bez marginesu;
- długości prompt/chosen/rejected po tokenizacji są raportowane.

## Kryteria akceptacji

DPO przechodzi do finalnej macierzy tylko, jeżeli:

- wygrywa z bazowym SFT i continued SFT na predefiniowanej metryce;
- poprawa nie jest ograniczona do score modelu użytego do tworzenia preferencji;
- probe embedder potwierdza brak szkody lub poprawę;
- efekt utrzymuje się w co najmniej dwóch seedach redukowanego runu;
- ręczna ocena potwierdza kierunek zmiany.
