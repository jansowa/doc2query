# Task 03 — Baseline’y SFT/QLoRA dla Bielika

> [Centralny rejestr zadań i statusów](README.md). Każda zmiana statusu lub zakresu tego zadania musi aktualizować rejestr w tym samym commicie.

## Status

`TODO`

## Cel

Zaimplementować stabilny trening passage→query i uruchomić serię tanich baseline’ów, zanim projekt przejdzie do DPO lub RL.

## Zależności

Taski 01–02.

## Modele

Przygotuj konfiguracje:

- `bielik_1_5b_base_or_instruct.yaml` — szybkie eksperymenty;
- `bielik_4_5b_base.yaml`;
- `bielik_4_5b_instruct.yaml`;
- `bielik_minitron_7b_instruct.yaml`;
- `bielik_pl_minitron_7b_instruct.yaml`.

Nie uruchamiaj 7B w pełnej skali w tym tasku. Zaimplementuj tylko smoke test i config.

## Ładowanie modelu

Wspieraj:

- 4-bit NF4;
- double quant;
- BF16, jeśli sprzęt wspiera, w przeciwnym razie FP16;
- `prepare_model_for_kbit_training`;
- gradient checkpointing;
- `use_cache=false`;
- automatyczne wykrywanie modułów linear do LoRA;
- jawne logowanie listy target modules i liczby parametrów trainable.

Nie zakładaj nazw modułów bez odczytania architektury. Test powinien przerwać run, jeśli LoRA nie objęła oczekiwanych warstw.

## Format danych

Użyj prompt-completion. Loss tylko na completion. Prompt i output muszą być oddzielne w dataset.

Baseline B0:

```text
Wygeneruj jedno polskie zapytanie wyszukiwawcze, na które można odpowiedzieć na podstawie pasażu.

Pasaż:
{passage}

Zapytanie:
```

Completion: dokładnie naturalne query.

Baseline B1 dodaje instrukcję o niekopiowaniu długich fragmentów, ale bez styl/focus controls.

## Weighted/balanced SFT

Zaimplementuj dwie opcje:

1. `BalancedBatchSampler` wyrównujący buckety:
   - style;
   - focus position;
   - overlap quantile;
   - długość passage;
2. `WeightedSFTTrainer`, który skaluje loss completion na poziomie przykładu.

Wagi muszą być znormalizowane, ograniczone `min/max` i logowane. Zwykły SFT pozostaje domyślnym kontrolnym baseline’em.

## Konfiguracje pamięci

Start dla 4.5B/16 GB:

```yaml
max_length: 768
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
gradient_checkpointing: true
packing: false
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
learning_rate: 1.0e-4
num_train_epochs: 1
```

Wykonaj memory probe dla 512, 768 i 1024. Raportuj realny peak VRAM, nie tylko estymację.

## Eksperymenty

### S00 — prompting bez treningu

- 5 tys. passage dev;
- greedy i sampling;
- stały prompt;
- pełna ewaluacja intrinsic.

### S01 — tiny smoke

- tiny model lub 1.5B;
- 100–1000 rekordów;
- 20 kroków;
- sprawdzenie spadku loss i zapisu adaptera.

### S02 — 1.5B 10k

Cel: strojenie techniczne, nie wynik finalny.

### S03 — 1.5B 50k

Porównaj LR, rank i max length na małej macierzy.

### S04 — 4.5B 50k base vs instruct

Identyczne dane, seed i budżet tokenów.

### S05 — 4.5B 50k ordinary vs balanced vs weighted

Sprawdź, czy kontrola rozkładu danych poprawia overlap/style bez utraty grounding.

## Checkpointing i wznowienie

- atomowy zapis;
- możliwość resume;
- zapis adaptera i tokenizer config;
- zapis próbki generacji na stałym panelu 100 passage;
- walidacja po ustalonej liczbie kroków;
- early stopping wyłącznie na predefiniowanej metryce dev.

## Wymagane skrypty

- `scripts/train_sft.py`
- `scripts/run_memory_probe.py`
- `scripts/generate_panel.py`
- `scripts/compare_sft_runs.py`

## Testy

- completion-only masking;
- truncation nie usuwa completion;
- prompt nie jest liczony do loss;
- LoRA target modules nie są puste;
- liczba trainable params jest rozsądna;
- save/load adapter daje te same logits w tolerancji;
- resume nie resetuje schedulera;
- weighted loss odpowiada ręcznemu obliczeniu na toy batchu.

## Kryteria akceptacji

- smoke test przechodzi na małym modelu;
- 1.5B generuje poprawny format i reaguje na trening;
- 4.5B QLoRA mieści się w 16 GB dla co najmniej jednego użytecznego configu;
- base vs instruct są porównane na identycznym pipeline;
- powstaje tabela z loss, intrinsic metrics, peak VRAM, throughput i probe-embedder score, jeżeli Task 04 jest już gotowy;
- nie ma podstaw do przejścia do DPO, dopóki SFT baseline nie jest stabilny.
