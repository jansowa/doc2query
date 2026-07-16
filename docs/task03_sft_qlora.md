# Task 03: baseline'y SFT/QLoRA

Pipeline trenuje causal LM na oddzielnych polach `prompt` i `completion`. Collator
ustawia `-100` dla całego promptu i paddingu, więc loss obejmuje wyłącznie query.
Przy truncation najpierw ograniczana jest część promptu; completion wraz z EOS
pozostaje obecne. Baseline `b0` używa minimalnej instrukcji, a `b1` dodaje zakaz
długiego kopiowania bez kontrolek stylu i focus.

## Dane i dostęp do modeli

Wejściem jest wynik `invert_doc2query_pairs` z Task 01 w JSONL albo Parquet.
Rekordy muszą zawierać `passage`, `query` i `split`; buckety `query_style`,
`focus_bucket`, `content_lemma_overlap` oraz długość pasażu zasilają balanced i
weighted SFT. Limity 10k/50k są deterministycznymi, zagnieżdżonymi podzbiorami
wybieranymi po `pair_id` i seedzie, a nie pierwszymi rekordami pliku.

Konfiguracje modeli znajdują się w `configs/model/`. Wszystkie wyłączają
`trust_remote_code` i przypinają pełny commit. Bieliki są gated: przed runem
trzeba zaakceptować warunki w oficjalnym repozytorium Hugging Face i zalogować
lokalne środowisko. Kod nie omija kontroli dostępu i nie zapisuje wag w Git.

## Instalacja i walidacja

Na stacji GPU najpierw należy zainstalować koło PyTorch zgodne ze sterownikiem,
zgodnie z głównym README, a potem:

```bash
uv sync --all-groups
uv run doc2query doctor --output reports/hardware.json
uv run doc2query config validate --config configs/experiments/s02_1_5b_10k.yaml
uv run pytest -q tests/test_training_sft.py
```

CPU smoke ma wyłączone 4-bit. Właściwe runy Bielika wymagają CUDA i konfiguracji
NF4/double-quant; próba użycia QLoRA na CPU kończy się przed pobraniem modelu z
jednoznacznym błędem.

## Komendy eksperymentów

S00 generuje osobne artefakty greedy i sampling. Pełną ewaluację intrinsic
dostarczy Task 04; te pliki są wejściem do tego harnessu.

```bash
uv run python scripts/generate_panel.py \
  --config configs/experiments/s00_prompting.yaml \
  --mode greedy --output runs/S00/panel_greedy.jsonl
uv run python scripts/generate_panel.py \
  --config configs/experiments/s00_prompting.yaml \
  --mode sampling --output runs/S00/panel_sampling.jsonl
```

S01 wykonuje 20 kroków i zapisuje checkpointy, adapter, tokenizer, panel,
`sft_summary.json`, rozkład wag oraz `run_manifest.json`:

```bash
uv run python scripts/train_sft.py \
  --config configs/experiments/s01_tiny_smoke.yaml \
  --resume-if-available
```

Właściwe baseline'y uruchamia się analogicznie:

```bash
uv run python scripts/train_sft.py --config configs/experiments/s02_1_5b_10k.yaml --resume-if-available
uv run python scripts/train_sft.py --config configs/experiments/s03_1_5b_50k.yaml --resume-if-available
uv run python scripts/train_sft.py --config configs/experiments/s04_4_5b_base.yaml --resume-if-available
uv run python scripts/train_sft.py --config configs/experiments/s04_4_5b_instruct.yaml --resume-if-available
uv run python scripts/train_sft.py --config configs/experiments/s05_4_5b_ordinary.yaml --resume-if-available
uv run python scripts/train_sft.py --config configs/experiments/s05_4_5b_balanced.yaml --resume-if-available
uv run python scripts/train_sft.py --config configs/experiments/s05_4_5b_weighted.yaml --resume-if-available
```

S04 base/instruct ma ten sam deterministyczny podzbiór, seed, liczbę kroków i
padding do `max_length`, co utrzymuje ten sam maksymalny budżet tokenów.

## Memory probe, resume i porównanie

Każda długość jest mierzona w osobnym procesie, aby peak VRAM poprzedniego
wariantu nie zanieczyścił kolejnego:

```bash
uv run python scripts/run_memory_probe.py \
  --config configs/experiments/s04_4_5b_instruct.yaml \
  --lengths 512 768 1024 --steps 2
```

Raport zawiera realne `torch.cuda.max_memory_allocated()` i
`max_memory_reserved()`. OOM pozostaje wpisem `failed`, a kolejny wariant nadal
jest uruchamiany.

Do zwykłych treningów używaj zawsze `--resume-if-available`. Przy pierwszym
uruchomieniu flaga rozpoczyna trening od zera. Po przerwaniu kolejne wywołanie
tej samej komendy automatycznie wybiera kompletny `checkpoint-N` o najwyższym
numerze; nie trzeba podawać ścieżki. Stan optymalizatora, schedulera, RNG i
`trainer_state.json` jest wczytywany przez Trainer.

Plik `resume_identity.json` zapisuje fingerprint danych, model i revision,
seed, LoRA, kwantyzację oraz ustawienia wpływające na trajektorię treningu.
Wznowienie zostaje przerwane, jeśli ten podpis nie pasuje. Chroni to przed
przypadkowym użyciem checkpointu po zmianie danych lub configu. Niedokończony
checkpoint po awarii jest ignorowany, jeżeli istnieje wcześniejszy kompletny.
Checkpoint najpierw trafia do ukrytego katalogu staging, a jego publiczna nazwa
pojawia się przez pojedyncze `os.replace`.

```bash
uv run python scripts/compare_sft_runs.py \
  runs/S04-base/sft_summary.json runs/S04-instruct/sft_summary.json \
  --output reports/sft/s04_base_vs_instruct.md
```

Tabela nie uzupełnia brakujących wyników: `intrinsic_metrics` i
`probe_embedder_score` pozostają `None`, dopóki Task 04 nie zapisze rzeczywistych
pomiarów. Nie wolno przechodzić do DPO tylko na podstawie spadku loss.

## Balanced i weighted SFT

`BalancedBatchSampler` oversampluje rzadkie wartości czterech osi: stylu,
focus, kwantyla overlapu i kwantyla długości. `WeightedSFTTrainer` najpierw liczy
średni completion loss osobno dla każdego przykładu, a dopiero potem stosuje
wagę. Wagi są ograniczone przez `weight_min/weight_max`, normalizowane do
średniej 1 i zawsze zapisywane wraz z licznościami bucketów. Ordinary SFT jest
kontrolą i pozostaje ustawieniem domyślnym.
