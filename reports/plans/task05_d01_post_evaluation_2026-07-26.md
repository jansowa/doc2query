# Plan i raport implementacji post-D01 evaluation

## Stan

Pipeline CPU/GPU został zaimplementowany i przetestowany na mockach CPU. Nie
uruchomiono treningów D01, generacji Bielikiem, primary/shadow scoringu ani
probe embeddera. Nie otwarto żadnego finalnego testu. Oficjalnym jedynym
wejściem ewaluacyjnym jest `dev_intrinsic_rank10` z zamrożonego manifestu Task
04; kod odrzuca każdą inną nazwę subsetu.

Implementacja obejmuje:

- passage-level JSONL journal naprawiający wyłącznie urwaną ostatnią linię,
  pełną identity model/config/adapter/cohort/control/seed i atomowy final JSONL;
- dokładne wznowienie bez ponownego generowania trwałych grup oraz odzyskiwalne
  archiwum niezgodnej częściowej trajektorii;
- pełne frozen rekordy: source query, pozytyw(y), co najmniej 10 hard
  negative'ów, ID, kontrolki, próby, seedy i provenance;
- wspólny crash-safe scoring Harnessu dla primary/shadow, metryki formy i
  intencji z osobnym `unknown`, lexical/copy, retrieval obu sędziów,
  disagreement, diversity i slice'y;
- matched W05/W06 baseline K=4, paired bootstrap po passage/source query i
  fail-closed raport Markdown + JSON;
- materializację probe inputs dopiero po zmierzonym scoringu i kompletnym
  matched intrinsic report. Materializator stosuje przypiętą kalibrację
  `HN0+filter/drop` do identity-aligned primary negative scores i odrzuca
  brakujące score'y, wyczerpane grupy, różny budżet oraz final-test provenance.

Retry/deduplikacja jest stanowa i ma per-attempt seed. Dlatego trajectory batch
size wynosi świadomie 1; większy batch zmieniałby losową trajektorię po crashu.
Scoring pozostaje batchowany. Runner nocny uruchamia tylko trening i
`generation-only`; scoring nie został automatycznie dołożony do okna 24 h.

## Komendy po ukończeniu adapterów

Wspólne zmienne:

```bash
PYTHON=.venv-gpu/bin/python
FROZEN=data/processed/v1/evaluation/task04-v1/manifest.json
PRIMARY=configs/reranker/primary_polish_roberta_v3_cuda.yaml
SHADOW=configs/reranker/shadow_bge_v2_m3.yaml
```

Generacja wariantów i matched baseline'ów (każda komenda jest wznawialna):

```bash
$PYTHON scripts/run_d01_postprocess.py generation-only \
  --config configs/experiments/d01_1_5b_style_dev_generation_s42.yaml \
  --frozen-manifest "$FROZEN" --subset dev_intrinsic_rank10 \
  --adapter runs/D01-1.5B-STYLE-50K-S42/adapter \
  --output runs/D01-1.5B-STYLE-DEV-GENERATION-S42/generation/controlled.jsonl

$PYTHON scripts/run_d01_postprocess.py generation-only \
  --config configs/experiments/d01_w05_matched_dev_generation_s42.yaml \
  --frozen-manifest "$FROZEN" --subset dev_intrinsic_rank10 \
  --adapter runs/W05-1.5B-50K-8GB/adapter \
  --output runs/D01-W05-MATCHED-DEV-GENERATION-S42/generation/uncontrolled.jsonl

$PYTHON scripts/run_d01_postprocess.py generation-only \
  --config configs/experiments/d01_4_5b_style_dev_generation_s42.yaml \
  --frozen-manifest "$FROZEN" --subset dev_intrinsic_rank10 \
  --adapter runs/D01-4.5B-STYLE-50K-S42/adapter \
  --output runs/D01-4.5B-STYLE-DEV-GENERATION-S42/generation/controlled.jsonl

$PYTHON scripts/run_d01_postprocess.py generation-only \
  --config configs/experiments/d01_w06_matched_dev_generation_s42.yaml \
  --frozen-manifest "$FROZEN" --subset dev_intrinsic_rank10 \
  --adapter runs/W06-4.5B-INSTRUCT-50K-8GB-BS1-L512/adapter \
  --output runs/D01-W06-MATCHED-DEV-GENERATION-S42/generation/uncontrolled.jsonl
```

Scoring jest osobnym postprocessem. Przykład dla jednego ramienia (powtórzyć
dla czterech artefaktów):

```bash
$PYTHON scripts/run_d01_postprocess.py score \
  --generations runs/D01-1.5B-STYLE-DEV-GENERATION-S42/generation/controlled.jsonl \
  --generation-summary runs/D01-1.5B-STYLE-DEV-GENERATION-S42/generation/controlled.jsonl.summary.json \
  --output-dir reports/measurements/task05_d01/D01-1.5B-STYLE-50K-S42 \
  --primary-judge "$PRIMARY" --shadow-judge "$SHADOW" \
  --primary-judge-device cuda --shadow-judge-device cuda \
  --corpus-index artifacts/task04/p03/bm25_train_v1
```

Raport porównawczy wyprowadza generation token ceiling z liczby zapisanych
query oraz przypiętego `max_new_tokens`; nie przyjmuje ręcznie deklarowanego
budżetu.

```bash
$PYTHON scripts/run_d01_postprocess.py compare \
  --baseline-summary reports/measurements/task05_d01/W05/summary.json \
  --baseline-rows reports/measurements/task05_d01/W05/per_generation.jsonl \
  --variant-summary reports/measurements/task05_d01/D01-1.5B-STYLE-50K-S42/summary.json \
  --variant-rows reports/measurements/task05_d01/D01-1.5B-STYLE-50K-S42/per_generation.jsonl \
  --comparison-contract configs/evaluation/comparison_contract_v1.yaml \
  --output-json reports/measurements/task05_d01/1.5b_comparison.json \
  --output-markdown reports/measurements/task05_d01/1.5b_comparison.md
```

Materializacja probe inputs (bez uruchamiania treningu):

Komenda odmawia pracy, dopóki matched report nie ma kompletu trzech
guardraili P-04 (`corpus_round_trip_at_20`, sentence-level source hit i format)
z 95% CI w granicy non-inferiority.

```bash
$PYTHON scripts/run_d01_postprocess.py materialize-probe-inputs \
  --generations runs/D01-1.5B-STYLE-DEV-GENERATION-S42/generation/controlled.jsonl \
  --generation-summary runs/D01-1.5B-STYLE-DEV-GENERATION-S42/generation/controlled.jsonl.summary.json \
  --scoring-summary reports/measurements/task05_d01/D01-1.5B-STYLE-50K-S42/summary.json \
  --scoring-rows reports/measurements/task05_d01/D01-1.5B-STYLE-50K-S42/per_generation.jsonl \
  --comparison-report reports/measurements/task05_d01/1.5b_comparison.json \
  --probe-recipe configs/evaluation/probe_v1.yaml \
  --output artifacts/task05_d01/probe_inputs/1.5b.jsonl
```

## Pozostałe bramki

1. Ukończyć dwa treningi D01 i potwierdzić adapter fingerprints.
2. Wygenerować cztery artefakty na tej samej pełnej frozen-dev kohorcie.
3. Ukończyć primary i shadow scoring oraz oba matched intrinsic reporty.
4. Zamrozić identyczny budżet tokenów/par/pasaży/K i dopiero zmaterializować
   probe inputs.
5. W osobnej sesji uruchomić porównywalne probe embeddery i bootstrap CI.
6. Nie promować modelu na podstawie samego rerankera; do czasu probe decyzja
   raportu pozostaje `not_measured`.
