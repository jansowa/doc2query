# Task 04: frozen evaluation harness

The harness never rewrites `data/processed/v1`. It freezes ordered ID lists and
three independent hashes (source file, ID list, selected canonical records) under
`data/processed/v1/evaluation/task04-v1/`. Re-running the freezer against an
existing manifest fails; `--verify` validates all hashes.

```bash
.venv/bin/python scripts/freeze_evaluation_sets.py
.venv/bin/python scripts/freeze_evaluation_sets.py --verify
```

The full dev/test populations remain available for lexical, format and
embedder analyses. Retrieval metrics whose contract requires at least ten hard
negatives use `dev_intrinsic_rank10`, `test_intrinsic_rank10`,
`test_embedder_rank10`, or the fixed `test_generator_panel_rank10`. The
manifest reports every exclusion and its reason. For split v1 this excludes
9674 dev and 9640 test records after the Task 01 cross-split-negative cleanup;
the records are not deleted or reassigned.

## One-command checkpoint evaluation

The command generates one greedy query and four fixed sampling candidates per
passage, unloads the generator, scores source-vs-negative retrieval, computes
lexical/format/focus/diversity metrics and slices, then creates JSONL, JSON,
Markdown and HTML artifacts:

```bash
.venv/bin/python scripts/evaluate_generator.py \
  --config configs/experiments/w03_1_5b_10k_lr2e4_seed42.yaml \
  --adapter runs/W03-1.5B-10K-LR2E4-S42/adapter \
  --frozen-manifest data/processed/v1/evaluation/task04-v1/manifest.json \
  --subset test_generator_panel_rank10 \
  --primary-judge configs/reranker/primary_polish_roberta_v3.yaml \
  --shadow-judge configs/reranker/shadow_bge_v2_m3.yaml \
  --judge-device cuda \
  --output-dir reports/evaluation/W03-1.5B-10K-LR2E4-S42
```

`generation_report.json` records decoding parameters, throughput and peak
VRAM. `evaluation_manifest.json` records the resolved training config,
checkpoint, test fingerprint, judges and code provenance. A supplied
`--generations` artifact skips generation but still enforces the frozen test
fingerprint in all comparison outputs. Metrics that were not run remain
`null` and are listed under `unmeasured`.

## Probe embedder

`configs/evaluation/probe_v1.yaml` is the frozen comparison budget. It uses the
same pinned Polish encoder, tokenizer, positive/hard-negative sampling, number
of steps and seed for natural-query upper/control, copy negative control and
each synthetic generator. Smoke overrides are explicitly non-comparable.

```bash
.venv/bin/python scripts/train_probe_embedder.py \
  --recipe configs/evaluation/probe_v1.yaml \
  --train-input data/processed/v1/train.parquet \
  --frozen-manifest data/processed/v1/evaluation/task04-v1/manifest.json \
  --test-subset test_embedder_rank10 \
  --query-source natural \
  --output-dir runs/probe-natural-v1
```

Synthetic runs additionally require `--synthetic-generations`. Comparison uses
natural frozen queries and reports Recall@1/5/10/100, MRR@10, nDCG@10, MAP,
hard-negative win rate, latency and model size. Paired bootstrap rejects
different test fingerprints. Variant ranking is emitted only when measured
probe metrics exist; intrinsic reward never substitutes for them.

## Human panel

`scripts/human_evaluation.py export` creates randomized blind A/B CSV and JSONL
for up to 300 frozen panel records. The form covers answerability, naturalness,
retrieval usefulness, copying, answer leakage, preference and target fragment.
The import path reports Cohen kappa for two raters or Fleiss kappa for more.

The HTML/Markdown report includes at least 100 side-by-side examples when the
input contains them and a dedicated reward-hacking/failure-mode section.
