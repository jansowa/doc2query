# Task 04: frozen evaluation harness

The harness never rewrites `data/processed/v1`. It freezes ordered ID lists and
three independent hashes (source file, ID list, selected canonical records) under
`data/processed/v1/evaluation/task04-v1/`. Re-running the freezer against an
existing manifest fails; `--verify` validates all hashes.

```bash
.venv/bin/python scripts/freeze_evaluation_sets.py
.venv/bin/python scripts/freeze_evaluation_sets.py --verify
```

Harness v1.1 separates two retrieval protocols. `candidate_pool_ranking` is a
generator diagnostic over known positives and inherited or deterministically
backfilled negatives; every metric starts with `pool_`. `corpus_retrieval`
searches a complete frozen documents artifact; every metric starts with
`corpus_`, and this protocol is the retrieval basis for generator and probe
comparisons. The older `*_rank10` sets remain usable as diagnostic panels, but
they are not a substitute for full-corpus retrieval.

## Frozen corpus indexes

Build BM25 and auxiliary bi-encoder indexes into separate immutable
directories. Their manifests record the source hash, corpus fingerprint,
candidate count, backend parameters and index hash. The auxiliary encoder
manifest additionally records the pinned model revision, license and
`trust_remote_code` policy.

```bash
.venv/bin/python scripts/build_corpus_index.py \
  --config configs/evaluation/corpus_retrieval_v1.yaml \
  --documents data/processed/v1/documents.parquet \
  --backend bm25 \
  --analysis-cache data/interim/text_analysis.sqlite \
  --output-dir data/processed/v1/evaluation/corpus-bm25-v1

.venv/bin/python scripts/build_corpus_index.py \
  --config configs/evaluation/corpus_retrieval_v1.yaml \
  --documents data/processed/v1/documents.parquet \
  --backend auxiliary_biencoder \
  --output-dir data/processed/v1/evaluation/corpus-aux-bi-v1
```

BM25 uses cached Polish content lemmas. The auxiliary index is brute-force, so
FAISS is not required. A comparison index must be frozen only after its
relevance and ambiguity thresholds have been calibrated on dev.
The default BM25 contract requires the CPU-only `pl_core_news_lg` model; use
the `simple` backend only for smoke tests and never mix its index fingerprint
with the comparison index.

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
  --corpus-index data/processed/v1/evaluation/corpus-bm25-v1 \
  --judge-device cuda \
  --output-dir reports/evaluation/W03-1.5B-10K-LR2E4-S42
```

`generation_report.json` records decoding parameters, throughput and peak
VRAM. `evaluation_manifest.json` records the resolved training config,
checkpoint, test fingerprint, judges and code provenance. A supplied
`--generations` artifact skips generation but still enforces the frozen test
fingerprint in all comparison outputs. The report contains `pool_recall_at_*`,
`pool_mrr`, `pool_ndcg_at_10` and the explicit diagnostic pool size separately
from `corpus_round_trip_at_1/5/20/100`, full corpus size, source margin,
`corpus_effective_candidate_count`, ambiguity rate and round-trip/pool-margin
correlations. A generator comparison refuses summaries without the same
measured corpus-index fingerprint. Metrics that were not run remain `null` and
are listed under `unmeasured`.

## Probe embedder

`configs/evaluation/probe_v1.yaml` is the frozen comparison budget. It uses the
same pinned Polish encoder, tokenizer, positive/hard-negative sampling, number
of steps and seed for natural-query gold-data control, copy negative control and
each synthetic generator. Smoke overrides are explicitly non-comparable.

```bash
.venv/bin/python scripts/train_probe_embedder.py \
  --recipe configs/evaluation/probe_v1.yaml \
  --train-input data/processed/v1/train.parquet \
  --frozen-manifest data/processed/v1/evaluation/task04-v1/manifest.json \
  --test-subset test_embedder \
  --corpus data/processed/v1/documents.parquet \
  --query-source natural \
  --output-dir runs/probe-natural-v1
```

Synthetic runs additionally require `--synthetic-generations`. Probe
evaluation always indexes the supplied complete corpus rather than rebuilding
a small pool from test positives and negatives. It reports
`corpus_recall_at_1/5/10/100`, `corpus_mrr_at_10`, `corpus_ndcg_at_10`,
`corpus_map`, explicit corpus size, corpus fingerprint, latency and model size.
Recall@K is rejected when the corpus contains fewer than K documents. Paired
bootstrap rejects different test or corpus fingerprints. Variant ranking is
emitted only when measured probe metrics exist; intrinsic reward never
substitutes for them.

## Human panel

`scripts/human_evaluation.py export` creates randomized blind A/B CSV and JSONL
for up to 300 frozen panel records. The form covers answerability, naturalness,
retrieval usefulness, copying, answer leakage, preference and target fragment.
The import path reports Cohen kappa for two raters or Fleiss kappa for more.

The HTML/Markdown report includes at least 100 side-by-side examples when the
input contains them and a dedicated reward-hacking/failure-mode section.
