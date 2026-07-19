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

### P-03 hard-negative contract and fail-closed calibration

The comparison recipe now declares `recipe_version: probe-v1.1-p03` and the
versioned `probe-negatives-v1` contract. Its comparison path defaults to
`hn0_filter` plus `drop`. No numeric `possible_false_negative` threshold is
stored in the repository: Task 02 has not yet produced the required dev-only
calibration artifact. The three calibration fields in
`configs/evaluation/probe_v1.yaml` are deliberately `null`; a probe command
therefore exits before loading a model and writes `p03_preflight.json` with
status `blocked`.

The accepted calibration JSON contract pins all of the following:

- artifact ID and canonical payload fingerprint;
- `fit_split` beginning with `dev` and containing no `test`;
- fingerprint of the development records and SHA-256 of the source scores;
- primary reranker name and full revision;
- finite raw-pair-logit threshold, `greater_than_or_equal` operator and a
  documented threshold-selection method.

The loader rejects a missing/tampered artifact, a final-test fit, a different
judge, score space, ID or fingerprint. `test_native_pl` and
`test_translated_msmarco_pl` are evaluation-only and never enter this
preflight.

The deterministic strategies are:

- `hn0`: inherited negatives without filtering, retained only for the
  one-time W05 diagnostic;
- `hn0_filter`: score all inherited negatives with the frozen primary judge,
  flag scores at or above the pinned threshold, apply `drop | demote |
  keep+log`, then choose deterministically;
- `hn1_bm25`: mine a pinned number of candidates from the frozen P-01 BM25
  index, exclude known positives, apply the same calibrated policy and retain
  BM25 rank/score provenance.

`demote` removes a flagged candidate from the explicit paired-negative role
and may retain one as an ordinary in-batch negative. Reports contain candidate,
flag and action counts/rates separately for `natural`, `copy_control`, and
each synthetic `generator_id`. Probe summaries repeat the recipe version,
strategy, policy, threshold, calibration ID/fingerprint and BM25 fingerprint.
The comparison helper refuses any drift in those fields.

After Task 02 creates the approved artifact exclusively on dev, pin its path,
ID and fingerprint in the recipe. HN1 additionally requires a project-local
BM25 index and its fingerprint. Use project-local caches:

```bash
UV_CACHE_DIR="$PWD/.uv-cache" \
HF_HOME="$PWD/.cache/huggingface" \
TRANSFORMERS_CACHE="$PWD/.cache/huggingface/transformers" \
uv run python scripts/train_probe_embedder.py \
  --recipe configs/evaluation/probe_v1.yaml \
  --train-input data/processed/v1/train.parquet \
  --frozen-manifest data/processed/v1/evaluation/task04-v1/manifest.json \
  --corpus data/processed/v1/documents.parquet \
  --primary-judge-config configs/reranker/primary_polish_roberta_v3.yaml \
  --query-source synthetic \
  --generator-id W05-1.5B-50K-8GB \
  --synthetic-generations reports/evaluation/W05-1.5B-50K-8GB/generations.jsonl \
  --output-dir runs/probe-w05-hn0-filter-v1
```

The W05 HN0/HN0+filter/HN1 sensitivity run is diagnostic only. It has not been
run because neither the calibration threshold artifact nor a frozen BM25
index exists. Its blocker is recorded in
`reports/blockers/task04_p03_w05_sensitivity.md`; no final test was opened to
select a threshold or negative recipe.

## Native Polish holdout (P-02)

The source audit, licensing notes, contamination risks, frozen artifact
fingerprints and the non-networking import procedure are in
[`datasets/native_pl_holdout.md`](datasets/native_pl_holdout.md). The frozen
contract is `configs/evaluation/native_pl_holdout_v1.yaml`; measured
fingerprints are recorded in
`configs/evaluation/native_pl_holdout_v1_fingerprints.json`.

Freeze or verify the two named sets without downloading a model:

```bash
UV_CACHE_DIR="$PWD/.uv-cache" uv run python scripts/freeze_native_pl_holdout.py \
  --translated-manifest data/processed/v1/evaluation/task04-v1/manifest.json \
  --polqa-test data/raw/native_pl/polqa/<revision>/test.csv \
  --polqa-passages data/raw/native_pl/polqa/<revision>/passages.jsonl \
  --output-dir data/processed/v1/evaluation/task04-native-pl-v1

UV_CACHE_DIR="$PWD/.uv-cache" uv run python scripts/freeze_native_pl_holdout.py \
  --output-dir data/processed/v1/evaluation/task04-native-pl-v1 --verify
```

Omitting `--polqa-test` is safe for contract testing: translated MS MARCO-PL is
materialized, while native remains `missing_source_artifact` with `null`
hashes. The production v1 manifest includes the pinned PolQA test and corpus.
The importer never turns validation/train rows into a test. Existing manifests
are immutable.

`quick` selects 100 queries, `medium` 500 and `full` all frozen IDs. Selection
is deterministic and every profile has its own ID hash. Quick/medium use only
the deduplicated judged passages of their selected queries and are diagnostics;
the frozen translated corpora contain 996 and 5,069 documents respectively.
Full uses the frozen 7,097,288-document PolQA corpus and is the comparison
profile. It has been materialized, but no full index or probe has been run.
Cross-profile comparison is forbidden.

One probe training can evaluate both origins:

```bash
UV_CACHE_DIR="$PWD/.uv-cache" uv run python scripts/train_probe_embedder.py \
  --recipe configs/evaluation/probe_v1.yaml \
  --train-input data/processed/v1/train.parquet \
  --frozen-manifest data/processed/v1/evaluation/task04-v1/manifest.json \
  --holdout-manifest data/processed/v1/evaluation/task04-native-pl-v1/manifest.json \
  --holdout-profile quick \
  --corpus data/processed/v1/documents.parquet \
  --query-source natural \
  --output-dir runs/probe-natural-v1
```

The result and `embedder_report.md/json` always show
`test_native_pl` and `test_translated_msmarco_pl` separately. Missing native
sets `report_status: incomplete`; missing metrics remain `null`/`NOT MEASURED`.
The native test is final-test-only and prohibited for tuning.

`translationese-surface-v1` reports English residue, three explicit calque
patterns, punctuation spacing and a weak ASCII-only flag. It is deterministic
and model-free, but is explicitly labeled as a distribution diagnostic rather
than proof of translation or naturalness.

## Human panel

`scripts/human_evaluation.py export` creates randomized blind A/B CSV and JSONL
for up to 300 frozen panel records. The form covers answerability, naturalness,
retrieval usefulness, copying, answer leakage, preference and target fragment.
The import path reports Cohen kappa for two raters or Fleiss kappa for more.

The HTML/Markdown report includes at least 100 side-by-side examples when the
input contains them and a dedicated reward-hacking/failure-mode section.
