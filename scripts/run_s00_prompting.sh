#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${DOC2QUERY_PYTHON:-$ROOT_DIR/.venv-gpu/bin/python}"
CONTRACT="configs/evaluation/s00_prompting_v1.yaml"
OUTPUT_DIR="runs/S00-prompting-v1"
DERIVED_MANIFEST="$OUTPUT_DIR/cohort/manifest.json"
SUBSET="dev_s00_5000"
GREEDY_BATCH_SIZE="${S00_GREEDY_BATCH_SIZE:-32}"
SAMPLING_BATCH_SIZE="${S00_SAMPLING_BATCH_SIZE:-8}"
MIN_PROMPT_BATCH_SIZE="${S00_MIN_BATCH_SIZE:-1}"
SCORING_BATCH_SIZE="${S00_SCORING_BATCH_SIZE:-64}"
BM25_WORKERS="${S00_BM25_WORKERS:-8}"
SCORING_PROGRESS_EVERY="${S00_SCORING_PROGRESS_EVERY:-100}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "S00 requires the GPU environment; set DOC2QUERY_PYTHON or create .venv-gpu." >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
exec 9>"$OUTPUT_DIR/runner.lock"
if ! flock -n 9; then
  echo "Another optimized S00 runner holds $OUTPUT_DIR/runner.lock." >&2
  exit 3
fi

export HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TOKENIZERS_PARALLELISM=false

echo "[S00 runtime] greedy_batch_size=$GREEDY_BATCH_SIZE sampling_batch_size=$SAMPLING_BATCH_SIZE min_batch_size=$MIN_PROMPT_BATCH_SIZE" >&2
echo "[S00 scoring] batch_size=$SCORING_BATCH_SIZE bm25_workers=$BM25_WORKERS progress_every=$SCORING_PROGRESS_EVERY" >&2

"$PYTHON_BIN" scripts/run_s00_prompting.py \
  --contract "$CONTRACT" \
  --output-dir "$OUTPUT_DIR" \
  --prepare-only

if [[ ! -f data/processed/v1/evaluation/corpus-bm25-v1/manifest.json ]]; then
  echo "[S00 preflight] building the missing development-only Harness v1.1 BM25 index" >&2
  "$PYTHON_BIN" scripts/build_corpus_index.py \
    --config configs/evaluation/corpus_retrieval_v1.yaml \
    --documents data/processed/v1/documents.parquet \
    --backend bm25 \
    --analysis-cache data/interim/text_analysis.sqlite \
    --output-dir data/processed/v1/evaluation/corpus-bm25-v1
fi

"$PYTHON_BIN" scripts/run_s00_prompting.py \
  --contract "$CONTRACT" \
  --output-dir "$OUTPUT_DIR" \
  --greedy-batch-size "$GREEDY_BATCH_SIZE" \
  --sampling-batch-size "$SAMPLING_BATCH_SIZE" \
  --min-batch-size "$MIN_PROMPT_BATCH_SIZE"

for STRATEGY in zero_shot few_shot; do
  REPORT_DIR="reports/evaluation/S00-${STRATEGY}-dev-v1"
  if [[ -f "$REPORT_DIR/result.json" ]]; then
    echo "[S00 harness] reusing completed $REPORT_DIR" >&2
    continue
  fi
  "$PYTHON_BIN" scripts/evaluate_generator.py \
    --config configs/experiments/s00_prompting.yaml \
    --frozen-manifest "$DERIVED_MANIFEST" \
    --subset "$SUBSET" \
    --generations "$OUTPUT_DIR/${STRATEGY}.generations.jsonl" \
    --primary-judge configs/reranker/primary_polish_roberta_v3_p03_gpu.yaml \
    --shadow-judge configs/reranker/shadow_bge_v2_m3.yaml \
    --corpus-index data/processed/v1/evaluation/corpus-bm25-v1 \
    --judge-device cuda \
    --scoring-batch-size "$SCORING_BATCH_SIZE" \
    --bm25-workers "$BM25_WORKERS" \
    --progress-every "$SCORING_PROGRESS_EVERY" \
    --output-dir "$REPORT_DIR"
done
