#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export HF_HOME="$ROOT/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_ASSETS_CACHE="$HF_HOME/assets"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"
export TMPDIR="$ROOT/.cache/tmp"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

mkdir -p logs "$TMPDIR"
exec .venv/bin/python scripts/evaluate_generator.py \
  --config configs/experiments/w06_4_5b_50k_8gb_bs8.yaml \
  --adapter runs/W06-4.5B-INSTRUCT-50K-8GB-BS8-L512/checkpoint-3125 \
  --frozen-manifest data/processed/v1/evaluation/task04-v1/manifest.json \
  --subset test_generator_panel_rank10 \
  --primary-judge configs/reranker/primary_polish_roberta_v3.yaml \
  --judge-device cuda \
  --output-dir reports/evaluation/W06-4.5B-INSTRUCT-50K-8GB-BS8-L512-CKPT3125 \
  >>"$ROOT/logs/w06_evaluation.log" 2>&1
