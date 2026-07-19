#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/.uv-cache}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_ASSETS_CACHE="$HF_HOME/assets"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$HF_HOME/token}"
export TMPDIR="$ROOT/.cache/tmp"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=1200
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

case "$HF_HOME" in
  "$ROOT"/*) ;;
  *) echo "Campaign blocked: HF_HOME must be on the project partition: $ROOT" >&2; exit 2 ;;
esac
case "$UV_CACHE_DIR" in
  "$ROOT"/*) ;;
  *) echo "Campaign blocked: UV_CACHE_DIR must be on the project partition: $ROOT" >&2; exit 2 ;;
esac

mkdir -p \
  "$HF_HUB_CACHE" "$HF_ASSETS_CACHE" "$TMPDIR" logs \
  reports/base_1_5b_campaign

LOG="$ROOT/logs/base_1_5b_campaign.log"
STATUS="$ROOT/reports/base_1_5b_campaign/status.tsv"
LOCK="$ROOT/reports/base_1_5b_campaign/queue.lock"
exec 9>"$LOCK"
if ! flock -n 9; then
  printf '[%s] Another base campaign owns %s; exiting.\n' \
    "$(date --iso-8601=seconds)" "$LOCK" | tee -a "$LOG"
  exit 3
fi

configs=(
  configs/experiments/b01_1_5b_10k_l768_lr2e4_s42.yaml
  configs/experiments/b02_1_5b_10k_l1024_lr2e4_s42.yaml
  configs/experiments/b03_1_5b_10k_r16_lr2e4_s42.yaml
  configs/experiments/b04_1_5b_10k_r32_lr2e4_s42.yaml
  configs/experiments/b05_1_5b_10k_attention_lr2e4_s42.yaml
  configs/experiments/b06_1_5b_10k_eb32_lr2e4_s42.yaml
  configs/experiments/b07_1_5b_10k_dropout0_lr2e4_s42.yaml
)

usage() {
  echo "usage: $0 [--dry-run|--help]"
}

case "${1:-}" in
  --help|-h)
    usage
    exit 0
    ;;
  --dry-run)
    "${DOC2QUERY_PYTHON:-.venv/bin/python}" - "${configs[@]}" <<'PY'
import sys
from pathlib import Path
from doc2query.config import load_config

for value in sys.argv[1:]:
    config = load_config(Path(value))
    print(
        config.run.experiment_id,
        config.model.name_or_path,
        config.model.revision,
        config.data.max_train_examples,
        config.training.max_length,
        config.training.learning_rate,
        config.lora.r,
        config.lora.target_modules,
        config.training.gradient_accumulation_steps,
        config.lora.dropout,
    )
PY
    exit 0
    ;;
  "")
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

if [[ ! -f "$STATUS" ]]; then
  printf 'started_at\tfinished_at\tname\texit_code\n' >"$STATUS"
fi

record_step() {
  local name="$1"
  shift
  local started finished rc
  started="$(date --iso-8601=seconds)"
  printf '\n[%s] START %s\n' "$started" "$name" | tee -a "$LOG"
  set +e
  "$@" >>"$LOG" 2>&1
  rc=$?
  set -e
  finished="$(date --iso-8601=seconds)"
  printf '[%s] END %s rc=%s\n' "$finished" "$name" "$rc" | tee -a "$LOG"
  printf '%s\t%s\t%s\t%s\n' "$started" "$finished" "$name" "$rc" >>"$STATUS"
  return "$rc"
}

if [[ -n "${DOC2QUERY_PYTHON:-}" ]]; then
  PYTHON="$DOC2QUERY_PYTHON"
  if [[ ! -x "$PYTHON" ]]; then
    echo "Campaign blocked: DOC2QUERY_PYTHON is not executable: $PYTHON" >&2
    exit 2
  fi
else
  if ! record_step gpu-environment bash scripts/bootstrap_gpu_env.sh; then
    tail -n 20 "$LOG" >&2
    exit 2
  fi
  PYTHON="$ROOT/.venv-gpu/bin/python"
fi
export DOC2QUERY_PYTHON="$PYTHON"
HF_CLI="$(dirname "$PYTHON")/hf"
if [[ ! -x "$HF_CLI" ]]; then
  echo "Campaign blocked: Hugging Face CLI is absent next to $PYTHON." >&2
  exit 2
fi

ensure_snapshot() {
  local repo="$1"
  local revision="$2"
  local slug="models--${repo//\//--}"
  local destination="$HF_HUB_CACHE/$slug"
  local home_source="$HOME/.cache/huggingface/hub/$slug"
  if [[ -d "$destination/snapshots/$revision" ]]; then
    printf '[%s] CACHE HIT %s@%s\n' "$(date --iso-8601=seconds)" "$repo" "$revision" \
      | tee -a "$LOG"
    return 0
  fi
  if [[ -d "$home_source/snapshots/$revision" ]]; then
    printf '[%s] COPY %s@%s from user cache to project cache\n' \
      "$(date --iso-8601=seconds)" "$repo" "$revision" | tee -a "$LOG"
    mkdir -p "$destination"
    cp -a "$home_source/." "$destination/"
  else
    printf '[%s] DOWNLOAD %s@%s into project cache\n' \
      "$(date --iso-8601=seconds)" "$repo" "$revision" | tee -a "$LOG"
    HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
      "$HF_CLI" download "$repo" \
      --revision "$revision" \
      --cache-dir "$HF_HUB_CACHE" \
      --quiet >>"$LOG" 2>&1
  fi
  if [[ ! -d "$destination/snapshots/$revision" ]]; then
    echo "Campaign blocked: exact snapshot was not materialized: $repo@$revision" >&2
    return 2
  fi
}

if ! record_step gpu-preflight \
  "$PYTHON" -c \
  'import torch
if torch.version.cuda is None:
    raise SystemExit("GPU preflight blocked: selected Python has a CPU-only Torch build.")
if not torch.cuda.is_available():
    raise SystemExit("GPU preflight blocked: CUDA Torch is installed, but CUDA is unavailable in this process.")
print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory)'; then
  tail -n 20 "$LOG" >&2
  exit 2
fi

record_step cache-bielik-base \
  ensure_snapshot \
  speakleash/Bielik-1.5B-v3 \
  4b25049621bf3952a1fc9314c89773102eda0333
record_step cache-probe \
  ensure_snapshot \
  sdadas/polish-reranker-base-ranknet \
  a7c66d41a8097ca02e75616d0951c941d94ff6a1
record_step cache-primary \
  ensure_snapshot \
  sdadas/polish-reranker-roberta-v3 \
  e6471da541f4e7be33845b6d57248a8d8bde27e8

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# This is the only probe comparison in the queue. It remains dev-only and
# writes no generator or final-model decision.
record_step p03-w05-sensitivity \
  bash scripts/run_p03_w05_sensitivity.sh

# Required technical memory measurements, safe to repeat after interruption.
record_step memory-probe-768-1024 \
  "$PYTHON" scripts/run_memory_probe.py \
  --config configs/experiments/s01_1_5b_8gb_smoke.yaml \
  --lengths 768 1024 \
  --steps 2 \
  --output-dir reports/base_1_5b_campaign/memory_probes

# W01/W02/W03 already cover LR 1e-4/5e-5/2e-4 at 10k, W04 adds seed 43,
# and W05 is the 50k baseline. The following runs are single-factor technical
# ablations against W03. They are not ranked here and do not open final tests.
for config in "${configs[@]}"; do
  record_step "train-$(basename "$config" .yaml)" \
    "$PYTHON" scripts/train_sft.py \
    --config "$config" \
    --resume-if-available
done

printf '[%s] Base 1.5B technical queue complete. No winner was selected; P-04/probe remains required.\n' \
  "$(date --iso-8601=seconds)" | tee -a "$LOG"
