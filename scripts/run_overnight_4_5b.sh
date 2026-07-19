#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Keep large downloads and temporary files on the project filesystem, not $HOME.
export HF_HOME="$ROOT/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_ASSETS_CACHE="$HF_HOME/assets"
# Reuse only the small existing credential file; model blobs stay under $ROOT.
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"
export TMPDIR="$ROOT/.cache/tmp"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=1200
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_ASSETS_CACHE" "$TMPDIR" logs reports/overnight_4_5b

LOG="$ROOT/logs/overnight_4_5b.log"
STATUS="$ROOT/reports/overnight_4_5b/status.tsv"
LOCK="$ROOT/reports/overnight_4_5b/queue.lock"
exec 9>"$LOCK"
if ! flock -n 9; then
  printf '[%s] Another overnight queue owns %s; exiting.\n' \
    "$(date --iso-8601=seconds)" "$LOCK" | tee -a "$LOG"
  exit 3
fi

if [[ ! -f "$STATUS" ]]; then
  printf 'started_at\tfinished_at\tname\texit_code\n' >"$STATUS"
fi

record_step() {
  local name="$1"
  shift
  local started finished rc
  started="$(date --iso-8601=seconds)"
  printf '\n[%s] START %s\n' "$started" "$name" | tee -a "$LOG"
  "$@" >>"$LOG" 2>&1
  rc=$?
  finished="$(date --iso-8601=seconds)"
  printf '[%s] END %s rc=%s\n' "$finished" "$name" "$rc" | tee -a "$LOG"
  printf '%s\t%s\t%s\t%s\n' "$started" "$finished" "$name" "$rc" >>"$STATUS"
  return "$rc"
}

if ! record_step gpu-preflight \
  .venv/bin/python -c \
  'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory)'; then
  exit 2
fi

smoke_configs=(
  configs/experiments/w06_4_5b_smoke_bs2.yaml
  configs/experiments/w06_4_5b_smoke_bs4.yaml
  configs/experiments/w06_4_5b_smoke_bs8.yaml
  configs/experiments/w06_4_5b_smoke_bs16.yaml
)
full_configs=(
  configs/experiments/w06_4_5b_50k_8gb_bs2.yaml
  configs/experiments/w06_4_5b_50k_8gb_bs4.yaml
  configs/experiments/w06_4_5b_50k_8gb_bs8.yaml
  configs/experiments/w06_4_5b_50k_8gb_bs16.yaml
)
selected=""
best_throughput="0"
max_safe_reserved_bytes=7350000000

for index in "${!smoke_configs[@]}"; do
  smoke_config="${smoke_configs[$index]}"
  full_config="${full_configs[$index]}"
  name="$(basename "$smoke_config" .yaml)"
  smoke_dir="runs/${name}"
  if record_step "smoke-${name}" \
    .venv/bin/python scripts/train_sft.py \
    --config "$smoke_config" \
    --max-steps 3 \
    --output-dir "$smoke_dir" \
    --no-panel \
    --resume-if-available; then
    read -r throughput peak_reserved < <(
      .venv/bin/python -c \
      'import json,sys; d=json.load(open(sys.argv[1])); print(d["throughput_examples_per_second"], d["peak_vram_reserved_bytes"])' \
      "$smoke_dir/sft_summary.json"
    )
    printf '[%s] Candidate %s throughput=%s examples/s peak_reserved=%s bytes.\n' \
      "$(date --iso-8601=seconds)" "$full_config" "$throughput" "$peak_reserved" \
      | tee -a "$LOG"
    if awk -v throughput="$throughput" -v best="$best_throughput" \
      -v peak="$peak_reserved" -v limit="$max_safe_reserved_bytes" \
      'BEGIN { exit ! (throughput > best && peak <= limit) }'; then
      selected="$full_config"
      best_throughput="$throughput"
    fi
    continue
  fi
  if ! tail -n 300 "$LOG" | grep -Eqi \
    'CUDA out of memory|CUDA error: out of memory|OutOfMemoryError'; then
    printf '[%s] Smoke failed for a reason other than OOM; lower lengths will not help.\n' \
      "$(date --iso-8601=seconds)" | tee -a "$LOG"
    break
  fi
done

if [[ -z "$selected" ]]; then
  printf '[%s] No safe 4.5B configuration passed smoke; queue stopped without starting a long run.\n' \
    "$(date --iso-8601=seconds)" | tee -a "$LOG"
  exit 4
fi

printf '[%s] Selected %s at %s examples/s for the full resumable run.\n' \
  "$(date --iso-8601=seconds)" "$selected" "$best_throughput" | tee -a "$LOG"

record_step "train-$(basename "$selected" .yaml)" \
  .venv/bin/python scripts/train_sft.py \
  --config "$selected" \
  --resume-if-available
train_rc=$?

printf '[%s] Overnight queue complete; training rc=%s.\n' \
  "$(date --iso-8601=seconds)" "$train_rc" | tee -a "$LOG"
exit "$train_rc"
