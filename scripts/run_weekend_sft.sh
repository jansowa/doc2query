#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p logs reports/weekend_sft
LOG="logs/weekend_sft.log"
STATUS="reports/weekend_sft/status.tsv"
touch "$LOG"
if [[ ! -f "$STATUS" ]]; then
  printf 'started_at\tfinished_at\tname\texit_code\n' >"$STATUS"
fi

export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=600
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

run_step() {
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
  return 0
}

if ! .venv/bin/python -c \
  'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))' \
  >>"$LOG" 2>&1; then
  printf '[%s] CUDA unavailable; queue aborted\n' "$(date --iso-8601=seconds)" | tee -a "$LOG"
  exit 2
fi

run_step memory-probe-512 \
  .venv/bin/python scripts/run_memory_probe.py \
  --config configs/experiments/s01_1_5b_8gb_smoke.yaml \
  --lengths 512 --steps 2 \
  --output-dir reports/weekend_sft/memory_probes

run_step smoke-20 \
  .venv/bin/python scripts/train_sft.py \
  --config configs/experiments/s01_1_5b_8gb_smoke.yaml \
  --resume-if-available

for config in \
  configs/experiments/w01_1_5b_10k_lr1e4_seed42.yaml \
  configs/experiments/w02_1_5b_10k_lr5e5_seed42.yaml \
  configs/experiments/w03_1_5b_10k_lr2e4_seed42.yaml \
  configs/experiments/w04_1_5b_10k_lr1e4_seed43.yaml \
  configs/experiments/w05_1_5b_50k_8gb.yaml
do
  run_step "train-$(basename "$config" .yaml)" \
    .venv/bin/python scripts/train_sft.py \
    --config "$config" \
    --resume-if-available
done

printf '[%s] Weekend queue complete\n' "$(date --iso-8601=seconds)" | tee -a "$LOG"
