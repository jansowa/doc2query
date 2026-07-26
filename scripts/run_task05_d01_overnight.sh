#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

PYTHON=${DOC2QUERY_PYTHON:-$ROOT/.venv-gpu/bin/python}
GATE_SUMMARY="$ROOT/artifacts/task04/hn_full_gate_v1/summary.json"
SMOKE_CONFIG="$ROOT/configs/experiments/d01_1_5b_style_smoke_s42.yaml"
TRAIN_CONFIG="$ROOT/configs/experiments/d01_1_5b_style_50k_s42.yaml"
GENERATION_CONFIG="$ROOT/configs/experiments/d01_1_5b_style_dev_generation_s42.yaml"
TRAIN_DIR="$ROOT/runs/D01-1.5B-STYLE-50K-S42"
GENERATION_DIR="$ROOT/runs/D01-1.5B-STYLE-DEV-GENERATION-S42/generation"
GENERATION_OUTPUT="$GENERATION_DIR/controlled.jsonl"
SMOKE_4_5B_CONFIG="$ROOT/configs/experiments/d01_4_5b_style_smoke_s42.yaml"
TRAIN_4_5B_CONFIG="$ROOT/configs/experiments/d01_4_5b_style_50k_s42.yaml"
GENERATION_4_5B_CONFIG="$ROOT/configs/experiments/d01_4_5b_style_dev_generation_s42.yaml"
TRAIN_4_5B_DIR="$ROOT/runs/D01-4.5B-STYLE-50K-S42"
GENERATION_4_5B_DIR="$ROOT/runs/D01-4.5B-STYLE-DEV-GENERATION-S42/generation"
GENERATION_4_5B_OUTPUT="$GENERATION_4_5B_DIR/controlled.jsonl"
LOG_DIR="$ROOT/logs/task05_d01_overnight"
REPORT_DIR="$ROOT/reports/measurements/task05_d01_overnight"
LOG="$LOG_DIR/queue.log"
STATUS="$REPORT_DIR/status.tsv"

mkdir -p "$LOG_DIR" "$REPORT_DIR"
exec 9>"$REPORT_DIR/queue.lock"
if ! flock -n 9; then
  printf '[%s] Another D01 queue owns the lock; exiting.\n' "$(date --iso-8601=seconds)" | tee -a "$LOG"
  exit 3
fi

if [[ ! -x $PYTHON ]]; then
  echo "D01 requires the project GPU environment: $PYTHON" >&2
  exit 2
fi

export HF_HOME=${HF_HOME:-$ROOT/.cache/huggingface}
export HF_HUB_CACHE=$HF_HOME/hub
export HF_ASSETS_CACHE=$HF_HOME/assets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1

if [[ ! -f $STATUS ]]; then
  printf 'started_at\tfinished_at\tname\texit_code\n' >"$STATUS"
fi

run_step() {
  local name=$1
  shift
  local started finished rc
  started=$(date --iso-8601=seconds)
  printf '\n[%s] START %s\n' "$started" "$name" | tee -a "$LOG"
  set +e
  "$@" >>"$LOG" 2>&1
  rc=$?
  set -e
  finished=$(date --iso-8601=seconds)
  printf '[%s] END %s rc=%s\n' "$finished" "$name" "$rc" | tee -a "$LOG"
  printf '%s\t%s\t%s\t%s\n' "$started" "$finished" "$name" "$rc" >>"$STATUS"
  return "$rc"
}

# Task 05 experiments remain closed until the currently running full HN gate
# has produced its dev-only, inference-only measured artifact.  This check
# deliberately does not infer a winning negative recipe from the measurement.
run_step hn-gate-preflight "$PYTHON" - "$GATE_SUMMARY" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"full HN gate is not complete: missing {path}")
summary = json.loads(path.read_text(encoding="utf-8"))
contract = summary.get("contract", {})
if summary.get("status") != "measured":
    raise SystemExit("full HN gate summary is not measured")
if contract.get("evaluation_subset") != "dev_intrinsic_rank10":
    raise SystemExit("full HN gate did not use the frozen dev-only cohort")
if summary.get("final_tests_used") != [] or summary.get("training_runs") != []:
    raise SystemExit("full HN gate provenance is not inference-only/dev-only")
if not summary.get("artifact_fingerprint"):
    raise SystemExit("full HN gate summary has no artifact fingerprint")
if int(summary.get("common_legal_query_count", 0)) < 1:
    raise SystemExit("full HN gate has no common legal query cohort")
print(
    "HN gate complete:",
    summary["artifact_fingerprint"],
    "legal_queries=",
    summary["common_legal_query_count"],
)
PY

run_step gpu-preflight "$PYTHON" - <<'PY'
import sys
import torch

if torch.version.cuda != "12.4":
    raise SystemExit(f"expected CUDA 12.4 torch build, got {torch.__version__} / {torch.version.cuda}")
if not torch.cuda.is_available():
    raise SystemExit(f"CUDA is unavailable from {sys.executable}")
properties = torch.cuda.get_device_properties(0)
print(
    f"GPU ready: {properties.name}, VRAM={properties.total_memory / 1024**3:.2f} GiB, "
    f"torch={torch.__version__}"
)
PY

run_step disk-preflight "$PYTHON" - "$ROOT" <<'PY'
import shutil
import sys

free = shutil.disk_usage(sys.argv[1]).free
minimum = 20 * 1024**3
if free < minimum:
    raise SystemExit(f"D01 requires at least 20 GiB free; found {free / 1024**3:.2f} GiB")
print(f"disk ready: {free / 1024**3:.2f} GiB free")
PY

run_step config-preflight "$PYTHON" - "$SMOKE_CONFIG" "$TRAIN_CONFIG" "$GENERATION_CONFIG" <<'PY'
import sys
from pathlib import Path
from doc2query.config import load_config

for value in sys.argv[1:]:
    config = load_config(Path(value))
    if not config.generation.controlled or [mode.value for mode in config.generation.focus_modes] != ["none"]:
        raise SystemExit(f"D01 must enable controls without focus: {value}")
    print(value, config.run.experiment_id)
PY

run_step config-preflight-4.5b \
  "$PYTHON" - "$SMOKE_4_5B_CONFIG" "$TRAIN_4_5B_CONFIG" "$GENERATION_4_5B_CONFIG" <<'PY'
import sys
from pathlib import Path
from doc2query.config import load_config

for value in sys.argv[1:]:
    config = load_config(Path(value))
    focus_modes = [mode.value for mode in config.generation.focus_modes]
    if not config.generation.controlled or focus_modes != ["none"]:
        raise SystemExit(f"D01 must enable controls without focus: {value}")
    if "Bielik-4.5B-v3.0-Instruct" not in config.model.name_or_path:
        raise SystemExit(f"D01 4.5B config resolved the wrong model: {value}")
    print(value, config.run.experiment_id)
PY

run_step smoke-train \
  "$PYTHON" scripts/train_sft.py \
  --config "$SMOKE_CONFIG" \
  --no-panel \
  --resume-if-available

run_step train-50k \
  "$PYTHON" scripts/train_sft.py \
  --config "$TRAIN_CONFIG" \
  --resume-if-available

# Run both main training jobs before diagnostics so a fixed 24 h window first
# secures the two reusable adapters.  W06 already proved this 4.5B recipe on
# the same physical 8 GB GPU; the smoke still fails closed on environment drift.
run_step smoke-train-4.5b \
  "$PYTHON" scripts/train_sft.py \
  --config "$SMOKE_4_5B_CONFIG" \
  --no-panel \
  --resume-if-available

run_step train-50k-4.5b \
  "$PYTHON" scripts/train_sft.py \
  --config "$TRAIN_4_5B_CONFIG" \
  --resume-if-available

GENERATION_SUMMARY=${GENERATION_OUTPUT%.jsonl}.summary.json
if [[ -f $GENERATION_OUTPUT && -f $GENERATION_SUMMARY ]]; then
  printf '[%s] Diagnostic controlled generation is already complete; skipping.\n' \
    "$(date --iso-8601=seconds)" | tee -a "$LOG"
else
  if [[ -f $GENERATION_OUTPUT || -f $GENERATION_SUMMARY ]]; then
    archive_suffix=$(date +%Y%m%dT%H%M%S)
    [[ ! -f $GENERATION_OUTPUT ]] || \
      mv "$GENERATION_OUTPUT" "$GENERATION_OUTPUT.incomplete-$archive_suffix"
    [[ ! -f $GENERATION_SUMMARY ]] || \
      mv "$GENERATION_SUMMARY" "$GENERATION_SUMMARY.incomplete-$archive_suffix"
  fi
  mkdir -p "$GENERATION_DIR"
  run_step generate-dev-diagnostic \
    "$PYTHON" -m doc2query.cli generate \
    --config "$GENERATION_CONFIG" \
    --adapter "$TRAIN_DIR/adapter" \
    --output "$GENERATION_OUTPUT"
fi

GENERATION_4_5B_SUMMARY=${GENERATION_4_5B_OUTPUT%.jsonl}.summary.json
if [[ -f $GENERATION_4_5B_OUTPUT && -f $GENERATION_4_5B_SUMMARY ]]; then
  printf '[%s] 4.5B diagnostic controlled generation is already complete; skipping.\n' \
    "$(date --iso-8601=seconds)" | tee -a "$LOG"
else
  if [[ -f $GENERATION_4_5B_OUTPUT || -f $GENERATION_4_5B_SUMMARY ]]; then
    archive_suffix=$(date +%Y%m%dT%H%M%S)
    [[ ! -f $GENERATION_4_5B_OUTPUT ]] || \
      mv "$GENERATION_4_5B_OUTPUT" \
        "$GENERATION_4_5B_OUTPUT.incomplete-$archive_suffix"
    [[ ! -f $GENERATION_4_5B_SUMMARY ]] || \
      mv "$GENERATION_4_5B_SUMMARY" \
        "$GENERATION_4_5B_SUMMARY.incomplete-$archive_suffix"
  fi
  mkdir -p "$GENERATION_4_5B_DIR"
  run_step generate-dev-diagnostic-4.5b \
    "$PYTHON" -m doc2query.cli generate \
    --config "$GENERATION_4_5B_CONFIG" \
    --adapter "$TRAIN_4_5B_DIR/adapter" \
    --output "$GENERATION_4_5B_OUTPUT"
fi

printf '[%s] D01 overnight queue complete. This is a dev diagnostic, not a final-test result.\n' \
  "$(date --iso-8601=seconds)" | tee -a "$LOG"
