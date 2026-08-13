#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$ROOT/configs/preferences/task06_candidate_execution_design_v1.yaml"
AUDIT="$ROOT/reports/measurements/task06/candidate_execution_design_v1/id_only_audit.json"
PREFLIGHT="$ROOT/reports/measurements/task06/candidate_execution_design_v1/preflight.json"
OUTPUT="$ROOT/artifacts/task06/candidate_smoke_v1"
PILOT_OUTPUT="$ROOT/artifacts/task06/candidate_pilot_v1"
LOGS="$OUTPUT/logs"
PYTHON="$ROOT/.venv-gpu/bin/python"
RUNNER="$ROOT/scripts/run_task06_smoke.py"
LOCK="$OUTPUT/run.lock"

run_stage() {
  local name="$1"
  shift
  mkdir -p "$LOGS"
  "$@" 2>&1 | tee "$LOGS/$name.log"
}

require_gpu() {
  nvidia-smi --query-gpu=index,memory.total,memory.used,utilization.gpu \
    --format=csv,noheader,nounits
  "$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"'
}

run_execution() {
  local stage="$1"
  local target="$2"
  local passages="$3"
  local exclude_args=()
  if [[ "$stage" == "pilot" ]]; then
    exclude_args=(--exclude-ids "$OUTPUT/cohort.ids.json")
  fi
  LOGS="$target/logs"
  LOCK="$target/run.lock"
  mkdir -p "$target"
  exec 9>"$LOCK"
  if ! flock -n 9; then
    echo "FAIL CLOSED: another Task 06 smoke runner holds $LOCK" >&2
    exit 1
  fi
  require_gpu
  export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
  export HF_HUB_CACHE="$HF_HOME/hub"
  export HF_ASSETS_CACHE="$HF_HOME/assets"
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export TOKENIZERS_PARALLELISM=false
  run_stage prepare "$PYTHON" "$RUNNER" prepare --design "$CONFIG" --output-root "$target" \
    --stage-name "$stage" --passages "$passages" "${exclude_args[@]}"
  run_stage generate_w06 "$PYTHON" "$RUNNER" generate --design "$CONFIG" \
    --output-root "$target" --role w06_anchor --stage-name "$stage" --passages "$passages"
  run_stage generate_d01 "$PYTHON" "$RUNNER" generate --design "$CONFIG" \
    --output-root "$target" --role d01_controlled --stage-name "$stage" --passages "$passages"
  run_stage score_w06 "$PYTHON" "$RUNNER" score --design "$CONFIG" \
    --output-root "$target" --role w06_anchor --device cuda --stage-name "$stage"
  run_stage score_d01 "$PYTHON" "$RUNNER" score --design "$CONFIG" \
    --output-root "$target" --role d01_controlled --device cuda --stage-name "$stage"
  run_stage natural "$PYTHON" "$RUNNER" natural --design "$CONFIG" \
    --output-root "$target" --device cuda --stage-name "$stage"
  run_stage select "$PYTHON" "$RUNNER" select --design "$CONFIG" \
    --output-root "$target" --device cuda --stage-name "$stage"
}

case "${1:-}" in
  preflight)
    "$ROOT/.venv/bin/python" "$ROOT/scripts/preflight_task06_candidate_execution.py" \
      --config "$CONFIG" --audit "$AUDIT" --output "$PREFLIGHT"
    ;;
  run-smoke)
    run_execution smoke "$OUTPUT" 32
    ;;
  run-pilot)
    run_execution pilot "$PILOT_OUTPUT" 512
    ;;
  status)
    if [[ -f "$PILOT_OUTPUT/selection/report.json" ]]; then
      "$ROOT/.venv/bin/python" -m json.tool "$PILOT_OUTPUT/selection/report.json"
    elif [[ -d "$PILOT_OUTPUT" ]]; then
      find "$PILOT_OUTPUT" -maxdepth 3 -type f -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' | sort
    elif [[ -f "$OUTPUT/selection/report.json" ]]; then
      "$ROOT/.venv/bin/python" -m json.tool "$OUTPUT/selection/report.json"
    elif [[ -d "$OUTPUT" ]]; then
      find "$OUTPUT" -maxdepth 3 -type f -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' | sort
    elif [[ -f "$PREFLIGHT" ]]; then
      "$ROOT/.venv/bin/python" -m json.tool "$PREFLIGHT"
    else
      echo "preflight_missing"
    fi
    ;;
  *)
    echo "usage: $0 {preflight|status|run-smoke|run-pilot}" >&2
    exit 2
    ;;
esac
