#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

PHASE=${1:-}
case "$PHASE" in
  preflight|run|compare|run-all) ;;
  *)
    echo "usage: $0 {preflight|run|compare|run-all}" >&2
    exit 2
    ;;
esac

PYTHON=${DOC2QUERY_PYTHON:-$ROOT/.venv-gpu/bin/python}
CONFIG=configs/evaluation/d01b_probe_dev_confirm_v1.yaml
RUN_ROOT=runs/task05_d01b_probe_dev_confirm_v2_batch2
MEASUREMENT_ROOT=reports/measurements/task05/d01b_probe_dev_confirm_v2_batch2
LOG_ROOT=logs/task05_d01b_probe_dev_confirm_v2_batch2
STATUS=$MEASUREMENT_ROOT/status.json
mkdir -p "$RUN_ROOT" "$MEASUREMENT_ROOT" "$LOG_ROOT"

exec 9>"$MEASUREMENT_ROOT/campaign.lock"
if ! flock -n 9; then
  echo "another D01b dev-confirm phase owns the campaign lock" >&2
  exit 3
fi
if [[ ! -x $PYTHON ]]; then
  echo "missing GPU environment: $PYTHON" >&2
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
export DOC2QUERY_PROGRESS=1

assert_gpu_idle() {
  local active query_rc pmon_rc attempt
  set +e
  active=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null)
  query_rc=$?
  set -e
  if [[ $query_rc -ne 0 ]]; then
    pmon_rc=1
    for attempt in 1 2 3; do
      set +e
      active=$(nvidia-smi pmon -c 1 2>/dev/null | awk '$3 == "C" || $3 == "C+G" {print $2 "," $10}')
      pmon_rc=$?
      set -e
      if [[ $pmon_rc -eq 0 ]]; then
        break
      fi
      sleep 1
    done
    if [[ $pmon_rc -ne 0 ]]; then
      echo "cannot verify GPU compute-process state; refusing D01b dev-confirm training" >&2
      return 5
    fi
  fi
  if [[ -n $active ]]; then
    echo "GPU has active compute processes; refusing D01b dev-confirm training" >&2
    echo "$active" >&2
    return 4
  fi
}

write_status() {
  local phase=$1 started=$2 finished=$3 rc=$4 log=$5
  "$PYTHON" - "$STATUS" "$phase" "$started" "$finished" "$rc" "$log" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "contract": "task05-d01b-probe-dev-confirm-runner-v1",
    "phases": {},
    "final_tests_used": [],
}
if path.is_file():
    payload = json.loads(path.read_text(encoding="utf-8"))
if (
    payload.get("contract") != "task05-d01b-probe-dev-confirm-runner-v1"
    or payload.get("final_tests_used") != []
):
    raise ValueError("D01b dev-confirm runner status identity drifted")
payload["phases"][sys.argv[2]] = {
    "started_at": sys.argv[3],
    "finished_at": sys.argv[4],
    "exit_code": int(sys.argv[5]),
    "log": sys.argv[6],
}
temporary = path.with_suffix(".json.tmp")
temporary.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
os.replace(temporary, path)
PY
}

run_logged() {
  local phase=$1
  shift
  local started finished rc log
  started=$(date --iso-8601=seconds)
  log=$LOG_ROOT/$phase.log
  printf '[%s] START %s\n' "$started" "$phase" | tee -a "$log"
  set +e
  ( set -e; "$@" ) >>"$log" 2>&1
  rc=$?
  set -e
  finished=$(date --iso-8601=seconds)
  printf '[%s] END %s rc=%s\n' "$finished" "$phase" "$rc" | tee -a "$log"
  write_status "$phase" "$started" "$finished" "$rc" "$log"
  return "$rc"
}

preflight() {
  "$PYTHON" scripts/run_d01b_probe_dev_confirm.py preflight --config "$CONFIG"
}

train_arm() {
  local id=$1 input=$2 generator=$3 seed=$4
  "$PYTHON" scripts/train_probe_embedder.py \
    --recipe configs/evaluation/probe_v1.yaml \
    --comparison-contract configs/evaluation/comparison_contract_v1.yaml \
    --train-input "$input" \
    --synthetic-generations "$input" \
    --generator-id "$generator" \
    --query-source synthetic \
    --frozen-manifest data/processed/v1/evaluation/task04-v1/manifest.json \
    --test-subset dev_intrinsic_rank10 \
    --corpus data/processed/v1/documents.parquet \
    --primary-judge-config configs/reranker/primary_polish_roberta_v3_p03_gpu_batch4.yaml \
    --seed "$seed" \
    --max-steps 4000 \
    --batch-size 2 \
    --train-prefix-limit 7936 \
    --checkpoint-interval-steps 100 \
    --evaluation-encode-batch-size 8 \
    --retrieval-query-batch-size 512 \
    --retrieval-device cuda \
    --output-dir "$RUN_ROOT/$id"
}

train_all() {
  local seed
  preflight
  for seed in 42 43 44; do
    train_arm \
      "D01B-PROBE-W05-DEV-CONFIRM-S${seed}-B2" \
      artifacts/task05/d01b_prospective_1_5b_v3/probe_inputs/w05_baseline.jsonl \
      W05-1.5B-50K-8GB \
      "$seed"
    train_arm \
      "D01B-PROBE-HYBRID-DEV-CONFIRM-S${seed}-B2" \
      artifacts/task05/d01b_prospective_1_5b_v3/probe_inputs/selected_hybrid.jsonl \
      D01B-PROSPECTIVE-V3-HYBRID-1.5B-S42 \
      "$seed"
  done
}

compare() {
  "$PYTHON" scripts/run_d01b_probe_dev_confirm.py compare --config "$CONFIG"
}

run_all() {
  train_all
  compare
}

case "$PHASE" in
  preflight)
    run_logged preflight preflight
    ;;
  run)
    assert_gpu_idle
    run_logged run train_all
    ;;
  compare)
    run_logged compare compare
    ;;
  run-all)
    assert_gpu_idle
    run_logged run-all run_all
    ;;
esac
