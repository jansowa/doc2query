#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

PHASE=${1:-}
case "$PHASE" in
  preflight|stage-seed42|run|compare|run-all) ;;
  *)
    echo "usage: $0 {preflight|stage-seed42|run|compare|run-all}" >&2
    exit 2
    ;;
esac

PYTHON=${DOC2QUERY_PYTHON:-$ROOT/.venv-gpu/bin/python}
CONFIG=configs/evaluation/d01b_scale_interaction_4_5b_trivia_dev_confirm_v1.yaml
RUN_ROOT=runs/task05_d01b_scale_interaction_4_5b_trivia_dev_confirm_v1
MEASUREMENT_ROOT=reports/measurements/task05/d01b_scale_interaction_4_5b_trivia_dev_confirm_v1
LOG_ROOT=logs/task05/d01b_scale_interaction_4_5b_trivia_dev_confirm_v1
MANIFEST=data/processed/task05/d01b_trivia_external_dev_v1/manifest.json
CORPUS=data/processed/task05/d01b_trivia_external_dev_v1/documents.parquet
BASE_INPUT=artifacts/task05/d01b_scale_interaction_4_5b_pilot_v1/probe_inputs/baseline.jsonl
HYBRID_INPUT=artifacts/task05/d01b_scale_interaction_4_5b_pilot_v1/probe_inputs/hybrid.jsonl
mkdir -p "$RUN_ROOT" "$MEASUREMENT_ROOT" "$LOG_ROOT"

exec 9>"$MEASUREMENT_ROOT/campaign.lock"
if ! flock -n 9; then
  echo "another D01b TriviaQA confirm phase owns the campaign lock" >&2
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

STATUS=$MEASUREMENT_ROOT/status.json
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
    "contract": "task05-d01b-scale-interaction-4.5b-trivia-dev-confirm-runner-v1",
    "phases": {},
    "pilot_retraining": False,
    "four_point_five_b_full_authorized": False,
    "final_tests_used": [],
}
if path.is_file():
    payload = json.loads(path.read_text(encoding="utf-8"))
if (
    payload.get("contract")
    != "task05-d01b-scale-interaction-4.5b-trivia-dev-confirm-runner-v1"
    or payload.get("pilot_retraining") is not False
    or payload.get("four_point_five_b_full_authorized") is not False
    or payload.get("final_tests_used") != []
):
    raise ValueError("TriviaQA confirm runner status identity drifted")
payload["phases"][sys.argv[2]] = {
    "started_at": sys.argv[3],
    "finished_at": sys.argv[4],
    "exit_code": int(sys.argv[5]),
    "log": sys.argv[6],
}
temporary = path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
      if [[ $pmon_rc -eq 0 ]]; then break; fi
      sleep 1
    done
    if [[ $pmon_rc -ne 0 ]]; then
      echo "cannot verify GPU state; refusing TriviaQA confirm" >&2
      return 5
    fi
  fi
  if [[ -n $active ]]; then
    echo "GPU has active compute processes; refusing TriviaQA confirm" >&2
    return 4
  fi
}

preflight() {
  "$PYTHON" scripts/run_d01b_trivia_confirm.py preflight --config "$CONFIG"
}

preflight_staged() {
  "$PYTHON" scripts/run_d01b_trivia_confirm.py preflight --config "$CONFIG" \
    --require-staged-seed42
}

stage_seed42() {
  "$PYTHON" scripts/run_d01b_trivia_confirm.py stage-seed42 --config "$CONFIG"
}

train_arm() {
  local id=$1 input=$2 generator=$3 seed=$4
  "$PYTHON" scripts/train_probe_embedder.py \
    --recipe configs/evaluation/probe_v1.yaml \
    --comparison-contract configs/evaluation/comparison_contract_v1.yaml \
    --train-input "$input" --synthetic-generations "$input" \
    --generator-id "$generator" --query-source synthetic \
    --frozen-manifest "$MANIFEST" --test-subset dev_d01b_trivia_external_v1 \
    --corpus "$CORPUS" \
    --primary-judge-config configs/reranker/primary_polish_roberta_v3_p03_gpu_batch4.yaml \
    --seed "$seed" --max-steps 1024 --batch-size 2 --train-prefix-limit 3072 \
    --checkpoint-interval-steps 64 --evaluation-encode-batch-size 8 \
    --retrieval-query-batch-size 512 --retrieval-device cuda \
    --output-dir "$RUN_ROOT/$id"
}

run_confirm() {
  local seed
  stage_seed42
  preflight_staged
  for seed in 42 43 44; do
    train_arm "D01B-TRIVIA-CONFIRM-W06-4.5B-S${seed}" "$BASE_INPUT" \
      W06-4.5B-INSTRUCT-50K-8GB-BS8-L512 "$seed"
    train_arm "D01B-TRIVIA-CONFIRM-HYBRID-4.5B-S${seed}" "$HYBRID_INPUT" \
      D01B-SCALE-PILOT-HYBRID-4.5B-S42 "$seed"
  done
}

compare() {
  preflight_staged
  "$PYTHON" scripts/run_d01b_trivia_confirm.py compare --config "$CONFIG"
}

run_all() {
  run_confirm
  compare
}

case "$PHASE" in
  preflight) run_logged preflight preflight ;;
  stage-seed42) run_logged stage-seed42 stage_seed42 ;;
  run) assert_gpu_idle; run_logged run run_confirm ;;
  compare) run_logged compare compare ;;
  run-all) assert_gpu_idle; run_logged run-all run_all ;;
esac
