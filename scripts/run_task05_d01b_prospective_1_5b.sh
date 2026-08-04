#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

PHASE=${1:-}
case "$PHASE" in
  preflight|prepare-cohort|generate|score|select-compare) ;;
  *)
    echo "usage: $0 {preflight|prepare-cohort|generate|score|select-compare}" >&2
    exit 2
    ;;
esac

PYTHON=${DOC2QUERY_PYTHON:-$ROOT/.venv-gpu/bin/python}
CONTRACT=${D01B_PROSPECTIVE_CONTRACT:-configs/evaluation/d01b_prospective_1_5b_v1.yaml}
ARTIFACT_ROOT=${D01B_PROSPECTIVE_ARTIFACT_ROOT:-artifacts/task05/d01b_prospective_1_5b_v1}
MEASUREMENT_ROOT=${D01B_PROSPECTIVE_MEASUREMENT_ROOT:-reports/measurements/task05/d01b_prospective_1_5b_v1}
LOG_ROOT=${D01B_PROSPECTIVE_LOG_ROOT:-logs/task05/d01b_prospective_1_5b_v1}
BASE_CONFIG=${D01B_PROSPECTIVE_BASE_CONFIG:-configs/experiments/d01b_prospective_w05_1_5b_s42.yaml}
CONTROLLED_CONFIG=${D01B_PROSPECTIVE_CONTROLLED_CONFIG:-configs/experiments/d01b_prospective_d01_1_5b_s42.yaml}
RUNNER_CONTRACT=${D01B_PROSPECTIVE_RUNNER_CONTRACT:-task05-d01b-prospective-runner-v1}
COHORT=$ARTIFACT_ROOT/cohort.materialized.json
BASE_GENERATIONS=$ARTIFACT_ROOT/generation/baseline.jsonl
CONTROLLED_GENERATIONS=$ARTIFACT_ROOT/generation/controlled.jsonl
BASE_SCORE=$MEASUREMENT_ROOT/scoring/baseline
CONTROLLED_SCORE=$MEASUREMENT_ROOT/scoring/controlled
mkdir -p "$ARTIFACT_ROOT" "$MEASUREMENT_ROOT" "$LOG_ROOT"

exec 9>"$MEASUREMENT_ROOT/campaign.lock"
if ! flock -n 9; then
  echo "another prospective D01b phase owns the campaign lock" >&2
  exit 3
fi
if [[ ! -x $PYTHON ]]; then
  echo "missing GPU environment: $PYTHON" >&2
  exit 2
fi

STATUS=$MEASUREMENT_ROOT/status.json
LOG=$LOG_ROOT/$PHASE.log
STARTED=$(date --iso-8601=seconds)

write_status() {
  local finished=$1 rc=$2
  "$PYTHON" - "$STATUS" "$PHASE" "$STARTED" "$finished" "$rc" "$LOG" "$RUNNER_CONTRACT" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {"schema_version": 1, "contract": sys.argv[7], "phases": {}, "final_tests_used": []}
if path.is_file():
    payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("contract") != sys.argv[7] or payload.get("final_tests_used") != []:
    raise ValueError("prospective runner status identity drifted")
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

run_phase() {
  local rc finished
  printf '[%s] START %s\n' "$STARTED" "$PHASE" | tee -a "$LOG"
  set +e
  "$@" >>"$LOG" 2>&1
  rc=$?
  set -e
  finished=$(date --iso-8601=seconds)
  printf '[%s] END %s rc=%s\n' "$finished" "$PHASE" "$rc" | tee -a "$LOG"
  write_status "$finished" "$rc"
  return "$rc"
}

assert_gpu_idle() {
  local active query_rc pmon_rc
  set +e
  active=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null)
  query_rc=$?
  set -e
  if [[ $query_rc -ne 0 ]]; then
    set +e
    active=$(nvidia-smi pmon -c 1 2>/dev/null | awk '$3 == "C" || $3 == "C+G" {print $2 "," $10}')
    pmon_rc=$?
    set -e
    if [[ $pmon_rc -ne 0 ]]; then
      echo "cannot verify GPU compute-process state; refusing phase $PHASE" >&2
      return 5
    fi
  fi
  if [[ -n $active ]]; then
    echo "GPU has active compute processes; refusing phase $PHASE" >&2
    echo "$active" >&2
    return 4
  fi
}

export HF_HOME=${HF_HOME:-$ROOT/.cache/huggingface}
export HF_HUB_CACHE=$HF_HOME/hub
export HF_ASSETS_CACHE=$HF_HOME/assets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1

preflight=("$PYTHON" scripts/run_d01b_prospective.py preflight --contract "$CONTRACT")
validate_cohort=("$PYTHON" scripts/run_d01b_prospective.py validate-cohort --contract "$CONTRACT" --cohort-manifest "$COHORT")

case "$PHASE" in
  preflight)
    run_phase "${preflight[@]}"
    ;;
  prepare-cohort)
    run_phase "$PYTHON" scripts/run_d01b_prospective.py prepare-cohort \
      --contract "$CONTRACT" --cohort-manifest "$COHORT"
    ;;
  generate)
    assert_gpu_idle
    run_phase bash -c '
      set -euo pipefail
      python=$1; contract=$2; cohort=$3; base=$4; controlled=$5; base_config=$6; controlled_config=$7
      "$python" scripts/run_d01b_prospective.py preflight --contract "$contract"
      "$python" scripts/run_d01b_prospective.py validate-cohort --contract "$contract" --cohort-manifest "$cohort"
      "$python" scripts/run_d01_postprocess.py generation-batched \
        --config "$base_config" \
        --frozen-manifest data/processed/v1/evaluation/task04-v1/manifest.json \
        --subset dev_intrinsic --cohort-manifest "$cohort" \
        --adapter runs/W05-1.5B-50K-8GB/adapter --output "$base" \
        --generation-batch-size 16
      "$python" scripts/run_d01_postprocess.py generation-batched \
        --config "$controlled_config" \
        --frozen-manifest data/processed/v1/evaluation/task04-v1/manifest.json \
        --subset dev_intrinsic --cohort-manifest "$cohort" \
        --adapter runs/D01-1.5B-STYLE-50K-S42/adapter --output "$controlled" \
        --generation-batch-size 16
      "$python" scripts/run_d01b_prospective.py validate-exact-k --contract "$contract" --summary "$base.summary.json"
      "$python" scripts/run_d01b_prospective.py validate-exact-k --contract "$contract" --summary "$controlled.summary.json"
    ' _ "$PYTHON" "$CONTRACT" "$COHORT" "$BASE_GENERATIONS" "$CONTROLLED_GENERATIONS" "$BASE_CONFIG" "$CONTROLLED_CONFIG"
    ;;
  score)
    assert_gpu_idle
    run_phase bash -c '
      set -euo pipefail
      python=$1; contract=$2; cohort=$3; base=$4; controlled=$5; base_score=$6; controlled_score=$7
      "$python" scripts/run_d01b_prospective.py preflight --contract "$contract"
      "$python" scripts/run_d01b_prospective.py validate-cohort --contract "$contract" --cohort-manifest "$cohort"
      "$python" scripts/run_d01b_prospective.py validate-exact-k --contract "$contract" --summary "$base.summary.json"
      "$python" scripts/run_d01b_prospective.py validate-exact-k --contract "$contract" --summary "$controlled.summary.json"
      score_one() {
        local generations=$1 output=$2
        "$python" scripts/run_d01_postprocess.py score \
          --generations "$generations" --generation-summary "$generations.summary.json" \
          --output-dir "$output" \
          --primary-judge configs/reranker/primary_polish_roberta_v3_cuda.yaml \
          --shadow-judge configs/reranker/shadow_bge_v2_m3.yaml \
          --primary-judge-device cuda --shadow-judge-device cuda \
          --corpus-index data/processed/v1/evaluation/corpus-bm25-v1
      }
      score_one "$base" "$base_score"
      score_one "$controlled" "$controlled_score"
      "$python" scripts/run_d01b_prospective.py validate-scoring --contract "$contract" --summary "$base_score/summary.json"
      "$python" scripts/run_d01b_prospective.py validate-scoring --contract "$contract" --summary "$controlled_score/summary.json"
    ' _ "$PYTHON" "$CONTRACT" "$COHORT" "$BASE_GENERATIONS" "$CONTROLLED_GENERATIONS" "$BASE_SCORE" "$CONTROLLED_SCORE"
    ;;
  select-compare)
    assert_gpu_idle
    run_phase bash -c '
      set -euo pipefail
      python=$1; contract=$2; cohort=$3; base_score=$4; controlled_score=$5; root=$6
      "$python" scripts/run_d01b_prospective.py preflight --contract "$contract"
      "$python" scripts/run_d01b_prospective.py validate-cohort --contract "$contract" --cohort-manifest "$cohort"
      "$python" scripts/run_d01b_prospective.py validate-scoring --contract "$contract" --summary "$base_score/summary.json"
      "$python" scripts/run_d01b_prospective.py validate-scoring --contract "$contract" --summary "$controlled_score/summary.json"
      "$python" scripts/run_d01b_prospective.py select-compare \
        --contract "$contract" --cohort-manifest "$cohort" \
        --baseline-rows "$base_score/per_generation.jsonl" \
        --controlled-rows "$controlled_score/per_generation.jsonl" \
        --output-json "$root/report.json" --output-markdown "$root/report.md" \
        --output-selected "$root/selected.jsonl" \
        --semantic-cache-dir "$root/semantic_cache" --semantic-device cuda
    ' _ "$PYTHON" "$CONTRACT" "$COHORT" "$BASE_SCORE" "$CONTROLLED_SCORE" "$MEASUREMENT_ROOT"
    ;;
esac
