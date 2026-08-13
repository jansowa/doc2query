#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

PHASE=${1:-}
case "$PHASE" in
  preflight|prepare-cohorts|generate|score|select|materialize|probe|compare|run-all) ;;
  *)
    echo "usage: $0 {preflight|prepare-cohorts|generate|score|select|materialize|probe|compare|run-all}" >&2
    exit 2
    ;;
esac

PYTHON=${DOC2QUERY_PYTHON:-$ROOT/.venv-gpu/bin/python}
CONFIG=configs/evaluation/d01b_scale_interaction_4_5b_pilot_v1.yaml
ARTIFACT_ROOT=artifacts/task05/d01b_scale_interaction_4_5b_pilot_v1
MEASUREMENT_ROOT=reports/measurements/task05/d01b_scale_interaction_4_5b_pilot_v1
LOG_ROOT=logs/task05/d01b_scale_interaction_4_5b_pilot_v1
PROBE_ROOT=runs/task05_d01b_scale_interaction_4_5b_pilot_v1
COHORT=$ARTIFACT_ROOT/cohort.materialized.json
EVAL_MANIFEST=$ARTIFACT_ROOT/evaluation/manifest.json
BASE_GENERATIONS=$ARTIFACT_ROOT/generation/baseline.jsonl
CONTROLLED_GENERATIONS=$ARTIFACT_ROOT/generation/controlled.jsonl
BASE_SCORE=$MEASUREMENT_ROOT/scoring/baseline
CONTROLLED_SCORE=$MEASUREMENT_ROOT/scoring/controlled
BASE_INPUT=$ARTIFACT_ROOT/probe_inputs/baseline.jsonl
HYBRID_INPUT=$ARTIFACT_ROOT/probe_inputs/hybrid.jsonl
mkdir -p "$ARTIFACT_ROOT" "$MEASUREMENT_ROOT" "$LOG_ROOT" "$PROBE_ROOT"

exec 9>"$MEASUREMENT_ROOT/campaign.lock"
if ! flock -n 9; then
  echo "another D01b scale-pilot phase owns the campaign lock" >&2
  exit 3
fi
if [[ ! -x $PYTHON ]]; then
  echo "missing project environment: $PYTHON" >&2
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
    "contract": "task05-d01b-scale-interaction-4.5b-pilot-runner-v1",
    "phases": {},
    "four_point_five_b_full_authorized": False,
    "final_tests_used": [],
}
if path.is_file():
    payload = json.loads(path.read_text(encoding="utf-8"))
if (
    payload.get("contract") != "task05-d01b-scale-interaction-4.5b-pilot-runner-v1"
    or payload.get("four_point_five_b_full_authorized") is not False
    or payload.get("final_tests_used") != []
):
    raise ValueError("scale-pilot runner status identity drifted")
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
  local name=$1
  shift
  local started finished rc log
  started=$(date --iso-8601=seconds)
  log=$LOG_ROOT/$name.log
  printf '[%s] START %s\n' "$started" "$name" | tee -a "$log"
  set +e
  ( set -e; "$@" ) >>"$log" 2>&1
  rc=$?
  set -e
  finished=$(date --iso-8601=seconds)
  printf '[%s] END %s rc=%s\n' "$finished" "$name" "$rc" | tee -a "$log"
  write_status "$name" "$started" "$finished" "$rc" "$log"
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
      echo "cannot verify GPU state; refusing $PHASE" >&2
      return 5
    fi
  fi
  if [[ -n $active ]]; then
    echo "GPU has active compute processes; refusing $PHASE" >&2
    return 4
  fi
}

preflight() {
  "$PYTHON" scripts/run_d01b_scale_pilot.py preflight --config "$CONFIG"
}

prepare_cohorts() {
  "$PYTHON" scripts/run_d01b_scale_pilot.py prepare-cohorts --config "$CONFIG"
  preflight
}

generate() {
  preflight
  "$PYTHON" scripts/run_d01_postprocess.py generation-batched \
    --config configs/experiments/d01b_scale_pilot_w06_4_5b_s42.yaml \
    --frozen-manifest data/processed/v1/evaluation/task04-v1/manifest.json \
    --subset dev_intrinsic --cohort-manifest "$COHORT" \
    --adapter runs/W06-4.5B-INSTRUCT-50K-8GB-BS8-L512/adapter \
    --output "$BASE_GENERATIONS" --generation-batch-size 8
  "$PYTHON" scripts/run_d01_postprocess.py generation-batched \
    --config configs/experiments/d01b_scale_pilot_d01_4_5b_s42.yaml \
    --frozen-manifest data/processed/v1/evaluation/task04-v1/manifest.json \
    --subset dev_intrinsic --cohort-manifest "$COHORT" \
    --adapter runs/D01-4.5B-STYLE-50K-S42/adapter \
    --output "$CONTROLLED_GENERATIONS" --generation-batch-size 8
  "$PYTHON" scripts/run_d01b_prospective.py validate-exact-k --contract "$CONFIG" --summary "$BASE_GENERATIONS.summary.json"
  "$PYTHON" scripts/run_d01b_prospective.py validate-exact-k --contract "$CONFIG" --summary "$CONTROLLED_GENERATIONS.summary.json"
}

score() {
  preflight
  score_one() {
    local generations=$1 output=$2
    "$PYTHON" scripts/run_d01_postprocess.py score \
      --generations "$generations" --generation-summary "$generations.summary.json" \
      --output-dir "$output" \
      --primary-judge configs/reranker/primary_polish_roberta_v3_cuda.yaml \
      --shadow-judge configs/reranker/shadow_bge_v2_m3.yaml \
      --primary-judge-device cuda --shadow-judge-device cuda \
      --corpus-index data/processed/v1/evaluation/corpus-bm25-v1 \
      --archive-incompatible
  }
  score_one "$BASE_GENERATIONS" "$BASE_SCORE"
  score_one "$CONTROLLED_GENERATIONS" "$CONTROLLED_SCORE"
  "$PYTHON" scripts/run_d01b_prospective.py validate-scoring --contract "$CONFIG" --summary "$BASE_SCORE/summary.json"
  "$PYTHON" scripts/run_d01b_prospective.py validate-scoring --contract "$CONFIG" --summary "$CONTROLLED_SCORE/summary.json"
}

select_candidates() {
  preflight
  "$PYTHON" scripts/run_d01b_prospective.py select-compare \
    --contract "$CONFIG" --cohort-manifest "$COHORT" \
    --baseline-rows "$BASE_SCORE/per_generation.jsonl" \
    --controlled-rows "$CONTROLLED_SCORE/per_generation.jsonl" \
    --output-json "$MEASUREMENT_ROOT/report.json" \
    --output-markdown "$MEASUREMENT_ROOT/report.md" \
    --output-selected "$MEASUREMENT_ROOT/selected.jsonl" \
    --semantic-cache-dir "$MEASUREMENT_ROOT/semantic_cache" --semantic-device cuda
}

materialize() {
  preflight
  "$PYTHON" scripts/run_d01b_prospective.py materialize-probe-inputs \
    --contract "$CONFIG" --report "$MEASUREMENT_ROOT/report.json" \
    --selected-rows "$MEASUREMENT_ROOT/selected.jsonl" \
    --baseline-rows "$BASE_SCORE/per_generation.jsonl" \
    --controlled-rows "$CONTROLLED_SCORE/per_generation.jsonl" \
    --probe-recipe configs/evaluation/probe_v1.yaml \
    --baseline-output "$BASE_INPUT" --hybrid-output "$HYBRID_INPUT" \
    --manifest-output "$ARTIFACT_ROOT/probe_inputs/manifest.json"
}

train_probe_arm() {
  local id=$1 input=$2 generator=$3
  if [[ -f $PROBE_ROOT/$id/result.json ]]; then return; fi
  "$PYTHON" scripts/train_probe_embedder.py \
    --recipe configs/evaluation/probe_v1.yaml \
    --comparison-contract configs/evaluation/comparison_contract_v1.yaml \
    --train-input "$input" --synthetic-generations "$input" \
    --generator-id "$generator" --query-source synthetic \
    --frozen-manifest "$EVAL_MANIFEST" --test-subset dev_d01b_scale_pilot_v1 \
    --corpus data/processed/v1/documents.parquet \
    --primary-judge-config configs/reranker/primary_polish_roberta_v3_p03_gpu_batch4.yaml \
    --seed 42 --max-steps 1024 --batch-size 2 --train-prefix-limit 3072 \
    --checkpoint-interval-steps 64 --evaluation-encode-batch-size 8 \
    --retrieval-query-batch-size 512 --retrieval-device cuda \
    --output-dir "$PROBE_ROOT/$id"
}

probe() {
  preflight
  train_probe_arm D01B-SCALE-PILOT-W06-PROBE-S42 "$BASE_INPUT" W06-4.5B-INSTRUCT-50K-8GB-BS8-L512
  train_probe_arm D01B-SCALE-PILOT-HYBRID-PROBE-S42 "$HYBRID_INPUT" D01B-SCALE-PILOT-HYBRID-4.5B-S42
}

compare() {
  preflight
  "$PYTHON" scripts/run_d01b_scale_pilot.py compare --config "$CONFIG"
}

run_all() {
  prepare_cohorts
  generate
  score
  select_candidates
  materialize
  probe
  compare
}

case "$PHASE" in
  preflight) run_logged preflight preflight ;;
  prepare-cohorts) run_logged prepare-cohorts prepare_cohorts ;;
  generate) assert_gpu_idle; run_logged generate generate ;;
  score) assert_gpu_idle; run_logged score score ;;
  select) assert_gpu_idle; run_logged select select_candidates ;;
  materialize) run_logged materialize materialize ;;
  probe) assert_gpu_idle; run_logged probe probe ;;
  compare) run_logged compare compare ;;
  run-all) assert_gpu_idle; run_logged run-all run_all ;;
esac
