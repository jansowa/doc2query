#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

PYTHON=${DOC2QUERY_PYTHON:-$ROOT/.venv-gpu/bin/python}
LOG_DIR=$ROOT/logs/task04_p05_dev_screen
RUN_ROOT=$ROOT/runs/task04_p05_dev_screen/dev_screen
INTERRUPTED_ROOT=$ROOT/runs/task04_p05_dev_screen/interrupted
mkdir -p "$LOG_DIR" "$RUN_ROOT" "$INTERRUPTED_ROOT"

export HF_HOME=$ROOT/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
export DOC2QUERY_PROGRESS=1

"$PYTHON" scripts/plan_p05_probe_matrix.py \
  --campaign-audit reports/measurements/task03_campaign_audit_2026-07-21.json \
  --budget reports/measurements/task04_p05_dev_screen/budget.k1.json \
  --comparison-contract configs/evaluation/comparison_contract_v1.yaml \
  --probe-recipe configs/evaluation/probe_v1.yaml \
  --train-input artifacts/task04/p05/common_cohort/gold_natural.jsonl \
  --frozen-dev-manifest data/processed/v1/evaluation/task04-v1/manifest.json \
  --corpus data/processed/v1/documents.parquet \
  --primary-judge-config configs/reranker/primary_polish_roberta_v3_p03_gpu.yaml \
  --w05-adapter runs/W05-1.5B-50K-8GB/adapter \
  --w05-synthetic-generations artifacts/task04/p05/common_cohort/w05_synthetic.jsonl \
  --mixed-50-50-generations artifacts/task04/p05/common_cohort/mixed_50_50.jsonl \
  --p05-materialization-manifest artifacts/task04/p05/common_cohort/materialization.json \
  --output-root runs/task04_p05_dev_screen \
  --output reports/measurements/task04_p05_dev_screen/plan.ready.json \
  > "$LOG_DIR/planner.json"

"$PYTHON" -c 'import json
from pathlib import Path
p=json.loads(Path("reports/measurements/task04_p05_dev_screen/plan.ready.json").read_text())
if p.get("execution_ready") is not True or p.get("blockers") or p.get("final_tests_used") != []:
    raise SystemExit("P-05 planner is not ready or references final tests")
screens=[run for arm in p["arms"] for run in arm["runs"] if run["stage"] == "dev_screen"]
if len(screens) != 3 or any(run["seed"] != 42 or run["evaluation_sets"] != ["dev_intrinsic_rank10"] for run in screens):
    raise SystemExit("P-05 dev_screen contract drift")
print("P-05 preflight ready: exactly three seed-42 dev_intrinsic_rank10 runs")'

if [[ ${P05_PREFLIGHT_ONLY:-0} == 1 ]]; then
  printf '[preflight-only] validation complete; no probe was started.\n'
  exit 0
fi

run_probe() {
  local arm=$1
  shift
  local output=$RUN_ROOT/${arm}-S42
  local log=$LOG_DIR/${arm}-S42.log
  if [[ -f $output/result.json ]]; then
    "$PYTHON" -c 'import json, sys
from pathlib import Path
p=json.loads(Path(sys.argv[1]).read_text())
if p.get("training", {}).get("status") != "measured":
    raise SystemExit("completed probe result has no measured training stage")' "$output/result.json"
    printf '[skip] %s already complete: %s\n' "$arm" "$output/result.json"
    return 0
  fi
  if [[ -d $output ]] && find "$output" -mindepth 1 -print -quit | grep -q .; then
    if [[ -f $output/train_summary.json && -d $output/model ]] \
      && find "$output/model" -mindepth 1 -print -quit | grep -q .; then
      printf '[resume] %s training is complete; restarting only its evaluation stage.\n' "$arm"
    else
      local stamp archive
      stamp=$(date +%Y%m%dT%H%M%S)-$$
      archive=$INTERRUPTED_ROOT/${arm}-S42-$stamp
      mv "$output" "$archive"
      printf '[restart] %s had no complete training checkpoint.\n' "$arm"
      printf '[archive] partial output preserved at %s\n' "$archive"
    fi
  fi
  printf '[start] %s %s\n' "$arm" "$(date --iso-8601=seconds)"
  "$PYTHON" scripts/train_probe_embedder.py \
    --recipe configs/evaluation/probe_v1.yaml \
    --comparison-contract configs/evaluation/comparison_contract_v1.yaml \
    --train-input artifacts/task04/p05/common_cohort/gold_natural.jsonl \
    --frozen-manifest data/processed/v1/evaluation/task04-v1/manifest.json \
    --test-subset dev_intrinsic_rank10 \
    --corpus data/processed/v1/documents.parquet \
    --seed 42 \
    --max-steps 250 \
    --train-prefix-limit 2486 \
    --primary-judge-config configs/reranker/primary_polish_roberta_v3_p03_gpu.yaml \
    --output-dir "$output" \
    "$@" 2>&1 | tee -a "$log"
  printf '[done] %s %s\n' "$arm" "$(date --iso-8601=seconds)"
}

run_probe P05-GOLD-NATURAL --query-source natural
run_probe P05-W05-SYNTHETIC \
  --query-source synthetic \
  --synthetic-generations artifacts/task04/p05/common_cohort/w05_synthetic.jsonl \
  --generator-id W05-1.5B-50K-8GB
run_probe P05-MIXED50 \
  --query-source synthetic \
  --synthetic-generations artifacts/task04/p05/common_cohort/mixed_50_50.jsonl \
  --generator-id P05-W05-NATURAL-SYNTHETIC-50-50

printf '[complete] all three P-05 dev_screen probes finished; dev_confirm was not opened.\n'
