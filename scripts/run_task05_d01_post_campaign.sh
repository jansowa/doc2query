#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

PHASE=${1:-}
case "$PHASE" in
  preflight|prepare-common-cohort|generate-matched-baselines|score|compare|materialize-probe-inputs) ;;
  *)
    echo "usage: $0 {preflight|prepare-common-cohort|generate-matched-baselines|score|compare|materialize-probe-inputs}" >&2
    exit 2
    ;;
esac

PYTHON=${DOC2QUERY_PYTHON:-$ROOT/.venv-gpu/bin/python}
CAMPAIGN="$ROOT/configs/evaluation/d01_campaign_v2.yaml"
LOG_DIR="$ROOT/logs/task05_d01_post_campaign"
STATUS_DIR="$ROOT/reports/measurements/task05_d01_postprocess_v2"
STATUS_TSV="$STATUS_DIR/status.tsv"
STATUS_JSON="$STATUS_DIR/status.json"
mkdir -p "$LOG_DIR" "$STATUS_DIR"
exec 9>"$STATUS_DIR/campaign.lock"
if ! flock -n 9; then
  echo "another post-D01 campaign phase owns $STATUS_DIR/campaign.lock" >&2
  exit 3
fi
if [[ ! -x $PYTHON ]]; then
  echo "missing project environment: $PYTHON" >&2
  exit 2
fi
if [[ ! -f $STATUS_TSV ]]; then
  printf 'started_at\tfinished_at\tphase\texit_code\tlog\n' >"$STATUS_TSV"
fi
LOG="$LOG_DIR/${PHASE}.log"
STARTED=$(date --iso-8601=seconds)

write_status_json() {
  local finished=$1 rc=$2
  "$PYTHON" - "$STATUS_JSON" "$PHASE" "$STARTED" "$finished" "$rc" "$LOG" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {"schema_version": 1, "phases": {}}
if path.is_file():
    payload = json.loads(path.read_text(encoding="utf-8"))
payload.setdefault("phases", {})[sys.argv[2]] = {
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
  printf '%s\t%s\t%s\t%s\t%s\n' "$STARTED" "$finished" "$PHASE" "$rc" "$LOG" >>"$STATUS_TSV"
  write_status_json "$finished" "$rc"
  return "$rc"
}

assert_gpu_idle() {
  local active
  active=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null || true)
  if [[ -n $active ]]; then
    echo "GPU has active compute processes; refusing phase $PHASE:" >&2
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

if pgrep -f '[r]un_task05_groq_audits.py' >/dev/null; then
  printf '[%s] Groq audit detected; its journals/results remain outside this campaign.\n' \
    "$(date --iso-8601=seconds)" | tee -a "$LOG"
fi

case "$PHASE" in
  preflight)
    run_phase bash -c '"$1" scripts/run_d01_postprocess.py audit --campaign-config "$2" && "$1" scripts/run_d01_postprocess.py preflight --campaign-config "$2"' _ "$PYTHON" "$CAMPAIGN"
    ;;
  prepare-common-cohort)
    run_phase "$PYTHON" scripts/run_d01_postprocess.py prepare-common-cohort \
      --campaign-config "$CAMPAIGN"
    ;;
  generate-matched-baselines)
    assert_gpu_idle
    run_phase bash -c '
      set -euo pipefail
      python=$1
      "$python" scripts/run_d01_postprocess.py preflight --campaign-config configs/evaluation/d01_campaign_v2.yaml
      "$python" scripts/run_d01_postprocess.py generation-only \
        --config configs/experiments/d01_w05_matched_dev_generation_s42.yaml \
        --frozen-manifest data/processed/v1/evaluation/task04-v1/manifest.json \
        --subset dev_intrinsic_rank10 \
        --adapter runs/W05-1.5B-50K-8GB/adapter \
        --output runs/D01-W05-MATCHED-DEV-GENERATION-S42/generation/uncontrolled.full.jsonl \
        --archive-incompatible
      "$python" scripts/run_d01_postprocess.py generation-only \
        --config configs/experiments/d01_w06_matched_dev_generation_s42.yaml \
        --frozen-manifest data/processed/v1/evaluation/task04-v1/manifest.json \
        --subset dev_intrinsic_rank10 \
        --adapter runs/W06-4.5B-INSTRUCT-50K-8GB-BS8-L512/adapter \
        --output runs/D01-W06-MATCHED-DEV-GENERATION-S42/generation/uncontrolled.full.jsonl \
        --archive-incompatible
    ' _ "$PYTHON"
    ;;
  score)
    assert_gpu_idle
    run_phase bash -c '
      set -euo pipefail
      python=$1
      "$python" scripts/run_d01_postprocess.py preflight --campaign-config configs/evaluation/d01_campaign_v2.yaml
      primary=configs/reranker/primary_polish_roberta_v3_cuda.yaml
      shadow=configs/reranker/shadow_bge_v2_m3.yaml
      corpus=artifacts/task04/p03/bm25_train_v1
      score_one() {
        local generations=$1 summary=$2 output=$3
        "$python" scripts/run_d01_postprocess.py score \
          --generations "$generations" --generation-summary "$summary" \
          --output-dir "$output" --primary-judge "$primary" --shadow-judge "$shadow" \
          --primary-judge-device cuda --shadow-judge-device cuda --corpus-index "$corpus" \
          --archive-incompatible
      }
      base=artifacts/task05/d01_postprocess_v2/common_exact_k_v1/recovered
      score_one "$base/D01-1.5B-STYLE-50K-S42/generations.exact_k4.jsonl" "$base/D01-1.5B-STYLE-50K-S42/generations.exact_k4.jsonl.summary.json" reports/measurements/task05_d01_postprocess_v2/scoring/D01-1.5B-STYLE-50K-S42
      score_one "$base/W05-1.5B-50K-8GB/generations.exact_k4.jsonl" "$base/W05-1.5B-50K-8GB/generations.exact_k4.jsonl.summary.json" reports/measurements/task05_d01_postprocess_v2/scoring/W05-1.5B-50K-8GB
      score_one "$base/D01-4.5B-STYLE-50K-S42/generations.exact_k4.jsonl" "$base/D01-4.5B-STYLE-50K-S42/generations.exact_k4.jsonl.summary.json" reports/measurements/task05_d01_postprocess_v2/scoring/D01-4.5B-STYLE-50K-S42
      score_one "$base/W06-4.5B-INSTRUCT-50K-8GB-BS8-L512/generations.exact_k4.jsonl" "$base/W06-4.5B-INSTRUCT-50K-8GB-BS8-L512/generations.exact_k4.jsonl.summary.json" reports/measurements/task05_d01_postprocess_v2/scoring/W06-4.5B-INSTRUCT-50K-8GB-BS8-L512
    ' _ "$PYTHON"
    ;;
  compare)
    run_phase bash -c '
      set -euo pipefail
      python=$1; root=reports/measurements/task05_d01_postprocess_v2
      "$python" scripts/run_d01_postprocess.py compare --baseline-summary "$root/scoring/W05-1.5B-50K-8GB/summary.json" --baseline-rows "$root/scoring/W05-1.5B-50K-8GB/per_generation.jsonl" --variant-summary "$root/scoring/D01-1.5B-STYLE-50K-S42/summary.json" --variant-rows "$root/scoring/D01-1.5B-STYLE-50K-S42/per_generation.jsonl" --comparison-contract configs/evaluation/comparison_contract_v1.yaml --output-json "$root/comparisons/d01_1_5b_vs_w05.json" --output-markdown "$root/comparisons/d01_1_5b_vs_w05.md"
      "$python" scripts/run_d01_postprocess.py compare --baseline-summary "$root/scoring/W06-4.5B-INSTRUCT-50K-8GB-BS8-L512/summary.json" --baseline-rows "$root/scoring/W06-4.5B-INSTRUCT-50K-8GB-BS8-L512/per_generation.jsonl" --variant-summary "$root/scoring/D01-4.5B-STYLE-50K-S42/summary.json" --variant-rows "$root/scoring/D01-4.5B-STYLE-50K-S42/per_generation.jsonl" --comparison-contract configs/evaluation/comparison_contract_v1.yaml --output-json "$root/comparisons/d01_4_5b_vs_w06_bs8.json" --output-markdown "$root/comparisons/d01_4_5b_vs_w06_bs8.md"
    ' _ "$PYTHON"
    ;;
  materialize-probe-inputs)
    run_phase bash -c '
      set -euo pipefail
      python=$1; root=reports/measurements/task05_d01_postprocess_v2
      base=artifacts/task05/d01_postprocess_v2/common_exact_k_v1/recovered
      "$python" scripts/run_d01_postprocess.py materialize-probe-inputs --generations "$base/D01-1.5B-STYLE-50K-S42/generations.exact_k4.jsonl" --generation-summary "$base/D01-1.5B-STYLE-50K-S42/generations.exact_k4.jsonl.summary.json" --scoring-summary "$root/scoring/D01-1.5B-STYLE-50K-S42/summary.json" --scoring-rows "$root/scoring/D01-1.5B-STYLE-50K-S42/per_generation.jsonl" --comparison-report "$root/comparisons/d01_1_5b_vs_w05.json" --output artifacts/task05/d01_postprocess_v2/probe_inputs/d01_1_5b.jsonl
      "$python" scripts/run_d01_postprocess.py materialize-probe-inputs --generations "$base/D01-4.5B-STYLE-50K-S42/generations.exact_k4.jsonl" --generation-summary "$base/D01-4.5B-STYLE-50K-S42/generations.exact_k4.jsonl.summary.json" --scoring-summary "$root/scoring/D01-4.5B-STYLE-50K-S42/summary.json" --scoring-rows "$root/scoring/D01-4.5B-STYLE-50K-S42/per_generation.jsonl" --comparison-report "$root/comparisons/d01_4_5b_vs_w06_bs8.json" --output artifacts/task05/d01_postprocess_v2/probe_inputs/d01_4_5b.jsonl
    ' _ "$PYTHON"
    ;;
esac
