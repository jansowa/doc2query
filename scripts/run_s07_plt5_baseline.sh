#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
PYTHON=${DOC2QUERY_PYTHON:-$ROOT/.venv-gpu/bin/python}
TOOL_PYTHON=${DOC2QUERY_TOOL_PYTHON:-$ROOT/.venv/bin/python}
CONTRACT=configs/evaluation/s07_contract_v1.yaml
CONFIG=configs/experiments/s07_plt5_base_50k.yaml
SMOKE_CONFIG=configs/experiments/s07_tiny_smoke.yaml
REPORT_ROOT=reports/measurements/task03_s07
LOG_ROOT=logs/task03_s07
PROBE_OUTPUT=runs/task03_s07/dev_screen/S07-PLT5-BASE-50K-S42
PROBE_GENERATIONS=artifacts/task03/s07/p05/s07_synthetic.jsonl
TRAIN_TIMEOUT=${S07_TRAIN_TIMEOUT:-8h}
START_AT=${S07_START_AT:-auto}
STOP_AFTER=${S07_STOP_AFTER:-probe_train}
if [[ ${S07_STOP_AFTER_MEMORY_PROBE:-0} == 1 ]]; then
  STOP_AFTER=memory_probe
fi
declare -A STAGE_ORDER=(
  [preflight]=0 [smoke]=1 [memory_probe]=2 [train]=3
  [harness]=4 [probe_generation]=5 [probe_train]=6
)
if [[ $START_AT == auto ]]; then
  if [[ -f $PROBE_OUTPUT/result.json ]]; then
    START_AT=probe_train
  elif [[ -f $PROBE_GENERATIONS ]] \
    && [[ $(wc -l < "$PROBE_GENERATIONS") -eq 2486 ]] \
    && [[ -f reports/evaluation/S07-PLT5-BASE-50K-S42-dev/result.json ]]; then
    START_AT=probe_train
  elif [[ -f reports/evaluation/S07-PLT5-BASE-50K-S42-dev/result.json ]]; then
    START_AT=probe_generation
  elif [[ -f runs/S07-PLT5-BASE-50K-S42/sft_summary.json ]] \
    && [[ -d runs/S07-PLT5-BASE-50K-S42/model ]]; then
    START_AT=harness
  else
    START_AT=preflight
  fi
  printf '[S07 resume] Auto-selected START_AT=%s from completed artifacts.\n' "$START_AT"
fi
if [[ ! -v STAGE_ORDER[$START_AT] ]]; then
  printf 'Invalid S07_START_AT=%s; expected one of: %s\n' \
    "$START_AT" "${!STAGE_ORDER[*]}" >&2
  exit 2
fi
if [[ ! -v STAGE_ORDER[$STOP_AFTER] ]]; then
  printf 'Invalid S07_STOP_AFTER=%s; expected one of: %s\n' \
    "$STOP_AFTER" "${!STAGE_ORDER[*]}" >&2
  exit 2
fi
if (( STAGE_ORDER[$START_AT] > STAGE_ORDER[$STOP_AFTER] )); then
  printf 'S07_START_AT=%s occurs after S07_STOP_AFTER=%s\n' "$START_AT" "$STOP_AFTER" >&2
  exit 2
fi
stage_enabled() {
  (( STAGE_ORDER[$1] >= STAGE_ORDER[$START_AT] \
     && STAGE_ORDER[$1] <= STAGE_ORDER[$STOP_AFTER] ))
}
mkdir -p "$REPORT_ROOT" "$LOG_ROOT" artifacts/task03/s07/p05
export HF_HOME=$ROOT/.cache/huggingface
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
export DOC2QUERY_PROGRESS=${DOC2QUERY_PROGRESS:-1}

scripts/bootstrap_gpu_env.sh

if stage_enabled preflight; then
  printf '[S07 stage] preflight started.\n'
  "$PYTHON" scripts/preflight_s07.py --contract "$CONTRACT" --output "$REPORT_ROOT/preflight.json" 2>&1 | tee "$LOG_ROOT/preflight.log"
  "$TOOL_PYTHON" -m ruff check src scripts tests 2>&1 | tee "$LOG_ROOT/ruff.log"
  "$TOOL_PYTHON" -m mypy src scripts 2>&1 | tee "$LOG_ROOT/mypy.log"
  "$TOOL_PYTHON" -m pytest -q 2>&1 | tee "$LOG_ROOT/pytest.log"
fi

if stage_enabled smoke; then
  printf '[S07 stage] smoke started.\n'
  "$PYTHON" scripts/build_tiny_seq2seq_fixture.py --output artifacts/task03/s07/tiny-random-t5
  "$PYTHON" scripts/train_sft.py --config "$SMOKE_CONFIG" --resume-if-available 2>&1 | tee "$LOG_ROOT/smoke.log"
  if [[ ! -f $REPORT_ROOT/smoke_generations.jsonl ]]; then
    "$PYTHON" scripts/generate_panel.py --config "$SMOKE_CONFIG" \
      --model-checkpoint runs/S07-TINY-SEQ2SEQ-SMOKE/model \
      --input data/processed/v1/doc2query_dev.parquet \
      --output "$REPORT_ROOT/smoke_generations.jsonl" 2>&1 | tee "$LOG_ROOT/smoke_generation.log"
  fi
  "$PYTHON" -c 'import json, math
from pathlib import Path
s=json.loads(Path("runs/S07-TINY-SEQ2SEQ-SMOKE/sft_summary.json").read_text())
if s["global_step"] != 20 or not Path(s["model_path"]).is_dir():
    raise SystemExit("S07 smoke training gate failed")
if not all(math.isfinite(float(v)) for v in s["loss"].values() if v is not None):
    raise SystemExit("S07 smoke produced non-finite loss")
if s["panel"]["generations"] < 1:
    raise SystemExit("S07 smoke generation gate failed")'
fi

if (( STAGE_ORDER[$STOP_AFTER] < STAGE_ORDER[memory_probe] )); then
  printf '[S07] Requested stage range %s..%s complete.\n' "$START_AT" "$STOP_AFTER"
  exit 0
fi

if ! "$PYTHON" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
  "$PYTHON" -c 'import json; from pathlib import Path
p=Path("reports/measurements/task03_s07/gate_status.json")
p.write_text(json.dumps({"status":"blocked_before_memory_probe","reason":"CUDA unavailable","full_train_started":False,"final_tests_used":[],"dev_confirm_opened":False,"p06_opened":False},indent=2)+"\n")'
  printf '[S07] Cheap gates passed; CUDA is unavailable, so memory probe and full training were not started.\n'
  exit 0
fi

if [[ ${S07_ALLOW_MODEL_DOWNLOAD:-0} != 1 ]]; then
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
fi
if stage_enabled memory_probe; then
  printf '[S07 stage] memory_probe started.\n'
  "$PYTHON" scripts/run_memory_probe.py --config "$CONFIG" --lengths 512 --steps 2 \
    --output-dir "$REPORT_ROOT/memory_probes" 2>&1 | tee -a "$LOG_ROOT/memory_probe.log"
  "$PYTHON" -c 'import json
from pathlib import Path
p=json.loads(Path("reports/measurements/task03_s07/memory_probes/S07-PLT5-BASE-50K-S42/memory_probe.json").read_text())
probe=p["probes"][0]
if probe["status"] not in {"ok","already_complete"} or probe.get("peak_vram_reserved_bytes",0)<=0:
    raise SystemExit("S07 memory gate failed")
import torch
fraction=probe["peak_vram_reserved_bytes"]/torch.cuda.get_device_properties(0).total_memory
if fraction > 0.90:
    raise SystemExit(f"S07 memory gate failed: reserved fraction {fraction:.4f} > 0.90")'
fi

if [[ $STOP_AFTER == memory_probe ]]; then
  printf '[S07] Memory gate passed; requested stage range complete.\n'
  exit 0
fi

if stage_enabled train; then
  printf '[S07 stage] train started/resumed.\n'
  timeout --foreground "$TRAIN_TIMEOUT" "$PYTHON" scripts/train_sft.py --config "$CONFIG" \
    --resume-if-available 2>&1 | tee -a "$LOG_ROOT/train.log"
fi
if stage_enabled harness; then
  printf '[S07 stage] harness started/resumed.\n'
  "$PYTHON" scripts/evaluate_generator.py --config "$CONFIG" \
  --model-checkpoint runs/S07-PLT5-BASE-50K-S42/model \
  --frozen-manifest data/processed/v1/evaluation/task04-v1/manifest.json \
  --subset dev_intrinsic_rank10 --max-examples 5000 \
  --output-dir reports/evaluation/S07-PLT5-BASE-50K-S42-dev \
  --primary-judge configs/reranker/primary_polish_roberta_v3_s07_gpu.yaml \
  --shadow-judge configs/reranker/shadow_bge_v2_m3.yaml \
  --corpus-index data/processed/v1/evaluation/corpus-bm25-v1 \
  --generation-batch-size "${S07_GENERATION_BATCH_SIZE:-16}" \
  --scoring-batch-size "${S07_SCORING_BATCH_SIZE:-16}" \
  --bm25-workers "${S07_BM25_WORKERS:-2}" \
  --primary-judge-device "${S07_PRIMARY_JUDGE_DEVICE:-cuda}" \
  --shadow-judge-device "${S07_SHADOW_JUDGE_DEVICE:-cuda}" \
  --archive-incompatible-scoring \
  2>&1 | tee -a "$LOG_ROOT/harness.log"
fi
if stage_enabled probe_generation; then
  printf '[S07 stage] probe_generation started/resumed.\n'
  "$PYTHON" scripts/generate_s07_probe_queries.py --config "$CONFIG" \
  --model-checkpoint runs/S07-PLT5-BASE-50K-S42/model \
  --input artifacts/task04/p05/common_cohort/gold_natural.jsonl \
  --output artifacts/task03/s07/p05/s07_synthetic.jsonl \
  --generator-id S07-PLT5-BASE-50K-S42 --limit 2486 \
  --batch-size "${S07_PROBE_GENERATION_BATCH_SIZE:-32}" \
  2>&1 | tee -a "$LOG_ROOT/probe_generation.log"
fi
if stage_enabled probe_train; then
if [[ ! -f $PROBE_OUTPUT/result.json && -d $PROBE_OUTPUT ]] \
  && find "$PROBE_OUTPUT" -mindepth 1 -print -quit | grep -q . \
  && [[ ! -f $PROBE_OUTPUT/train_summary.json || ! -d $PROBE_OUTPUT/model ]] \
  && [[ ! -f $PROBE_OUTPUT/training_checkpoint.pt ]]; then
  stamp=$(date +%Y%m%dT%H%M%S)-$$
  archive=runs/task03_s07/interrupted/S07-PLT5-BASE-50K-S42-$stamp
  mkdir -p "$(dirname "$archive")"
  mv "$PROBE_OUTPUT" "$archive"
  printf '[S07 probe] Incomplete non-resumable training archived at %s; restarting probe stage.\n' "$archive"
fi
if [[ ! -f $PROBE_OUTPUT/result.json ]]; then
  printf '[S07 probe] Starting/resuming probe: batch_size=%s, checkpoint_interval=%s steps.\n' \
    "${S07_PROBE_BATCH_SIZE:-6}" "${S07_PROBE_CHECKPOINT_STEPS:-25}"
  "$PYTHON" scripts/train_probe_embedder.py \
  --recipe configs/evaluation/probe_v1.yaml \
  --comparison-contract configs/evaluation/comparison_contract_v1.yaml \
  --train-input artifacts/task04/p05/common_cohort/gold_natural.jsonl \
  --frozen-manifest data/processed/v1/evaluation/task04-v1/manifest.json \
  --test-subset dev_intrinsic_rank10 --corpus data/processed/v1/documents.parquet \
  --query-source synthetic --synthetic-generations artifacts/task03/s07/p05/s07_synthetic.jsonl \
  --generator-id S07-PLT5-BASE-50K-S42 --seed 42 --max-steps 250 --train-prefix-limit 2486 \
    --batch-size "${S07_PROBE_BATCH_SIZE:-6}" \
    --checkpoint-interval-steps "${S07_PROBE_CHECKPOINT_STEPS:-25}" \
    --evaluation-encode-batch-size "${S07_PROBE_ENCODE_BATCH_SIZE:-64}" \
    --retrieval-query-batch-size "${S07_RETRIEVAL_QUERY_BATCH_SIZE:-512}" \
    --retrieval-device "${S07_RETRIEVAL_DEVICE:-cuda}" \
    --primary-judge-config configs/reranker/primary_polish_roberta_v3_s07_gpu.yaml \
    --output-dir "$PROBE_OUTPUT" 2>&1 | tee -a "$LOG_ROOT/probe.log"
fi
fi
printf '[S07] Requested stage range %s..%s complete. dev_confirm, final tests and P-06 remain closed.\n' \
  "$START_AT" "$STOP_AFTER"
