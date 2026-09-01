#!/usr/bin/env bash
# Kolejka probe embedderów Task 07: materializacja wejść + trening per ramię.
#
#   scripts/queue_task07_probe_embedders.sh
#
# Protokół screen z Task 05 (500 kroków, bs 4, seed 42, dev_intrinsic_rank10);
# preregistracja: reports/preregistrations/task07_probe_screen_v1.md.
# Wznawialna: gotowe ramiona (result.json) są pomijane; restart dokańcza.
set -uo pipefail

cd "$(dirname "$0")/.."
[ -f pyproject.toml ] || { echo "nie jestem w katalogu repozytorium" >&2; exit 2; }

GPU_PY=".venv-gpu/bin/python"
INPUTS="artifacts/task07/probe_inputs_v1"
RUN_ROOT="runs/task07_probe"
# Kolejność wg wartości informacyjnej: baza, ramiona DPO/antykolapsowe, potem SFT.
ARMS="start defect_dpo rpo beta02 divch bottom_dpo nearmiss_dpo defect_csft defect_wsft nearmiss_csft nearmiss_wsft bottom_csft bottom_wsft"

for path in "$GPU_PY" configs/evaluation/probe_v1.yaml \
  configs/reranker/primary_polish_roberta_v3_p03_gpu.yaml \
  data/processed/v1/evaluation/task04-v1/manifest.json \
  data/processed/v1/documents.parquet; do
  [ -e "$path" ] || { echo "brak wymaganego wejścia: $path" >&2; exit 2; }
done

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# Nie startuj obok cudzej roboty na GPU (trening/generacja/ewaluacja).
while pgrep -f "doc2query.cli (train|generate|evaluate)" >/dev/null 2>&1; do
  echo "[probe] GPU zajęte, czekam 120 s ($(date '+%H:%M:%S'))"
  sleep 120
done

if [ ! -f "$INPUTS/manifest.json" ]; then
  echo "[probe] materializacja wejść ($(date '+%H:%M:%S'))"
  rm -rf "${INPUTS}.czesciowe" 2>/dev/null || true
  [ -d "$INPUTS" ] && mv "$INPUTS" "${INPUTS}.czesciowe"
  PYTHONPATH=src "$GPU_PY" scripts/build_task07_probe_inputs.py --output-dir "$INPUTS" \
    || { echo "[probe] materializacja padła" >&2; exit 1; }
fi

PAIRS=$("$GPU_PY" -c "import json;print(json.load(open('$INPUTS/manifest.json'))['pairs_per_arm'])")
echo "[probe] budżet na ramię: $PAIRS par"

for arm in $ARMS; do
  out="$RUN_ROOT/$arm"
  input="$INPUTS/$arm.jsonl"
  [ -f "$input" ] || { echo "[probe] brak wejścia $input, pomijam $arm" >&2; continue; }
  if [ -f "$out/result.json" ]; then
    echo "[probe] $arm gotowe, pomijam"
    continue
  fi
  attempt=1
  while [ "$attempt" -le 3 ]; do
    echo "[probe] $arm start, próba $attempt ($(date '+%H:%M:%S'))"
    if PYTHONPATH=src "$GPU_PY" scripts/train_probe_embedder.py \
      --recipe configs/evaluation/probe_v1.yaml \
      --comparison-contract configs/evaluation/comparison_contract_v1.yaml \
      --train-input "$input" \
      --synthetic-generations "$input" \
      --generator-id "T07-PROBE-$(echo "$arm" | tr '[:lower:]' '[:upper:]')-S42" \
      --query-source synthetic \
      --frozen-manifest data/processed/v1/evaluation/task04-v1/manifest.json \
      --test-subset dev_intrinsic_rank10 \
      --corpus data/processed/v1/documents.parquet \
      --primary-judge-config configs/reranker/primary_polish_roberta_v3_p03_gpu.yaml \
      --seed 42 \
      --max-steps 500 \
      --batch-size 4 \
      --train-prefix-limit "$PAIRS" \
      --checkpoint-interval-steps 50 \
      --evaluation-encode-batch-size 32 \
      --retrieval-query-batch-size 512 \
      --retrieval-device cuda \
      --output-dir "$out"; then
      echo "[probe] $arm koniec ($(date '+%H:%M:%S'))"
      break
    fi
    echo "[probe] $arm padło, wznowię za 120 s" >&2
    sleep 120
    attempt=$((attempt + 1))
  done
done

echo "[probe] kolejka zakończona ($(date '+%H:%M:%S'))"
