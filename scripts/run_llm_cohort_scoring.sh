#!/usr/bin/env bash
# Oceń wznawialnie kohortę tokenową (korpus nagrody albo teacher) na GPU.
#
# Usage: run_llm_cohort_scoring.sh <cohort-dir> <experiment-id>
#
# Każdy etap jest wznawialny i idempotentny: skrypt można w dowolnym momencie
# przerwać i uruchomić ponownie z tymi samymi argumentami. Scoring wznawia się
# z fsyncowanego dziennika (`scoring/scoring.journal.jsonl`), więc maksymalna
# strata to jeden batch ośmiu rekordów. Katalog jest chroniony `flock`.
#
# ADR: reports/decisions/task06_llm_cohort_gpu_scoring_amendment_2026-08-16.md
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COHORT="${1:?usage: $0 <cohort-dir> <experiment-id>}"
EXPERIMENT_ID="${2:?usage: $0 <cohort-dir> <experiment-id>}"
PYTHON="$ROOT/.venv-gpu/bin/python"
GENERATIONS="$COHORT/scoring_inputs/generations.jsonl"
OUTPUT="$COHORT/scoring"

test -x "$PYTHON" || { echo "brak środowiska GPU: $PYTHON" >&2; exit 2; }
test -f "$GENERATIONS" || { echo "brak wejścia scoringu: $GENERATIONS" >&2; exit 2; }

mkdir -p "$COHORT/logs" "$OUTPUT"
exec 9>"$COHORT/scoring.lock"
flock -n 9 || { echo "inny proces ocenia $COHORT" >&2; exit 3; }

if [ -f "$OUTPUT/summary.json" ] && [ -f "$OUTPUT/per_generation.jsonl" ]; then
  echo "scoring już ukończony dla $COHORT; nie ruszam zamrożonego artefaktu"
  exit 0
fi

export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_ASSETS_CACHE="$HF_HOME/assets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "[scoring] $EXPERIMENT_ID ($(wc -l < "$GENERATIONS") rekordów, dziennik wznawialny)"
cd "$ROOT"
"$PYTHON" "$ROOT/scripts/score_llm_cohort.py" \
  --generations "$GENERATIONS" \
  --output-dir "$OUTPUT" \
  --experiment-id "$EXPERIMENT_ID" \
  2>&1 | tee -a "$COHORT/logs/scoring.log"
echo "done: $OUTPUT"
