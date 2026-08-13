#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$ROOT/configs/preferences/task06_same_prompt_expansion_v1.yaml"
DESIGN="$ROOT/configs/preferences/task06_candidate_execution_design_v1.yaml"
OUTPUT="$ROOT/artifacts/task06/same_prompt_expansion_v1"
PYTHON="$ROOT/.venv-gpu/bin/python"
RUNNER="$ROOT/scripts/run_task06_smoke.py"

mkdir -p "$OUTPUT/logs"
exec 9>"$OUTPUT/run.lock"
flock -n 9 || { echo "another same-prompt runner is active" >&2; exit 1; }
nvidia-smi --query-gpu=index,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader,nounits
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_ASSETS_CACHE="$HF_HOME/assets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

"$PYTHON" "$RUNNER" expand --design "$CONFIG" --output-root "$OUTPUT" \
  2>&1 | tee "$OUTPUT/logs/generate.log"
"$PYTHON" "$RUNNER" score --design "$DESIGN" --output-root "$OUTPUT" \
  --role d01_controlled --device cuda --stage-name pilot \
  --experiment-id TASK06-PREFERENCE-D01-SAME-PROMPT \
  2>&1 | tee "$OUTPUT/logs/score.log"
