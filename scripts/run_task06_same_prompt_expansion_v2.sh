#!/usr/bin/env bash
# Generate and score the frozen same-prompt expansion v2 cohort, then apply the
# frozen diversity gate.  The cohort must already be frozen on CPU by
# scripts/freeze_task06_same_prompt_expansion_v2.py.
#
# Every stage is resumable and idempotent: interrupt with Ctrl-C at any time and
# rerun this exact command.  Generation and scoring resume from their fsynced
# journals (at most one batch of work is repeated) and a finished stage is
# skipped instead of recomputed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$ROOT/configs/preferences/task06_same_prompt_expansion_v2.yaml"
DESIGN="$ROOT/configs/preferences/task06_candidate_execution_design_v1.yaml"
GATE_POLICY="$ROOT/configs/preferences/task06_same_prompt_diversity_gate_v1.yaml"
OUTPUT="$ROOT/artifacts/task06/same_prompt_expansion_v2"
PYTHON="$ROOT/.venv-gpu/bin/python"
RUNNER="$ROOT/scripts/run_task06_smoke.py"
GENERATIONS="$OUTPUT/d01_controlled/generations.jsonl"

test -f "$OUTPUT/cohort.manifest.json" || {
  echo "freeze the v2 cohort first: scripts/freeze_task06_same_prompt_expansion_v2.py" >&2
  exit 1
}
mkdir -p "$OUTPUT/logs"
exec 9>"$OUTPUT/run.lock"
flock -n 9 || { echo "another same-prompt v2 runner is active" >&2; exit 1; }
nvidia-smi --query-gpu=index,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader,nounits
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_ASSETS_CACHE="$HF_HOME/assets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "[stage 1/3] generation (resumable journal)"
"$PYTHON" "$RUNNER" expand --design "$CONFIG" --output-root "$OUTPUT" \
  2>&1 | tee -a "$OUTPUT/logs/generate.log"

echo "[stage 2/3] primary/shadow/corpus scoring (resumable journal)"
"$PYTHON" "$RUNNER" score --design "$DESIGN" --output-root "$OUTPUT" \
  --role d01_controlled --device cuda --stage-name pilot \
  --experiment-id TASK06-PREFERENCE-D01-SAME-PROMPT-V2 \
  2>&1 | tee -a "$OUTPUT/logs/score.log"

if [ -f "$OUTPUT/diversity_gate/manifest.json" ]; then
  echo "[stage 3/3] diversity gate already applied; leaving the frozen artifact untouched"
else
  echo "[stage 3/3] diversity gate (CPU, frozen thresholds)"
  "$PYTHON" "$ROOT/scripts/apply_task06_same_prompt_diversity_gate.py" \
    --generations "$GENERATIONS" \
    --policy "$GATE_POLICY" \
    --output-dir "$OUTPUT/diversity_gate" \
    2>&1 | tee -a "$OUTPUT/logs/diversity_gate.log"
fi
echo "done: $OUTPUT"
