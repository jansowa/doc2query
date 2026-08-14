#!/usr/bin/env bash
# Freeze (if needed), generate, score and gate one same-prompt Task 06 cohort.
#
# Usage: run_task06_same_prompt_cohort.sh <cohort-config> <output-dir>
#
# Every stage is resumable and idempotent, so the script may be interrupted at any
# time and restarted with the same arguments: generation and scoring resume from
# their fsynced journals and a finished diversity gate is left untouched.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:?usage: $0 <cohort-config> <output-dir>}"
OUTPUT="${2:?usage: $0 <cohort-config> <output-dir>}"
DESIGN="$ROOT/configs/preferences/task06_candidate_execution_design_v1.yaml"
GATE_POLICY="$ROOT/configs/preferences/task06_same_prompt_diversity_gate_v1.yaml"
PYTHON="$ROOT/.venv-gpu/bin/python"
RUNNER="$ROOT/scripts/run_task06_smoke.py"
GENERATIONS="$OUTPUT/d01_controlled/generations.jsonl"

test -f "$CONFIG" || { echo "missing cohort config: $CONFIG" >&2; exit 2; }
test -x "$PYTHON" || { echo "missing GPU environment: $PYTHON" >&2; exit 2; }
EXPERIMENT_ID="$("$PYTHON" - "$CONFIG" <<'PY'
import sys
import yaml

config = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
print(config["generator"]["experiment_id"])
PY
)"
test -n "$EXPERIMENT_ID" || { echo "cohort config has no generator.experiment_id" >&2; exit 2; }

mkdir -p "$OUTPUT/logs"
exec 9>"$OUTPUT/run.lock"
flock -n 9 || { echo "another runner owns $OUTPUT" >&2; exit 3; }
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_ASSETS_CACHE="$HF_HOME/assets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

if [ ! -f "$OUTPUT/cohort.manifest.json" ]; then
  echo "[stage 0/3] quality-blind cohort freeze (CPU)"
  "$PYTHON" "$ROOT/scripts/freeze_task06_same_prompt_expansion_v2.py" \
    --config "$CONFIG" --output-dir "$OUTPUT" 2>&1 | tee -a "$OUTPUT/logs/freeze.log"
fi

echo "[stage 1/3] generation for $EXPERIMENT_ID (resumable journal)"
"$PYTHON" "$RUNNER" expand --design "$CONFIG" --output-root "$OUTPUT" \
  2>&1 | tee -a "$OUTPUT/logs/generate.log"

echo "[stage 2/3] primary/shadow/corpus scoring (resumable journal)"
"$PYTHON" "$RUNNER" score --design "$DESIGN" --output-root "$OUTPUT" \
  --role d01_controlled --device cuda --stage-name pilot \
  --experiment-id "$EXPERIMENT_ID" \
  2>&1 | tee -a "$OUTPUT/logs/score.log"

if [ -f "$OUTPUT/diversity_gate/manifest.json" ]; then
  echo "[stage 3/3] diversity gate already applied; leaving the frozen artifact untouched"
else
  echo "[stage 3/3] diversity gate (CPU, frozen thresholds)"
  "$PYTHON" "$ROOT/scripts/apply_task06_same_prompt_diversity_gate.py" \
    --generations "$GENERATIONS" --policy "$GATE_POLICY" \
    --output-dir "$OUTPUT/diversity_gate" 2>&1 | tee -a "$OUTPUT/logs/diversity_gate.log"
fi
echo "done: $OUTPUT"
