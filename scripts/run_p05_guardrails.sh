#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

PYTHON_BIN=${DOC2QUERY_PYTHON:-$ROOT/.venv-gpu/bin/python}
if [[ ! -x "$PYTHON_BIN" ]]; then
  printf 'Missing GPU Python: %s\nSet DOC2QUERY_PYTHON to the proven CUDA environment.\n' "$PYTHON_BIN" >&2
  exit 2
fi

export HF_HOME=${HF_HOME:-$ROOT/.cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
export TOKENIZERS_PARALLELISM=false

LOG_DIR=$ROOT/logs/task04_p05_guardrails
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/run.log") 2>&1

"$PYTHON_BIN" scripts/measure_p05_dev_guardrails.py --phase shared

for arm in P05-GOLD-NATURAL-S42 P05-MIXED50-S42 P05-W05-SYNTHETIC-S42; do
  "$PYTHON_BIN" scripts/measure_p05_dev_guardrails.py \
    --phase roundtrip \
    --arm-dir "runs/task04_p05_dev_screen/dev_screen/$arm"
done

"$PYTHON_BIN" scripts/measure_p05_dev_guardrails.py --phase merge
"$PYTHON_BIN" scripts/build_p05_p04_reports.py
