#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

PYTHON=${DOC2QUERY_PYTHON:-$ROOT/.venv-gpu/bin/python}
if [[ ! -x $PYTHON ]]; then
  echo "HN gate requires the project GPU environment: $PYTHON" >&2
  echo "Create or repair it with: bash scripts/bootstrap_gpu_env.sh" >&2
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

"$PYTHON" - <<'PY'
import sys

import torch

if torch.version.cuda != "12.4":
    raise SystemExit(
        f"HN gate requires the pinned CUDA 12.4 torch build; got "
        f"torch={torch.__version__}, torch.version.cuda={torch.version.cuda}, "
        f"python={sys.executable}"
    )
if not torch.cuda.is_available():
    raise SystemExit(
        f"HN gate cannot see CUDA from {sys.executable}; "
        "run this wrapper outside the Codex sandbox and verify nvidia-smi"
    )
properties = torch.cuda.get_device_properties(0)
print(
    f"HN gate GPU ready: torch={torch.__version__}, device={properties.name}, "
    f"VRAM={properties.total_memory / 1024**3:.2f} GiB, python={sys.executable}",
    flush=True,
)
PY

exec "$PYTHON" scripts/run_hn_full_gate.py \
  --root "$ROOT" \
  --config configs/evaluation/hn_full_gate_v1.yaml
