#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/.uv-cache}"
GPU_VENV="${DOC2QUERY_GPU_VENV:-$ROOT/.venv-gpu}"
PYTHON="$GPU_VENV/bin/python"

usage() {
  echo "usage: $0 [--check|--dry-run|--help]"
}

case "${1:-}" in
  --help|-h)
    usage
    exit 0
    ;;
  --check)
    MODE=check
    ;;
  --dry-run)
    MODE=dry-run
    ;;
  "")
    MODE=install
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

case "$GPU_VENV" in
  "$ROOT"/*) ;;
  *)
    echo "GPU environment blocked: DOC2QUERY_GPU_VENV must be on the project partition." >&2
    exit 2
    ;;
esac
case "$UV_CACHE_DIR" in
  "$ROOT"/*) ;;
  *)
    echo "GPU environment blocked: UV_CACHE_DIR must be on the project partition." >&2
    exit 2
    ;;
esac

check_stack() {
  [[ -x "$PYTHON" ]] || return 1
  "$PYTHON" - <<'PY'
from importlib.metadata import PackageNotFoundError, version

try:
    import torch
except ImportError:
    raise SystemExit(1) from None

expected = {
    "bitsandbytes": "0.49.2",
    "peft": "0.19.1",
    "pl_core_news_lg": "3.8.0",
    "torch": "2.6.0",
    "transformers": "5.13.1",
    "trl": "0.29.1",
}
try:
    actual = {name: version(name).split("+", maxsplit=1)[0] for name in expected}
except PackageNotFoundError:
    raise SystemExit(1) from None
if actual != expected or torch.version.cuda != "12.4":
    raise SystemExit(1)
print(
    f"GPU Python ready: {torch.__version__}, CUDA build {torch.version.cuda}, "
    f"interpreter {__import__('sys').executable}"
)
PY
}

if check_stack; then
  exit 0
fi
if [[ "$MODE" == check ]]; then
  echo "GPU Python is absent or does not match the pinned CUDA 12.4 stack: $PYTHON" >&2
  exit 1
fi

command -v uv >/dev/null || {
  echo "GPU environment blocked: uv is not available on PATH." >&2
  exit 2
}

if [[ ! -x "$PYTHON" ]]; then
  uv venv --python 3.11 "$GPU_VENV"
fi

install_args=(
  --python "$PYTHON"
  --group data
  --group training
  --group retrieval
  --group nlp
  --group evaluation
  --editable .
  --no-sources
  --torch-backend cu124
  --constraint configs/environment/gpu-cu124.constraints.txt
  "https://github.com/explosion/spacy-models/releases/download/pl_core_news_lg-3.8.0/pl_core_news_lg-3.8.0-py3-none-any.whl"
)
if [[ "$MODE" == dry-run ]]; then
  install_args=(--dry-run "${install_args[@]}")
fi
uv pip install "${install_args[@]}"

if [[ "$MODE" == dry-run ]]; then
  exit 0
fi
check_stack || {
  echo "GPU environment blocked: installation did not produce the pinned CUDA stack." >&2
  exit 2
}
