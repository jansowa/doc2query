#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/.uv-cache}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_ASSETS_CACHE="$HF_HOME/assets"
export TMPDIR="$ROOT/.cache/tmp"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

case "$HF_HOME" in
  "$ROOT"/*) ;;
  *) echo "P-03 BLOCKED: HF_HOME must be on the project partition: $ROOT" >&2; exit 2 ;;
esac
case "$UV_CACHE_DIR" in
  "$ROOT"/*) ;;
  *) echo "P-03 BLOCKED: UV_CACHE_DIR must be on the project partition: $ROOT" >&2; exit 2 ;;
esac

mkdir -p "$TMPDIR" "$ROOT/logs"

MODE=()
case "${1:-}" in
  "")
    ;;
  --dry-run)
    MODE=(--dry-run)
    shift
    ;;
  --smoke)
    MODE=(--mock-smoke)
    shift
    ;;
  --help|-h)
    exec "$ROOT/.venv/bin/python" scripts/p03_w05_sensitivity.py --help
    ;;
  *)
    echo "usage: $0 [--dry-run|--smoke|--help]" >&2
    exit 2
    ;;
esac
if (($#)); then
  echo "usage: $0 [--dry-run|--smoke|--help]" >&2
  exit 2
fi

exec "$ROOT/.venv/bin/python" scripts/p03_w05_sensitivity.py \
  --root "$ROOT" \
  --config configs/evaluation/p03_w05_sensitivity.yaml \
  "${MODE[@]}"
