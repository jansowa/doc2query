#!/usr/bin/env bash
# Pipeline wad na serwerze inferencji — jedna komenda, wartości tylko tutaj.
#
#   scripts/run_defect_pipeline.sh <BASE_URL> <MODEL> <API_KEY> [CONCURRENCY]
#
# przykład:
#   scripts/run_defect_pipeline.sh http://10.0.0.5:8000/v1 qwen3.8-27b sk-xxxx 16
#
# Wznawialny (journal per wywołanie); FRESH=1 zaczyna od zera. Wyniki:
# artifacts/task06/defect_pipeline_v1/verdicts/ — katalog do przeniesienia
# z powrotem na maszynę główną po zakończeniu.
set -euo pipefail

[ -f pyproject.toml ] || cd doc2query
[ -f pyproject.toml ] || { echo "uruchom z katalogu repozytorium (albo katalog wyżej)" >&2; exit 2; }

BASE_URL="${1:?podaj BASE_URL, np. http://host:8000/v1}"
MODEL="${2:?podaj nazwę modelu, np. qwen3.8-27b}"
API_KEY="${3:?podaj API key}"
CONCURRENCY="${4:-12}"

INPUT="artifacts/task06/defect_pipeline_v1/input/groups.jsonl"
OUT="artifacts/task06/defect_pipeline_v1/verdicts"
[ -f "$INPUT" ] || { echo "brak $INPUT — rozpakuj defect_pipeline_input.tar.gz w katalogu repo" >&2; exit 2; }

if [ "${FRESH:-0}" = "1" ] && [ -d "$OUT" ]; then
  archive="${OUT}.przerwane-$(date +%Y%m%dT%H%M%S)"
  mv "$OUT" "$archive"
  echo "FRESH=1: poprzedni journal odłożony do $archive"
fi

exec uv run python scripts/task06_defect_pipeline_remote.py \
  --groups "$INPUT" \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --api-key "$API_KEY" \
  --concurrency "$CONCURRENCY" \
  --output-dir "$OUT"
