#!/usr/bin/env bash
# Trzy ramiona Task 07 po kolei, wznawialnie, na zamrożonym planie.
#
# Ramię z gotowym `run_manifest.json` jest pomijane, a przerwane wznawia się z
# checkpointu (wagi trenowalne + stan AdamW). Kolejność jest stała: DPO pierwsze,
# bo tylko ono zależy od precomputowanych logprobów referencji.
#
# Użycie:  scripts/run_task07_arms.sh [katalog_runów]
# Wymaga GPU, więc idzie przez .venv-gpu (główne .venv ma torch CPU-only).
set -euo pipefail

cd "$(dirname "$0")/.."
[ -f pyproject.toml ] || { echo "nie jestem w katalogu repozytorium" >&2; exit 2; }

RUNS_DIR="${1:-runs}"
HANDOFF="artifacts/task07/handoff_v3_bottom"
PLAN="$HANDOFF/plan/dpo_plan.json"
PACKAGED="$HANDOFF/packaged"
REFERENCE="$HANDOFF/reference_logprobs/reference_logprobs.manifest.json"
ADAPTER="runs/D01-4.5B-STYLE-50K-S42/adapter"
CONFIG="configs/experiments/d01_4_5b_style_50k_s42.yaml"
PYTHON=".venv-gpu/bin/python"

for path in "$PLAN" "$REFERENCE" "$ADAPTER" "$CONFIG" "$PYTHON"; do
  [ -e "$path" ] || { echo "brak wymaganego wejścia: $path" >&2; exit 2; }
done

run_arm() {
  local arm="$1" out="$2"
  shift 2
  if [ -f "$out/run_manifest.json" ]; then
    echo "[$arm] gotowe, pomijam ($out/run_manifest.json)"
    return 0
  fi
  echo "[$arm] start → $out ($(date '+%H:%M:%S'))"
  PYTHONPATH=src "$PYTHON" -m doc2query.cli train dpo \
    --config "$CONFIG" \
    --plan "$PLAN" \
    --packaged-dir "$PACKAGED" \
    --adapter "$ADAPTER" \
    --output-dir "$out" \
    --arm "$arm" \
    --checkpoint-every 25 \
    "$@"
  echo "[$arm] koniec ($(date '+%H:%M:%S'))"
}

run_arm dpo "$RUNS_DIR/T07-V3-DPO-S42" --reference-logprobs "$REFERENCE"
run_arm continued_sft "$RUNS_DIR/T07-V3-CSFT-S42"
run_arm score_weighted_continued_sft "$RUNS_DIR/T07-V3-WSFT-S42"

echo "wszystkie trzy ramiona zakończone"
