#!/usr/bin/env bash
# Kolejka bezobsługowa: intrinsics generatora dla siedmiu punktów Task 07.
#
# `doc2query evaluate generator` na subsecie ROZWOJOWYM dev_intrinsic_rank10 dla
# adaptera startowego (D01) i sześciu ramion (bottom/near_miss × dpo/csft/wsft).
# To pomiar pomocniczy (metryki powierzchniowe), nie probe i nie bramka — służy
# do jakościowego porównania generacji zanim wystartuje właściwa ewaluacja probe.
# Żaden zbiór testowy nie jest dotykany. Punkt gotowy (report.json) jest pomijany.
set -euo pipefail

cd "$(dirname "$0")/.."
[ -f pyproject.toml ] || { echo "nie jestem w katalogu repozytorium" >&2; exit 2; }

CONFIG="configs/experiments/d01_4_5b_style_50k_s42.yaml"
MANIFEST="data/processed/v1/evaluation/task04-v1/manifest.json"
SUBSET="dev_intrinsic_rank10"
GPU_PY=".venv-gpu/bin/python"
OUT="runs/task07_generator_eval_v1"

for path in "$CONFIG" "$MANIFEST" "$GPU_PY"; do
  [ -e "$path" ] || { echo "brak wymaganego wejścia: $path" >&2; exit 2; }
done

declare -A POINTS=(
  [start]="runs/D01-4.5B-STYLE-50K-S42/adapter"
  [bottom_dpo]="runs/T07-V3-DPO-S42/adapter"
  [bottom_csft]="runs/T07-V3-CSFT-S42/adapter"
  [bottom_wsft]="runs/T07-V3-WSFT-S42/adapter"
  [nearmiss_dpo]="runs/T07-NM-DPO-S42/adapter"
  [nearmiss_csft]="runs/T07-NM-CSFT-S42/adapter"
  [nearmiss_wsft]="runs/T07-NM-WSFT-S42/adapter"
)

for name in start bottom_dpo bottom_csft bottom_wsft nearmiss_dpo nearmiss_csft nearmiss_wsft; do
  adapter="${POINTS[$name]}"
  out="$OUT/$name"
  if [ -f "$out/result.json" ]; then
    echo "[eval] $name gotowe, pomijam"
    continue
  fi
  [ -d "$adapter" ] || { echo "[eval] brak adaptera $adapter" >&2; exit 2; }
  echo "[eval] $name start ($(date '+%H:%M:%S'))"
  PYTHONPATH=src "$GPU_PY" -m doc2query.cli evaluate generator \
    --config "$CONFIG" \
    --frozen-manifest "$MANIFEST" \
    --subset "$SUBSET" \
    --adapter "$adapter" \
    --output-dir "$out"
  echo "[eval] $name koniec ($(date '+%H:%M:%S'))"
done

echo "[eval] wszystkie punkty policzone"
