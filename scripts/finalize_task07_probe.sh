#!/usr/bin/env bash
# Postprocessing po kolejce probe: analizy parowane vs start (CPU, bez GPU).
set -uo pipefail
cd "$(dirname "$0")/.."

REF="runs/task07_probe/start"
OUT="runs/task07_probe/paired_vs_start"
[ -f "$REF/result.json" ] || { echo "brak wyniku ramienia start" >&2; exit 1; }
mkdir -p "$OUT"

for dir in runs/task07_probe/*/; do
  arm="$(basename "$dir")"
  [ "$arm" = "start" ] && continue
  [ "$arm" = "paired_vs_start" ] && continue
  [ -f "$dir/result.json" ] || continue
  [ -f "$OUT/$arm.json" ] && continue
  echo "[finalize] parowane $arm vs start"
  uv run python scripts/compare_task07_probe_paired.py \
    --candidate "$dir" --reference "$REF" \
    --output "$OUT/$arm.json" >/dev/null || echo "[finalize] $arm padło" >&2
done
echo "[finalize] gotowe: $(ls "$OUT" 2>/dev/null | wc -l) porównań w $OUT"
