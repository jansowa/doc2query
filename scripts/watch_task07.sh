#!/usr/bin/env bash
# Jeden podgląd wszystkiego, co liczy się w Task 07: GPU, ramiona, kolejki lokalne.
#
#   scripts/watch_task07.sh          # odświeża co 20 s
#   scripts/watch_task07.sh 0        # jeden zrzut i wyjście
#
# Czyta wyłącznie artefakty i logi, więc nie dotyka żadnego runu.
set -uo pipefail

cd "$(dirname "$0")/.."
INTERVAL="${1:-20}"

snapshot() {
  printf '\033[2J\033[H'
  echo "Task 07 — $(date '+%Y-%m-%d %H:%M:%S')"
  if command -v nvidia-smi > /dev/null; then
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,power.draw,power.limit,temperature.gpu \
      --format=csv,noheader,nounits |
      awk -F', ' '{printf "GPU %s%% · %s/%s MiB · %s/%s W · %s°C\n", $3, $1, $2, $4, $5, $6}'
  fi
  echo

  echo "RAMIONA (run_manifest.json = gotowe)"
  for run in runs/T07-*/; do
    [ -d "$run" ] || continue
    name=$(basename "$run")
    if [ -f "$run/run_manifest.json" ]; then
      uv run python - "$run/run_manifest.json" "$name" <<'PY' 2>/dev/null || echo "  $name: gotowe"
import json, sys
manifest = json.load(open(sys.argv[1]))
start, end = manifest["dev"]["start"], manifest["dev"]["end"]
line = (
    f"  {sys.argv[2]:<20} GOTOWE  NLL {start['mean_chosen_nll_per_token']:.4f}"
    f" -> {end['mean_chosen_nll_per_token']:.4f}"
)
if "policy_margin_accuracy" in end:
    line += f" | margin {end['policy_margin_accuracy']:.4f}"
print(line)
PY
    else
      steps=$(wc -l < "$run/history.jsonl" 2>/dev/null || echo 0)
      ckpt=$([ -d "$run/checkpoint" ] && echo " (checkpoint)" || echo "")
      echo "  $name: w toku, $steps kroków w historii$ckpt"
    fi
  done
  echo

  echo "PROCESY"
  for pattern in "train dpo" "evaluate generator" "precompute_task07" "prepare_lexical_contrast" "run_probe"; do
    count=$(ps -eo cmd --no-headers | grep -c -- "$pattern" || true)
    [ "$count" -gt 0 ] && echo "  $pattern: $count"
  done
  pgrep -f "train dpo|evaluate generator|precompute_task07|prepare_lexical_contrast|run_probe" > /dev/null ||
    echo "  (nic nie liczy na GPU)"
  echo

  echo "OSTATNIE LINIE LOGÓW"
  for log in logs/task07_defect_arms.log logs/lexical_worklist.log logs/task07_probe.log; do
    [ -f "$log" ] || continue
    last=$(grep -aE "\[arms\]|\[lexical\]|\[probe\]|krok |Error|Traceback" "$log" | tail -1)
    [ -n "$last" ] && printf '  %-34s %s\n' "$(basename "$log"):" "${last:0:90}"
  done
}

if [ "$INTERVAL" = "0" ]; then
  snapshot
  exit 0
fi
while true; do
  snapshot
  echo
  echo "Ctrl+C kończy podgląd; obliczeń to nie dotyczy."
  sleep "$INTERVAL"
done
