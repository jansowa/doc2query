#!/usr/bin/env bash
# Trzy ramiona Task 07 dla dowolnej autoryzowanej kohorty.
#
#   scripts/run_task07_arms_generic.sh <katalog_handoffu> <prefiks_runów>
#
# przykład:
#   scripts/run_task07_arms_generic.sh artifacts/task07/handoff_defect_v1 T07-DEF
#
# Wznawialne: ramię z run_manifest.json jest pomijane, przerwane wraca z
# checkpointu (wagi trenowalne + stan AdamW). Każde ramię idzie przez retry, a
# trwała porażka jednego nie przerywa pozostałych.
set -uo pipefail

cd "$(dirname "$0")/.."
[ -f pyproject.toml ] || { echo "nie jestem w katalogu repozytorium" >&2; exit 2; }

H="${1:?podaj katalog handoffu, np. artifacts/task07/handoff_defect_v1}"
PREFIX="${2:?podaj prefiks runów, np. T07-DEF}"
SEED="${SEED:-S42}"
CONFIG="configs/experiments/d01_4_5b_style_50k_s42.yaml"
ADAPTER="runs/D01-4.5B-STYLE-50K-S42/adapter"
PLAN="$H/plan/dpo_plan.json"
REFERENCE="$H/reference_logprobs/reference_logprobs.manifest.json"
GPU_PY=".venv-gpu/bin/python"

for path in "$PLAN" "$REFERENCE" "$ADAPTER" "$CONFIG" "$GPU_PY" "$H/packaged"; do
  [ -e "$path" ] || { echo "brak wymaganego wejścia: $path" >&2; exit 2; }
done

FAILED_ARMS=""

run_arm() {
  local arm="$1" out="$2"
  shift 2
  if [ -f "$out/run_manifest.json" ]; then
    echo "[arms] $arm gotowe, pomijam ($out)"
    return 0
  fi
  local attempt=1
  while [ "$attempt" -le 3 ]; do
    echo "[arms] $arm start, próba $attempt ($(date '+%H:%M:%S'))"
    if PYTHONPATH=src "$GPU_PY" -m doc2query.cli train dpo \
      --config "$CONFIG" --plan "$PLAN" --packaged-dir "$H/packaged" \
      --adapter "$ADAPTER" --output-dir "$out" --arm "$arm" \
      --checkpoint-every 25 "$@"; then
      echo "[arms] $arm koniec ($(date '+%H:%M:%S'))"
      return 0
    fi
    echo "[arms] $arm padło; wznowienie z checkpointu za 120 s" >&2
    sleep 120
    attempt=$((attempt + 1))
  done
  echo "[arms] $arm nieudane po trzech próbach — idę do następnego" >&2
  FAILED_ARMS="$FAILED_ARMS $arm"
  return 1
}

run_arm dpo "runs/${PREFIX}-DPO-${SEED}" --reference-logprobs "$REFERENCE" || true
run_arm continued_sft "runs/${PREFIX}-CSFT-${SEED}" || true
run_arm score_weighted_continued_sft "runs/${PREFIX}-WSFT-${SEED}" || true

if [ -n "$FAILED_ARMS" ]; then
  echo "[arms] ramiona nieudane:$FAILED_ARMS — ponowne uruchomienie je dokończy" >&2
  exit 1
fi
echo "[arms] wszystkie trzy ramiona zakończone ($(date '+%H:%M:%S'))"
