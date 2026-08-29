#!/usr/bin/env bash
# Strażnik kolejek na okno bez nadzoru: wskrzesza to, co padło, i milczy resztę czasu.
#
#   nohup scripts/watchdog_queues.sh > logs/watchdog.log 2>&1 &
#
# Obie kolejki są idempotentne (każdy etap pilnowany istnieniem wyjścia), więc
# ponowne uruchomienie nigdy nie powtarza policzonej pracy — może tylko dokończyć.
# Strażnik kończy się sam, gdy oba cele są osiągnięte, i tak samo sam się kończy
# po `MAX_HOURS`, żeby nie żyć w nieskończoność po cichym błędzie.
set -uo pipefail  # bez -e: strażnik ma przeżyć błąd tego, czego pilnuje

cd "$(dirname "$0")/.."
[ -f pyproject.toml ] || { echo "nie jestem w katalogu repozytorium" >&2; exit 2; }

INTERVAL="${INTERVAL:-600}"
MAX_HOURS="${MAX_HOURS:-30}"
EVAL_OUT="runs/task07_generator_eval_v1"
EVAL_POINTS="start bottom_dpo bottom_csft bottom_wsft nearmiss_dpo nearmiss_csft nearmiss_wsft"
DEFECT_DONE="artifacts/task07/handoff_defect_v1/reference_logprobs/run_summary.json"

started=$(date +%s)

eval_complete() {
  for name in $EVAL_POINTS; do
    [ -f "$EVAL_OUT/$name/result.json" ] || return 1
  done
  return 0
}

running() {  # $1 = wzorzec procesu
  pgrep -f "$1" > /dev/null
}

echo "[strażnik] start $(date '+%Y-%m-%d %H:%M:%S'), interwał ${INTERVAL}s, limit ${MAX_HOURS}h"

while true; do
  now=$(date +%s)
  if [ $(((now - started) / 3600)) -ge "$MAX_HOURS" ]; then
    echo "[strażnik] limit ${MAX_HOURS}h osiągnięty, kończę"
    exit 0
  fi

  eval_ok=1
  eval_complete || eval_ok=0
  defect_ok=0
  [ -f "$DEFECT_DONE" ] && defect_ok=1

  if [ "$eval_ok" = 1 ] && [ "$defect_ok" = 1 ]; then
    echo "[strażnik] oba cele osiągnięte $(date '+%H:%M:%S'), kończę"
    exit 0
  fi

  if [ "$eval_ok" = 0 ] && ! running "queue_task07_generator_eval.sh"; then
    echo "[strażnik] kolejka intrinsics nie żyje, wskrzeszam $(date '+%H:%M:%S')"
    nohup scripts/queue_task07_generator_eval.sh >> logs/task07_generator_eval.log 2>&1 &
    sleep 30
  fi

  if [ "$defect_ok" = 0 ] && ! running "queue_after_defect_verdicts.sh"; then
    echo "[strażnik] kolejka pipeline'u wad nie żyje, wskrzeszam $(date '+%H:%M:%S')"
    nohup scripts/queue_after_defect_verdicts.sh >> logs/defect_after.log 2>&1 &
    sleep 30
  fi

  sleep "$INTERVAL"
done
