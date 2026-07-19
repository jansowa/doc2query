#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$ROOT/logs/overnight_4_5b.log"
STATUS="$ROOT/reports/overnight_4_5b/status.tsv"
PATTERN="START |END |Selected |queue complete|Overnight queue|No safe|\\{'loss'|\\{'eval_loss'|train_runtime|train_loss"

if [[ ! -f "$LOG" ]]; then
  echo "Log does not exist yet: $LOG" >&2
  exit 1
fi

case "${1:-snapshot}" in
  --follow|-f)
    echo "Following filtered metrics from $LOG (Ctrl-C stops only this viewer)."
    tail -n 40 -F "$LOG" \
      | tr '\r' '\n' \
      | rg --line-buffered "$PATTERN"
    ;;
  --raw)
    echo "Following raw log from $LOG (Ctrl-C stops only this viewer)."
    tail -n 80 -F "$LOG"
    ;;
  snapshot)
    echo "Service:"
    systemctl --user is-active doc2query-overnight-4-5b.service 2>/dev/null || true
    echo
    echo "Completed queue steps:"
    if [[ -f "$STATUS" ]]; then
      tail -n 12 "$STATUS"
    else
      echo "No status file yet."
    fi
    echo
    echo "Recent metrics and transitions:"
    tr '\r' '\n' <"$LOG" | rg "$PATTERN" | tail -n 40
    echo
    echo "Saved full checkpoints:"
    find "$ROOT/runs" -maxdepth 2 -type d -path '*/W06-*/checkpoint-*' -printf '%p\n' \
      | sort -V \
      | tail -n 20
    ;;
  *)
    echo "Usage: $0 [snapshot|--follow|-f|--raw]" >&2
    exit 2
    ;;
esac
