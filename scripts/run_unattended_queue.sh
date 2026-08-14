#!/usr/bin/env bash
# Sequential, fail-tolerant supervisor for a long unattended compute window.
#
# Usage: run_unattended_queue.sh [queue.tsv]
#
# Design rules (a failure must never cost the rest of the window):
#   * one job at a time, never concurrently with any other GPU process;
#   * a failing job is retried, then skipped — it never stops the queue;
#   * every job runs in its own process group under a hard time limit;
#   * jobs are resumable, so a retry continues instead of restarting;
#   * the queue stops early only when free disk falls below the guard;
#   * no job may power off, reboot, or touch final-test paths;
#   * finished jobs are remembered, so the queue can be relaunched safely.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
QUEUE_FILE="${1:-$ROOT/configs/unattended_queue_2026-08-14.tsv}"
QUEUE_ROOT="${DOC2QUERY_QUEUE_ROOT:-$ROOT/runs/unattended_queue_2026-08-14}"
MIN_FREE_GB="${DOC2QUERY_QUEUE_MIN_FREE_GB:-20}"
MAX_GPU_TEMP="${DOC2QUERY_QUEUE_MAX_GPU_TEMP:-86}"
GPU_IDLE_WAIT_SECONDS="${DOC2QUERY_QUEUE_GPU_WAIT:-7200}"
COOLDOWN_WAIT_SECONDS="${DOC2QUERY_QUEUE_COOLDOWN_WAIT:-1800}"
POLL_SECONDS="${DOC2QUERY_QUEUE_POLL:-5}"
RETRY_SLEEP_SECONDS="${DOC2QUERY_QUEUE_RETRY_SLEEP:-60}"
KILL_GRACE_SECONDS="${DOC2QUERY_QUEUE_KILL_GRACE:-60}"

test -f "$QUEUE_FILE" || { echo "missing queue file: $QUEUE_FILE" >&2; exit 2; }
mkdir -p "$QUEUE_ROOT/logs" "$QUEUE_ROOT/done"
EVENTS="$QUEUE_ROOT/queue.events.jsonl"
HEARTBEAT="$QUEUE_ROOT/heartbeat.txt"
SUMMARY="$QUEUE_ROOT/queue.summary.json"

exec 9>"$QUEUE_ROOT/queue.lock"
if ! flock -n 9; then
  echo "another unattended queue is already running" >&2
  exit 3
fi

export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_ASSETS_CACHE="$HF_HOME/assets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

log() { printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$QUEUE_ROOT/queue.log"; }

heartbeat() { printf '%s %s\n' "$(date --iso-8601=seconds)" "${1:-alive}" >"$HEARTBEAT"; }

emit_event() {
  python3 - "$EVENTS" "$@" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
keys = ("job", "attempt", "exit_code", "started_at", "finished_at", "duration_seconds", "outcome")
row = dict(zip(keys, sys.argv[2:]))
for key in ("attempt", "exit_code"):
    if key in row:
        row[key] = int(row[key])
if "duration_seconds" in row:
    row["duration_seconds"] = float(row["duration_seconds"])
with path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
PY
}

free_gb() { df --output=avail -BG "$ROOT" | tail -1 | tr -dc '0-9'; }

gpu_temp() {
  local value
  value=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
  printf '%s' "${value:-0}"
}

gpu_busy() {
  local apps
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
  if [ $? -ne 0 ]; then
    return 0  # cannot verify: treat as busy so we never collide with an unseen job
  fi
  [ -n "$apps" ]
}

wait_for_gpu() {
  local waited=0
  while gpu_busy; do
    if [ "$waited" -ge "$GPU_IDLE_WAIT_SECONDS" ]; then
      log "GPU still busy after ${waited}s; skipping this job"
      return 1
    fi
    [ $((waited % 300)) -eq 0 ] && log "waiting for the GPU to become idle (${waited}s)"
    heartbeat "waiting-for-gpu"
    sleep 30
    waited=$((waited + 30))
  done
  return 0
}

wait_for_cooldown() {
  local waited=0 temperature
  while :; do
    temperature=$(gpu_temp)
    [ "${temperature:-0}" -lt "$MAX_GPU_TEMP" ] && return 0
    if [ "$waited" -ge "$COOLDOWN_WAIT_SECONDS" ]; then
      log "GPU still at ${temperature}C after ${waited}s; continuing anyway"
      return 0
    fi
    log "GPU at ${temperature}C >= ${MAX_GPU_TEMP}C; cooling down"
    heartbeat "cooldown"
    sleep 60
    waited=$((waited + 60))
  done
}

# Run one command in its own process group under a hard limit; kill the whole group
# on expiry so no orphaned python keeps the GPU.
run_limited() {
  local limit=$1 logfile=$2 command=$3 pgid rc waited=0
  set -m
  bash -c "$command" >>"$logfile" 2>&1 &
  pgid=$!
  set +m
  while kill -0 "$pgid" 2>/dev/null; do
    if [ "$waited" -ge "$limit" ]; then
      log "time limit ${limit}s reached; terminating process group $pgid"
      kill -TERM -"$pgid" 2>/dev/null
      sleep "$KILL_GRACE_SECONDS"
      kill -KILL -"$pgid" 2>/dev/null
      wait "$pgid" 2>/dev/null
      return 124
    fi
    [ $((waited % 600)) -eq 0 ] && heartbeat "running"
    sleep "$POLL_SECONDS"
    waited=$((waited + POLL_SECONDS))
  done
  wait "$pgid"
  rc=$?
  return "$rc"
}

write_summary() {
  python3 - "$SUMMARY" "$EVENTS" "$QUEUE_ROOT/done" "$1" <<'PY'
import json
import os
import sys
from pathlib import Path

summary_path, events_path, done_dir, state = (Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4])
events = []
if events_path.is_file():
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
payload = {
    "schema_version": 1,
    "contract": "task06-unattended-compute-queue-v1",
    "state": state,
    "completed_jobs": sorted(path.name for path in done_dir.iterdir()) if done_dir.is_dir() else [],
    "attempts": len(events),
    "failed_attempts": [row for row in events if row.get("exit_code") not in (0, None)],
    "gpu_hours_by_job": {},
    "final_tests_used": [],
}
for row in events:
    job = row.get("job", "unknown")
    payload["gpu_hours_by_job"][job] = round(
        payload["gpu_hours_by_job"].get(job, 0.0) + float(row.get("duration_seconds", 0.0)) / 3600.0, 3
    )
temporary = summary_path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, summary_path)
PY
}

trap 'log "supervisor received a termination signal"; write_summary interrupted; heartbeat interrupted; exit 143' TERM INT

log "queue start: $QUEUE_FILE (free $(free_gb)GB, GPU $(gpu_temp)C)"
heartbeat "start"
write_summary running

while IFS=$'\t' read -r name limit attempts command || [ -n "${name:-}" ]; do
  case "${name:-}" in ''|'#'*) continue ;; esac
  if [ -z "${limit:-}" ] || [ -z "${attempts:-}" ] || [ -z "${command:-}" ]; then
    log "SKIP $name: malformed queue row"
    continue
  fi
  if [ -f "$QUEUE_ROOT/done/$name" ]; then
    log "SKIP $name: already completed"
    continue
  fi
  case "$command" in
    *poweroff*|*shutdown*|*reboot*|*final_test*|*test_native_pl*)
      log "REFUSE $name: forbidden command"
      emit_event "$name" 0 2 "$(date --iso-8601=seconds)" "$(date --iso-8601=seconds)" 0 refused
      continue
      ;;
  esac
  available=$(free_gb)
  if [ "${available:-0}" -lt "$MIN_FREE_GB" ]; then
    log "STOP: only ${available}GB free, below the ${MIN_FREE_GB}GB guard"
    write_summary stopped_low_disk
    heartbeat "stopped-low-disk"
    exit 0
  fi

  logfile="$QUEUE_ROOT/logs/$name.log"
  attempt=1
  while [ "$attempt" -le "$attempts" ]; do
    if ! wait_for_gpu; then
      emit_event "$name" "$attempt" 5 "$(date --iso-8601=seconds)" "$(date --iso-8601=seconds)" 0 gpu_busy
      break
    fi
    wait_for_cooldown
    started=$(date --iso-8601=seconds)
    started_epoch=$(date +%s)
    log "START $name attempt $attempt/$attempts (limit ${limit}s, free ${available}GB, $(gpu_temp)C)"
    printf '\n===== %s attempt %s =====\n' "$started" "$attempt" >>"$logfile"
    run_limited "$limit" "$logfile" "$command"
    rc=$?
    finished=$(date --iso-8601=seconds)
    duration=$(( $(date +%s) - started_epoch ))
    if [ "$rc" -eq 0 ]; then
      log "OK $name after ${duration}s"
      : >"$QUEUE_ROOT/done/$name"
      emit_event "$name" "$attempt" "$rc" "$started" "$finished" "$duration" completed
      break
    fi
    log "FAIL $name attempt $attempt rc=$rc after ${duration}s (see $logfile)"
    emit_event "$name" "$attempt" "$rc" "$started" "$finished" "$duration" failed
    attempt=$((attempt + 1))
    [ "$attempt" -le "$attempts" ] && sleep "$RETRY_SLEEP_SECONDS"
  done
  write_summary running
  heartbeat "between-jobs"
done <"$QUEUE_FILE"

log "queue finished"
write_summary finished
heartbeat "finished"
