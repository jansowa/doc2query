#!/usr/bin/env bash
# Power the machine off once the unattended queue has nothing left to do.
#
# Usage: poweroff_when_queue_drained.sh [queue.tsv]
#
# Intended to run from cron a few minutes off the queue-relaunch schedule.  It is
# deliberately paranoid, because powering off while work remains would waste days:
#   * refuses while the queue lock is held (a supervisor is running);
#   * refuses while any GPU compute process exists, or if the GPU cannot be queried;
#   * refuses unless every queue job is either completed or abandoned after
#     DOC2QUERY_POWEROFF_MAX_ATTEMPTS failed attempts;
#   * refuses unless at least one job actually completed;
#   * requires the drained state to hold for DOC2QUERY_POWEROFF_STABLE_SECONDS
#     before acting, so a relaunch in progress always wins;
#   * refuses if the operator dropped a "no_poweroff" file into the queue root;
#   * logs the decision and its evidence before powering off.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
QUEUE_FILE="${1:-$ROOT/configs/unattended_queue_2026-08-14.tsv}"
QUEUE_ROOT="${DOC2QUERY_QUEUE_ROOT:-$ROOT/runs/unattended_queue_2026-08-14}"
MAX_ATTEMPTS="${DOC2QUERY_POWEROFF_MAX_ATTEMPTS:-6}"
STABLE_SECONDS="${DOC2QUERY_POWEROFF_STABLE_SECONDS:-3600}"
DRY_RUN="${DOC2QUERY_POWEROFF_DRY_RUN:-0}"
STATE="$QUEUE_ROOT/drained_since.txt"
LOG="$QUEUE_ROOT/poweroff.log"

test -f "$QUEUE_FILE" || { echo "missing queue file: $QUEUE_FILE" >&2; exit 2; }
mkdir -p "$QUEUE_ROOT"
say() { printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$LOG"; }

# Never power off a machine that is merely running the test suite.
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY_RUN" = "0" ]; then
  say "hold: refusing to power off from inside a test run"
  exit 0
fi

if [ -f "$QUEUE_ROOT/no_poweroff" ]; then
  say "hold: $QUEUE_ROOT/no_poweroff exists"
  exit 0
fi

# A supervisor holding the queue lock means work is in flight.
if [ -f "$QUEUE_ROOT/queue.lock" ]; then
  if ! flock -n "$QUEUE_ROOT/queue.lock" true; then
    say "busy: the queue supervisor is running"
    rm -f "$STATE"
    exit 0
  fi
fi

gpu_apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
if [ $? -ne 0 ]; then
  say "hold: cannot query the GPU, refusing to decide"
  rm -f "$STATE"
  exit 0
fi
if [ -n "$gpu_apps" ]; then
  say "busy: GPU compute processes present"
  rm -f "$STATE"
  exit 0
fi

verdict=$(python3 - "$QUEUE_FILE" "$QUEUE_ROOT" "$MAX_ATTEMPTS" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

queue_file, queue_root, max_attempts = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
jobs = []
for line in queue_file.read_text(encoding="utf-8").splitlines():
    if not line.strip() or line.lstrip().startswith("#"):
        continue
    fields = line.split("\t")
    if len(fields) == 4:
        jobs.append(fields[0])
done = {path.name for path in (queue_root / "done").iterdir()} if (queue_root / "done").is_dir() else set()
failures: Counter[str] = Counter()
events_path = queue_root / "queue.events.jsonl"
if events_path.is_file():
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("outcome") in {"failed", "gpu_busy"}:
            failures[str(row.get("job"))] += 1
pending = [
    job for job in jobs if job not in done and failures.get(job, 0) < max_attempts
]
abandoned = [job for job in jobs if job not in done and failures.get(job, 0) >= max_attempts]
print(
    json.dumps(
        {
            "jobs": len(jobs),
            "completed": len([job for job in jobs if job in done]),
            "abandoned": abandoned,
            "pending": pending,
            "drained": not pending and len(jobs) > 0 and any(job in done for job in jobs),
        },
        sort_keys=True,
    )
)
PY
)
if [ -z "$verdict" ]; then
  say "hold: could not evaluate the queue state"
  exit 0
fi
drained=$(printf '%s' "$verdict" | python3 -c "import json,sys; print('yes' if json.load(sys.stdin)['drained'] else 'no')")
if [ "$drained" != "yes" ]; then
  say "pending work: $verdict"
  rm -f "$STATE"
  exit 0
fi

now=$(date +%s)
if [ ! -f "$STATE" ]; then
  printf '%s\n' "$now" >"$STATE"
  say "queue drained; starting the ${STABLE_SECONDS}s confirmation window: $verdict"
  exit 0
fi
since=$(cat "$STATE" 2>/dev/null | tr -dc '0-9')
since=${since:-$now}
held=$((now - since))
if [ "$held" -lt "$STABLE_SECONDS" ]; then
  say "queue drained for ${held}s; waiting for ${STABLE_SECONDS}s"
  exit 0
fi

say "queue drained for ${held}s; powering off. evidence: $verdict"
if [ "$DRY_RUN" != "0" ]; then
  say "dry run: would have run 'sudo -n systemctl poweroff'"
  exit 0
fi
if ! sudo -n systemctl poweroff; then
  say "poweroff refused: install the sudoers rule from the ADR, keeping the machine on"
  exit 1
fi
