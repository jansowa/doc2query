#!/usr/bin/env bash
# Czeka na koniec kolejki probe, liczy analizy parowane i WYŁĄCZA komputer.
# Uruchamiać przez sudo:
#   sudo setsid nohup scripts/shutdown_after_task07_probe.sh <UID_właściciela> \
#     >> logs/shutdown_watch.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
OWNER="${1:?podaj nazwę użytkownika, np. jsowa}"

echo "[shutdown-watch] start $(date '+%F %T')"
# Czekaj, aż zniknie proces kolejki ORAZ trening probe (wznowienia wliczone).
while pgrep -f "queue_task07_probe_embedders.sh" >/dev/null 2>&1 \
   || pgrep -f "train_probe_embedder.py" >/dev/null 2>&1; do
  sleep 120
done
echo "[shutdown-watch] kolejka zakończona $(date '+%F %T'); postprocessing"

runuser -u "$OWNER" -- bash scripts/finalize_task07_probe.sh \
  || echo "[shutdown-watch] postprocessing padł — wyniki surowe zostają na dysku" >&2

echo "[shutdown-watch] wyłączam za 2 minuty ($(date '+%F %T'))"
sync
sleep 120
shutdown -h now
