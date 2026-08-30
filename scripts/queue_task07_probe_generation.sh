#!/usr/bin/env bash
# Generacja zapytań probe dla wszystkich punktów Task 07 — praca na całą noc.
#
#   scripts/queue_task07_probe_generation.sh
#
# Dla każdego punktu (start SFT + dziewięć ramion) generuje po 4 kontrolowane
# zapytania na każdy z 1 984 pasaży kohorty probe. To wejście do probe embeddera,
# czyli do jedynego kryterium rozstrzygającego Task 07.
#
# Punkt z gotowym `generated.summary.json` jest pomijany, więc restart nie
# powtarza pracy. Nieudany punkt zostawia po sobie plik częściowy — kolejka go
# kasuje przed ponowną próbą, bo generacja nie ma journala i musi zacząć od zera.
set -uo pipefail

cd "$(dirname "$0")/.."
[ -f pyproject.toml ] || { echo "nie jestem w katalogu repozytorium" >&2; exit 2; }

CONFIG="configs/experiments/task07_probe_generation_v1.yaml"
GPU_PY=".venv-gpu/bin/python"
OUT="runs/task07_probe_gen_v1"

for path in "$CONFIG" "$GPU_PY" "data/processed/task07/probe_cohort_v1/passages.jsonl"; do
  [ -e "$path" ] || { echo "brak wymaganego wejścia: $path" >&2; exit 2; }
done

# Kolejność: najpierw punkt startowy (bez niego żadne porównanie nie ma sensu),
# potem ramiona DPO każdej kohorty, na końcu kontrole.
POINTS="
start:runs/D01-4.5B-STYLE-50K-S42/adapter
defect_dpo:runs/T07-DEF-DPO-S42/adapter
bottom_dpo:runs/T07-V3-DPO-S42/adapter
nearmiss_dpo:runs/T07-NM-DPO-S42/adapter
defect_csft:runs/T07-DEF-CSFT-S42/adapter
bottom_csft:runs/T07-V3-CSFT-S42/adapter
nearmiss_csft:runs/T07-NM-CSFT-S42/adapter
defect_wsft:runs/T07-DEF-WSFT-S42/adapter
bottom_wsft:runs/T07-V3-WSFT-S42/adapter
nearmiss_wsft:runs/T07-NM-WSFT-S42/adapter
"

FAILED=""
for entry in $POINTS; do
  name="${entry%%:*}"
  adapter="${entry#*:}"
  target="$OUT/$name/generated.jsonl"
  if [ -f "$OUT/$name/generated.summary.json" ]; then
    echo "[probe] $name gotowe, pomijam"
    continue
  fi
  if [ ! -d "$adapter" ]; then
    echo "[probe] brak adaptera $adapter — pomijam punkt $name" >&2
    FAILED="$FAILED $name(brak-adaptera)"
    continue
  fi
  attempt=1
  while [ "$attempt" -le 3 ]; do
    echo "[probe] $name start, próba $attempt ($(date '+%H:%M:%S'))"
    rm -f "$target"
    mkdir -p "$OUT/$name"
    if PYTHONPATH=src "$GPU_PY" -m doc2query.cli generate \
      --config "$CONFIG" --adapter "$adapter" --output "$target"; then
      echo "[probe] $name koniec ($(date '+%H:%M:%S'))"
      break
    fi
    echo "[probe] $name padło; ponawiam za 120 s" >&2
    sleep 120
    attempt=$((attempt + 1))
  done
  [ -f "$OUT/$name/generated.summary.json" ] || FAILED="$FAILED $name"
done

if [ -n "$FAILED" ]; then
  echo "[probe] punkty nieudane:$FAILED — ponowne uruchomienie je dokończy" >&2
  exit 1
fi
echo "[probe] wszystkie punkty wygenerowane ($(date '+%H:%M:%S'))"
