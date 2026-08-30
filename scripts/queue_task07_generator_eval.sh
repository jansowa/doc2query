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
# Bez configów zamrożonych sędziów `evaluate generator` policzy generacje i padnie
# na scoringu — a wtedy punkt nigdy nie dostaje result.json i strażnik wskrzesza
# go w nieskończoność. Sędziowie są tu wyłącznie oceniający, nigdy trenowani.
PRIMARY_JUDGE="configs/reranker/primary_polish_roberta_v3_cuda.yaml"
SHADOW_JUDGE="configs/reranker/shadow_bge_v2_m3.yaml"
GPU_PY=".venv-gpu/bin/python"
OUT="runs/task07_generator_eval_v1"

# --- odporność na jednorazowe potknięcia -------------------------------------
# `set -e` sam z siebie zabija kolejkę przy pierwszym błędzie, a przy runie bez
# nadzoru to znaczy: jedno chwilowe OOM i przez dobę nic się nie policzy.
# `retry` ponawia etap z rosnącą przerwą, `stage` idzie dalej mimo trwałej
# porażki — kolejne etapy i tak sprawdzają swoje wejścia, a wznowienie
# dokończy resztę.
FAILED_STAGES=""

retry() {
  local label="$1" attempts="${2:-3}"
  shift 2
  local attempt=1
  until "$@"; do
    if [ "$attempt" -ge "$attempts" ]; then
      echo "[kolejka] $label: $attempts prób bez powodzenia, idę dalej" >&2
      FAILED_STAGES="$FAILED_STAGES $label"
      return 1
    fi
    echo "[kolejka] $label: próba $attempt padła, ponawiam za $((attempt * 120)) s" >&2
    sleep $((attempt * 120))
    attempt=$((attempt + 1))
  done
  return 0
}

stage() {
  retry "$@" || true
}

for path in "$CONFIG" "$MANIFEST" "$GPU_PY" "$PRIMARY_JUDGE" "$SHADOW_JUDGE"; do
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
  if [ ! -d "$adapter" ]; then
    echo "[eval] brak adaptera $adapter — pomijam punkt" >&2
    FAILED_STAGES="$FAILED_STAGES $name(brak-adaptera)"
    continue
  fi
  echo "[eval] $name start ($(date '+%H:%M:%S'))"
  stage "$name" 3 env PYTHONPATH=src "$GPU_PY" -m doc2query.cli evaluate generator \
    --config "$CONFIG" \
    --frozen-manifest "$MANIFEST" \
    --subset "$SUBSET" \
    --adapter "$adapter" \
    --primary-judge "$PRIMARY_JUDGE" \
    --shadow-judge "$SHADOW_JUDGE" \
    --output-dir "$out"
  echo "[eval] $name koniec ($(date '+%H:%M:%S'))"
done

if [ -n "$FAILED_STAGES" ]; then
  echo "[eval] punkty nieudane:$FAILED_STAGES — ponowne uruchomienie je dokończy" >&2
fi
echo "[eval] przebieg zakończony"
