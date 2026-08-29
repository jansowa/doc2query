#!/usr/bin/env bash
# Zakolejkuj zarejestrowaną ablację top-vs-near-miss (tasks/07_dpo_training.md §Ablacje).
#
# Czeka, aż skończą się ramiona wariantu bottom, a potem prowadzi pełny łańcuch
# near_miss: handoff → pakowanie → długości tokenów → plan → precompute → trzy
# ramiona. Każdy krok jest pilnowany istnieniem wyjścia, więc skrypt można
# przerywać i uruchamiać ponownie bez utraty pracy. Autoryzacja treningu:
# reports/decisions/task07_training_authorization_2026-08-28.md (te same pasaże,
# prompty i chosen co wariant bottom; różni się wyłącznie strona rejected).
set -euo pipefail

cd "$(dirname "$0")/.."
[ -f pyproject.toml ] || { echo "nie jestem w katalogu repozytorium" >&2; exit 2; }

H="artifacts/task07/handoff_v3_nearmiss"
PAIRS="artifacts/task06/v3_pairs_v1/near_miss/pairs.jsonl"
CONFIG="configs/experiments/d01_4_5b_style_50k_s42.yaml"
ADAPTER="runs/D01-4.5B-STYLE-50K-S42/adapter"
PLAN_CFG="configs/train/task07_dpo_plan_v3_nearmiss.yaml"
PLAN="$H/plan/dpo_plan.json"
GPU_PY=".venv-gpu/bin/python"

for path in "$PAIRS" "$CONFIG" "$ADAPTER" "$GPU_PY"; do
  [ -e "$path" ] || { echo "brak wymaganego wejścia: $path" >&2; exit 2; }
done

echo "[kolejka] czekam na koniec ramion bottom ($(date '+%H:%M:%S'))"
for run in T07-V3-DPO-S42 T07-V3-CSFT-S42 T07-V3-WSFT-S42; do
  until [ -f "runs/$run/run_manifest.json" ]; do sleep 60; done
  echo "[kolejka] $run gotowe"
done

if [ ! -f "$H/packaged/manifest.json" ]; then
  echo "[kolejka] handoff near_miss ($(date '+%H:%M:%S'))"
  uv run python scripts/build_task07_handoff_v3.py \
    --pairs "$PAIRS" --handoff-dir "$H/inputs" --packaged-dir "$H/packaged"
fi

if [ ! -f "$H/token_lengths/token_lengths.manifest.json" ]; then
  echo "[kolejka] długości tokenów ($(date '+%H:%M:%S'))"
  uv run python scripts/measure_task07_token_lengths.py \
    --preference-train "$H/inputs/preference_train.jsonl" \
    --preference-dev "$H/inputs/preference_dev.jsonl" \
    --packaged-dir "$H/packaged" --output-dir "$H/token_lengths"
fi

if [ ! -f "$PLAN_CFG" ]; then
  echo "[kolejka] config planu ($(date '+%H:%M:%S'))"
  uv run python scripts/build_task07_dpo_plan_config.py \
    --plan-id task07-dpo-plan-v3-nearmiss-s42 \
    --token-length-manifest "$H/token_lengths/token_lengths.manifest.json" \
    --token-length-records "$H/token_lengths/token_lengths.jsonl" \
    --weight-manifest "$H/inputs/weight_manifest.json" \
    --output "$PLAN_CFG"
fi

dataset_args=(
  --task06-manifest "$H/packaged/manifest.json"
  --preference-train "$H/packaged/preference_train.jsonl"
  --preference-dev "$H/packaged/preference_dev.jsonl"
  --continued-sft-train "$H/packaged/continued_sft_train.jsonl"
  --continued-sft-dev "$H/packaged/continued_sft_dev.jsonl"
  --weighted-sft-train "$H/packaged/weighted_sft_train.jsonl"
  --weighted-sft-dev "$H/packaged/weighted_sft_dev.jsonl"
)

if [ ! -f "$PLAN" ]; then
  echo "[kolejka] plan ($(date '+%H:%M:%S'))"
  uv run python scripts/plan_dpo_controls.py "${dataset_args[@]}" \
    --config "$PLAN_CFG" \
    --token-length-manifest "$H/token_lengths/token_lengths.manifest.json" \
    --token-length-records "$H/token_lengths/token_lengths.jsonl" \
    --output "$PLAN" > "$H/plan/plan_stdout.json"
fi

if [ ! -f "$H/reference_logprobs/run_summary.json" ]; then
  echo "[kolejka] precompute logprobów referencji, GPU ($(date '+%H:%M:%S'))"
  PYTHONPATH=scripts:src "$GPU_PY" scripts/precompute_task07_reference_logprobs.py \
    "${dataset_args[@]}" --plan "$PLAN" \
    --output-dir "$H/reference_logprobs"
fi

run_arm() {
  local arm="$1" out="$2"
  shift 2
  if [ -f "$out/run_manifest.json" ]; then
    echo "[kolejka] $arm gotowe, pomijam"
    return 0
  fi
  echo "[kolejka] $arm start → $out ($(date '+%H:%M:%S'))"
  PYTHONPATH=src "$GPU_PY" -m doc2query.cli train dpo \
    --config "$CONFIG" --plan "$PLAN" --packaged-dir "$H/packaged" \
    --adapter "$ADAPTER" --output-dir "$out" --arm "$arm" \
    --checkpoint-every 25 "$@"
  echo "[kolejka] $arm koniec ($(date '+%H:%M:%S'))"
}

run_arm dpo runs/T07-NM-DPO-S42 \
  --reference-logprobs "$H/reference_logprobs/reference_logprobs.manifest.json"
run_arm continued_sft runs/T07-NM-CSFT-S42
run_arm score_weighted_continued_sft runs/T07-NM-WSFT-S42

echo "[kolejka] ablacja near_miss zakończona ($(date '+%H:%M:%S'))"
