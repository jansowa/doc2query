#!/usr/bin/env bash
# Kolejka ADR task07_anti_collapse_v1: cztery ramiona + pomiar różnorodności.
#
#   scripts/queue_task07_anti_collapse.sh
#
# Sekwencja: EQ102 (kontrola kroków) → BETA02 (plan + precompute + trening) →
# RPO (regularyzator NLL λ=1,0) → DIVCH (handoff + plan + precompute + trening)
# → generacja różnorodności dla BETA02/RPO/DIVCH tym samym protokołem, którym
# zmierzono kolaps. Każdy etap pilnowany istnieniem wyjścia; restart dokańcza.
set -uo pipefail

cd "$(dirname "$0")/.."
[ -f pyproject.toml ] || { echo "nie jestem w katalogu repozytorium" >&2; exit 2; }

CONFIG="configs/experiments/d01_4_5b_style_50k_s42.yaml"
GEN_CONFIG="configs/experiments/task07_probe_generation_v1.yaml"
ADAPTER="runs/D01-4.5B-STYLE-50K-S42/adapter"
GPU_PY=".venv-gpu/bin/python"
BOTTOM="artifacts/task07/handoff_v3_bottom"
DEFECT="artifacts/task07/handoff_defect_v1"
DIVCH="artifacts/task07/handoff_defect_diverse_v1"

for path in "$CONFIG" "$GEN_CONFIG" "$ADAPTER" "$GPU_PY" "$BOTTOM/plan/dpo_plan.json" \
  "$DEFECT/plan/dpo_plan.json" "artifacts/task06/defect_diverse_v1/pairs_trainable.jsonl"; do
  [ -e "$path" ] || { echo "brak wymaganego wejścia: $path" >&2; exit 2; }
done

train() { # train <out> <plan> <packaged> <ref|-> <ekstra...>
  local out="$1" plan="$2" packaged="$3" ref="$4"
  shift 4
  if [ -f "$out/run_manifest.json" ]; then
    echo "[anty] $(basename "$out") gotowe, pomijam"
    return 0
  fi
  local extra=()
  [ "$ref" != "-" ] && extra+=(--reference-logprobs "$ref")
  local attempt=1
  while [ "$attempt" -le 3 ]; do
    echo "[anty] $(basename "$out") start, próba $attempt ($(date '+%H:%M:%S'))"
    if PYTHONPATH=src "$GPU_PY" -m doc2query.cli train dpo \
      --config "$CONFIG" --plan "$plan" --packaged-dir "$packaged" \
      --adapter "$ADAPTER" --output-dir "$out" --arm dpo \
      --checkpoint-every 25 "${extra[@]}" "$@"; then
      echo "[anty] $(basename "$out") koniec ($(date '+%H:%M:%S'))"
      return 0
    fi
    echo "[anty] $(basename "$out") padło, wznowię za 120 s" >&2
    sleep 120
    attempt=$((attempt + 1))
  done
  return 1
}

precompute() { # precompute <handoff> <plan>
  local handoff="$1" plan="$2"
  [ -f "$handoff/reference_logprobs/run_summary.json" ] && return 0
  echo "[anty] precompute $(basename "$handoff") ($(date '+%H:%M:%S'))"
  PYTHONPATH=scripts:src "$GPU_PY" scripts/precompute_task07_reference_logprobs.py \
    --task06-manifest "$handoff/packaged/manifest.json" \
    --preference-train "$handoff/packaged/preference_train.jsonl" \
    --preference-dev "$handoff/packaged/preference_dev.jsonl" \
    --continued-sft-train "$handoff/packaged/continued_sft_train.jsonl" \
    --continued-sft-dev "$handoff/packaged/continued_sft_dev.jsonl" \
    --weighted-sft-train "$handoff/packaged/weighted_sft_train.jsonl" \
    --weighted-sft-dev "$handoff/packaged/weighted_sft_dev.jsonl" \
    --plan "$plan" --output-dir "$handoff/reference_logprobs"
}

# --- 1. EQ102: kontrola konfundenta kroków ------------------------------------
train runs/T07-V3-DPO-EQ102-S42 "$BOTTOM/plan/dpo_plan.json" "$BOTTOM/packaged" \
  "$BOTTOM/reference_logprobs/reference_logprobs.manifest.json" --max-steps 102 || true

# --- 2. BETA02: nowy plan (beta 0,2) + precompute + trening -------------------
BETA_PLAN_CFG="configs/train/task07_dpo_plan_defect_beta02.yaml"
BETA_PLAN="$DEFECT/plan_beta02/dpo_plan.json"
if [ ! -f "$BETA_PLAN_CFG" ]; then
  uv run python scripts/build_task07_dpo_plan_config.py \
    --plan-id task07-dpo-plan-defect-beta02-s42 --beta 0.2 \
    --token-length-manifest "$DEFECT/token_lengths/token_lengths.manifest.json" \
    --token-length-records "$DEFECT/token_lengths/token_lengths.jsonl" \
    --weight-manifest "$DEFECT/inputs/weight_manifest.json" \
    --output "$BETA_PLAN_CFG"
fi
if [ ! -f "$BETA_PLAN" ]; then
  mkdir -p "$(dirname "$BETA_PLAN")"
  uv run python scripts/plan_dpo_controls.py \
    --task06-manifest "$DEFECT/packaged/manifest.json" \
    --preference-train "$DEFECT/packaged/preference_train.jsonl" \
    --preference-dev "$DEFECT/packaged/preference_dev.jsonl" \
    --continued-sft-train "$DEFECT/packaged/continued_sft_train.jsonl" \
    --continued-sft-dev "$DEFECT/packaged/continued_sft_dev.jsonl" \
    --weighted-sft-train "$DEFECT/packaged/weighted_sft_train.jsonl" \
    --weighted-sft-dev "$DEFECT/packaged/weighted_sft_dev.jsonl" \
    --config "$BETA_PLAN_CFG" \
    --token-length-manifest "$DEFECT/token_lengths/token_lengths.manifest.json" \
    --token-length-records "$DEFECT/token_lengths/token_lengths.jsonl" \
    --output "$BETA_PLAN" > "$(dirname "$BETA_PLAN")/plan_stdout.json"
fi
if [ ! -f "$DEFECT/reference_logprobs_beta02/run_summary.json" ]; then
  echo "[anty] precompute beta02 ($(date '+%H:%M:%S'))"
  PYTHONPATH=scripts:src "$GPU_PY" scripts/precompute_task07_reference_logprobs.py \
    --task06-manifest "$DEFECT/packaged/manifest.json" \
    --preference-train "$DEFECT/packaged/preference_train.jsonl" \
    --preference-dev "$DEFECT/packaged/preference_dev.jsonl" \
    --continued-sft-train "$DEFECT/packaged/continued_sft_train.jsonl" \
    --continued-sft-dev "$DEFECT/packaged/continued_sft_dev.jsonl" \
    --weighted-sft-train "$DEFECT/packaged/weighted_sft_train.jsonl" \
    --weighted-sft-dev "$DEFECT/packaged/weighted_sft_dev.jsonl" \
    --plan "$BETA_PLAN" --output-dir "$DEFECT/reference_logprobs_beta02"
fi
train runs/T07-DEF-DPO-BETA02-S42 "$BETA_PLAN" "$DEFECT/packaged" \
  "$DEFECT/reference_logprobs_beta02/reference_logprobs.manifest.json" || true

# --- 3. RPO: istniejący plan defect + regularyzator NLL λ=1,0 -----------------
train runs/T07-DEF-DPO-RPO-S42 "$DEFECT/plan/dpo_plan.json" "$DEFECT/packaged" \
  "$DEFECT/reference_logprobs/reference_logprobs.manifest.json" \
  --nll-coefficient 1.0 || true

# --- 4. DIVCH: handoff z różnicowanym chosen + plan + precompute + trening ----
if [ ! -f "$DIVCH/packaged/manifest.json" ]; then
  uv run python scripts/build_task07_handoff_v3.py \
    --pairs artifacts/task06/defect_diverse_v1/pairs_trainable.jsonl \
    --handoff-dir "$DIVCH/inputs" --packaged-dir "$DIVCH/packaged"
fi
if [ ! -f "$DIVCH/token_lengths/token_lengths.manifest.json" ]; then
  uv run python scripts/measure_task07_token_lengths.py \
    --preference-train "$DIVCH/inputs/preference_train.jsonl" \
    --preference-dev "$DIVCH/inputs/preference_dev.jsonl" \
    --packaged-dir "$DIVCH/packaged" --output-dir "$DIVCH/token_lengths"
fi
DIVCH_PLAN_CFG="configs/train/task07_dpo_plan_defect_diverse.yaml"
DIVCH_PLAN="$DIVCH/plan/dpo_plan.json"
if [ ! -f "$DIVCH_PLAN_CFG" ]; then
  uv run python scripts/build_task07_dpo_plan_config.py \
    --plan-id task07-dpo-plan-defect-diverse-s42 \
    --token-length-manifest "$DIVCH/token_lengths/token_lengths.manifest.json" \
    --token-length-records "$DIVCH/token_lengths/token_lengths.jsonl" \
    --weight-manifest "$DIVCH/inputs/weight_manifest.json" \
    --output "$DIVCH_PLAN_CFG"
fi
if [ ! -f "$DIVCH_PLAN" ]; then
  mkdir -p "$(dirname "$DIVCH_PLAN")"
  uv run python scripts/plan_dpo_controls.py \
    --task06-manifest "$DIVCH/packaged/manifest.json" \
    --preference-train "$DIVCH/packaged/preference_train.jsonl" \
    --preference-dev "$DIVCH/packaged/preference_dev.jsonl" \
    --continued-sft-train "$DIVCH/packaged/continued_sft_train.jsonl" \
    --continued-sft-dev "$DIVCH/packaged/continued_sft_dev.jsonl" \
    --weighted-sft-train "$DIVCH/packaged/weighted_sft_train.jsonl" \
    --weighted-sft-dev "$DIVCH/packaged/weighted_sft_dev.jsonl" \
    --config "$DIVCH_PLAN_CFG" \
    --token-length-manifest "$DIVCH/token_lengths/token_lengths.manifest.json" \
    --token-length-records "$DIVCH/token_lengths/token_lengths.jsonl" \
    --output "$DIVCH_PLAN" > "$(dirname "$DIVCH_PLAN")/plan_stdout.json"
fi
precompute "$DIVCH" "$DIVCH_PLAN"
train runs/T07-DEF-DPO-DIVCH-S42 "$DIVCH_PLAN" "$DIVCH/packaged" \
  "$DIVCH/reference_logprobs/reference_logprobs.manifest.json" || true

# --- 5. Pomiar różnorodności nowych ramion tym samym protokołem ---------------
for entry in \
  beta02:runs/T07-DEF-DPO-BETA02-S42/adapter \
  rpo:runs/T07-DEF-DPO-RPO-S42/adapter \
  divch:runs/T07-DEF-DPO-DIVCH-S42/adapter; do
  name="${entry%%:*}"
  adapter="${entry#*:}"
  out="runs/task07_probe_gen_v1/$name"
  [ -f "$out/generated.summary.json" ] && { echo "[anty] generacja $name gotowa"; continue; }
  [ -d "$adapter" ] || { echo "[anty] brak adaptera $adapter, pomijam $name" >&2; continue; }
  echo "[anty] generacja $name ($(date '+%H:%M:%S'))"
  rm -f "$out/generated.jsonl"
  mkdir -p "$out"
  PYTHONPATH=src "$GPU_PY" -m doc2query.cli generate \
    --config "$GEN_CONFIG" --adapter "$adapter" --output "$out/generated.jsonl" || true
done

echo "[anty] kolejka zakończona ($(date '+%H:%M:%S'))"
