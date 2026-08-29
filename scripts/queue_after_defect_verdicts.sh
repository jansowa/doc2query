#!/usr/bin/env bash
# Kolejka po przyniesieniu verdictów z serwera: składanie → bramki → przygotowanie.
#
#   scripts/queue_after_defect_verdicts.sh            # czeka na verdicts i rusza
#
# Czeka, aż pojawi się journal verdictów, po czym bez pytania wykonuje wszystko,
# co NIE jest treningiem: składa pary wg ADR (+ amendment), liczy audyt
# anty-skrótowy, buduje handoff Task 07, długości tokenów, plan i precompute
# logprobów referencji. Zatrzymuje się PRZED treningiem — ADR §7.4 wymaga
# osobnej autoryzacji właściciela dla nowej kohorty.
#
# Każdy etap pilnowany istnieniem wyjścia, więc restart nie powtarza pracy.
set -euo pipefail

cd "$(dirname "$0")/.."
[ -f pyproject.toml ] || { echo "nie jestem w katalogu repozytorium" >&2; exit 2; }

VERDICTS="${1:-artifacts/task06/defect_pipeline_v1/verdicts/verdicts.journal.jsonl}"
PAIRS_DIR="artifacts/task06/defect_pairs_v1"
H="artifacts/task07/handoff_defect_v1"
CONFIG="configs/experiments/d01_4_5b_style_50k_s42.yaml"
ADAPTER="runs/D01-4.5B-STYLE-50K-S42/adapter"
PLAN_CFG="configs/train/task07_dpo_plan_defect_v1.yaml"
PLAN="$H/plan/dpo_plan.json"
GPU_PY=".venv-gpu/bin/python"

echo "[kolejka] czekam na $VERDICTS ($(date '+%H:%M:%S'))"
until [ -f "$VERDICTS" ]; do sleep 60; done
echo "[kolejka] verdicty są: $(wc -l < "$VERDICTS") wpisów"

if [ ! -f "$PAIRS_DIR/pairs.jsonl" ]; then
  echo "[kolejka] składanie par + audyt anty-skrótowy ($(date '+%H:%M:%S'))"
  uv run python scripts/assemble_defect_pairs.py --journal "$VERDICTS" --output-dir "$PAIRS_DIR"
fi

# ADR §7.2: klasa z AUC audytu anty-skrótowego powyżej progu NIE wchodzi do
# treningu bez amendmentu. Kohortę trenowalną buduje osobny filtr, a pełny
# zbiór par zostaje nietknięty jako artefakt pomiarowy.
if [ ! -f "$PAIRS_DIR/pairs_trainable.jsonl" ]; then
  echo "[kolejka] filtr klas zablokowanych przez audyt ($(date '+%H:%M:%S'))"
  uv run python scripts/filter_defect_pairs_by_audit.py --pairs-dir "$PAIRS_DIR"
fi

pairs=$(wc -l < "$PAIRS_DIR/pairs_trainable.jsonl")
echo "[kolejka] par trenowalnych: $pairs"
if [ "$pairs" -lt 200 ]; then
  echo "[kolejka] za mało par na sensowny handoff — zatrzymuję się, obejrzyj summary.json" >&2
  exit 0
fi

if [ ! -f "$H/packaged/manifest.json" ]; then
  echo "[kolejka] handoff Task 07 ($(date '+%H:%M:%S'))"
  uv run python scripts/build_task07_handoff_v3.py \
    --pairs "$PAIRS_DIR/pairs_trainable.jsonl" --handoff-dir "$H/inputs" --packaged-dir "$H/packaged"
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
    --plan-id task07-dpo-plan-defect-v1-s42 \
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
  mkdir -p "$H/plan"
  uv run python scripts/plan_dpo_controls.py "${dataset_args[@]}" \
    --config "$PLAN_CFG" \
    --token-length-manifest "$H/token_lengths/token_lengths.manifest.json" \
    --token-length-records "$H/token_lengths/token_lengths.jsonl" \
    --output "$PLAN" > "$H/plan/plan_stdout.json"
fi

if [ ! -f "$H/reference_logprobs/run_summary.json" ] && [ -x "$GPU_PY" ]; then
  # Na 8 GB dwa zadania GPU się nie mieszczą: czekamy, aż zwolni je inna kolejka.
  while pgrep -f "doc2query.cli (evaluate generator|train dpo)" > /dev/null; do
    echo "[kolejka] GPU zajęte przez inny run, czekam ($(date '+%H:%M:%S'))"
    sleep 120
  done
  echo "[kolejka] precompute logprobów referencji, GPU ($(date '+%H:%M:%S'))"
  PYTHONPATH=scripts:src "$GPU_PY" scripts/precompute_task07_reference_logprobs.py \
    "${dataset_args[@]}" --plan "$PLAN" --output-dir "$H/reference_logprobs"
fi

cat <<'INFO'

[kolejka] gotowe wszystko, co nie jest treningiem.

Trening nowej kohorty wymaga OSOBNEJ autoryzacji właściciela (ADR §7.4).
Przed nią obejrzyj:
  artifacts/task06/defect_pairs_v1/summary.json        — pass-rate per klasa
  artifacts/task06/defect_pairs_v1/shortcut_audit.json — AUC (>0,80 blokuje klasę)
Po autoryzacji trzy ramiona idą tak jak dla poprzednich kohort, wskazując
--plan artifacts/task07/handoff_defect_v1/plan/dpo_plan.json
INFO
