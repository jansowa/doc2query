#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

export D01B_PROSPECTIVE_CONTRACT=configs/evaluation/d01b_prospective_1_5b_v2.yaml
export D01B_PROSPECTIVE_ARTIFACT_ROOT=artifacts/task05/d01b_prospective_1_5b_v2
export D01B_PROSPECTIVE_MEASUREMENT_ROOT=reports/measurements/task05/d01b_prospective_1_5b_v2
export D01B_PROSPECTIVE_LOG_ROOT=logs/task05/d01b_prospective_1_5b_v2
export D01B_PROSPECTIVE_BASE_CONFIG=configs/experiments/d01b_prospective_v2_w05_1_5b_s42.yaml
export D01B_PROSPECTIVE_CONTROLLED_CONFIG=configs/experiments/d01b_prospective_v2_d01_1_5b_s42.yaml
export D01B_PROSPECTIVE_RUNNER_CONTRACT=task05-d01b-prospective-runner-v2

exec "$ROOT/scripts/run_task05_d01b_prospective_1_5b.sh" "$@"
