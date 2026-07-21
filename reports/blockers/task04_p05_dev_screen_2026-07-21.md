# Task 04 P-05 dev-screen preflight

Date: 2026-07-21

Status: `READY_FOR_USER_GPU_RUN`; no probe or final test executed.

The CPU-only campaign audit passed with nine required arms complete and I02,
I04 and I05 deferred under the pinned early-stop ADR. P-04/P-05 planning was
run in plan-only mode and emitted zero execution commands because the three
materialized common-cohort inputs and their manifest did not yet exist.

Review of the imported materializer found that it labeled its outputs
HN0+filter/drop without proving that both natural and W05 queries had passed
the frozen filter. The materializer now fails closed unless both input
fingerprint manifests declare the same recomputed post-filter eligible-ID
SHA-256. `scripts/prepare_p05_eligible_inputs.py` was added to create that
proof with the pinned dev-only calibration and primary judge.

GPU eligibility preparation completed against only:

- `data/processed/v1/train.parquet`;
- the existing W05 train generations;
- the existing P-03 common-cohort IDs;
- `configs/evaluation/probe_v1.yaml` and the CPU primary-judge config.

Natural filtering retained 9,965 of the 9,973 P-03 common examples and W05
filtering retained all 9,973. After intersection, deterministic K=1 selection
removed 12 repeated-document pairs and four divisibility-tail records. The
materialized comparison cohort contains exactly 9,944 unique pair IDs and
9,944 unique document IDs in each arm. Its fingerprint is
`d89b799a...df67b5c`; the full budget is 4,608,000 padded tokens, 9,944 pairs,
9,944 passages and K=1.

The mixed arm is exactly 50/50 both in `dev_screen` (1,243 natural + 1,243
W05) and in the full prefix (4,972 + 4,972). The plan-only planner passes with
no blockers, emits only `dev_intrinsic_rank10` for development evaluation and
retains `final_tests_used=[]`.

The attempted gold probe before the machine restart produced no model or
result. A later attempt was deliberately interrupted after CUDA reported a
missing deterministic cuBLAS workspace setting; it is not a measurement.
`scripts/run_p05_dev_screen.sh` now pins `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
shows progress for filtering, training, corpus encoding and query evaluation,
and runs only the three seed-42 25% screens. It is safe to invoke again after
an interruption: validated completed arms are skipped, a completed training
stage is identity-checked and reused while evaluation restarts, and output
interrupted before a complete training summary is moved recoverably under
`runs/task04_p05_dev_screen/interrupted/` before only that arm is restarted.
Logs are appended across invocations. No materialization-time result, probe
measurement, promotion, or generator selection is claimed yet. S00 and S07
remain `required_unexecuted`.

The older tentative 9,968-pair budget and the intermediate 9,960-pair
eligibility budget are not comparison-authoritative. Only
`reports/measurements/task04_p05_dev_screen/budget.k1.json` may be used by the
runner and planner.
