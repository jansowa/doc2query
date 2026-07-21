# Task 04 P-05 dev-screen preflight

Date: 2026-07-21

Status: `PREPARING_ELIGIBILITY`; no probe or final test executed.

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

CPU preparation was started against only:

- `data/processed/v1/train.parquet`;
- the existing W05 train generations;
- the existing P-03 common-cohort IDs;
- `configs/evaluation/probe_v1.yaml` and the CPU primary-judge config.

At commit time it had not produced `eligibility_audit.json`, so no
materialization, probe training, dev measurement, promotion, or generator
selection is claimed. `final_tests_used=[]`; S00 and S07 remain
`required_unexecuted`.

The tentative 9,968-pair budget is not authoritative. The preparator writes
`budget.measured.json` from the actual dual-source eligible intersection; only
that measured budget may be passed to the materializer and planner.
