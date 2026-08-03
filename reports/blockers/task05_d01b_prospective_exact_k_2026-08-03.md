# Task 05 D01b prospective 1.5B — exact-K generation blocker

Date: 2026-08-03

Status: `STOPPED_FAIL_CLOSED`

Scope: the preregistered unseen `dev_intrinsic` cohort only;
`final_tests_used=[]`.

## Outcome

The already started `generate` phase ended with exit code 1 before scoring.
The controlled arm completed all 2000 passage groups with exactly four
queries per group (8000 queries, zero exhausted groups). The W05 baseline arm
produced 7547 queries and exhausted at least one slot in 320 of 2000 groups
after the preregistered maximum of three attempts per slot. Its summary
records 2510 duplicate outputs and one invalid output.

The runner's exact-K validator rejected the baseline summary as designed.
No prospective query text was manually inspected while preparing this report,
no judge or corpus scoring was started, the selector was not applied, no gate
was evaluated and no probe input was materialized.

The durable status and logs remain under the ignored runtime paths:

- `reports/measurements/task05_d01b_prospective_1_5b_v1/status.json`;
- `logs/task05_d01b_prospective_1_5b_v1/generate.log`;
- `artifacts/task05/d01b_prospective_1_5b_v1/generation/`.

Their identities retain `final_tests_used=[]`. Existing generation artifacts
must not be deleted or rewritten.

## Interpretation

This is an operational contract failure, not a generator-quality result. The
frozen cohort requires exactly four baseline and four controlled queries for
every one of the same 2000 groups, while the frozen baseline retry budget did
not produce four distinct valid queries for all groups. Scoring a smaller
post-generation cohort, padding with duplicates, increasing retries or
changing decoding would amend the preregistered protocol after generation and
is therefore forbidden for this validation.

Re-running
`bash scripts/run_task05_d01b_prospective_1_5b.sh generate` cannot repair the
contract: the identity-bound journal is already complete and deterministically
reconstructs the same incomplete summary before the exact-K check fails.

## Required next decision

The current prospective validation remains stopped. Any retry requires a new
ADR and config, frozen before generation on a still-unseen development cohort,
with an explicit exact-K completion policy. Such a retry must preserve selector
commit `2164822`, all selector weights and thresholds, reserved-shadow
independence, the preregistered guardrails and `final_tests_used=[]`. It must
not use the failed cohort's quality outputs for tuning.

The completed run took about 31.0 minutes for W05 and 19.7 minutes for D01
(about 50.7 minutes total generation time on the recorded GPU trajectory).
No further GPU phase is authorized by this blocker report.

## Verification

- `ruff check .`: passed;
- `ruff format --check` on the five new/changed Python API files: passed;
- `pytest`: 251 passed, 16 warnings;
- targeted `mypy` on the five new/changed Python API files: passed.

Global `ruff format --check .` still reports 12 pre-existing unrelated files;
they were not reformatted in this task.
