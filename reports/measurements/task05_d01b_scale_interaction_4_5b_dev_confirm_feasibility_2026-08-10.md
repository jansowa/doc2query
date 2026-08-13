# Task 05: D01b 4.5B dev-confirm feasibility

Date: 2026-08-10

## Outcome

The prospective development confirm is blocked before preregistration and
before any expensive run. The complete ID-only audit found exactly 591 legal,
previously unseen eligible development records. They are disjoint from
`dev_intrinsic_rank10`, prospective cohorts v1/v2/v3, and both scale-pilot
cohorts. No raw ID or record text was emitted or manually inspected; the audit
logic accessed only IDs and hard-negative counts. No final-test manifest,
record, prediction, or metric was opened.

The reserve is fingerprinted in both frozen selection order and sorted ID
order in the machine-readable evidence. Eligibility used only ID provenance
and the already frozen requirement of at least five inherited hard negatives;
it did not use quality outcomes or select favorable records.

## Planning-only sensitivity

The pilot result was used only to estimate sensitivity. The calculation
recovers a paired-query standard deviation of `0.21810233528521697` from the
aggregate pilot difference and its two-sided 95% CI. With 591 queries, a
two-sided 97.5% interval has projected half-width `0.020108814663528603`.
Centered on the pilot effect, the projected interval is
`[0.0006291054152610179, 0.04084673474231822]`, so it cannot provide a
meaningful confirm against the unchanged `+0.01` practical-effect threshold.

Even the deliberately optimistic and methodologically unjustified assumption
that all three seed estimates are independent divides the half-width by
`sqrt(3)` but produces a lower bound of only `0.009128090519717087`. The
one-seed pilot cannot estimate seed variance. A normal-approximation planning
calculation requires about 3922 untouched evaluation queries for 80% power or
5121 for 90% power at the pilot point effect. These are planning estimates,
not new gates or guarantees.

## Boundary

No confirm config was frozen and no runner or operator command was created,
because doing so would imply a methodologically executable stage. The pilot
must not be rerun. Resampling, cross-validation, seed replication, or reuse of
previously evaluated query IDs cannot manufacture independent confirmatory
queries. Final tests remain closed.

The state remains:

- `selection_claim=null`;
- `retained_for_finalist_freeze=false`;
- `four_point_five_b_full_authorized=false`;
- `expensive_run_authorized=false`;
- `final_tests_used=[]`.

Machine-readable evidence:
`reports/measurements/task05/d01b_scale_interaction_4_5b_dev_confirm_feasibility_v1.json`.

## Validation

- full CPU suite: `447 passed`, 16 warnings;
- focused feasibility tests: `3 passed`;
- Ruff: passed for the repository;
- mypy: `src` and all three changed Python files passed;
- repository-wide `mypy src tests` was executed and still reports 19 existing
  test-typing errors in six unchanged test files;
- `git diff --check`: passed.
