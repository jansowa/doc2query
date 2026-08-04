# Task 05 D01b prospective 1.5B v2 — exact-K generation blocker

Date: 2026-08-04

Status: `STOPPED_FAIL_CLOSED`

Scope: prospective v2 development cohort only; `final_tests_used=[]`.

## Outcome

The `generate` phase ran from 2026-08-03 20:43:24+02:00 to
21:57:39+02:00 and ended with exit code 1 before scoring. Controlled completed
8000/8000 queries with no exhausted group. W05 completed 7912/8000 queries:
1938 groups had four distinct outputs, 36 had three and 26 had two. Thus 62
of 2000 groups lacked exact K and 88 slots were missing after the frozen limit
of 16 attempts per slot.

W05 used 12,785 attempts and recorded 4,873 duplicate outputs with zero
invalid outputs. Controlled used 8,050 attempts, recorded 50 duplicates and
zero invalid outputs. No query text was manually inspected for this report.
No judge/corpus scoring, selector, comparison, probe materialization, 4.5B or
final test was run.

## Decision

Do not retry v2 and do not increase the distinct-output retry ceiling. Repeated
novelty conditioning changes the effective W05 distribution, increases cost
and still does not guarantee exact K. The owner accepted a new prospective v3
contract in which four independent W05 slots preserve valid duplicates as a
measured property of the baseline. Only invalid outputs are retried. Controlled
generation, the frozen selector and all quality thresholds remain unchanged.

Existing v2 artifacts and journals remain immutable under:

- `artifacts/task05/d01b_prospective_1_5b_v2/`;
- `reports/measurements/task05/d01b_prospective_1_5b_v2/`;
- `logs/task05/d01b_prospective_1_5b_v2/`.

All retain `final_tests_used=[]`.
