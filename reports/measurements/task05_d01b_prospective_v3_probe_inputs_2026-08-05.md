# Task 05 D01b prospective v3 — equal-budget probe inputs

Date: 2026-08-05

Status: `MATERIALIZED_AND_CPU_VALIDATED`

## Authorization and source

Materialization used only the completed prospective v3 report with decision
`authorize_equal_budget_probe_inputs`. The frozen best-four-of-eight selector,
primary score rows and `configs/evaluation/probe_v1.yaml` were not changed or
recomputed. Generation, scoring and `select-compare` were not repeated.

The materializer requires all seven preregistered gates, the report and input
SHA-256 identities, `probe_materialization_authorized=true`, HN0+filter/drop,
the pinned Task 02 calibration and `final_tests_used=[]`.

## Materialized common budget

The two arms are:

- observed W05 baseline: 7936 pairs;
- prospectively selected hybrid: 7936 pairs.

Each arm contains 1984 unique positive passages and exactly four queries per
passage. HN0+filter/drop at the pinned threshold `8.617486953735352` removed
nine whole passage groups from the shared cohort because at least one arm lost
all negatives for a pair. Eight selected-hybrid groups and nine W05 groups
were individually ineligible; their union contained nine groups. The existing
P-05 one-group-per-document rule then removed seven repeated-positive groups.
Both decisions were applied symmetrically, yielding a common equal budget.

Artifacts:

- `artifacts/task05/d01b_prospective_1_5b_v3/probe_inputs/w05_baseline.jsonl`,
  SHA-256 `c0c2bdc8bf1d99772bdc760dcce8225e56f44bbc785b759fa7dd2ac8752260b6`;
- `artifacts/task05/d01b_prospective_1_5b_v3/probe_inputs/selected_hybrid.jsonl`,
  SHA-256 `ac9e7b3b76822fdd1ca2000264609b3db0ae74cb474d2e13434ba7e27c2a626d`;
- `artifacts/task05/d01b_prospective_1_5b_v3/probe_inputs/manifest.json`.

The manifest records `training_started=false`, `training_authorized=false`,
`four_point_five_b_authorized=false` and `final_tests_used=[]`.

## CPU validation

- 7936 unique `pair_id` values in each arm;
- zero pairs without a retained hard negative;
- 1984 positive passage IDs in each arm, all with K=4;
- manifest SHA-256 values match the files;
- targeted pytest: 6 passed;
- full pytest: 257 passed, 16 third-party warnings;
- Ruff check: passed;
- Ruff format check for changed Python files: passed;
- targeted mypy: passed;
- shell syntax and CLI help: passed.

The repository-wide format check still reports pre-existing formatting drift
in files outside this change; those files were not modified.

## Remaining gate

At the time of this materialization the v3 ADR authorized inputs only. A later
prospective decision, `task05_d01b_probe_dev_screen_v1.md`, has now frozen and
authorized only the two-arm `dev_screen`; its CPU preflight passed. Probe
training still has not started. `dev_confirm`, final tests and 4.5B remain
forbidden.
