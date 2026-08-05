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
  SHA-256 `a409c7e05085b76cf10c40a1897e5070dd9841590c5d7e81c16de840a33ac49c`;
- `artifacts/task05/d01b_prospective_1_5b_v3/probe_inputs/selected_hybrid.jsonl`,
  SHA-256 `333917369ac76e8d0e8f760f402daf66b1f42f460a1af7c5f004cdb6d8acdda3`;
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

The v3 ADR explicitly authorizes input materialization only. It does not define
or authorize the costly two-arm probe run, its evaluation subset, seed matrix,
comparison/CI report, runtime command or promotion decision. Therefore no
training command is declared and no probe training was started. A prospective
training/evaluation decision must be frozen before such a command can be
prepared; final tests and 4.5B remain forbidden.
