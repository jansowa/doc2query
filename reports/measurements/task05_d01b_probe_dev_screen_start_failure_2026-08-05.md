# D01b probe dev-screen: first-start failure and input repair

Date: 2026-08-05

## Observed failure

The first operator invocation
`bash scripts/run_task05_d01b_probe_dev_screen.sh run-all` ran from
09:25:28 to 09:25:44 Europe/Warsaw and exited with `rc=1`. The preregistered
preflight passed and the frozen primary judge loaded, but probe pair
preparation produced an empty training set. The failure occurred before the
first optimizer step. No checkpoint, `result.json`, retrieval output or probe
comparison was produced.

The runner then incorrectly continued to the second arm and comparison. They
failed respectively with the same empty-training-set error and a missing
`result.json`. These follow-on errors are not probe results.

## Root cause and repair

The existing frozen probe loader accepts synthetic records only when
`mode="deterministic"` and `candidate_index=0`. The equal-budget materializer
had supplied the other compatibility fields but omitted these two, so all
7936 rows in each arm were ignored by the loader.

The materializer now writes both fields. The D01b preflight verifies them
fail-closed, and its test verifies that the materialized fixture is visible to
the real probe loader. The shell runner now executes its multi-command phase
with errexit enabled, so it stops at the first failing arm and does not attempt
later arms or comparison.

Only CPU materialization was repeated. Generation, generator scoring,
selection and probe training were not repeated. The cohorts and frozen budget
remain unchanged: 7936 pairs, 1984 unique positive documents and exact K=4 in
each arm; the dev-screen prefix remains 1984 pairs and 496 documents.

Repaired artifact SHA-256 values:

- manifest: `477086cfb6c43fed1a3edaaaeb975f2b8f559a6bfeebfa25df3c01a822da6409`;
- W05: `c0c2bdc8bf1d99772bdc760dcce8225e56f44bbc785b759fa7dd2ac8752260b6`;
- selected hybrid: `ac9e7b3b76822fdd1ca2000264609b3db0ae74cb474d2e13434ba7e27c2a626d`.

The post-repair CPU preflight completed with `rc=0` at 09:32:48. It verified
the exact common prefix, pinned development panel of 6598 natural queries,
zero training/evaluation source-ID overlap and the updated input hashes.
Direct loader validation observed 7936 usable synthetic rows in each arm.
The full CPU suite passed with 259 tests; repository-wide Ruff, formatting of
the changed Python files and targeted mypy checks passed. The repository-wide
mypy command still reports 19 pre-existing errors in six unrelated test files.

## Authorization and next action

The accepted preregistration still authorizes only the 1.5B dev-only screen.
The exact retry command is:

```bash
bash scripts/run_task05_d01b_probe_dev_screen.sh run-all
```

Expected wall time remains approximately 7–8 hours on this machine. There is
no checkpoint to resume. `dev_confirm`, 4.5B and all final tests remain
forbidden. `final_tests_used=[]`.
