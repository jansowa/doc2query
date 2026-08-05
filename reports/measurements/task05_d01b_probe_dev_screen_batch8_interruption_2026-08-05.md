# D01b probe dev-screen: batch-8 machine interruption

Date: 2026-08-05

## Observed state

The repaired `run-all` invocation started at 10:33:13 Europe/Warsaw. Its
preflight passed with the repaired input hashes and `final_tests_used=[]`.
W05 negative filtering completed for all 1984 prefix pairs, after which probe
training started with batch size 8 and the preregistered 250-step budget.

The machine shut down without a clean phase exit. Consequently, the runner
status still contains the earlier failed start and the new log has no `END`
record. The last progress record is step 84/250. The last complete rolling
checkpoint is step 50 at:

`runs/task05_d01b_probe_dev_screen_v1/D01B-PROBE-W05-DEV-SCREEN-S42/training_checkpoint.pt`

It is 1,488,858,534 bytes and was written at 10:36:17 Europe/Warsaw. There is
no W05 `result.json`, no hybrid training artifact and no comparison result.
Printed progress after step 50 is not recoverable completed work.

## Authorized restart

The project owner attributed the interruption to excessive GPU load and
explicitly instructed the training batch size to be halved. The prospective
amendment
`task05_d01b_probe_dev_screen_batch4_amendment_2026-08-05.md` therefore freezes
batch size 4 and 500 steps for both arms. This preserves 2000 consumed examples
and the original 1,152,000-token budget per arm.

Because batch size and step count are part of the checkpoint identity, the
batch-8 checkpoint is retained but must not be resumed. The amended execution
uses new `v2_batch4` run, measurement and log roots and distinct `B4` arm IDs.
All other training and evaluation settings remain frozen.

The amended CPU preflight completed with `rc=0` from 10:43:44 to 10:43:49
Europe/Warsaw. It verified both 7936-row inputs, the common 1984-pair prefix,
496 passages per prefix, the 6598-query natural dev panel, zero source-ID
overlap, the unchanged 1,152,000-token budget and runtime recipe fingerprint
`40dcf99747c3a50f31dfca18127c74a2a398353bb99022a167bd81f921b25726`.
No training artifact exists in the new `v2_batch4` run root.
The full CPU suite passed with 259 tests; repository-wide Ruff, changed-file
format checks, shell syntax validation and targeted mypy checks also passed.

No final test was opened. `dev_confirm` and 4.5B remain unauthorized, and
`final_tests_used=[]`.
