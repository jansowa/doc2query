# D01b probe dev-screen: hybrid encode batch 64 interruption

Date: 2026-08-05

## Preserved state

After the corrupt-shard repair, the W05 arm completed its natural-development
evaluation. It has 100/100 valid corpus embedding shards, a complete cache
manifest, query and retrieval journals, and `result.json`. W05 training and
evaluation must not be repeated.

The hybrid arm subsequently completed all 500 batch-4 training steps and saved
its model, negative audit and `train_summary.json`. Its `encode_corpus` phase
used execution batch size 64 and persisted 42/100 valid shards before the
machine shut down. The shards form the exact continuous zero-based prefix
0–41, contain no zero-byte or invalid files and use the manifest's frozen
`chunk_size=24064`. Hybrid has no `result.json`; comparison has not run.

## Execution-only amendment

At the project owner's explicit instruction, the next resume uses
`evaluation_encode_batch_size=32`. Training batch size remains 4 and retrieval
query batch size remains 512. The smaller encode batch does not enter the
model, dataset or metric identity and does not change shard boundaries.

The cache resume test now explicitly creates shards with one encode batch and
repairs a missing shard with a different smaller batch. It verifies that only
the missing shard's rows are encoded and all existing shards remain reusable.
The real artifact audit verified 100 valid W05 shards and 42 valid hybrid
shards before this change.

The amended real CPU preflight completed with `rc=0` from 21:41:28 to
21:41:30 Europe/Warsaw. The full CPU suite passed with 260 tests;
repository-wide Ruff, changed-file formatting, shell syntax and targeted mypy
also pass. No GPU computation was started during validation.

After CPU validation, the next `run-all` may reuse the complete W05 result,
complete hybrid training and hybrid shards 1–42. It should start missing corpus
encoding at shard 43/100, then finish hybrid evaluation and the preregistered
comparison. Estimated remaining time is approximately 4–5 hours.

No final test was opened. `dev_confirm` and 4.5B remain unauthorized;
`final_tests_used=[]`.
