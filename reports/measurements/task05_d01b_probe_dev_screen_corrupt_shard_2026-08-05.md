# D01b probe dev-screen: corrupt W05 corpus shard

Date: 2026-08-05

## Completed work and failure

The batch-4 W05 arm completed all 500 training steps and saved its model,
negative audit and `train_summary.json`. The summary records batch size 4 and
the frozen 1,152,000-token comparison budget. No W05 `result.json` exists yet,
the hybrid arm has not started, and no comparison was run.

An earlier machine shutdown interrupted W05 corpus encoding after five of 100
shards and left `chunk-00005.pt` as a zero-byte file. The operator's next
`run-all`, from 15:17:55 to 18:15:30 Europe/Warsaw, reused the completed
training and five apparently present shards, then encoded shards 7–100. The
cache completion check considered filename presence sufficient. Loading the
finished index therefore failed on the empty shard with:

```text
RuntimeError: mmap can only be used with files saved with
torch.save(_use_new_zipfile_serialization=True)
```

Artifact validation found exactly 99 valid shards and one invalid shard with
zero-based index 5. The cache manifest reports 2,404,263 corpus rows and 100
shards. This is a technical resume failure, not an experimental result.

## Repair and resume boundary

Shard reuse now requires a non-empty, loadable floating-point rank-2 tensor
with the exact expected row count. A missing or invalid shard is re-encoded
only when the pinned model and corpus texts are available. Shard writes now
flush and `fsync` the temporary file, replace the destination atomically and
`fsync` the containing directory.

A regression test creates a zero-byte middle shard and verifies that only its
two fixture rows are re-encoded while the other shards are reused. The full CPU
suite passed with 260 tests; repository-wide Ruff, changed-file formatting and
targeted mypy also pass. The real D01b CPU preflight completed with `rc=0` at
19:14:36 Europe/Warsaw.

The next `run-all` may reuse the completed W05 training and its 99 valid corpus
shards. It must repair only shard 6/100 before continuing W05 evaluation, then
run the untouched hybrid arm and the preregistered comparison. Estimated
remaining wall time is approximately 4–5 hours on this machine.

No final test was opened. `dev_confirm` and 4.5B remain unauthorized;
`final_tests_used=[]`.
