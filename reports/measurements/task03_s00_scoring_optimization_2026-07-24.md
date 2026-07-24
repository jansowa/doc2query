# S00 scoring optimization benchmark — 2026-07-24

Status: `IMPLEMENTED_BENCHMARKED_NOT_FULLY_SCORED`

Scope: frozen development cohort only; `final_tests_used=[]`.

## Problem and implementation

The first S00 scoring attempt was interrupted after roughly ten hours. It had
no durable scored rows because the evaluator buffered all 25,000 outputs in
memory and wrote `per_generation.jsonl` only at the end. The old loop also
called each reranker separately for every output and ran separate BM25 SQL
paths for top-k, effective-candidate count and the known-positive score.

The optimized evaluator keeps the Harness v1.1 metrics and frozen P-04
contract unchanged. It now:

- flattens candidate-pool, focus, reference and shadow pairs into bounded
  reranker batches;
- starts BM25 work in parallel with GPU scoring;
- uses eight bounded read-only SQLite workers on this host;
- materializes BM25 scores once per query, then derives top-100, the exact
  effective-candidate count and known-positive score from that materialization;
- commits every completed scoring batch to `scoring.journal.jsonl`, calls
  `fsync`, validates a resume identity covering the complete generation input,
  judges and corpus index, and safely truncates only a crash-damaged final
  line;
- reports durable rows, throughput, percentage and ETA. At the default batch
  64, interruption loses at most the currently executing batch.

Execution controls are `S00_SCORING_BATCH_SIZE` (default 64),
`S00_BM25_WORKERS` (default 8) and `S00_SCORING_PROGRESS_EVERY` (default 100).
They are execution parameters, not experimental arms.

## CPU BM25 benchmark

All queries came from the completed zero-shot frozen-dev generation artifact.
No final data were opened.

- On 8 queries, the legacy implementation took 23.22 s.
- The final one-pass implementation took 12.81 s with one worker and 4.05 s
  with eight workers.
- Returned metric dictionaries were exactly equal to the legacy dictionaries
  on the benchmark records.
- On an earlier 24-query worker probe, 4/8/12 workers took
  22.33/13.45/14.20 s. Eight workers are therefore the measured default for
  the 6-core/12-thread Ryzen 5 5600X; twelve caused contention.

## GPU and end-to-end benchmark

Both pinned judges were executed on the local RTX 3060 Ti over 64 real
zero-shot dev outputs. The standalone batched generated-query workload
(876 pairs including focus sentences) took 11.17 s for primary and 13.91 s
for shadow. This is a throughput diagnostic, not a quality result.

The final end-to-end 64-record benchmark, including primary, reference,
focus, shadow, lexical/format metrics and concurrent eight-worker BM25, took
25.57 s (`2.50 records/s`). A linear projection for one 25,000-output arm is
2.77 h after model loading. This projection is not a completed S00 runtime;
query complexity, thermal state and disk contention can change the full-run
time. The earlier pre-optimization end-to-end implementation took 52.08 s on
the same 64-record panel and projected 5.65 h.

## Verified state and next command

The generation journal is complete: 50,000/50,000 completions, comprising
25,000 zero-shot and 25,000 few-shot outputs. Full Harness scoring of both
arms remains unexecuted after the interruption, so there is no S00 quality
result and no S07/P-06 decision.

CPU verification after implementation: `ruff` clean, `mypy` clean and
181 tests passed. Tests cover metric equivalence, durable batch resume,
identity rejection and recovery from a truncated final journal line.

Resume the same runner with:

```bash
bash scripts/run_s00_prompting.sh
```

It reuses all completed generation and any durable scoring journal. It does
not run `dev_confirm`, P-05 guardrails or final tests.
