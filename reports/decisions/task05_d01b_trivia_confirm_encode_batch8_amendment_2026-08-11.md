# Amendment: D01b TriviaQA confirm conservative corpus-encoding batch

Date: 2026-08-11

Status: accepted by the owner before the next incomplete arm was trained

## Trigger

The first `run-all` attempt ended externally while W06 seed 43 was encoding the
external corpus with `evaluation_encode_batch_size=32`. It left 88 of 100
crash-safe embedding shards and no traceback or runner `END` record. The owner
reported that this machine has previously powered off under batch sizes that
were too aggressive.

The resumed attempt reused the saved shards and completed W06 seed 43. The
owner then stopped the runner with `Ctrl-C` while Hybrid seed 43 was filtering
training negatives, before that arm's training began, and explicitly required
the evaluation encode batch to be reduced to 8.

## Execution-only change

For every subsequent invocation of the frozen confirm runner:

- reduce `evaluation_encode_batch_size` from 32 to 8;
- keep training batch 2, max length 192, 1024 steps and seeds 42/43/44;
- keep primary-judge filtering batch 4 and retrieval query batch 512;
- keep the same models, revisions, train inputs, external query cohort, corpus,
  metrics, bootstrap, 97.5% interval, practical-effect threshold and
  guardrails.

The embedding cache identity is based on the model, recipe and corpus rather
than the inner encoding microbatch. Complete cache shards and completed
per-query retrieval rows remain valid and are reused. An incomplete shard is
never accepted because shards are written atomically. Changing the inner batch
does not authorize regeneration, retraining of seed 42, threshold tuning,
selection, Task 06/09 promotion, full 4.5B training or final-test access.

## Preserved state and restart

At acceptance, complete result artifacts exist for W06 seed 42, Hybrid seed 42
and W06 seed 43. Hybrid seed 43 has no completed training artifact. The same
crash-safe operator command remains the only authorized entry point:

```bash
bash scripts/run_task05_d01b_scale_interaction_4_5b_trivia_dev_confirm.sh run-all
```

The amended preflight must return `status=verified` and bind batch 8 before the
operator restarts. Existing results must not be deleted merely to homogenize an
execution-only microbatch. `final_tests_used=[]` remains mandatory.
