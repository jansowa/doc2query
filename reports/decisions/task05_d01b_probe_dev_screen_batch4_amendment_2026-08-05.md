# ADR amendment: D01b probe dev-screen batch 4 restart

Date: 2026-08-05

Status: accepted by the project owner before restart

## Context

The first valid batch-8 execution of the preregistered D01b probe dev screen
was interrupted by a machine shutdown during W05 training. The log reached
step 84/250 and the last complete rolling checkpoint is step 50. The project
owner reports that this machine can shut down when GPU load is too high and
explicitly requires halving the training batch size.

The batch-8 checkpoint is not compatible with a changed training recipe and
must not be resumed under the amended identity. It remains preserved under
the original run root as evidence of the interrupted execution.

## Amendment

Keep every input, seed, model revision, negative policy, maximum length,
evaluation panel, corpus, bootstrap setting and decision gate from
`task05_d01b_probe_dev_screen_v1.md`. Change only the training execution
budget as follows for both arms:

- `batch_size`: 8 to 4;
- `max_steps`: 250 to 500;
- output, measurement and log roots: a new `v2_batch4` namespace;
- arm IDs: add the `B4` execution suffix.

This preserves 2000 training examples consumed per arm and the frozen
`probe-budget-v1` token count:

```text
500 steps * 4 examples * 192 tokens * 3 encoded sequences = 1,152,000
```

The smaller in-batch negative pool is an explicit consequence of the
operator-mandated memory reduction and will be identical for both arms. No
gradient accumulation or learning-rate change is introduced because neither
belongs to the accepted probe-v1.1 recipe.

## Boundaries

Restart both arms from their common initial model state; do not mix the
batch-8 W05 checkpoint with batch-4 outputs. A result is comparison-eligible
only when both batch-4 arms complete the amended equal budget. The amendment
does not authorize `dev_confirm`, 4.5B, generation, generator rescoring,
selection changes or any final test. `final_tests_used=[]` remains mandatory.
