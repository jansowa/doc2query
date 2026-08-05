# ADR amendment: D01b probe corpus encoding batch 32

Date: 2026-08-05

Status: accepted by the project owner before the next resume

## Context

The batch-4 D01b probe execution completed the W05 arm, including its natural
development evaluation. The hybrid arm completed all 500 training steps and
persisted 42 of 100 corpus embedding shards before another machine shutdown.
The project owner reports that the `encode_corpus` phase still places excessive
load on this machine and explicitly requires lowering its batch size.

## Amendment

Change only `evaluation_encode_batch_size` from 64 to 32. Keep training batch
size 4, 500 training steps, model weights, inputs, corpus, seed, retrieval query
batch size 512, retrieval device, metrics, bootstrap and all decision gates
unchanged.

This is an execution-only memory control. The existing corpus cache manifests
retain `chunk_size=24064`; batch size 32 only divides each missing shard into
smaller forward passes. Every existing shard remains identity-checked and is
reused. Specifically, the complete W05 result and the first 42 valid hybrid
shards must not be recomputed.

## Boundaries

The resume may finish only the missing hybrid corpus shards, hybrid query
evaluation and the preregistered comparison. It does not authorize retraining,
generation, rescoring, `dev_confirm`, 4.5B or final tests.
`final_tests_used=[]` remains mandatory.
