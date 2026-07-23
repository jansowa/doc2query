# P-05 dev-screen audit — 2026-07-23

Status: `INCOMPLETE_GUARDRAILS`; no arm promoted and no final test opened.

All three seed-42 probe runs completed on the same 6,598-query frozen
`dev_intrinsic_rank10` subset, the same 2,404,263-document corpus, the same
probe recipe and the same four-dimensional 25% budget. The stored dataset
label says `test_translated_msmarco_pl`, but its fingerprint
`235d9b81…ffab6` is exactly the frozen development subset. This is a metadata
defect, not final-test access; the runtime labeling path has been corrected.

| Arm | nDCG@10 | MRR@10 | R@10 | R@100 |
|---|---:|---:|---:|---:|
| gold natural | 0.052762 | 0.048471 | 0.080929 | 0.180380 |
| mixed 50/50 | 0.057568 | 0.052691 | 0.089542 | 0.195771 |
| W05 synthetic | 0.048618 | 0.045032 | 0.074346 | 0.167235 |

The preregistered 10,000-sample paired-query bootstrap (seed `20260721`) gives:

- mixed 50/50 minus gold: `+0.004806`, 95% CI
  `[+0.000876, +0.008692]`;
- W05 synthetic minus gold: `-0.004143`, 95% CI
  `[-0.007937, -0.000361]`.

Mixed 50/50 is better than gold on this single-seed reduced probe, while
synthetic-only is worse. Neither passes the preregistered practical-effect
rule requiring the lower CI bound to be at least `+0.01`; this is therefore
not a finalist or scale decision.

P-04 also requires paired development guardrails for
`corpus_round_trip_at_20`, `sentence_level_source_hit` and
`format_valid_rate`. They were not produced by the probe runner. Because a
missing guardrail fails closed, `dev_confirm` is not authorized. The next
in-scope step is to measure and assemble these three guardrails on frozen dev,
then run the P-04 decision engine. Starting an independent Task 05/06
experiment before that would leave the current gate unresolved.
