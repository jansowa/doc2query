# P-05/P-04 frozen-dev guardrail progress — 2026-07-23

Status: `COMPLETE_NO_PROMOTION`; no arm is authorized for
`dev_confirm` and no final test was opened.

The measurement uses the same 6,598-query `dev_intrinsic_rank10` cohort,
fingerprint `235d9b81…ffab6`, as all three completed seed-42 probe runs.
`scripts/run_p05_dev_screen.sh` was not invoked.

## Measured guardrails

| Arm | `corpus_round_trip_at_20` | hits / 6,598 |
|---|---:|---:|
| gold natural | 0.116854 | 771 |
| mixed 50/50 | 0.130797 | 863 |
| W05 synthetic | 0.112307 | 741 |

The exact round-trip backfill reused the stored probe results for queries
whose status was already determined by Recall@10/100 or single-positive MAP.
Only 157/144/142 multi-positive queries respectively required rescoring from
the existing frozen corpus-embedding caches. This was a read-only CPU
backfill; it did not repeat training, corpus encoding or the completed probe.

The shared natural dev queries have `format_valid_rate = 1.000000` (6,598 /
6,598 valid, zero violations). Raw artifact fingerprints:

- gold round-trip: `4ca73c29…db0e7`;
- mixed round-trip: `26fdb75b…4c797`;
- synthetic round-trip: `20f5ec09…b1c2`;
- format: `78e1a592…c3097b`.

## Completed sentence measurement and fail-closed decision

The canonical `sentence_level_source_hit` compares the best sentence score
from the known source passage(s) with the hardest inherited-negative score
under pinned `sdadas/polish-reranker-roberta-v3`. Whole-passage scores cannot
be substituted. The completed pinned-primary measurement gives
`sentence_level_source_hit=0.894665` on the 6,598 shared frozen natural dev
queries. Its raw artifact SHA-256 is `1996a995…c40ab`.

The completed fail-closed P-04 engine returns `non_inferior_only` for mixed and
synthetic-only, with `errors=[]`. All guardrails pass, but both variant lower
CI bounds for the primary `corpus_ndcg_at_10` effect remain below the
preregistered `+0.01` threshold. Therefore no arm may proceed to
`dev_confirm`. The final decision and exact CIs are recorded in
[`p04_gate_decision_2026-07-23.md`](p04_gate_decision_2026-07-23.md).

The resumable runner that produced the result scores the sentence guardrail,
reuses completed round-trip files, merges the per-query artifacts, builds
paired-query 10,000-sample bootstrap reports and invokes the decision engine:

```bash
DOC2QUERY_PYTHON="$PWD/.venv-gpu/bin/python" scripts/run_p05_guardrails.sh
```

It writes progress to the terminal and appends it to
`logs/task04_p05_guardrails/run.log`. It never accesses a final-test subset.
