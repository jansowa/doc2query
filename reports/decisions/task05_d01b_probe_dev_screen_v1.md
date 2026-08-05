# ADR: D01b prospective v3 equal-budget probe dev screen

Date: 2026-08-05

Status: accepted and preregistered before probe training

## Scope

This decision authorizes exactly one development-only `dev_screen` comparison
of the prospectively selected D01b v3 1.5B hybrid against its observed W05
1.5B anchor. It follows the already accepted Task 04 P-03/P-04 contracts and
does not authorize `dev_confirm`, 4.5B, final tests or Task 06.

## Frozen training inputs and common budget

Use the CPU-validated materialization under
`artifacts/task05/d01b_prospective_1_5b_v3/probe_inputs/`. Both complete arms
contain 7936 pairs, 1984 unique positive passages and exact K=4 after the
common HN0+filter/drop and one-group-per-document policy.

For the P-04 25% `dev_screen`, use the exact first 1984 ordered rows of each
arm. The files are sorted by prospective evaluation group and pair identity,
so this prefix contains the same 496 positive passages in both arms and exact
K=4. Do not resample, shuffle before prefix selection, backfill or compensate
with more steps.

The frozen per-arm budget is:

- `token_count=1152000`;
- `pair_count=1984`;
- `unique_passage_count=496`;
- `queries_per_passage=4`;
- `definition_version=probe-budget-v1`.

Train `sdadas/polish-reranker-base-ranknet` revision
`a7c66d41a8097ca02e75616d0951c941d94ff6a1` with
`configs/evaluation/probe_v1.yaml`, seed 42, 250 steps, batch 8, max length
192 and the existing in-batch loss. Use the accepted HN0+filter/drop recipe
and pinned primary judge. Rechecking retained negatives is permitted only as
the existing probe-loader validation; it may not change the common prefix. Any
dropped pair or budget drift stops both arms.

## Frozen development evaluation

Evaluate both trained probes on the complete `dev_intrinsic_rank10` panel of
6598 natural queries and the frozen full development corpus. The training
cohort was selected from `dev_intrinsic` only after excluding this complete
rank-10 panel; the preflight must verify zero source-example-ID intersection.

Use the existing natural-query P-04 guardrail artifact, pinned by SHA-256, for
`sentence_level_source_hit` and `format_valid_rate`. Compute each probe's
`corpus_round_trip_at_20` and `corpus_ndcg_at_10` on the shared natural panel.
Run a paired query bootstrap with 10000 PCG64 samples and seed 20260721.

The hybrid is `eligible` for a separately authorized `dev_confirm` only if:

- the lower 95% CI for `corpus_ndcg_at_10` difference is at least `+0.01`;
- the lower CI for `corpus_round_trip_at_20` is at least `-0.02`;
- the existing answerability and format non-inferiority gates pass;
- every P-04 identity and equal-budget check passes.

`non_inferior_only`, `rejected` or `incomplete` stops promotion. A passing
screen records eligibility only; it does not itself start `dev_confirm`.

## Operational limits

The runner must be resumable after completed training, reject an active GPU,
write separate logs and outputs per arm, and never reference
`test_native_pl`, `test_translated_msmarco_pl`, `test_embedder` or any other
final-test subset. `final_tests_used=[]` is mandatory in every contract and
decision artifact.
