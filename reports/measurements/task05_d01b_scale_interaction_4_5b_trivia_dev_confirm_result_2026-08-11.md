# Task 05: D01b 4.5B TriviaQA development-confirm result

Date: 2026-08-11

## Outcome

The crash-safe operator run completed with `rc=0` at
`2026-08-11T23:25:57+02:00`. All six W06-versus-D01b arm/seed runs contain a
complete training summary, 100 corpus-embedding shards, 8000 paired-query rows
and `result.json`. Seed 42 training was reused without retraining; seeds 43 and
44 were trained under the frozen matched budget. `final_tests_used=[]`.

The preregistered external-development confirm passed. The decision artifact
records `decision=eligible_for_finalist_freeze_review`,
`retained_for_finalist_freeze=true`,
`selection_claim=external_dev_confirm_passed_pending_finalist_freeze_review`,
`task06_or_task09_promotion_authorized=false` and
`four_point_five_b_full_authorized=false`.

Authoritative artifacts:

- summary:
  `reports/measurements/task05/d01b_scale_interaction_4_5b_trivia_dev_confirm_v1/summary.json`,
  SHA-256 `037f611322b39f261c04fa41eed10814f108cb3632c17f30a63f472c9ca0edf6`;
- runner status:
  `reports/measurements/task05/d01b_scale_interaction_4_5b_trivia_dev_confirm_v1/status.json`,
  SHA-256 `085f6b88edd8a50325e97d2ec33be01702a5d20ad3b33f24c76b453e78eb10b1`.

## Frozen primary result

Metrics are Hybrid minus W06 after taking, for each query, the fixed mean over
seeds 42/43/44 and then bootstrapping 8000 query IDs 10000 times. The interval
is the preregistered two-sided 97.5% percentile interval.

| Metric | Difference | 97.5% CI | Gate |
|---|---:|---:|---|
| corpus nDCG@10 | +0.0478666 | [+0.0450118, +0.0508263] | pass; lower bound >= +0.01 |
| corpus Recall@10 | +0.0372332 | [+0.0343041, +0.0401587] | guardrail pass |
| corpus MRR@10 | +0.0887288 | [+0.0834149, +0.0941504] | guardrail pass |
| corpus MAP | +0.0299017 | [+0.0278405, +0.0320512] | guardrail pass |

All other reported retrieval metrics are also pointwise positive with positive
97.5% lower bounds: Recall@1 `+0.0140309`, Recall@5 `+0.0291477` and
Recall@100 `+0.0795191`.

## Seed-instability caveat

W06 seed 43 is a valid, complete run with the correct seed, recipe, train
fingerprint, model size, corpus and 8000 query IDs, but its training did not
converge: loss changed from `1.3430706` to `1.3889105`, and corpus nDCG@10 is
`0.00001897`. W06 seeds 42 and 44 obtained `0.0782124` and `0.0595843`.
Hybrid is materially more stable across the same seeds: `0.1052508`,
`0.1023983` and `0.0737665`.

This completed-run observation does not invalidate or replace the frozen
fixed-seed analysis; the protocol deliberately reports seed dispersion and
does not resample seeds. It does mean that the large official effect combines
two findings: higher retrieval quality and better training stability for the
Hybrid data under this small-probe recipe.

As a labelled post-hoc robustness diagnostic only, excluding the collapsed
seed 43 from both arms leaves seeds 42+44 at nDCG@10 difference `+0.0206102`
with two-sided 97.5% paired-query CI `[+0.0174116, +0.0237756]`. Seed 42 alone
is `+0.0270383` with CI `[+0.0225591, +0.0315889]`; seed 44 alone is
`+0.0141822` with CI `[+0.0100026, +0.0182966]`. These diagnostics support the
direction of the result but are not new promotion gates.

## Boundary and owner decision

The result retains D01b Hybrid 4.5B for a separate finalist-freeze review. It
does not by itself choose a production generator, authorize full 4.5B
training, open Task 06/09 execution, or open final tests. The next required
owner decision has two related parts because Hybrid is a two-generator
selection procedure, not a single checkpoint:

1. whether Task 06 should freeze the confirmed W06+D01 candidate pool and
   safe-anchor selector as its data-generation procedure;
2. which single SFT adapter should be the Task 07 optimization start. The
   controlled D01 adapter is the natural candidate if the goal is to retain
   explicit controls, while W06 remains the anchor and candidate source; this
   is a recommendation, not a result established by the probe confirm.

A positive decision needs its own prospective handoff ADR and Task 06
generation/scoring preflight. A negative decision preserves this result as
confirmed development evidence and stops this branch without changing the
threshold or rerunning the confirm.
