# ADR: Task 04 P-04 statistical and budget contract

Status: **ACCEPTED**

ADR ID: `ADR-task04-P04-statistical-budget`

Version: `1.0.0`

Machine-readable contract: `configs/evaluation/comparison_contract_v1.yaml`

## Decision objective

This contract is preregistered before any comparison-eligible probe run. It
separates development selection from the single final-test opening, separates
training variance from query-sampling uncertainty, and rejects comparisons
whose budgets or contract versions differ.

## Primary outcome and practical effect

The primary final outcome is probe-embedder `corpus_ndcg_at_10` on the full
frozen `test_native_pl` corpus. A variant may be called practically better only
when the paired-query 95% confidence interval for its absolute difference over
the control has lower bound at least `+0.01`. A positive point estimate alone
is insufficient.

The `0.01` absolute margin is a preregistered engineering threshold, not an
empirical result. Changing it creates a new ADR version and is forbidden after
comparison-eligible dev results have been inspected.

## Non-inferiority guardrails

All guardrails are evaluated as right-minus-left differences on frozen dev.
Their paired-query 95% CI lower bound must be at least the negative margin:

| Dimension | Metric | Margin |
|---|---|---:|
| grounding | `corpus_round_trip_at_20` | `0.02` |
| answerability | `sentence_level_source_hit` | `0.02` |
| format | `format_valid_rate` | `0.005` |

A missing guardrail is a failure, not zero and not a waiver. Grounding,
answerability and format cannot be traded away for the primary metric.

## Seeds, halving and uncertainty

Successive halving has two preregistered stages:

1. `dev_screen`: seed 42, 25% of the full budget, frozen `dev_intrinsic`; retain
   at most half of eligible variants after applying all guardrails.
2. `dev_confirm`: seeds 42, 43 and 44, 100% budget, frozen `dev_intrinsic`;
   freeze finalists before opening any final test.

Report every seed separately, then mean, sample standard deviation and range
across independent training seeds. Separately run 10,000 paired bootstrap
resamples over query IDs (seed `20260721`) and report the 95% percentile CI.
Never pool seed-to-seed variance into the query bootstrap or present one as a
substitute for the other. Ties after guardrails are resolved by lower training
variance, then lower measured generation cost.

## Exact comparison budget

Every run manifest must record `probe-budget-v1` and all four positive integer
dimensions:

- `token_count`: padded training-token ceiling across query, positive and one
  paired negative;
- `pair_count`: materialized query-positive training pairs after the common
  cohort and false-negative policy;
- `unique_passage_count`: unique positive document IDs in those pairs;
- `queries_per_passage`: exact uniform K after deduplication.

The comparison validator requires exact equality on all four dimensions and
the budget-definition version. It also checks the statistical-contract
version, fingerprint, ADR ID and ADR version. Missing metadata fails closed.
If filtering changes a cohort, intersect and rematerialize a common cohort;
do not compensate silently with extra epochs or examples.

## Data access and final-test opening

`dev_intrinsic` may be used for prompt and threshold tuning, halving and
finalist selection. `test_native_pl`, `test_translated_msmarco_pl` and
`test_embedder` may not be used for those decisions.

There is exactly one final opening, after finalist identities, configs, seeds,
code revision, dataset fingerprints and four-dimensional budgets are frozen.
At that opening evaluate all frozen finalists and controls on both full native
and translated tests and report every result. No retuning, reselection or new
variant may follow from final-test observations; any later study needs a new
untouched test and a new ADR.

## Consequences

The contract deliberately blocks comparisons made from legacy manifests that
lack P-04 metadata. Diagnostic and smoke runs remain reportable but are not
comparison-eligible. This ADR authorizes implementation and dev-only
comparisons, not D00–D12, Task 06, comparative probes in this change, or any
claim about an unrun experiment.
