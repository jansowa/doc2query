# ADR: D01b 4.5B scale-interaction external development confirm

Date: 2026-08-10

Status: accepted prospectively before external-cohort model evaluation

## Trigger and confirmatory claim

The one-seed 4.5B scale-interaction pilot produced an eligible screen but no
selection claim. The original unused MSMARCO development reserve was too small
for a meaningful multiplicity-adjusted confirm. The owner then supplied a
previously unused Polish TriviaQA retrieval dataset. Its cohort policy was
accepted before materialization and before any model was evaluated on this
source.

This ADR authorizes preparation of exactly one external-development confirm of
W06 4.5B versus the D01b 4.5B safe-anchor hybrid. Passing may set
`retained_for_finalist_freeze=true` for later review, but does not itself select
a final model, open Task 06/09, authorize full 4.5B training, or open final
tests.

## Frozen external cohort and corpus

Use only `dev_d01b_trivia_external_v1` from
`data/processed/task05/d01b_trivia_external_dev_v1/manifest.json`. The
prospective audit found 48317 eligible Polish queries after excluding 116
missing translations and 11980 queries without a positive above the strict
source threshold. Exact and 5-gram-Jaccard>=0.85 positive overlap against the
768 pilot training passages were both zero.

The deterministically selected cohort contains 8000 query IDs. All positives
with `pos_scores_stronger_reranker > 23.50` are relevant; all ten supplied
negatives remain in the record. The global corpus contains 139782 unique
documents. One query, regardless of its number of positives, is one
observation and one bootstrap unit.

The cohort snapshot is
`reports/preregistrations/task05_d01b_trivia_external_dev_v1.cohort.json`.
No TriviaQA row may be used for training, threshold tuning, selector changes,
or model choice before this confirm is complete. The downloaded card does not
declare a license, so this ADR permits internal evaluation only; publication
or redistribution remains blocked pending a separate license decision.

## Frozen training comparison

Reuse the already materialized matched pilot inputs without regeneration,
rescoring, reselection, or backfill:

- W06 baseline: 3072 pairs, 768 passages, K=4;
- D01b hybrid: 3072 pairs, 768 passages, K=4;
- probe: `sdadas/polish-reranker-base-ranknet` revision
  `a7c66d41a8097ca02e75616d0951c941d94ff6a1`;
- batch 2, max length 192, 1024 steps, 1179648 padded training tokens;
- HN0+filter/drop and the same frozen primary calibration;
- seeds 42, 43, and 44 for both arms.

The complete seed-42 models from the pilot are reused byte-for-byte and only
evaluated on the new external cohort. They must not be retrained. Seeds 43 and
44 start from the same base revision and use the exact seed-42 training budget
and inputs. Every arm/seed receives a separate crash-safe namespace.

## Primary endpoint, multiplicity, and guardrails

For each query and arm, average each metric over the fixed seeds 42/43/44,
then compute hybrid-minus-W06 paired differences. Bootstrap query IDs only,
using NumPy PCG64, 10000 samples, seed 20260721. Do not resample seeds.

The primary endpoint is `corpus_ndcg_at_10`. Use a two-sided 97.5% percentile
interval (`1.25%`, `98.75%`). The confirm passes the primary gate only when its
lower bound is at least the unchanged `+0.01` practical-effect threshold.

Report, without substituting them for the primary endpoint:

- corpus Recall@1, Recall@5, Recall@10, Recall@100;
- corpus MRR@10 and MAP;
- per-seed means, standard deviations, minima, maxima, and ranges;
- query-level positive counts and slices for original-only versus any mined
  positive provenance.

The following paired 97.5% non-inferiority guardrails must also pass:

- `corpus_recall_at_10`: lower bound at least `-0.02`;
- `corpus_mrr_at_10`: lower bound at least `-0.02`;
- `corpus_map`: lower bound at least `-0.02`.

All pilot intrinsic guardrails must remain frozen and previously passed. A
missing query, mismatched seed set, unequal corpus, incomplete retrieval,
changed metric definition, or absent slice stops comparison fail-closed.

## Operational boundary

CPU-only cohort materialization and preflight are authorized. Expensive GPU
evaluation/training remains blocked until preflight returns `status=verified`
and the owner explicitly invokes the single operator command that will be
printed in the preparation report. The runner must lock the campaign, stop on
the first error, reuse complete seed-42 training artifacts, resume valid
retrieval shards, and reject final-test references.

No pilot generation or pilot training is repeated. `final_tests_used=[]`
throughout this stage.
