# ADR: D01b 4.5B external TriviaQA development-cohort policy

Date: 2026-08-10

Status: accepted prospectively before cohort materialization or model evaluation

## Trigger and scope

The owner supplied the previously unused dataset
`mining-negatives/trivia-mined-negatives` after the original development
reserve was found insufficient. This ADR authorizes a metadata/ID-only audit
and deterministic materialization of an external Polish development cohort.
It does not authorize probe training, model evaluation, selection, Task 06/09
promotion, or final-test access.

The source is frozen as `data/raw/trivia-mined-negatives/train_pl.jsonl`,
SHA-256
`2ed9f62ae99b3c8e66274e70e9af975e10feaf31b1f154c7976ab24dccda10ac`.
The downloaded dataset card is frozen at SHA-256
`bb88a54c3e3b24f377d06f8665a75e0d4e4ea5bd120eec419204b3e6a50a5f3b`.
The card reports 60413 natural queries, exactly ten negatives per query,
441689 positives above the recommended strong-reranker threshold, and 48411
queries with at least one such positive.

## Frozen eligibility and relevance policy

Use `query_id` as the only sampling unit. A query is eligible exactly when:

- its ID is non-empty and unique;
- `query`, `pos`, `pos_id`, `pos_scores_stronger_reranker`, `pos_is_synthetic`,
  `neg`, `neg_id`, and `neg_selection_tier` have aligned valid shapes;
- it contains exactly ten negatives with unique non-empty IDs;
- at least one positive has `pos_scores_stronger_reranker > 23.50` (strict,
  matching the dataset card, not the inclusive MSMARCO rule);
- no retained positive ID is also a negative ID for that query;
- `translation_missing` is not true.

Retain every positive above the threshold as relevant, including its original
`pos_is_synthetic` provenance. Positives at or below the threshold are absent
from both relevance judgments and the frozen evaluation corpus. Retain all ten
provided negatives. Never rescore, relabel, weight, or filter using the local
primary/shadow judges.

Multiple positives do not create multiple statistical observations. All
metrics and bootstrap resampling use one row per `query_id`, with every
retained positive treated as relevant.

## Frozen selection

Order all eligible queries by ascending
`SHA-256("20260810:<query_id>")`, with `query_id` as tie-break, and select the
first 8000. This count was chosen before inspecting model outputs and exceeds
the planning estimates of 3922 queries for 80% power and 5121 for 90% power.
Do not backfill, stratify by score, inspect query text, or choose favorable
domains after materialization. Fewer than 8000 eligible queries stops the
stage fail-closed.

Materialize one canonical development record per selected query and one global
deduplicated corpus containing all retained positives and all ten negatives.
A document ID mapping to more than one distinct text stops materialization.
The materialized manifest must pin source, selected-ID, canonical-record, and
corpus fingerprints and record `final_tests_used=[]`.

## Leakage boundary

This source has not been used by the D01b pilot or the earlier MSMARCO
development cohorts. The audit must compare IDs and normalized exact-text
hashes against the frozen pilot probe-training passages. Any positive overlap
removes the whole query before deterministic selection; the exclusion count
and fingerprints must be reported. A near-duplicate audit against the same
training passages is mandatory in the final confirm preflight. No TriviaQA
record may be used for probe training, generator tuning, threshold tuning, or
candidate selection.

## Next gate

Only after this audit and materialization pass may a separate confirm ADR pin:
W06 4.5B versus D01b safe-anchor hybrid 4.5B, the already matched pilot
training inputs, seeds 42/43/44, reuse (not retraining) of the verified seed-42
models, matched training budgets for seeds 43/44, the external corpus, a
two-sided 97.5% paired-query interval, the unchanged `+0.01` primary threshold,
guardrails, recovery behavior, and one explicit operator command.

Final tests remain closed and the hybrid remains unselected throughout this
cohort-preparation stage.
