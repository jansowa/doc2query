# Task 05: D01b 4.5B TriviaQA development-confirm preparation

Date: 2026-08-10

## Outcome

The external-development confirm is fully preregistered and its CPU preflight
returns `status=verified`. No model evaluation, new probe training, pilot
retraining, or final-test access occurred.

The owner-provided `mining-negatives/trivia-mined-negatives` repository was
downloaded completely. `train_pl.jsonl` contains 60413 rows and is frozen at
SHA-256
`2ed9f62ae99b3c8e66274e70e9af975e10feaf31b1f154c7976ab24dccda10ac`.
The source files remain under ignored `data/raw/` and are not candidates for
Git inclusion.

## Prospective cohort audit

The cohort policy was recorded before materialization and before any model
was evaluated on TriviaQA. It freezes one query as one observation, the strict
source filter `pos_scores_stronger_reranker > 23.50`, all passing positives,
all ten negatives, deterministic seed `20260810`, and 8000 selected queries.

The metadata/ID audit found:

- 60413 source rows;
- 116 rows excluded because the Polish translation was missing;
- 11980 rows without any positive above the strict threshold;
- 48317 eligible Polish queries;
- 440476 retained positives in that eligible population;
- zero exact positive overlaps with the 768 pilot training passages;
- zero selected positive near-duplicates at 5-gram Jaccard >= 0.85.

The frozen cohort contains 8000 query IDs. Its global corpus contains 139782
unique documents and has SHA-256
`32af7d61ef0496175a7244ec73ad99bd00c5dd23d5bdd6015d850b2bd9aab845`.
The canonical records, ID list, corpus, and materialized manifest passed the
repository frozen-set loader. No query or passage text was manually inspected.

Tracked cohort evidence:
`reports/preregistrations/task05_d01b_trivia_external_dev_v1.cohort.json`.

## Frozen confirm

The confirm remains exactly W06 4.5B versus D01b safe-anchor hybrid 4.5B.
Both arms use the already materialized pilot inputs: 3072 pairs, 768 passages,
K=4, batch 2, max length 192, 1024 steps, and 1179648 padded tokens per seed.

Seeds are 42/43/44. The completed seed-42 model fingerprints and training
summaries were verified and staged using symlinks plus copied metadata; no old
evaluation output was copied and seed 42 will not be retrained. Only seeds 43
and 44 require new training.

For each query, metrics are averaged over the fixed three seeds before query
resampling. The primary endpoint is `corpus_ndcg_at_10`; the paired query
bootstrap uses NumPy PCG64, 10000 samples, seed 20260721, and a two-sided 97.5%
percentile interval. Its lower bound must remain at least `+0.01`. Frozen
97.5% non-inferiority guardrails for Recall@10, MRR@10, and MAP must each have
a lower bound of at least `-0.02`.

The dataset card does not declare a license, so the contract permits internal
evaluation only and blocks publication or redistribution pending a separate
license decision.

## Preflight and operator boundary

The real preflight verified:

- the completed pilot screen and its `eligible`-only scope;
- every source, ADR, cohort, corpus, input, recipe, model, and summary hash;
- exact matched arm budgets and passage IDs;
- 8000 frozen external queries and 139782 corpus documents;
- seed 42 staged without retraining, seeds 43/44 designated for new training;
- the 97.5% interval, unchanged `+0.01` threshold, and final-test prohibition;
- more than 94 GB free disk against a 30 GB fail-closed floor.

The single expensive operator entry point is:

`bash scripts/run_task05_d01b_scale_interaction_4_5b_trivia_dev_confirm.sh run-all`

It has not been invoked. The runner locks the campaign, checks GPU idleness,
reruns preflight, resumes complete training/evaluation artifacts, stops on the
first error, and writes separate arm/seed namespaces.

## Validation

- focused TriviaQA cohort/confirm tests: `16 passed`;
- complete CPU suite: `463 passed`, 16 warnings;
- Ruff: passed for the repository;
- mypy: passed for all 108 `src` files and all six changed Python files;
- shell syntax: passed;
- `git diff --check`: passed.

Current boundary: `selection_claim=null`,
`retained_for_finalist_freeze=false`,
`four_point_five_b_full_authorized=false`, and `final_tests_used=[]`.
