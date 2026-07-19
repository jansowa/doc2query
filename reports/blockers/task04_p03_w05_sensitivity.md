# Task 04 P-03 — resolved blocker / running W05 sensitivity check

Status: `UNBLOCKED / MEASUREMENT IN PROGRESS`

Date of audit: 2026-07-19

## Scope

This blocker applies only to the one-time diagnostic comparison
`W05: HN0 vs HN0+filter vs HN1 BM25`. It is not a generator comparison,
final result, P-04 run, or full HN0–HN3 campaign.

## Resolved prerequisites

The two original artifact blockers are resolved without opening any final test:

- frozen-dev calibration:
  - artifact ID `pfn-dev-v1-b455711ec36526b2`;
  - fingerprint
    `9ee4280f18e684b0dc3bb7fd885801b5ae8821af758e2845ab349c559613b3f4`;
  - threshold `8.617486953735352`, selected by the query-macro Youden-J maximum
    with a conservative highest-threshold tie break;
  - 16,272 dev queries, 21,241 known-positive pairs and 145,441 inherited
    hard-negative pairs;
  - final tests used for tuning: none.
- frozen train corpus:
  - artifact ID `train-corpus-v1`;
  - fingerprint
    `26e435e1d0413dc92e151a02e46f752747ecbdb0df20399318fc2c03223b0abd`;
  - 2,211,463 unique documents referenced by canonical train records.
- P-01 BM25 over that train corpus:
  - index fingerprint
    `e5df243227e8e877550c283e2f7c882fa931ee38d849d39e8f2e2a51dc182119`;
  - SQLite integrity check: `ok`;
  - normalizer `pl_core_news_lg==3.8.0`;
  - 2,211,463 documents and 1,129,538 terms.

The calibration and BM25 fingerprints are pinned in
`configs/evaluation/probe_v1.yaml`.

## Former blocker found by W05 preflight

The original preflight found:

- `reports/evaluation/W05-1.5B-50K-8GB/generations.jsonl` contains 100 unique
  example IDs from the frozen test panel and has zero intersection with
  canonical train IDs;
- `runs/W05-1.5B-50K-8GB/panel_generations.jsonl` likewise has zero train-ID
  coverage;
- `train_probe_embedder.py --query-source synthetic` requires generated
  queries keyed by train `example_id`; supplying either existing file would
  produce zero training pairs;
- the W05 adapter and checkpoint are present, but the pinned
  `speakleash/Bielik-1.5B-v3` base weights were initially absent from the
  project-local Hugging Face cache.

The exact snapshot was subsequently copied from the authenticated user cache
to the project partition. The proper dev-only run has started its resumable
10k train-query generation. No sensitivity result is available yet.

Using `test_native_pl`, the translated final test panel, the W05 test
generations, or natural queries as a substitute would invalidate the
sensitivity check. No such substitution was made.
The file hashes and intersection counts are frozen in
`reports/measurements/task04_p03_w05_preflight.json`.

## Historical evidence

- `configs/evaluation/probe_v1.yaml` previously selected one inherited
  negative by a deterministic hash but had no false-negative policy,
  threshold provenance or negative-recipe version.
- Task 02 contains generic robust-z and percentile calibrator code.
- The tracked W05 primary-reranker artifacts contain raw scores only for
  generated-query/source-positive pairs. They contain no inherited-negative
  scores, threshold, dev split fingerprint, threshold-selection method or
  calibration artifact fingerprint.
- `tasks/02_reranker_and_reward_proxies.md` and
  `docs/task02_rerankers.md` explicitly state that the full primary/shadow
  benchmark on project dev/test with hard negatives remains unmeasured.
- P-01 contains tested BM25 implementation and a frozen index contract, but no
  materialized BM25 index/manifest is present under the project artifacts.
- The W05 generator checkpoint/generations and frozen data splits are present,
  but they do not remove the two blockers above.

## Historical reason no threshold was created

A raw score percentile cannot be selected without an approved operating rule
or labelled development calibration target. Choosing one in P-03 would be an
ad-hoc threshold, contrary to the research contract. Scores from
`test_native_pl`, translated MS MARCO-PL test, or another final test were not
used and must not be used for this purpose.

## Implemented safe behavior

The `probe-negatives-v1` loader requires a pinned Task 02 JSON artifact fitted
exclusively on a `dev*` split. It verifies the artifact ID, canonical
fingerprint, development dataset fingerprint, source-score SHA-256, primary
judge revision, score space, finite threshold and documented selection
method. Missing or mismatched provenance blocks before model loading.

The code and mock-only tests implement deterministic HN0, HN0+filter and HN1
BM25, policies `drop`, `demote`, `keep+log`, per-source flag reports and strict
comparison compatibility checks.

## Conditions to unblock W05

1. Materialize deterministic W05 synthetic queries for one frozen train-ID
   list using checkpoint `runs/W05-1.5B-50K-8GB/checkpoint-3125`, the pinned
   base revision, and a recorded generation config/fingerprint. Do not use
   final-test records.
2. Confirm identical train records, example/token/step
   budget and seed for all three diagnostic arms.
3. Run only the three P-03 arms and report uncertainty. If the difference is
   material, or the result remains inconclusive pending the P-04 statistical
   contract, create an ADR before any generator comparison.

No sensitivity result or recipe choice is claimed. P-04 was not started.

## Implemented resumable runner

The complete fail-closed path is now:

```bash
HF_HOME="$PWD/.cache/huggingface" \
UV_CACHE_DIR="$PWD/.uv-cache" \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_p03_w05_sensitivity.sh
```

It freezes train IDs and their fingerprint, resumes one-query greedy W05
generation without duplicates, prepares a shared legal-negative cohort,
equalizes the padded token/step budget, resumes all three probe arms, evaluates
only frozen `dev_intrinsic_rank10`, performs paired-query bootstrap and writes
an ADR for a separated or inconclusive outcome. Mock smoke and dry-run modes
are `--smoke` and `--dry-run`.

The pinned
`speakleash/Bielik-1.5B-v3@4b25049621bf3952a1fc9314c89773102eda0333` now
passes the project-local cache check without bypassing gated access. The
runner streams progress, throughput, elapsed time and ETA to both console and
the combined campaign log. No completed sensitivity measurement or
hard-negative recipe selection is claimed.
