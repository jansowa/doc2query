# Task 04 P-03 — blocker W05 sensitivity check

Status: `BLOCKED` (diagnostic run not started)

Date of audit: 2026-07-19

## Scope

This blocker applies only to the one-time diagnostic comparison
`W05: HN0 vs HN0+filter vs HN1 BM25`. It is not a generator comparison,
final result, P-04 run, or full HN0–HN3 campaign.

## Evidence found in the repository

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

## Why no threshold was created

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

1. Task 02 must produce and approve a reproducible
   `possible_false_negative_threshold` artifact using only frozen development
   data, then pin its ID and fingerprint in `probe_v1.yaml`.
2. Build the P-01 BM25 index for the train corpus on the project partition and
   pin its index fingerprint for HN1.
3. Confirm one W05 checkpoint, identical train records, example/token/step
   budget and seed for all three diagnostic arms.
4. Run only the three P-03 arms and report uncertainty. If the difference is
   material, or the result remains inconclusive pending the P-04 statistical
   contract, create an ADR before any generator comparison.

No sensitivity result or recipe choice is claimed in this blocker.
