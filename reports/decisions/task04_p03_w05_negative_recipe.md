# ADR: P-03 W05 hard-negative recipe

Status: **ACCEPTED for pre-final probe comparisons**

Decision date: 2026-07-21

Scope: Task 04 P-03; frozen development data only

## Decision

Use `HN0+filter` (`strategy=hn0_filter`, `possible_false_negative_policy=drop`)
from `probe-negatives-v1` for the first comparison-eligible probe runs. Keep the
dev-only threshold and primary-judge provenance already pinned in
`configs/evaluation/probe_v1.yaml`. Do not use HN1 BM25 for those comparisons.

This does not select a generator and does not authorize final-test access.
HN0/HN0+filter/HN1/HN2/HN3 remains a separate full gate before Task 09.

## Evidence and rationale

The completed one-off W05 sensitivity measurement used 1,000 paired queries
from frozen `dev_intrinsic_rank10`; its outcome is `statistically_separated`.
No final, native, translated or embedder test was opened.

- HN1 BM25 was worse than HN0 for pool nDCG@10 by `-0.02964`, with paired
  query-bootstrap 95% CI `[-0.04472, -0.01322]`. It was also worse for MRR and
  hard-negative win rate, with both intervals wholly below zero.
- HN1 BM25 was worse than HN0+filter for pool nDCG@10 by `-0.03645`, 95% CI
  `[-0.05105, -0.02125]`; the MRR and win-rate intervals were also wholly below
  zero.
- HN0+filter versus HN0 was not statistically separated: nDCG@10 difference
  `+0.00682`, 95% CI `[-0.00348, 0.01659]`. It nevertheless applies the
  pre-registered safety policy and removed candidates flagged by the frozen
  dev-only false-negative calibration, dropping only 20 of 10,000 preparation
  examples.

HN0+filter is therefore the conservative choice: it does not show a dev loss
relative to unfiltered HN0 and enforces the false-negative policy. HN1 cannot
be justified for first comparisons because it introduces a measured loss.

## Provenance and limits

- Measurement:
  `reports/measurements/task04_p03_w05_sensitivity/sensitivity_report.json`
- Frozen dev fingerprint:
  `99772726437a25fe6acb3f4d916b18beb0617c08ddf18c7b8e60b319a2b1462a`
- Contract: `p03-w05-sensitivity-v1`
- Bootstrap: 2,000 paired query resamples, seed 42
- `final_tests_used`: `[]`

This is a single-seed diagnostic of the negative recipe. It is not evidence of
generator superiority or final embedder quality. P-04 governs all subsequent
comparisons.
