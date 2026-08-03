# ADR: prospective unseen validation of D01b with exact-K completion v2

Date: 2026-08-03

Status: accepted and preregistered before generation

## Context

The first prospective D01b cohort stopped before scoring because the W05 arm
produced only 7547 of 8000 required distinct queries. Its fixed ceiling of
three attempts per slot exhausted at least one slot in 320 of 2000 groups. No
prospective query text was manually inspected, no quality scoring or selection
was run and no final test was opened. The failure is recorded in
`reports/blockers/task05_d01b_prospective_exact_k_2026-08-03.md`.

The project owner accepted a ceiling of 16 attempts per slot for a new
prospective run. This is an operational completeness limit, not a quality
threshold. It was chosen to provide substantial headroom over the observed
duplicate rate while keeping a finite fail-closed budget. Any group still
lacking four distinct valid queries after 16 attempts makes the whole run
ineligible for scoring.

## Frozen selector and scientific contract

The selector remains the implementation and decision contract at commit
`2164822`:

- `src/doc2query/evaluation/d01_usefulness.py`, SHA-256
  `f8bb6ccd491a6e4f3fd721ca6368bcc75a450455082f5d0f6aa94a96e29c764c`;
- `configs/evaluation/d01b_usefulness_hybrid_v1.yaml`, SHA-256
  `0ba63995648c57c5d68ec23c6b0c54008036c914d8d75f241f5a07db8c84abd5`.

The best-of-eight candidate pool, exact K=4 selection, all feasibility rules,
copy thresholds, objective weights, PolDense representation and deterministic
tie-break remain unchanged. Primary and BM25 may participate in selection.
Reserved shadow must not influence feasibility, objective, tie-breaking or
candidate selection.

## Metadata-only cohort audit and decision

The source is `dev_intrinsic`. Exclude both `dev_intrinsic_rank10` and every
record selected for prospective v1. The v1 exclusion is reconstructed from its
preregistered SHA-256 selection rule (seed 20260802, minimum five hard
negatives, first 2000 records) and must reproduce selected ID-list SHA-256
`b540e24cf04dbc3638921173f116e80d7a7714523a597eb5cc1dea408ebfbc57`.

After both exclusions, 7674 records remain. Their hard-negative distribution
is 2: 3, 3: 11, 4: 69, 5: 124, 6: 353, 7: 933, 8: 2160 and 9: 4021.
The minimum remains five hard negatives, leaving 7591 eligible records.

Select 2000 records by ascending
`SHA-256("20260803:<example_id>")`, with `example_id` as the tie-break. The
selected ID-list SHA-256 is
`819141f6c236a371797eb3272de50e59f00f7d678e66433f233f645a66b6d80d`;
the canonical selected source-record SHA-256 is
`c49f8d45a2a54ab8a05f5454f6a684de4f7b14ba83cb21ca28ba1ed494063aa5`.
Intersection with rank-10 and prospective v1 is zero. All 2000 records have an
exact positive-ID and negative-ID match in the existing Task 02 natural-primary
artifact. No text, generation or quality field influenced selection.

The frozen cohort identity is stored in
`reports/preregistrations/task05_d01b_prospective_1_5b_v2.cohort.json`.
No final-test manifest or record was opened; `final_tests_used=[]`.

## Frozen generation and scoring

Only 1.5B is authorized. Generate four distinct valid W05 queries and four
distinct valid D01 controlled queries per passage. Seed 42, temperature 0.8,
top-p 0.95 and `max_new_tokens=64` remain unchanged. The only protocol change
from v1 is `max_attempts_per_query=16`, applied symmetrically to both arms.
Attempts and accepted outputs remain identity-bound and deterministic under the
existing original-frozen-index seed contract.

The base model, adapter fingerprints, primary, reserved shadow, PolDense model,
BM25 corpus index, revisions and trust settings remain those pinned by v1 and
are repeated in the machine-readable validation config.

## Preregistered gates and stopping rules

Use paired percentile bootstrap by passage/query group, 10,000 samples, seed
20260721 and a 95% interval. Every gate must pass:

- corpus round-trip@20 lower CI at least -0.02;
- sentence-level source hit lower CI at least -0.02;
- format validity lower CI at least -0.005;
- reserved-shadow Recall@1 lower CI at least -0.02;
- copy-risk rate no worse than baseline, with upper CI at most 0;
- semantic diversity non-inferior, with lower CI at least 0.

Primary/BM25 improvements are descriptive and may partly follow from selector
construction. Reserved shadow is the independent judge guardrail. Any identity
drift, missing metric, non-finite value, incomplete exact-K arm or failed gate
stops the run.

Only passage of every gate may authorize materialization of equal-budget 1.5B
hybrid-versus-W05 probe inputs. It does not itself authorize probe training,
4.5B, a selector change or any final test. Every artifact must contain an
identity/fingerprint and `final_tests_used=[]`.
