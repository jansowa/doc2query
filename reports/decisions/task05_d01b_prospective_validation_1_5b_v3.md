# ADR: prospective D01b v3 with duplicate-preserving W05 slots

Date: 2026-08-04

Status: accepted and preregistered before generation

## Context

Prospective v1 and v2 both stopped before scoring because novelty-conditioned
W05 generation could not produce four distinct valid queries for every frozen
passage. V2 still lacked 88 slots in 62 of 2000 groups after up to 16 attempts
per slot and recorded 4873 duplicate outputs. Controlled completed exact K.
No prospective query text was manually inspected and neither cohort reached
scoring or selection.

The owner accepted a v3 protocol that treats repeated valid W05 outputs as an
observed baseline property instead of repeatedly sampling until novelty. Four
independent W05 slots are always retained when format-valid, including exact
or normalized duplicates. Only invalid/empty completions are retried, up to the
already accepted ceiling of 16 attempts. Controlled generation continues to
deduplicate and uses the same ceiling.

## Frozen selector

The selector remains frozen at commit `2164822`:

- `src/doc2query/evaluation/d01_usefulness.py`, SHA-256
  `f8bb6ccd491a6e4f3fd721ca6368bcc75a450455082f5d0f6aa94a96e29c764c`;
- `configs/evaluation/d01b_usefulness_hybrid_v1.yaml`, SHA-256
  `0ba63995648c57c5d68ec23c6b0c54008036c914d8d75f241f5a07db8c84abd5`.

Candidate identities remain slot-specific and unique even when normalized text
is repeated. Duplicate candidates receive the existing semantic and lexical
diversity penalties through identical embeddings/lemmas. Selector weights,
feasibility rules, copy thresholds, natural-margin calibration and tie-breaking
do not change. Primary and BM25 may affect selection; shadow remains reserved
from all selection decisions.

## Metadata-only cohort

Use `dev_intrinsic` after excluding `dev_intrinsic_rank10` and every record
selected by prospective v1 and v2. Both earlier cohorts are reconstructed from
their frozen selection seeds and fingerprints. After exclusions, 5674 records
remain and 5591 have at least five inherited hard negatives.

Select 2000 by ascending `SHA-256("20260804:<example_id>")`, with
`example_id` as tie-break. The selected ID-list SHA-256 is
`070bee90616e135daaa974cf041d5a47462868de860859f8a8eec21287ff023d`;
the canonical selected source-record SHA-256 is
`918157a086bd9713331f50c463c4f30b13bcd84789498952a66491794db4d416`.
Intersection with rank-10, v1 and v2 is zero. Natural-primary positive and
negative ID sets match exactly for all 2000 records. No generation, text or
quality field influenced selection; `final_tests_used=[]`.

The cohort manifest is
`reports/preregistrations/task05_d01b_prospective_1_5b_v3.cohort.json`.

## Generation and scoring

Generate exactly four W05 slots and four D01 controlled slots per passage.
Both use seed 42, temperature 0.8, top-p 0.95, `max_new_tokens=64`, the same
base model/adapters and original-frozen-index seed contract.

- W05: `preserve_duplicate_slots=true`; retry only invalid output, maximum 16.
- D01: `preserve_duplicate_slots=false`; retry invalid or duplicate output,
  maximum 16.

The primary, reserved shadow, BM25 corpus and PolDense models, revisions,
prefixes and trust settings remain frozen as in v2. Each generation and score
artifact must expose identity, fingerprints, raw duplicate counts and
`final_tests_used=[]`.

## Comparison and gates

Apply the unchanged best-four-of-eight selector. The four observed W05 slots,
including any repeated text, are the safety anchor. Compare the hybrid against
that exact anchor on all 2000 groups.

Use paired percentile bootstrap by passage/query group, 10,000 samples, seed
20260721 and 95% intervals. Every gate must pass:

- corpus round-trip@20 lower CI at least -0.02;
- sentence-level source hit lower CI at least -0.02;
- format validity lower CI at least -0.005;
- reserved-shadow Recall@1 lower CI at least -0.02;
- copy-risk rate upper CI at most 0;
- semantic diversity lower CI at least 0;
- normalized within-group duplicate rate upper CI at most 0.

Duplicate rate is descriptive and a guardrail only; it may not be used to
retune selector weights or thresholds. Primary/BM25 improvements may be partly
construction-induced. Reserved shadow is the independent grounding guardrail.

Any identity drift, incomplete four-slot arm, missing/non-finite metric or
failed gate stops the run. Passing every gate may authorize equal-budget 1.5B
probe input materialization only. It does not authorize probe training, 4.5B,
selector changes or final tests.
