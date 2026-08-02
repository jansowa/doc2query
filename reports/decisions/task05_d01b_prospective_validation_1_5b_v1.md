# ADR: prospective unseen validation of the frozen D01b selector (1.5B)

Date: 2026-08-02

Status: accepted and preregistered before generation

## Context

The D01b best-of-eight selector was defined after inspecting the original
`dev_intrinsic_rank10` D01 comparison. Its retrospective result is useful for
forming a hypothesis, but it is permanently ineligible for promotion and
cannot authorize probe materialization. This ADR freezes a separate unseen,
non-final development validation before any query is generated or inspected.

The selector is the implementation and decision contract at commit `2164822`:

- `src/doc2query/evaluation/d01_usefulness.py`, SHA-256
  `f8bb6ccd491a6e4f3fd721ca6368bcc75a450455082f5d0f6aa94a96e29c764c`;
- `configs/evaluation/d01b_usefulness_hybrid_v1.yaml`, SHA-256
  `0ba63995648c57c5d68ec23c6b0c54008036c914d8d75f241f5a07db8c84abd5`.

No objective weight, feasibility rule, copy threshold, semantic model, metric
definition or tie-break may be changed after this preregistration in response
to prospective outputs. A contract incompatibility must stop the run and be
reported; it does not authorize retuning on this cohort.

## Metadata-only cohort audit and decision

The source population is exactly `dev_intrinsic` minus
`dev_intrinsic_rank10`. Neither SFT model trained on dev, and no final-test
manifest or record was opened. The 9674 available records have this hard-
negative distribution: 2: 3, 3: 11, 4: 69, 5: 168, 6: 464, 7: 1174,
8: 2761 and 9: 5024.

The prospective minimum is five inherited hard negatives. This leaves 9591
eligible records. All 9591 have a matching positive and exactly the expected
negative-ID intersection in the already existing Task 02 natural-primary
artifact. The cohort contains 2000 records selected by ascending
`SHA-256("20260802:<example_id>")`, with `example_id` as the deterministic
tie-break. The selection is independent of text, generation and quality.

The frozen cohort manifest is
`reports/preregistrations/task05_d01b_prospective_1_5b_v1.cohort.json`.
Its selected ID-list SHA-256 is
`b540e24cf04dbc3638921173f116e80d7a7714523a597eb5cc1dea408ebfbc57`;
its canonical selected source-record SHA-256 is
`3beac7deaffa9d4a839b40a0b705fa043dadb58f704c7de3bfb7fd5a6021ccf7`.
The cohort has zero intersection with `dev_intrinsic_rank10`.

## Frozen generation and scoring

Only the 1.5B comparison is authorized in this stage. For every selected
passage, generate exactly four sampled W05 baseline queries and four sampled
D01 controlled queries. Both use seed 42, temperature 0.8, top-p 0.95,
`max_new_tokens=64`, at most three attempts per slot and the existing
passage-index seed contract. The base model is
`speakleash/Bielik-1.5B-v3` revision
`4b25049621bf3952a1fc9314c89773102eda0333`; the exact adapters and generation
config fingerprints are pinned in the validation config.

Score the eight candidates with the frozen primary
`sdadas/polish-reranker-roberta-v3` revision
`e6471da541f4e7be33845b6d57248a8d8bde27e8`, reserved shadow
`BAAI/bge-reranker-v2-m3` revision
`953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`, and the pinned dev-inclusive
BM25 corpus index. Semantic diversity uses `OPI-PIB/PolDense-150M` revision
`b94ea7f951cc480369a85fa9021694eef80c3a00` with `[sts]: `, normalized
embeddings and `trust_remote_code=false`.

The D01b selection remains best four of eight with the four W05 queries as the
safety anchor. Feasible sets may not be worse than the anchor on group primary
Recall@1, corpus round-trip@20, sentence-level source hit or format validity,
and may not contain more copy-risk queries. The unchanged objective weights
are 0.35 natural-margin alignment, 0.30 semantic diversity, 0.10 lexical
diversity, 0.15 corpus specificity and 0.10 low copy density. Primary and BM25
may affect selection. Shadow scores must not affect feasibility, objective,
tie-breaking or candidate selection.

## Preregistered comparison gates

Compare the selected hybrid with the four W05 anchor queries on exactly the
same 2000 passage groups. Use paired percentile bootstrap by passage/query
group, 10,000 samples, seed 20260721, and a 95% interval. All gates must pass:

- corpus round-trip@20 lower CI at least -0.02;
- sentence-level source hit lower CI at least -0.02;
- format validity lower CI at least -0.005;
- reserved-shadow Recall@1 lower CI at least -0.02;
- copy-risk rate is no worse than baseline (upper CI at most 0);
- semantic diversity is non-inferior (lower CI at least 0).

Primary/BM25 improvements are descriptive and may partly follow from selector
construction. Reserved shadow is the independent judge guardrail. A missing
metric, identity mismatch, incomplete exact-K arm, non-finite value or failed
gate makes the decision fail closed.

Only if every gate passes may the report set
`probe_materialization_authorized=true` for equal-budget 1.5B hybrid versus
W05 inputs. This ADR does not authorize materializing or training that probe,
opening any final test, running 4.5B, or changing the selector. Every artifact
must carry an identity/fingerprint and `final_tests_used=[]`.
