# ADR: D01b difficulty-aware hybrid selection

Date: 2026-08-01

Status: accepted for retrospective development diagnostics only

## Context

Matched D01 comparisons showed that uncontrolled W05/W06 queries retrieve the
source passage substantially more often than the uniformly controlled arms.
This is not sufficient to conclude that they are better training examples:
source retrieval may reward extractive or otherwise easy queries, while the
downstream embedder benefits from grounded but non-trivial distinctions among
the positive and inherited hard negatives.

The frozen Task 02 primary scores provide a natural-query reference margin for
the same development examples, positive document and inherited negatives. They
therefore permit a diagnostic separation of:

1. grounding (does the synthetic query still identify an answerable source?);
2. difficulty (is its positive-versus-negative margin close to the natural
   query rather than merely maximized?);
3. diversity and copying.

The D01 dev results were inspected before this ADR. Consequently no result on
the existing D01 cohort can authorize promotion, probe materialization or a
final-test opening. It may only test engineering feasibility and determine a
future, prospectively frozen validation design.

## Decision

Implement a hybrid best-of-eight selector for each matched passage and model
size. Its candidate pool consists of the four uncontrolled baseline queries
and four controlled queries already generated and scored.

The four baseline queries are the safety anchor. A candidate subset of four is
feasible only when, at passage-group level, it is no worse than that anchor on:

- primary source Recall@1;
- corpus source round-trip@20;
- sentence-level source hit;
- format validity;
- count of natural-calibrated copy-risk queries.

Among feasible subsets, maximize a fixed weighted objective:

- `0.35` natural-margin alignment;
- `0.30` PolDense semantic diversity;
- `0.10` lexical diversity;
- `0.15` corpus specificity;
- `0.10` low copy density.

Natural-margin alignment compares the synthetic primary margin to the frozen
natural-query primary margin for the same positive document and intersecting
inherited negatives. It rewards similarity to natural difficulty, not a large
margin. The scale is the cohort IQR of natural margins with a positive floor.

Semantic diversity uses normalized `OPI-PIB/PolDense-150M` embeddings at pinned
revision `b94ea7f951cc480369a85fa9021694eef80c3a00` and the symmetric `[sts]: `
prefix. Query-to-passage retrieval prefixes are not used here.

Primary and BM25 signals may be used by the selector. The frozen shadow judge
must not influence feasibility, objective weights, candidate selection or
tie-breaking. It remains an independent diagnostic. Final tests remain closed.

## Interpretation and next gate

The retrospective report must expose:

- synthetic-minus-natural margin excess for each arm, requested form and
  requested intent;
- selected source composition;
- paired changes versus the baseline anchor for primary, shadow, corpus,
  answerability, copying and semantic diversity;
- `promotion_eligible=false`, `probe_materialization_authorized=false` and
  `final_tests_used=[]` unconditionally.

After the retrospective run, a separate ADR must define an unseen development
cohort or newly generated prospective validation before the selector can pass
the normal intrinsic gates. Only that future validation may authorize equal-
budget probe materialization. The existing failed D01 reports are not amended
or bypassed.
