# Task 05 D01b — usefulness and safe-anchor hybrid diagnostic

Date: 2026-08-02

Status: `RETROSPECTIVE_EXPLORATORY_COMPLETE`

This diagnostic was designed after the matched D01 dev aggregate had already
been inspected. It is not promotion-eligible, cannot authorize probe
materialization and used no final test.

## Question

Could the high source retrieval of uncontrolled W05/W06 queries merely reflect
queries that are too easy to provide useful contrastive training signal?

The diagnostic compares each synthetic primary margin with the frozen Task 02
natural-query margin for the same dev example, positive document and
intersecting inherited hard negatives. Positive `margin_excess` means that the
synthetic query is easier than the corresponding natural query for the frozen
primary judge.

## Difficulty diagnosis

| Arm | Mean margin excess | Median excess | Easier than natural | Sentence source hit |
|---|---:|---:|---:|---:|
| W05 1.5B uncontrolled | +0.267 | +0.137 | 53.29% | 86.17% |
| D01 1.5B controlled | -0.832 | -0.640 | 40.67% | 78.50% |
| W06 4.5B uncontrolled | +0.425 | +0.206 | 55.29% | 87.39% |
| D01 4.5B controlled | -0.781 | -0.554 | 41.90% | 78.45% |

The uncontrolled queries are moderately, not overwhelmingly, easier than the
natural references. The uniformly controlled arms are harder, but their lower
sentence hit shows that raw difficulty includes misgrounded or inapplicable
intent cases and is not automatically useful contrastive difficulty.

## Safe-anchor selection

For every passage, all 70 subsets of four candidates from the four baseline
and four controlled queries were evaluated. The all-baseline set was the
safety anchor. A subset was feasible only if it did not reduce group-level
primary Recall@1, corpus round-trip@20, sentence source hit, format validity or
copy-risk count. Among feasible subsets, the fixed D01b objective rewarded
natural-margin alignment, PolDense semantic diversity, lexical diversity,
corpus specificity and low copying. Shadow scores were excluded from all
selection decisions.

| Selected minus baseline anchor | 1.5B | 4.5B |
|---|---:|---:|
| Controlled-query share | 42.66% | 42.25% |
| Groups changed | 5035 / 5321 | 5055 / 5321 |
| Primary Recall@1 | +5.40 pp | +4.39 pp |
| Corpus round-trip@20 | +4.76 pp | +4.58 pp |
| Sentence source hit | +5.87 pp | +4.85 pp |
| **Reserved shadow Recall@1** | **+3.58 pp** | **+3.00 pp** |
| Shadow margin | -0.095 | -0.132 |
| Copy density | -0.018 | -0.018 |
| Semantic diversity | +0.022 | +0.023 |
| Lexical diversity | +0.061 | +0.063 |

All displayed changes use passage-group paired bootstrap with 10,000 samples.
The 95% CI for reserved shadow Recall@1 is `[+3.24, +3.93]` pp for 1.5B and
`[+2.67, +3.34]` pp for 4.5B. Shadow Recall@1 increases while shadow margin
decreases, which is consistent with retaining the source at rank one while
making inherited hard negatives less trivial.

The selected median margin excess is `+0.038` for 1.5B and `+0.079` for 4.5B;
the selected easier-than-natural rates are 51.91% and 53.53%. The hybrid is
therefore close to the natural-query difficulty centre instead of maximizing
the primary margin.

Machine-readable reports and selected diagnostic rows are under
`reports/measurements/task05_d01_postprocess_v2/usefulness/`. Their status is
`retrospective_exploratory_complete`, with `promotion_eligible=false`,
`probe_materialization_authorized=false` and `final_tests_used=[]`.

## Decision and next gate

The result supports a prospectively validated hybrid-selection direction. It
does not reopen the original D01 gate because both the selector and objective
were defined after observing D01 dev results, and primary/BM25 improvements are
partly enforced by construction. The independent shadow improvement is useful
evidence of generalization, not sufficient promotion evidence.

Before probe training:

1. freeze the current selector without retuning;
2. prepare a development cohort of passages not used by the 50k SFT runs and
   not drawn from any final-test split;
3. generate and score the same best-of-eight pool there;
4. require ordinary intrinsic non-inferiority plus reserved-shadow
   non-inferiority on that unseen cohort;
5. only then materialize equal-budget hybrid and baseline probe inputs.
