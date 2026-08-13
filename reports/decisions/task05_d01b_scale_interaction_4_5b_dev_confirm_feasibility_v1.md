# ADR: D01b 4.5B development-confirm feasibility

Date: 2026-08-10

Status: fail-closed; owner decision required

## Decision

Do not preregister or run the D01b 4.5B development confirm on the remaining
591 records. Do not reopen the pilot, reuse an evaluated development cohort,
weaken the `+0.01` practical-effect threshold after seeing the pilot, or open
final tests. The legal reserve is insufficient for a meaningful two-sided
97.5% confirm under the frozen comparison and multiplicity boundary.

The comparison remains W06 4.5B versus D01b safe-anchor hybrid 4.5B,
matched-budget, seeds 42/43/44. This ADR does not change that estimand or any
guardrail. Because feasibility failed, it deliberately does not create a
confirm config, crash-safe expensive runner, or operator `run-all` command.

## Evidence

The ID-only audit covers all 16272 `dev_intrinsic` IDs and reconstructs every
prior exclusion from pinned manifests. After excluding 6598 rank-10 IDs, 6000
prospective v1/v2/v3 IDs, 1000 pilot-generation IDs, and 2000 pilot-evaluation
IDs, exactly 591 records remain eligible under the frozen minimum of five hard
negatives. All intersections are zero. Raw IDs and record text were not
emitted; only counts and SHA-256 fingerprints were retained.

Planning from the aggregate pilot CI gives a projected 97.5% half-width of
`0.020108814663528603` at n=591. The projected lower bound at the pilot point
effect is `0.0006291054152610179`; even an optimistic independent-three-seed
calculation gives `0.009128090519717087`, below `+0.01`. Estimated evaluation
sizes are 3922 for 80% power and 5121 for 90% power. Pilot uncertainty and
unknown seed variance mean these figures should not be treated as guarantees.

## Owner choices

1. **Stop D01b promotion (recommended).** Preserve the scale pilot as an
   eligible screen without a confirmatory or finalist claim.
2. **Provide a genuinely new, untouched development population.** Before any
   outcome access, accept a new prospective ADR that pins provenance,
   leakage exclusions, at least the planned evaluation size, the unchanged
   comparison/threshold/97.5% interval, seeds 42/43/44, matched budgets,
   guardrails, and a crash-safe preflight. The new data must not be a final
   test or a relabeling of previously evaluated IDs.
3. **Authorize a new exploratory question.** A post-result threshold change
   or a 591-record analysis could only be a separately named exploratory
   screen. It cannot confirm this hypothesis, satisfy the current gate, or
   promote the hybrid to Task 06/09.

Using final tests, recycling prior cohorts, or treating repeated seeds and
bootstrap samples as new query observations are not legal options.

Until the owner chooses option 2 and supplies adequate untouched data, Task 05
is `BLOCKED`; Task 06/09 promotion and all final tests remain closed.
