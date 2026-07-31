# ADR: D01 anti-copy and semantic-diversity quality gate

Status: **ACCEPTED**

ADR ID: `ADR-task05-D01-copy-semantic-quality`

Version: `1.0.0`

## Context and timing

The original D01 matched report measured lexical copying but did not make it a
promotion guardrail, and it left embedding-based diversity unmeasured. A model
could therefore obtain strong retrieval and apparent between-query diversity
by copying different passage fragments.

This correction is registered before either matched D01 comparison or any D01
probe-input materialization. One controlled 1.5B arm had already completed
intrinsic scoring, so its observed values must not determine thresholds. Every
copy threshold below is derived mechanically from the matched frozen natural
dev references. D01 outputs and final-test data are forbidden calibration
inputs. The baseline-vs-variant results were not interpreted to choose margins.

## Decision

Use `OPI-PIB/PolDense-150M` at the full pinned revision recorded in the
machine-readable contract. The 150M checkpoint is the selected cost/quality
point for four-query groups; changing model size or revision creates a new
contract. `trust_remote_code=false` is mandatory.

PolDense distinguishes asymmetric retrieval from symmetric similarity:

- query-to-passage retrieval uses `[query]: ` on the query;
- query-to-query semantic similarity uses `[sts]: ` on every query.

D01 diversity is a symmetric query-to-query task, therefore its embeddings use
`[sts]: `. The retrieval prefix remains pinned in the contract to prevent a
future query-to-passage measurement from silently using the wrong input.

Calibrate upper-tail copy thresholds on the exact matched natural dev
references: the 95th percentile for copy density, normalized LCS and longest
contiguous copied n-gram, and the 99th percentile for query-to-passage length
ratio. A generated query of at least four words is high-risk when it exceeds
both overlap thresholds, exceeds the longest-span threshold, or is unusually
passage-like in length.

Both arms must remain within five percentage points of the natural high-risk
rate. The controlled variant must also be non-inferior to its baseline: the
upper 95% paired-bootstrap bound for the variant-minus-baseline risk rate may
not exceed two percentage points.

Semantic diversity is measured only on the intersection of passage groups for
which all four queries in both arms are below the anti-copy risk threshold.
This avoids rewarding diversity produced by copied fragments and avoids
selection bias from using a different clean cohort per arm. At least 80% of
matched groups must remain. Cosine `0.85` is the fixed semantic-cluster edge
threshold. On this common clean cohort, the controlled arm may
not lose more than 0.10 semantic clusters per four-query group and may not
increase maximum pairwise cosine by more than 0.02 at the upper 95% CI bound.
Passage-lemma-removed pairwise Jaccard is reported as a diagnostic, not as a
standalone promotion signal.

Embedding arrays are identity-bound and cached atomically. Every comparison
also materializes a blind, arm-hidden audit of 100 high-retrieval cases
prioritized by copy risk. Automatic scoring cannot fill or substitute this
human form.

## Consequences

The D01 comparison and probe-input gate fail closed if the quality contract is
missing, its ADR fingerprint drifts, PolDense embeddings are unavailable, or
any anti-copy/semantic guardrail fails. Existing generator and reranker scoring
does not need to be repeated because it already stores the required per-query
fields. Intrinsic success still does not establish the main outcome: the
matched probe embedder on natural queries remains required.

No final-test subset is opened by this decision.

## Source

- PolDense-150M model card and prefix contract:
  <https://huggingface.co/OPI-PIB/PolDense-150M>
- Pinned model revision:
  <https://huggingface.co/OPI-PIB/PolDense-150M/commit/b94ea7f951cc480369a85fa9021694eef80c3a00>
