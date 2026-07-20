# Task 04 P-03 — runtime recovery after HN1 normalizer failure

The first full P-03 attempt completed and persisted all 10,000 deterministic
W05 train generations. It then prepared HN0 and spent about five hours scoring
HN0+filter, but failed before HN1 because the new project `.venv-gpu` contained
spaCy without the index-pinned `pl_core_news_lg==3.8.0` package.

The environment bootstrap now installs the official 3.8.0 wheel into the
project environment/cache and treats its absence as an incomplete GPU stack.
P-03 loads the normalizer before any costly stage, so this dependency fails
early rather than after filtering.

Preparation now writes a contract- and SHA-256-pinned cache after each arm.
A complete cached HN0, HN0+filter or HN1 arm is reused after interruption;
partial or contract-drifted caches are rejected. The failed process had not
persisted HN0+filter, so that arm must be recomputed once.

The primary judge identity and revision remain unchanged. A fixed 199-pair
technical panel compared CPU batch 8 with GPU batch 32:

- maximum absolute logit difference: `1.7642974853515625e-05`;
- mean absolute logit difference: `3.764527526932146e-06`;
- classification changes at the pinned threshold
  `8.617486953735352`: `0`;
- peak GPU allocation: `2019069440` bytes.

A separate 100-record throughput smoke measured `8.5146` examples/s on GPU
versus `0.6789` examples/s for the bulk CPU implementation and roughly
`0.57` examples/s in the failed per-example run. These are runtime diagnostics,
not sensitivity results. No final test was used and no negative recipe was
selected.
