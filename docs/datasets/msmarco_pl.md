# `speakleash/msmarco_pl` ingestion contract

The dataset is pinned to revision `ffcfc5fbc254bea348a7871133a6a0fa9ca21cb5` in
`configs/data/msmarco_pl.yaml`. At the time of inspection it is private, not gated, and has no
declared license in its card metadata. Access must use the user's Hugging Face credentials; never
store a token in this repository. Confirm the license before redistributing data or derived text.

The Polish subset contains 388,699 queries, 500,732 positive passages, and exactly ten mined hard
negatives per query. It has only a `train` split. Task 01 must therefore create immutable,
leakage-safe train/dev/test splits before calibration or evaluation.

## Materialization and adaptation

Download only the pinned Polish file into the ignored `data/` tree:

```bash
hf download speakleash/msmarco_pl train_pl.jsonl \
  --repo-type dataset \
  --revision ffcfc5fbc254bea348a7871133a6a0fa9ca21cb5 \
  --local-dir data/raw/msmarco_pl

uv run python scripts/adapt_msmarco_pl.py \
  --input data/raw/msmarco_pl/train_pl.jsonl \
  --output data/interim/msmarco_pl.canonical.jsonl \
  --report reports/msmarco_pl_adaptation.json \
  --min-positive-score 23.50
```

The adapter streams JSONL and validates all parallel arrays. It maps `query_id`, `pos`, `pos_id`,
`pos_scores`, `pos_is_synthetic`, `neg`, `neg_id`, and `neg_scores` into the canonical nested
contract. It also records duplicate negative IDs and possible mojibake without silently repairing
text. Positives with the copied source score below `23.50` are removed by an explicit project data
policy; the boundary value is retained. Rows left without a positive are skipped, and both removed
positives and skipped rows are counted in `reports/msmarco_pl_adaptation.json`.

## Score semantics

The source `pos_scores`, `neg_scores`, and aggregate differences were computed on English texts and
copied into the Polish rows. The adapter therefore stores them as `source_en_score` with
`source_score_language: en`.

These values may be used for provenance, stratification, and a separately named difficulty slice.
They must not be used as Polish reranker calibration labels, grounding rewards, or substitutes for
scores recomputed by the frozen primary and shadow judges on Polish text.

## Evaluation policy

- report query-macro as the primary aggregate and pair-micro as a diagnostic;
- report `synthetic_positive` and `source_en_difficulty` slices;
- keep source document IDs in every scoring artifact;
- keep translated/MS MARCO evaluation separate from a native-Polish or manually verified holdout;
- flag encoding and translation artifacts instead of treating all queries as native Polish usage;
- collapse or explicitly report exact and near-duplicate candidates during Task 01 deduplication.
