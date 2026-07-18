# Task 02: frozen judges and reward proxies

The primary judge is `sdadas/polish-reranker-roberta-v3`, pinned to commit
`e6471da541f4e7be33845b6d57248a8d8bde27e8`. Its Hugging Face model card declares the
Gemma license, 443M parameters, and an 8192-token context. Review the Gemma terms before use:
<https://huggingface.co/sdadas/polish-reranker-roberta-v3>.

The optional fast Polish diagnostic is `sdadas/polish-reranker-base-ranknet`, pinned to commit
`a7c66d41a8097ca02e75616d0951c941d94ff6a1`. Its model card declares Apache-2.0, about
0.1B parameters, and a 512-token context:
<https://huggingface.co/sdadas/polish-reranker-base-ranknet>. It is useful for cheap panel
inspection, but it does not replace the stronger primary judge or an independent shadow judge.

The independent shadow judge is `BAAI/bge-reranker-v2-m3`, pinned to commit
`953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`. Its model card declares Apache-2.0 and an
8192-token context: <https://huggingface.co/BAAI/bge-reranker-v2-m3>.

Both configs disable remote code. The loader sets `eval()`, disables parameter gradients, and uses
`torch.inference_mode()`. There is deliberately no training API. Raw logits are calibrated per
model and never averaged. The benchmark emits an explicit disagreement report.

## Offline workflow

Task 01 must first provide a frozen dev split. Each canonical record needs a query, one or more
positives, and at least ten hard negatives. The benchmark evaluates every positive, reports
query-macro metrics as primary, and retains pair-micro metrics as a diagnostic. It never gives a
multi-positive query extra weight in the primary aggregate.

For `speakleash/msmarco_pl`, run the pinned streaming adapter documented in
`docs/datasets/msmarco_pl.md` before this workflow. Do not pass the raw `pos`/`neg` rows directly to
the benchmark.

```bash
uv run python scripts/benchmark_rerankers.py \
  --input data/processed/v1/dev.parquet \
  --judge-config configs/reranker/primary_polish_roberta_v3.yaml \
  --judge-config configs/reranker/shadow_bge_v2_m3.yaml \
  --output-dir reports/rerankers/dev

uv run python scripts/calibrate_reranker.py \
  --scores reports/rerankers/dev/scores.jsonl \
  --method robust_z --output reports/rerankers/calibration.json

uv run python scripts/assign_focus_labels.py \
  --input data/processed/v1/dev_doc2query.parquet \
  --judge-config configs/reranker/primary_polish_roberta_v3.yaml \
  --output data/processed/v1/dev_with_focus.jsonl

uv run python scripts/precompute_text_analysis.py \
  --input data/processed/v1/dev_doc2query.parquet \
  --cache data/processed/v1/text_analysis.sqlite --backend simple

uv run python scripts/calibrate_rewards.py \
  --input data/processed/v1/natural_reward_dev.parquet \
  --output reports/rewards/calibration.json
```

The simple normalizer is the reproducible cheap baseline. `spacy_pl` requires an installed Polish
spaCy model and always runs on CPU. Cache keys include the backend/model version and normalization
configuration. Passage analysis should be precomputed offline.

## Measurement status

The fast Polish diagnostic and primary judge weights were downloaded after Task 03 and used to
score the same panel of 100 W05 generated-query/source-passage pairs. This only measures raw
positive-pair logits; it is not the Task 02 retrieval benchmark because it has no hard-negative
margins and no independent shadow judge. The tracked
`tests/fixtures/task02_holdout_lexical_report.json` still records only the lexical smoke
correlation. Do not move past the Phase A gate until both frozen primary/shadow judges are measured
on natural dev/test and a human-labeled sample.
