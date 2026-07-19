# Task 02 — possible-false-negative calibration on frozen dev

Status: `MEASURED`

Date: 2026-07-19

## Contract

- fit data: `data/processed/v1/dev.parquet` only;
- final tests used for tuning: none;
- frozen primary: `sdadas/polish-reranker-roberta-v3`;
- revision: `e6471da541f4e7be33845b6d57248a8d8bde27e8`;
- score space: raw pair logit;
- selection: maximize query-macro Youden J over known-positive versus inherited
  hard-negative labels;
- exact ties: choose the highest threshold, preserving specificity.

The rule gives each development query equal total weight within each class, so
queries with multiple positives do not dominate calibration. The inherited
hard-negative class can contain label noise. Consequently,
`possible_false_negative` means only that the primary score lies in the
empirically positive-like region; it is not proof that the document is relevant.

## Measurement

- queries: 16,272;
- known-positive pairs: 21,241;
- inherited-negative pairs: 145,441;
- threshold (`>=`): `8.617486953735352`;
- query-macro true-positive rate: `0.9013852308378869`;
- query-macro false-positive rate: `0.0669031962588428`;
- Youden J: `0.8344820345790441`;
- query-bootstrap 95% CI for J at the selected threshold:
  `[0.8298256767882284, 0.8396126221301115]`.

## Pinned provenance

- artifact ID: `pfn-dev-v1-b455711ec36526b2`;
- artifact fingerprint:
  `9ee4280f18e684b0dc3bb7fd885801b5ae8821af758e2845ab349c559613b3f4`;
- frozen dev file fingerprint:
  `12b86da10c73707a58a8793111afcca0424981d738667a04c228014780a02a1d`;
- source-score SHA-256:
  `7a1f9f5949a26d3c09780ba30689fb259c1d20cd401a1e8860b8292461eef115`.

The machine-readable artifact is
`artifacts/task02/pfn_dev_v1/calibration.json`. Raw logits remain an untracked
large measurement artifact on the project partition.
