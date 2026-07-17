# Task 01: data audit, deduplication, and leakage-safe splits

The pipeline is streaming for source records and stores large indexes in SQLite/Parquet. It does not
load all document texts into RAM. Runtime data belongs below the ignored `data/` directory.

## End-to-end `speakleash/msmarco_pl` workflow

First materialize and adapt the pinned Polish source as described in
[`docs/datasets/msmarco_pl.md`](datasets/msmarco_pl.md). Then run:

```bash
mkdir -p data/interim data/processed/v1 reports

uv run python scripts/validate_dataset.py \
  --input data/interim/msmarco_pl.canonical.jsonl \
  --accepted data/interim/msmarco_pl.accepted.jsonl \
  --rejected data/interim/msmarco_pl.rejected.jsonl \
  --report reports/data_validation.json \
  --policy configs/data/validation_policy.yaml

uv run python scripts/build_document_index.py \
  --input data/interim/msmarco_pl.accepted.jsonl \
  --sqlite data/interim/documents.sqlite \
  --documents data/processed/v1/documents.parquet \
  --report reports/document_index.json

uv run python scripts/deduplicate_documents.py \
  --index data/interim/documents.sqlite \
  --output data/processed/v1/dedup_map.parquet \
  --report reports/deduplication.json \
  --resume-if-available

uv run python scripts/build_splits.py \
  --input data/interim/msmarco_pl.accepted.jsonl \
  --dedup-map data/processed/v1/dedup_map.parquet \
  --output-dir data/processed/v1 \
  --train-ratio 0.90 --dev-ratio 0.05 --test-ratio 0.05 \
  --seed 42 --version v1

for split in train dev test; do
  uv run python scripts/invert_doc2query_pairs.py \
    --input data/processed/v1/${split}.parquet \
    --output data/processed/v1/doc2query_${split}.parquet \
    --report reports/invert_${split}.json \
    --split ${split}
done

uv run python scripts/build_data_report.py \
  --input data/processed/v1/train.parquet \
  --input data/processed/v1/dev.parquet \
  --input data/processed/v1/test.parquet \
  --json reports/data_audit.json \
  --html reports/data_audit.html \
  --validation-report reports/data_validation.json \
  --dedup-report reports/deduplication.json \
  --split-manifest data/processed/v1/split_manifest.json \
  --tokenizer-config configs/data/bielik_tokenizers.yaml
```

The tokenizer audit may require accepting model access conditions on Hugging Face. Omitting
`--tokenizer-config` keeps the cheap CPU smoke path, but its whitespace-token diagnostic must not be
used to choose `max_length`.

## Safety and reproducibility

- Validation rules have explicit `warn`, `drop`, or `error` modes. Every rejection is written with
  its issues; an `error` policy violation makes the CLI exit non-zero after producing the report.
- Exact deduplication uses normalized SHA-256. Near-deduplication uses banded SimHash with bounded
  candidate sets persisted in SQLite; candidate-cap hits are reported.
- `--resume-if-available` starts from zero when no deduplication checkpoint exists and otherwise
  validates and resumes compatible `parents`, `simhash`, and LSH tables in the document index.
- Query nodes, positive-document nodes, and near-duplicate canonical IDs form indivisible split
  components. Assignment is deterministic for the seed and balances global/domain deficits.
- Cross-split hard negatives pointing at a positive in another split are removed by default.
  Records left with fewer than ten negatives are reported and must be reviewed before training.
- An existing `split_manifest.json` is treated as frozen; rebuilding in the same version directory
  fails instead of silently changing dev/test.
- `split_assignments.parquet`, fingerprints, parameters, and removal counts support replay/audit.

## Storage representation

Heterogeneous canonical records are stored in Parquet as compressed `record_json`. The shared
`read_records()` API transparently reconstructs dictionaries. This keeps a stable schema across
source-specific metadata while preserving streaming row groups. Document indexes remain normalized,
queryable SQLite during deduplication.

## Measurement status

The complete pipeline is covered by a synthetic end-to-end smoke test, including exact and near
duplicates, connected components, frozen deterministic splits, negative leakage cleanup,
multi-positive inversion, and JSON/HTML reports. The full 1.8 GB Polish source has not been processed
in this implementation change, so no real-data audit statistics are claimed yet.
