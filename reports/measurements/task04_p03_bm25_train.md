# Task 04 P-03 — pinned train-corpus BM25

Status: `MEASURED`

Date: 2026-07-19

The corpus contains only documents referenced by canonical train records.
Neither `test_native_pl` nor another final test was used to build or tune it.

## Frozen train corpus

- artifact ID: `train-corpus-v1`;
- queries: 292,907;
- unique documents: 2,211,463;
- artifact fingerprint:
  `26e435e1d0413dc92e151a02e46f752747ecbdb0df20399318fc2c03223b0abd`;
- ordered document-ID SHA-256:
  `53c4c68bd130e979471defeecc82e3b56065d6b47038d59a30a27ddb9f7c1f2e`;
- Parquet SHA-256:
  `d22f734a860c833f05104f005fb0061d732e17fa40e9debd11e61ef1f17b658f`.

## BM25

- backend: `bm25_sqlite`;
- normalizer: `pl_core_news_lg==3.8.0`;
- normalizer namespace: `spacy_pl:pl_core_news_lg:3.8.0:v1`;
- `k1=1.2`, `b=0.75`;
- documents: 2,211,463;
- terms: 1,129,538;
- average document length: `33.90394548767038`;
- SQLite integrity check: `ok`;
- database SHA-256:
  `4f52f2140187360b42345cd112c19bdc09a6067d7414c8236c546d2a6299b27b`;
- document fingerprint:
  `9511a1a948a06a9fc41121be87bd4a6d11238c9960eb55dd72c027bf6944c1f6`;
- index fingerprint:
  `e5df243227e8e877550c283e2f7c882fa931ee38d849d39e8f2e2a51dc182119`.

The configured relevance threshold is metadata for ambiguity diagnostics and
does not select HN1 candidates. HN1 uses the raw BM25 ordering. This artifact
is diagnostic P-03 infrastructure, not a final corpus-retrieval comparison.
