# Post-D01 campaign v2 — stan operatorski

## Zweryfikowany stan

Read-only audyt pełnego `dev_intrinsic_rank10` przeszedł fail-closed. Oba SFT
mają 50000 train records, global step 3125 i dataset fingerprint
`017a26ebcf6c5811d5c84498d44881d943c919680e9eed482a649409dfc06b73`.
Adapter SHA-256 wynosi:

- D01 1.5B: `dc2beb9092740d5981850ee59c9c7815b6a612cf77ca353af88fd0b056a98c14`;
- D01 4.5B: `71937228ea977d9d6a89613fe6f802fc3711dba9499a8e23c6c1e4e21e77a867`.

Generacje zachowują wszystkie 6598 passage, fingerprint kohorty
`235d9b81e04ddc5e74bd2bbe884055dd74f03b6706e6030e88a4f918ac2ffab6`,
frozen order, seed/control contract i `final_tests_used=[]`. D01 1.5B ma
26386 query, 1 invalid attempt, 86 duplicate attempts i 6 exhausted groups;
D01 4.5B ma 26384 query, 0 invalid attempts, 110 duplicate attempts i 8
exhausted groups. Jest to wyłącznie wynik techniczny.

Preflight potwierdził W05 oraz rzeczywisty W06
`W06-4.5B-INSTRUCT-50K-8GB-BS8-L512`; fikcyjny adapter BS1 nie jest używany.
Zweryfikował też pełne SHA indeksu BM25 i dokumentów: corpus fingerprint
`e5df243227e8e877550c283e2f7c882fa931ee38d849d39e8f2e2a51dc182119`,
2211463 dokumenty, protokół `corpus_retrieval`.

## Fazy

Każde wywołanie wykonuje dokładnie jedną fazę i ma osobny lock, log oraz
trwały status:

```bash
bash scripts/run_task05_d01_post_campaign.sh preflight
bash scripts/run_task05_d01_post_campaign.sh generate-matched-baselines
bash scripts/run_task05_d01_post_campaign.sh prepare-common-cohort
bash scripts/run_task05_d01_post_campaign.sh score
bash scripts/run_task05_d01_post_campaign.sh compare
bash scripts/run_task05_d01_post_campaign.sh materialize-probe-inputs
```

`prepare-common-cohort` wolno uruchomić dopiero po obu pełnych baseline
generation. Finalnie filtruje osobne kopie wszystkich czterech ramion do
wspólnego exact K=4; nie modyfikuje źródeł. `score` wymaga primary, shadow i
zweryfikowanego corpus index. Comparator odrzuca brak metryki, różny ID/order,
cohort fingerprint, K, seed, retry lub token ceiling. Probe inputs powstaną
tylko po kompletnym scoringu, przejściu wszystkich P-04 guardraili i
identity-aligned HN0+filter/drop.

Logi: `logs/task05_d01_post_campaign/<phase>.log`.
Statusy: `reports/measurements/task05_d01_postprocess_v2/status.tsv` oraz
`status.json`. Finalna kohorta i odzyskane artefakty trafią do
`artifacts/task05/d01_postprocess_v2/common_exact_k_v1/`.

Nie uruchomiono matched baseline generation, scoringu, comparison, probe input
materialization ani probe training. Nie odczytano finalnych testów. Audyt Groq
pozostaje niezależny i nie jest tu agregowany ani interpretowany.
