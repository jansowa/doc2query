# P-05/P-04 dev-screen gate decision — 2026-07-23

Status: `COMPLETE_NO_PROMOTION`; no arm is authorized for `dev_confirm` and
no final test was opened.

The fail-closed `task04-p04-v1` engine consumed two complete comparison
reports over the same 6,598-query frozen `dev_intrinsic_rank10` cohort,
fingerprint `235d9b81…ffab6`, with paired-query bootstrap `n=10,000`, seed
`20260721`. Both reports have `errors=[]`, `final_tests_used=[]` and the same
four-dimensional comparison budget.

## Absolute seed-42 metrics

| Arm | corpus nDCG@10 | round-trip@20 | sentence source hit | format valid |
|---|---:|---:|---:|---:|
| gold natural | 0.052762 | 0.116854 | 0.894665 | 1.000000 |
| mixed 50/50 | 0.057568 | 0.130797 | 0.894665 | 1.000000 |
| W05 synthetic | 0.048618 | 0.112307 | 0.894665 | 1.000000 |

Sentence source hit and format validity are properties of the shared frozen
natural dev evaluation queries, so their paired differences between
probe-training arms are exactly zero. Round-trip@20 is arm-specific known
positive hit@20 under each trained probe.

## Paired differences versus gold

| Arm | Metric | Difference | 95% CI | Required lower bound | Pass |
|---|---|---:|---:|---:|---|
| mixed 50/50 | corpus nDCG@10 | +0.004806 | [+0.000901, +0.008712] | +0.010 | no |
| mixed 50/50 | round-trip@20 | +0.013944 | [+0.006972, +0.021067] | -0.020 | yes |
| mixed 50/50 | sentence source hit | 0.000000 | [0.000000, 0.000000] | -0.020 | yes |
| mixed 50/50 | format valid | 0.000000 | [0.000000, 0.000000] | -0.005 | yes |
| W05 synthetic | corpus nDCG@10 | -0.004143 | [-0.008005, -0.000318] | +0.010 | no |
| W05 synthetic | round-trip@20 | -0.004547 | [-0.011216, +0.002122] | -0.020 | yes |
| W05 synthetic | sentence source hit | 0.000000 | [0.000000, 0.000000] | -0.020 | yes |
| W05 synthetic | format valid | 0.000000 | [0.000000, 0.000000] | -0.005 | yes |

## Decision

The engine returns `non_inferior_only` for both variants. All non-inferiority
guardrails pass, but neither lower confidence bound reaches the preregistered
minimum practical effect `+0.01` for `corpus_ndcg_at_10`. Consequently:

- `dev_confirm_authorized_arms=[]`;
- no finalist or scale claim is made;
- `dev_confirm` must not be run for these arms;
- final tests remain unopened;
- a new campaign or changed threshold requires a separate, prospective ADR.

Machine-readable artifacts are stored beside this report under `p04_gate/`.
Their SHA-256 fingerprints are:

- mixed report `874b9074…a1b3e`, decision `38aefbae…1139d`;
- synthetic report `899d8e4f…8d24d`, decision `29db5b6c…1ee24`;
- decision summary `cd9c9374…e162`;
- shared sentence/format raw artifact `1996a995…c40ab`.
