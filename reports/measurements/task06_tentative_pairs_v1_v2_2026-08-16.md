# Pomiar: tentative pary chosen/rejected z kohort v1+v2 (2026-08-16)

ADR (prerejestrowany, progi zamrożone przed odczytem jakiejkolwiek pary):
[`task06_tentative_pair_policy_v1.md`](../decisions/task06_tentative_pair_policy_v1.md).
Kontrakt: `configs/preferences/task06_tentative_pair_policy_v1.yaml`
(`policy_id=task06-tentative-pair-policy-v1`).
Artefakty: `artifacts/task06/same_prompt_expansion_v{1,2}/tentative_pairs/`
(`pairs.jsonl`, `group_outcomes.jsonl`, `report.json`, `manifest.json`).
Etap w całości CPU, bez ładowania modeli. `final_tests_used=[]`.

## Wynik

| kohorta | grupy | `eligible` bramki | pary | odsetek par wśród `eligible` |
|---|---|---|---|---|
| same_prompt_expansion_v1 | 500 | 362 | **202** | 55,8% |
| same_prompt_expansion_v2 | 500 | 466 | **245** | 52,6% |
| **razem** | 1000 | 828 | **447** | 54,0% |

**447 par to mniej niż 500 wymaganych przez rozwojową bramkę dual-LLM.**
Uruchamia to ścieżkę niedoboru zapisaną prospektywnie w ADR (punkt 3 sekcji
„Kolejność wykonania”): audyt obejmuje wszystkie uzyskane pary, niedobór jest
raportowany, a **żadnego progu nie wolno poluzować**. Rozszerzenie budowy par na
kohortę v3 (2791 grup `eligible`) wymaga osobnej decyzji właściciela zapisanej
jako amendment — nie jest podejmowane tą sesją.

## Dlaczego grupy nie dały pary

| przyczyna | v1 | v2 | razem |
|---|---|---|---|
| `group_not_gate_eligible` (bramka różnorodności) | 138 | 34 | 172 |
| `no_admissible_chosen` | 71 | 94 | 165 |
| `no_candidate_below_margin_gap` | 35 | 53 | 88 |
| `shadow_veto` | 37 | 48 | 85 |
| `no_admissible_rejected` | 17 | 25 | 42 |
| `near_duplicate_query_pair` | 0 | 1 | 1 |

Obserwacje (opisowe, nie zmieniają żadnego progu):

- najczęstszą przyczyną poza samą bramką jest brak dopuszczalnego `chosen`
  (165 grup) — czyli w grupie żaden reprezentant nie miał jednocześnie
  dodatniego marginesu primary, round-tripu korpusowego w top-20, poprawnego
  formatu i braku copy-risk;
- **veto shadow zadziałało 85 razy (10,3% grup `eligible`)**. To jest niezależna,
  mierzalna niezgodność sędziów na parach, których primary był pewny; rząd
  wielkości zgadza się z 9,81% disagreement zmierzonym w bramce HN (Task 04);
- zamrożony próg `min_margin_gap = 1.0` odrzucił 88 grup, w których primary nie
  rozróżnił kandydatów o więcej niż jedną jednostkę log-odds.

## Rozkład zbudowanych par (v1, 202 pary)

Margines primary między `chosen` a `rejected`: min 1,0069, p25 1,3297,
p50 2,0589, p75 3,4564, max 11,8364. Rozkład jest silnie skośny w prawo, co jest
oczekiwane przy strategii `top_vs_near_miss` z twardym progiem 1,0.

Etykiety typu `rejected` (jedna para może mieć kilka; wyłącznie raport, nie
selekcja): `lower_primary_margin` 202, `shadow_agrees` 202,
`possible_ambiguous_query` 201, `lower_content_jaccard_than_chosen` 108,
`weak_corpus_round_trip` 40, `judge_rank_disagreement` 19, `copy_risk` 7.

Dwie rzeczy warte uwagi przy interpretacji:

- `shadow_agrees` = 202/202 jest **konstrukcyjne, nie potwierdzające**: veto
  shadow usuwa dokładnie te pary, w których shadow nie zgadza się z primary,
  więc wśród par, które przeżyły, zgodność musi wynosić 100%. Nie wolno tego
  raportować jako niezależnego potwierdzenia jakości — niezależną liczbą jest
  odsetek weta (10,3%) liczony na grupach przed filtrem;
- `possible_ambiguous_query` = 201/202 pokazuje, że flaga
  `corpus_possibly_ambiguous_query` jest niemal stała w tej kohorcie i jako
  etykieta typu błędu nie niesie informacji różnicującej.

## Czego ten pomiar nie robi

Nie zbudowano próbki audytowej, nie uruchomiono Groq, nie policzono żadnego
`total_score`, nie skalibrowano żadnej wagi, nie zbudowano par z kohort v3–v11 i
nie autoryzowano treningu DPO (`task07_training_authorized=false`).
`final_tests_used=[]`.
