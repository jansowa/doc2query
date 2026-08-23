# Amendment: liczebność próby audytu Groq dla polityki v2.1 (2026-08-23)

## Status

**Amendment wykonawczy, spisany przed pierwszym requestem audytu v2.1.** Dotyczy
jednej bramki w loaderze konfiguracji audytu i niczego więcej. Autoryzacja:
właściciel, ta sama sesja, po zamrożeniu ADR
[`task06_defect_pair_policy_v2_1.md`](task06_defect_pair_policy_v2_1.md).

## Problem

`load_llm_audit_config` egzekwowała `pair_count == 500` z komunikatem
„owner-approved pilot audit must contain exactly 500 pairs". Liczba 500 pochodzi
z waiveru z 2026-08-12, który zastąpił ręczny przegląd 500 par audytem dual-LLM,
i była twardą bramką, żeby nikt nie zmienił liczebności audytu po cichu.

ADR v2.1 wyprowadza liczebność komórki bramkowej **z rachunku mocy**, nie z
waiveru: 800 par daje moc 0,964 dla P3, 1,000 dla P2 i ~1,000 dla P4' przy
zmierzonych założeniach osi A, a punkt decyzyjny reguły Cloppera–Pearsona to
≤ 16/800 dla P3 i ≥ 262/800 dla P2. Przy 500 parach moc P3 spada do 0,793, czyli
poniżej progu, przy którym warto uruchamiać audyt bez mechanizmu eskalacji
(którego ADR świadomie nie tworzy).

## Zmiana

Bramka przestaje pinować jedną liczbę, a zaczyna pinować **zbiór liczebności
zamrożonych prospektywnie przez ADR-y właściciela**:

```python
APPROVED_PAIR_COUNTS = frozenset({500, 800})
```

- **500** — rozwojowa bramka pilota (waiver 2026-08-12; audyty v1 i v2 pozostają
  odtwarzalne bit w bit, bo ich configi się nie zmieniają);
- **800** — komórka bramkowa v2.1 z rachunku mocy (ADR §4–5).

Bramka pozostaje **fail-closed**: każda inna liczba jest odrzucana, więc nie da
się zmienić liczebności audytu bez nowej prospektywnej decyzji. Test
`test_groq_audit_sample_size_is_pinned_to_adr_frozen_values` sprawdza, że 501
nadal jest odrzucane.

## Czego amendment NIE zmienia

Prompt, rubryka, `prompt_version`, modele i ich `reasoning_effort`, `batch_size`,
`api` (temperatura 0, `response_format`, `max_completion_tokens_per_pair`,
timeout), `limits_per_model`, `retry`, `quota_scheduler`, `resume_policy`,
`required_outputs`, `disagreement_policy`, `assignment`, `blind_order_policy`,
`role`, `human_evidence_claimed`. Config `task06_groq_preference_audit_v2_1.json`
różni się od v1 **wyłącznie** w polach `pair_count`, `status`, `owner_waiver` i
`scope_note`, co jest sprawdzane maszynowo testem
`test_groq_audit_config_v2_1_changes_only_the_sample_size`.

Nie zmienia też żadnej reguły decyzyjnej, progu ani predykcji, nie dotyka
artefaktów audytów v1/v2 i nie autoryzuje treningu.

`task07_training_authorized=false`, `final_tests_used=[]`.
