# Amendment: czytnik audytu Groq przyjmuje eksport par v2 (2026-08-21)

## Status i autoryzacja

**Prospektywny amendment**, spisany **przed uruchomieniem audytu v2** i przed odczytem
jakiejkolwiek oceny sędziów. Właściciel autoryzował 2026-08-21 wariant „adaptacja
czytnika” (opcja A z przedstawionych trzech), z warunkiem wprost: **bez zmiany promptu,
rubryki, modeli, limitów i reguł decyzyjnych**.

Amendment dotyczy wyłącznie zadania V2-05 i jest domknięciem długu nazwanego w §12 ADR
[`task06_defect_pair_policy_v2.md`](task06_defect_pair_policy_v2.md). Nie zmienia polityki
par v2, nie zmienia predykcji P1–P4, nie zmienia progów, nie autoryzuje treningu
(`task07_training_authorized=false`) i nie otwiera testów finalnych
(`final_tests_used=[]`).

## Problem

`src/doc2query/preferences/groq_pair_audit.py` czyta eksport audytowy pod nazwami pól
polityki v1. Eksport par v2
(`artifacts/task06/preference_audit_v3_defect_pairs/`, kontrakt
`task06-defect-pair-audit-blind-export-v2`) trzech z nich nie dostarcza:

1. **manifest** ma własny numer kontraktu, więc walidacja modelu v1 go odrzuca;
2. **`primary_margin_gap_band`** nie istnieje — polityka v2 celowo nie stratyfikuje po
   marginesie (§6.2 ADR), bo margines nie porządkuje par;
3. **`rejected_failure_types`** nosi w v2 nazwę **`rejected_defect_labels`** — etykiety
   opisują nazwany defekt, nie porażkę względem porządku marginesowego.

Bez adaptacji czytnik przerywa pracę na błędzie brakującego pola, jeszcze przed wysłaniem
pierwszego requestu.

## Ustalenie, na którym opiera się ta decyzja

Przejrzano **każde** użycie obu pól w czytniku (linie 668–669, 691, 725–727, 764, 799).
Wchodzą wyłącznie do trzech miejsc opisowych: kopii do wiersza wyniku oraz dwóch tabelek
przekrojowych (`agreement_by_*`). **Nie wpływają na żadną ocenę, żaden werdykt konsensusu,
żaden licznik `consensus_*`, żadną bramkę ani żadną predykcję.** Predykcje P1–P4 liczą się
z pól całkowicie niezależnych od tych dwóch.

## Decyzja: dokładnie trzy zmiany, wszystkie mechaniczne

1. **Dyspozytor manifestu.** Czytnik rozpoznaje kontrakt eksportu i waliduje manifest
   właściwym modelem (v1 albo v2). Używa wyłącznie pól obecnych w obu:
   `sample.path`, `sampled_pair_count`, `policy_id`, `policy_sha256`,
   `audit_ids_fingerprint`, `development_gate_met`.
2. **Wymiar przekroju zależny od kontraktu.** Eksport v1 → `primary_margin_gap_band`
   (bez zmian). Eksport v2 → **`axis`**. To jest jedyna zmiana o treści merytorycznej i
   jest wymuszona konstrukcyjnie: w v2 pasm marginesu nie ma, a osi w v1 nie ma.
3. **Nazwa pola etykiet zależna od kontraktu.** v1 → `rejected_failure_types`,
   v2 → `rejected_defect_labels`.

Nazwy kluczy w wyniku analizy idą za kontraktem eksportu, żeby **nie unieważnić
zamrożonego artefaktu v1**: dla v1 pozostają `agreement_by_primary_margin_gap_band`,
`decided_agreement_by_primary_margin_gap_band`, `agreement_by_rejected_failure_type`; dla
v2 są to `agreement_by_axis`, `decided_agreement_by_axis`,
`agreement_by_rejected_defect_label`.

### Gwarancja odtwarzalności v1

Ścieżka v1 zachowuje wszystkie dotychczasowe nazwy kluczy i wszystkie wartości. Jedyna
różnica w payloadzie analizy v1 to **jeden nowy klucz opisowy `export_contract`**; żaden
istniejący klucz nie zmienia nazwy ani wartości. Zapisuję to wprost, zamiast twierdzić
bajtową identyczność, której nowy klucz nie spełnia.

## Co pozostaje nietknięte

- **prompt i rubryka sędziów** — bez jednego znaku zmiany; wersja promptu nadal wchodzi
  do `identity.json`;
- **modele** (`gpt-oss-120b`, `qwen3.6-27b`), ich kolejność, temperatura i schemat
  odpowiedzi;
- **budżety i limity** dzienne, serializacja globalna, retry, wznawianie, ledgery,
  naprawa ID po przedrostku, obsługa `out_of_schema`;
- **reguły decyzyjne**: definicje `consensus_supports_automatic`,
  `consensus_contradicts_automatic`, `abstained`, `disagreement`,
  `eligible_for_automatic_acceptance`, kubełki pewności, sposób liczenia CI;
- kontrakt `pair_count=500` i sprawdzenie zgodności z manifestem;
- `human_evidence_claimed=false`, `safe_anchor_selection_signal=false`,
  `task07_training_authorized=false`, `final_tests_used=[]`.

## Konsekwencje

- Amendment jest spisany **przed** pierwszym requestem audytu v2 i nie wolno go rozszerzać
  po zobaczeniu wyników. Każda dalsza zmiana czytnika po odczycie ocen wymaga nowego,
  prospektywnego dokumentu.
- Bramka V2-05 pozostaje dokładnie taka, jaką zamroził ADR V2-03: wiążące P1–P4,
  fail-closed, bez poluzowania progów przy niedowiezieniu.
- Zamrożone artefakty audytu v1 (`artifacts/task06/preference_audit_v2/`, jego ledgery i
  analiza) nie są przeliczane ani nadpisywane.
