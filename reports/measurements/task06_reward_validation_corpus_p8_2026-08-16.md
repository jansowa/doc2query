# Pomiar: P8 na ocenionym korpusie walidacyjnym nagrody (2026-08-16)

ADR predykcji (zapisany **przed** generacją i przed dostępem do GPU):
[`task06_reward_validation_corpus_v1.md`](../decisions/task06_reward_validation_corpus_v1.md).
Amendment autoryzujący GPU:
[`task06_llm_cohort_gpu_scoring_amendment_2026-08-16.md`](../decisions/task06_llm_cohort_gpu_scoring_amendment_2026-08-16.md).
Artefakty: `artifacts/task06/reward_validation_corpus_v1/scoring/` (1440/1440
rekordów, `elapsed_seconds=175.3`, batch 8), `measurement_p8.json`.

Scoring poszedł **tą samą** ścieżką co kandydaci lokalnego generatora
(`evaluate_intrinsic_records`: primary builder, shadow niezależna kontrola,
corpus round-trip na zamrożonym BM25 o 2,4 mln dokumentów). Nie napisano żadnego
nowego kodu oceniającego.

Uwaga o wznawianiu: pierwszy run przerwał się na 944/1440 rekordach po
zakończeniu poprzedniej sesji. Ponowne uruchomienie tej samej komendy wypisało
`[intrinsic resume] 944/1,440 rows durable` i dokończyło bez powtarzania
pracy — mechanizm dziennika działa w praktyce, nie tylko w projekcie.

## Wynik P8

| predykcja | wynik | próg | ocena |
|---|---|---|---|
| **P8a** `ungrounded` ma niższy primary score niż `good_specific` | **0.9556** (172/180, 0 remisów) | ≥ 0.85 | **PASS** |
| **P8b** `too_general` ma słabszy corpus round-trip niż `good_specific` | 124/180 słabszych, 54 remisy, 2 odwrócone | **brak progu w ADR** | kierunkowo potwierdzona, bez werdyktu |

P8b była zapisana kierunkowo, bez liczby, więc **nie orzekam PASS/FAIL** —
dopisanie progu po zobaczeniu danych byłoby kalibracją po fakcie. Rozkład
`corpus_round_trip_at_20` w parach (good, too_general): `(1,0)` 124 razy,
`(0,0)` 43, `(1,1)` 11, `(0,1)` 2. Średnie: **0.750 vs 0.072**. 43 remisy to
przypadki, w których również `good_specific` nie wróciło w top-20 — to informacja
o trudności korpusu, nie o metryce.

Kontrola shadow (niezależna, nie część P8): `ungrounded` niżej niż
`good_specific` w **96.7%** grup. Dwa niezależne sędziowie zgadzają się co do
kierunku, więc sygnał gruntowania nie jest artefaktem jednego modelu.

## Profile klas według zamrożonych sygnałów

| klasa | primary (`pool_positive_score`) | `pool_margin` | round-trip @20 | @100 |
|---|---|---|---|---|
| `copy_verbatim` | **11.99** | **7.71** | 0.989 | 1.000 |
| `good_specific` | 10.67 | 5.21 | 0.750 | 0.883 |
| `too_general` | 5.93 | 1.12 | 0.072 | 0.156 |
| `ungrounded` | 4.08 | **−0.48** | 0.272 | 0.422 |

Dwie rzeczy warte podkreślenia:

**`ungrounded` ma ujemny średni margines.** Primary stawia twardy negatyw wyżej
niż pozytyw, gdy zapytanie pyta o coś, czego w pasażu nie ma. Sygnał gruntowania
działa więc nie tylko rankingowo, ale i w wartości bezwzględnej.

**Sygnały sędziowskie nagradzają kopiowanie najmocniej z wszystkich klas.**
`copy_verbatim` bije `good_specific` na primary, na marginesie i na round-trip.
W **95/180 grup (52.8%)** dosłowna kopia fragmentu pasażu ma **najwyższy**
`pool_margin` w całej grupie.

## Konsekwencja dla zamrożonej polityki par (weryfikacja krzyżowa)

Polityka `task06-tentative-pair-policy-v1` (zamrożona osobno tego samego dnia)
używa `pool_margin` jako **jedynego** sygnału budującego. Powyższa liczba
oznacza, że bez ochrony przed kopiowaniem ponad połowa par wybrałaby jako
`chosen` dosłowną kopię pasażu, a DPO uczyłoby się kopiować zamiast pytać.

Polityka ma tę ochronę: `copy_risk.reject_chosen_on_copy_risk: true`
(odziedziczone z Task 05: `copy_density > 0.6`, `normalized_lcs > 0.8`,
`minimum_query_words: 4`). Zmierzona skuteczność na tym korpusie:

| klasa | guard odrzuca jako `chosen` |
|---|---|
| `copy_verbatim` | **180/180 = 100%** |
| `good_alternative` (poprawna, forma keyword) | 53/180 = **29.4%** |
| `good_specific` | 5/180 = 2.8% |
| `near_duplicate_of_good` | 5/180 = 2.8% |
| `too_general` | 0/180 = 0% |

Guard jest więc **konieczny i skuteczny** wobec wzorca, który sygnały sędziowskie
nagradzają — łapie wszystkie 180 kopii. Ma jednak mierzalny koszt: odrzuca 29.4%
poprawnych, krótkich zapytań w formie `keyword_query`. Dekompozycja przyczyn dla
`good_alternative`: `normalized_lcs` samodzielnie 15, `copy_density` samodzielnie
9, `minimum_query_words` samodzielnie 13, reszta w kombinacjach (razem 53).
Powód jest strukturalny: zapytania kluczowe mają 2–5 słów, więc krótki wspólny
podciąg z pasażem daje wysoki `normalized_lcs` przy zerowej intencji kopiowania,
a próg `minimum_query_words: 4` wyklucza legalne frazy 2–3-słowowe.

Nic z tego **nie zmieniam**: `copy_risk` jest odziedziczonym, zamrożonym
kontraktem Task 05, a polityka par została zamrożona przed odczytem par.
Zapisuję to jako przesłankę na wypadek, gdyby przyszły audyt dual-LLM pokazał
niedoreprezentowanie formy `keyword_query` wśród `chosen` — wtedy będzie znana
zmierzona przyczyna i będzie ona wymagała własnego, prospektywnego ADR.

## Granice

- Etykiety pochodzą z konstrukcji autora (model), nie od człowieka; to nie jest
  human evidence ani panel kalibracyjny Task 02.
- Pomiar jest diagnostyką komponentów: nie zmienił żadnego progu, nie kalibrował
  bramki różnorodności, nie dotknął progu `source_en_score >= 23.50`.
- Korpus nie wszedł do frozen train, do kohort preferencyjnych ani do par;
  niczego na nim nie trenowano.
- `final_tests_used=[]`.
