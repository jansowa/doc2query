# Autoryzacja treningu Task 07 na parach v3 (2026-08-28)

## Status

**Decyzja właściciela**, wypowiedziana 2026-08-28: „Autoryzuję trening DPO na
parach v3". Ten dokument ją zapisuje i wyznacza jej zakres. Od tej chwili
`task07_training_authorized=true` — ale **tylko** w zakresie opisanym w §2.

Nie jest to wynik żadnego pomiaru i nie wolno go tak cytować. Nowe artefakty
runtime nie mogą przepisywać `task07_training_authorized: false` jako pola
kontraktowego tam, gdzie dotyczy treningu ramion Task 07; zamknięte artefakty
sprzed tej daty pozostają bez zmian i bez reinterpretacji.

## 1. Co poprzedzało decyzję

| przesłanka | stan |
|---|---|
| polityka par v3 | zamrożony ADR, jednomyślność 6/6 |
| kalibracja selektora na etykietach z konstrukcji | 0,9793 na klasach osądzalnych |
| pary | 2 730 (2 461 train / 269 dev), klastrowo rozłączne |
| plan DPO | `task07-dpo-plan-v3-bottom-s42`, fingerprint `b1ab25b7…` |
| logproby referencji | 2 461/2 461, zero truncacji, zwalidowane |
| memory probe | 2,72 GiB precompute, 4,14 GiB trening (768) |
| Groq jako bramka | zdjęty [amendmentem](task06_v3_groq_role_amendment_2026-08-27.md) |
| ślepy spot-check 50 par | **niewykonany** (arkusz wygenerowany) |

## 2. Zakres autoryzacji

Wolno:

1. uruchomić **trzy ramiona** z planu (`dpo`, `continued_sft`,
   `score_weighted_continued_sft`) na kohorcie treningowej par v3, przy
   dopasowanym budżecie 154 kroków i 1 087 057 tokenów;
2. wybierać punkty na podstawie **dev**, w tym LR/beta z ablacji §Ablacje;
3. zapisywać adaptery i manifesty runów.

Nie wolno, i ta decyzja tego nie zmienia:

- **dotykać zbiorów testowych** — `final_tests_used=[]` obowiązuje dalej, kryterium
  rozstrzygającym pozostaje probe embedder na naturalnych zamrożonych zapytaniach;
- rozszerzać kohorty: pule v4–v11 pozostają zamknięte, 1 453 nieoglądanych par v2.1
  pozostaje nieoglądanych;
- reinterpretować bramki v2/v2.1 (`INCONCLUSIVE` zostaje `INCONCLUSIVE`);
- traktować wyniku treningu jako dowodu jakości selektora v3 — to osobne pytanie.

## 3. Warunek operacyjny, nie bramka

Amendment §2.3 przewidywał ślepy spot-check 50 par **przed** treningiem jako
kontrolę sanity. Właściciel autoryzował trening, mając ten spot-check niewykonany;
autoryzacja jest jego decyzją i nie wymaga uzupełnienia. Zapisuję jednak wprost:
spot-check pozostaje do wykonania, a jeśli wypadnie źle, jest to argument
przeciw parom, nie przeciw wynikowi treningu, i wtedy trzeba go rozpatrzyć osobno.
Spot-check nigdy nie jest panelem AGENTS.md §9.3 i nie wolno go tak raportować.

## 4. Ograniczenie objętości, zapisane wprost

Kohorta ma 2 461 par treningowych, a ablacje w `tasks/07_dpo_training.md` zakładają
20k/50k/100k. Ta rozbieżność nie jest rozwiązana i pozostaje zapisanym
ograniczeniem interpretacyjnym: brak efektu przy 2,5 tys. par nie dowodzi braku
efektu DPO, a obecność efektu wymaga kontroli continued SFT przy tym samym budżecie
— dlatego ramiona kontrolne są obowiązkowe, nie opcjonalne.

`final_tests_used=[]`.
