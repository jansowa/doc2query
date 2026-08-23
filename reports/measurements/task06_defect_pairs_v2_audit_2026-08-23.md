# Pomiar: ślepy audyt dual-LLM par defektowych v2 — bramka V2-05 (2026-08-23)

Polityka i predykcje (zamrożone **przed zbudowaniem pierwszej pary v2**):
[`task06_defect_pair_policy_v2.md`](../decisions/task06_defect_pair_policy_v2.md).
Amendment czytnika: [`task06_groq_audit_reader_axis_amendment_2026-08-21.md`](../decisions/task06_groq_audit_reader_axis_amendment_2026-08-21.md).
Eksport: `artifacts/task06/preference_audit_v3_defect_pairs/` (500 ślepych par,
orientacja 250/250, 500/500 zobowiązań zweryfikowanych).
Artefakty audytu: `.../groq_dual_llm/{analysis.json,pair_verdicts.jsonl,ledgers/}`.
Status runu: **`complete`**, 250/250 requestów u obu sędziów, 1000 ocen, zero
`audit_id` do naprawy, zero par bez werdyktu.

Audyt wykonano w dwóch oknach dziennych budżetów Groq (2026-08-21 i 2026-08-22),
bez zmiany promptu, rubryki, modeli, limitów ani reguł decyzyjnych.

## Werdykt: bramka V2-05 **niedowieziona** — dwie z czterech predykcji nie przeszły

| # | predykcja | próg | wynik v2 | baseline v1 | werdykt |
|---|---|---|---|---|---|
| P1 | `chosen` nieodpowiadalne, **u każdego sędziego** | ≤ 5,0% | `gpt-oss` **4,80%** / `qwen` **5,20%** | 16,6% / 18,8% | **FAIL** (qwen) |
| P2 | `consensus_supports_automatic` | ≥ 30% i CI > 24,4% | **30,80%**, CI [26,78%; 35,05%] | 24,4% | **PASS** |
| P3 | `consensus_contradicts_automatic` | ≤ 3,1% | **3,20%**, CI [1,84%; 5,14%] | 6,2% | **FAIL** |
| P4 | kontrast osi (nieodpowiadalne `rejected` A − B), u każdego sędziego | ≥ +20 pp | **+45,9 pp** / **+56,7 pp** | — | **PASS** |
| P5 | remisy (bez progu, kierunkowo) | — | 63,2% / 34,2% | 51,8%+9,2% / 32,2% | n/d |

Zgodnie z §11 ADR: **wszystkie cztery** wiążące predykcje muszą być dowiezione.
Nie są, więc pary v2 **nie zastępują niczego, nie idą do żadnego treningu, a
polityka wraca do projektowania**. Zabronione i niewykonywane: obniżanie progów,
zmiana cięcia osi B, zmiana kwot, zmiana rubryki sędziów, dobieranie sędziego
ani powtórzenie audytu na nowej próbce dla lepszej liczby.

### Obie porażki są o **jedną parę**

- P1 (`qwen`): 26 nieodpowiadalnych `chosen` na 500. Przy 25 byłoby dokładnie
  5,00% i predykcja by przeszła.
- P3: 16 sprzeczności na 500. Przy 15 byłoby 3,00%.

Zapisuję to, bo jest prawdą o danych, a **nie** jako argument za przejściem
bramki. Prerejestrowany próg punktowy jest progiem punktowym; fakt, że przedział
ufności obu wielkości zawiera próg (P1 `qwen` CI [3,42%; 7,53%], P3 CI [1,84%;
5,14%]), tnie w obie strony i tak samo dobrze pokazuje, że próba n=500 nie ma
mocy rozstrzygać na tym poziomie. To jest wada projektu predykcji, którą wolno
naprawić wyłącznie **nowym, prospektywnym** ADR — nie ponownym odczytem tych
danych.

## Co v2 jednak dowiozła: poprawa jest realna i duża

Polityka v2 poprawiła **każdy** mierzony wymiar wobec v1:

| miara | v1 | v2 |
|---|---|---|
| nieodpowiadalne `chosen` (`gpt-oss` / `qwen`) | 16,6% / 18,8% | **4,8% / 5,2%** |
| zgodność sędziego z automatem (`gpt-oss` / `qwen`) | 0,718 / 0,708 | **0,886 / 0,778** |
| zgodność **między** modelami | 0,879 | **0,983** |
| wykluczone z automatycznej akceptacji | 75,6% | **69,2%** |
| konsensus wspiera automat | 24,4% | **30,8%** |
| konsensus przeczy automatowi | 6,2% | **3,2%** |

Spadek nieodpowiadalnych `chosen` **3,5×** jest bezpośrednim potwierdzeniem, że
przypięty sędzia odpowiadalności (Qwen3.8-27B FP8) robi to, co obiecywał w
kalibracji, na niezależnej populacji i według niezależnych sędziów. Zgodność
między modelami 0,983 mówi, że to, co zostaje, nie jest szumem sędziów.

### P4 potwierdza mechanizm ponad wszelką wątpliwość

Oś A dobiera `rejected` m.in. po werdykcie `no`, oś B **wymaga** po tej stronie
`yes`. Niezależni sędziowie widzą dokładnie tę różnicę: nieodpowiadalne
`rejected` to 49,0% (oś A) wobec 3,1% (oś B) według `gpt-oss` i 63,0% wobec 6,3%
według `qwen`. Kontrast dwukrotnie przekracza próg u obu sędziów. Kotwiczenie
par w defektach **działa** jako mechanizm — to jest najmocniejszy wynik tego
audytu i nie unieważnia go niedowieziona bramka.

### Oś B jest słabym ogniwem

Zgodność konsensusu z automatem w osi A wynosi **0,974** (n=154), a w osi B
**0,250** (n=16). Podobnie w przekroju etykiet defektu: `judge_rank_disagreement`
1,000 (n=48), `judge_unanswerable` 0,993 (n=142), `lower_primary_margin` 0,984
(n=126), `weak_corpus_round_trip` 0,957 (n=69), ale `high_lexical_overlap`
**0,250** (n=16). Łatwość leksykalna, na której stoi oś B, **nie jest** dla
sędziów powodem, by uznać zapytanie za gorsze. Nakłada się to na wcześniejszy
niedobór podaży osi B (192 zamiast kwoty 250) i sugeruje, że problemem nie jest
cięcie `content_jaccard`, tylko sama hipoteza, że wyższy overlap czyni parę
preferencyjną. Liczby są małe (n=16 rozstrzygniętych), więc jest to przesłanka
do projektu, nie rozstrzygnięcie.

### Remisy wzrosły u `gpt-oss`

63,2% wobec 51,8%+9,2% w v1 (u `qwen` bez zmian: 34,2% wobec 32,2%). Predykcja
P5 jawnie nie stawiała progu i zakładała, że remisy nie spadną — nie spadły.
Pary defektowe bywają subtelne, zwłaszcza w osi B, gdzie obie strony są
odpowiadalne z konstrukcji. Abstencja konsensusu to 327/500 (65,4%).

## Konsekwencje

1. **Task 07 pozostaje zamknięty.** `task07_training_authorized=false` bez
   zmian; pary v2 nie są danymi treningowymi.
2. Artefakty v1/v1.1 i eksporty audytów pozostają nietknięte; ten audyt niczego
   nie unieważnia.
3. Wynik jest **negatywny wobec bramki, a nie wobec kierunku**: mechanizm osi
   defektowych potwierdzony (P4), filtr odpowiadalności potwierdzony (spadek
   3,5×), konsensus poprawiony (P2). Do decyzji właściciela pozostaje, czy
   projektować v2.1 — i przy jakiej mocy próby — czy uznać próg P1/P3 za
   niewłaściwie postawiony i przeprojektować samą regułę decyzyjną. Obie ścieżki
   wymagają nowego prospektywnego ADR **przed** jakimkolwiek kolejnym odczytem.
4. Nie jest to human evidence i nie wolno tak tego raportować.

`final_tests_used=[]`. `task07_training_authorized=false`.
