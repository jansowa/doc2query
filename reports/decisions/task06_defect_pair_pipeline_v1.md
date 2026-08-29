# ADR: pipeline par preferencyjnych z wadami nazwanymi z konstrukcji, v1 (2026-08-29)

## Status

**Prospektywny ADR, zamrożony przed uruchomieniem masowej generacji** na serwerze
inferencji właściciela (Qwen3.8-27B). Autoryzacja kierunku: decyzje właściciela z
2026-08-28 (teacher w generacji par, klasa problemu w metadanych) i 2026-08-29
(oś keywordowa, lematyzacja stanzą, „możemy iść w tym kierunku"). Projekt i
eksploracje: [plan](../plans/task07_defect_pair_pipeline_design_2026-08-29.md),
próbki `artifacts/task06/teacher_defect_explore_v1/` i
`artifacts/task06/teacher_pipeline_explore_v2/`.

Ten ADR zamraża: klasy, role populacji, prompty (przez `prompt_version` w
skrypcie), progi filtrów, reguły weryfikacji i kryteria składania. Zmiana
któregokolwiek elementu po obejrzeniu pass-rate'ów wymaga nowego ADR.
`final_tests_used=[]`.

## 1. Wejście i zakres

- **Grupy**: dokładnie te, z których pochodzą pary v3 (bundle turnieju,
  3 619 grup; pary buduje się dla grup obecnych w handoffie bottom — 2 730).
  Pasaż, prompt i `chosen` (zwycięzca turnieju) pozostają bajt w bajt te same.
  Zero nowych pasaży; kohorty v4–v11 pozostają zamknięte.
- **Serwer**: wyłącznie inferencja teachera (klasyfikacja, mutacje,
  answerability, potwierdzenia). Składanie par i pomiary lematyczne wykonują
  się lokalnie, deterministycznie, po przeniesieniu verdictów.

## 2. Populacje negatywów i klasy (metadane obowiązkowe)

Każda para niesie `defect_class`, `negative_population`
(`mined_organic` | `mutated_synthetic` | `off_topic_organic`) i `pair_class`
(`defect` | `lexical_contrast`). Klasy wad:

| `defect_class` | rozstrzyga | warunek przyjęcia negatywu |
|---|---|---|
| `copy_phrasing` | **wyłącznie mechanicznie**: ciągły wspólny fragment ≥5 słów z pasażem (po normalizacji NFKC, lowercase, tokeny `\w+`) | LCS ≥5 **i** answerability pasażu = TAK |
| `not_answerable` | teacher (klasyfikacja/mutacja) + answerability | answerability = **NIE** i pokrycie powierzchniowe ≥0,34 (na temat) |
| `too_general` | teacher + anty-równoważność | answerability dowolne, Jaccard słów vs `chosen` ≤0,6 |
| `answer_leak` | teacher + obecność tokenów faktu | answerability = TAK; forma zgodna z kontrolką (regex) |
| `off_topic` | już rozstrzygnięte (pary v3) | bez zmian; populacja kotwicząca |

LLM **nigdy** nie rozstrzyga `copy_phrasing` (eksploracja: nadgorliwość na
krótkich frazach) — jego etykieta tej klasy jest ignorowana i reklasyfikowana
mechanicznie.

## 3. Kolejność pozyskiwania negatywu w grupie (zamrożona)

1. **Kopalnia** (`mined_organic`): teacher klasyfikuje kandydatów studenckich
   spoza (`chosen`, obecny `rejected`). Kandydat przyjęty do klasy po przejściu
   filtrów §4 i weryfikacji §5.
2. **Mutacja** (`mutated_synthetic`): dla klas bez organicznej podaży w grupie —
   minimalna edycja `chosen` (prompty `mutate` z `prompt_version`
   `task06-teacher-pipeline-pl-v2`); nigdy wolna generacja.
3. Limit: **≤1 para na (grupę, klasę)**; grupa bez ważnego negatywu danej klasy
   nie wystawia pary tej klasy (fail-closed, bez dogenerowywania w kółko —
   dozwolona jest najwyżej jedna runda regeneracji mutacji na klasę).

## 4. Filtry deterministyczne S2 (progi zamrożone)

- równoważność: Jaccard zbiorów słów (`\w+`, lowercase) `rejected` vs `chosen`
  ≤0,6; dodatkowo identyczność po normalizacji odrzuca;
- długość: 2–24 słowa i stosunek długości do `chosen` w [0,4; 2,5];
- forma: `keyword_query` nie zaczyna się od słowa pytajnego i nie kończy „?";
  `full_question` przeciwnie (regex; lista słów pytajnych w skrypcie);
- `copy_phrasing`: LCS ≥5 (jak §2); pozostałe klasy wymagają LCS <5.

## 5. Weryfikacja LLM (serwer; głosy zapisywane w metadanych)

- **Answerability** (prompt `answerable` w skrypcie): pytanie „czy pasaż zawiera
  odpowiedź", zwraca `true/false`; liczona dla `chosen` (musi być TAK — inaczej
  grupa wypada w całości) i dla każdego kandydata na negatyw (wymóg wg klasy §2).
- **Potwierdzenie preferencji**: rubryka `R3_holistic` z polityki v3, **obie
  kolejności pozycji**, wymagana jednomyślność 2/2 na stronę `chosen`; remis
  lub rozjazd odrzuca negatyw.
- Wszystko temperature 0; `reasoning_effort`/`chat_template_kwargs` wg endpointu.

## 6. Klasa pary `lexical_contrast`

Budowana lokalnie z verdictów, na lematach (stanza pl, UPOS content words):

- `rejected`: kandydat klasy `not_answerable` o **maksymalnym** pokryciu
  lematycznym pasażu, wymagane ≥0,6;
- `chosen`: kandydat studencki zaklasyfikowany `ok` + answerability TAK o
  **minimalnym** pokryciu, wymagane ≤0,4; może różnić się od zwycięzcy turnieju
  (pozostaje tekstem studenta); potwierdzenie preferencji §5 obowiązuje;
- brak kandydata spełniającego próg po dowolnej stronie ⇒ grupa nie wystawia
  pary tej klasy. Teacher nigdy nie pisze strony `chosen`.

## 7. Bramki przed użyciem w treningu

1. Raport pass-rate per (klasa, populacja) — bez zmiany progów po obejrzeniu.
2. **Audyt anty-skrótowy**: regresja logistyczna na cechach powierzchniowych
   (długość w słowach, udział znaków interpunkcyjnych, pierwsze słowo,
   pokrycie powierzchniowe) rozdzielająca `chosen`/`rejected`; AUC >0,80 na
   klasie syntetycznej ⇒ klasa nie wchodzi do treningu bez amendmentu.
3. Ślepy spot-check właściciela ≥30 nowych par (kontrola operacyjna, nie panel).
4. Trening na nowej kohorcie wymaga **osobnej autoryzacji właściciela** —
   autoryzacja z 2026-08-28 obejmowała wyłącznie kohortę v3 bottom (2 461 par);
   dla porządku obejmujemy nią też wykonany wariant near_miss przez jego
   rejestrację jako ablacji w tym samym zakresie.
5. Kryterium rozstrzygające pozostaje probe embedder; metadane klasowe służą do
   slice'ów, nie do selekcji post-hoc.

## 8. Budżet i wykonanie

Szacunek: ~2 730 grup × (≈5 klasyfikacji + ≤3 mutacje + ≤7 answerability +
≤8 potwierdzeń) ≈ **50–65k wywołań**; przy tempie turnieju v3 (86k w 3,7 h) —
jedna sesja serwera. Runner wznawialny po journalu, równoległość konfigurowalna,
fail-fast po serii błędów; klucz/URL/model podawane wyłącznie argumentami.
