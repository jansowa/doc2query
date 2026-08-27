# Pomiar: kalibracja selektora preferencji v3 (2026-08-27)

Protokół zamrożony **przed pierwszym wywołaniem**:
[`task06_judge_selected_pair_policy_v3.md`](../decisions/task06_judge_selected_pair_policy_v3.md) §6.
Kod: `src/doc2query/preferences/pair_selector_v3.py`,
runner: `scripts/run_task06_v3_selector_calibration.py` i `scripts/run_v3_calibration.sh`.
Artefakty: `artifacts/task06/v3_selector_calibration_v1/{direct,reasoning}/`
(`judgments.journal.jsonl`, `run_summary.json`, `calibration_report.json`).

Sędzia: `qwen3.8-27b` FP8 na endpoincie vLLM operatora (adres wyłącznie parametrem
CLI). Zbiór: **1 800 par** o kierunku **znanym z konstrukcji** — 2 klasy dobre × 5 klas
zepsutych × 180 pasaży korpusu walidacyjnego nagrody. Każda para: 3 rubryki × 2
kolejności = 6 głosów, `temperature=0`.

Ten pomiar **nie zbudował ani jednej pary v3** i nie autoryzuje treningu.
`task07_training_authorized=false`, `final_tests_used=[]`.

## 1. Wykonanie

| ramię | wywołania | porażki | tempo | czas | pary kompletne |
|---|---|---|---|---|---|
| bez rozumowania | 10 800 | **0** | 5,81/s | 31 min | 1 800 |
| z rozumowaniem | 10 498 | **302** (2,8%) | 0,44/s | **405 min** | 1 568 |

Ramię z rozumowaniem jest **13× wolniejsze** i traci 232 pary na obcięciu budżetu
1 024 tokenów (myślenie zjada limit przed werdyktem).

## 2. Wynik pozorny i dlaczego jest mylący

Zagregowana czystość na wszystkich pięciu klasach zepsutych, przy jednomyślności 6/6:
**0,8498** (zatrzymane 1 205 par z 1 800, 181 ze złym kierunkiem). Przejście z 2 do 6
głosów kupowało zaledwie +2,5 pp czystości (0,824 → 0,850) za jedną trzecią
wydajności — co wyglądało na wniosek, że spójność sędziego nie informuje o
poprawności, a filtr agregacyjny nie działa.

**Ten wniosek jest błędny i winna jest konstrukcja zbioru, nie sędzia.** Rozbicie
czystości per klasa zepsuta (6/6, ramię bez rozumowania) pokazuje dwie zupełnie różne
populacje:

| klasa zepsuta | zatrzymane | czystość |
|---|---|---|
| `ungrounded` | 265 | **1,0000** |
| `copy_verbatim` | 227 | **1,0000** |
| `too_general` | 282 | **0,9433** |
| `wrong_form` | 242 | 0,8099 |
| `wrong_focus` | 189 | **0,3704** |

`wrong_focus` i `wrong_form` są zepsute **względem kontrolki, której sędzia nie
widzi** — żądanego fragmentu pasażu i żądanej formy. Zapytanie o inny fragment bywa
samo w sobie lepsze, więc sędzia je wybiera i z własnej perspektywy ma rację: nikt mu
nie powiedział, o co pytamy. Ślepość, którą ADR nakłada w §3, czyni te dwie klasy
**nierozstrzygalnymi z definicji**, a 0,3704 to wynik poniżej losowego, bo
wrong-focus bywa lepiej sformułowane.

Potwierdza to zachowanie rubryk osobno (ramię bez rozumowania):

| rubryka | klasy rozstrzygalne | klasy nierozstrzygalne ślepo |
|---|---|---|
| `R1_grounding` | 0,871 | **0,521** (poziom losowy) |
| `R2_retrieval_usefulness` | 0,878 | 0,610 |
| `R3_holistic` | **0,936** | 0,631 |

Wcześniejsze „R1 ma tylko 0,731" było artefaktem mieszania populacji: na parach, gdzie
ugruntowanie obu stron jest identyczne, rubryka ugruntowania musi wybierać losowo.

## 3. Wynik właściwy: krzywa na klasach rozstrzygalnych

`ungrounded`, `copy_verbatim`, `too_general` — 1 080 par kompletnych, ramię bez
rozumowania:

| reguła agregacji | zatrzymane | wydajność | **czystość** | zły kierunek |
|---|---|---|---|---|
| ≥2 głosy (1 rubryka spójna) | 1 059 | 0,981 | 0,9547 | 48 |
| ≥4 głosy (2 z 3 rubryk) | 1 033 | 0,956 | 0,9584 | 43 |
| **≥6 głosów (jednomyślność)** | **774** | **0,717** | **0,9793** | **16** |

Ramię z rozumowaniem daje na tych klasach 0,9749 przy ≥4 i **0,9871** przy ≥6, ale przy
wydajności 0,581 i 13× koszcie.

Dla kontrastu, te same progi na klasach nierozstrzygalnych ślepo dają czystość
**0,619 / 0,628 / 0,617** — płaską wobec progu. To jest podręcznikowy obraz błędu
systematycznego: zaostrzanie kryterium spójności nie poprawia niczego, bo sędzia
konsekwentnie odpowiada na inne pytanie niż to, które zadaje konstrukcja.

Głosy liczą się parami (rubryka wnosi dwa głosy tylko przy zgodzie obu kolejności),
więc osiągalne progi są parzyste: 2, 4, 6. Nieparzyste dawały te same zbiory i zostały
usunięte z raportu.

## 4. Obciążenie pozycyjne

Udział par, w których zamiana kolejności odwraca werdykt:

| rubryka | bez rozumowania | z rozumowaniem |
|---|---|---|
| `R1_grounding` | 0,124 | 0,115 |
| `R2_retrieval_usefulness` | 0,091 | 0,110 |
| `R3_holistic` | 0,097 | **0,059** |

Obciążenie jest realne i rzędu 10%, co uzasadnia wymóg z §3 ADR: bez zamiany pozycji
co dziesiąta para byłaby rozstrzygnięta pozycją, nie treścią. Rozumowanie zmniejsza je
wyłącznie w rubryce holistycznej.

## 5. Wnioski wiążące dla amendmentu progu

1. **Selektor v3 jest sprawny w zakresie, który widzi**: ugruntowanie, kopiowanie i
   ogólność. Przy jednomyślności czystość 0,979, a dwie z trzech klas rozpoznane
   bezbłędnie na kilkuset parach.
2. **Selektor v3 nie jest i nie będzie sędzią zgodności z formą i focusem** — ślepo
   nie da się tego ocenić. Oś C zostaje poza wydaniem, teraz z **pomiarem** zamiast
   argumentu o zepsutym etykieciarzu, a par v3 nie wolno kredytować kontrolą formy.
3. **Rozumowanie odrzucone**: +0,8 pp czystości przy 6/6 wobec 13× kosztu, 2,8%
   porażek i 232 par utraconych na obcięciu budżetu. Wynik negatywny, zmierzony.
4. Kalibracja **nie mierzyła** wyniku probe ani jakości DPO. Mierzyła zgodność
   selektora z etykietami z konstrukcji na zbiorze rozmyślnie trudnym.

`task07_training_authorized=false`, `final_tests_used=[]`.
