# Ślepy spot-check właściciela na 50 parach v3 (2026-08-28)

## Status

**Kontrola operacyjna** z amendmentu
[`task06_v3_groq_role_amendment_2026-08-27.md`](../decisions/task06_v3_groq_role_amendment_2026-08-27.md)
§2.3. **Nie jest to panel AGENTS.md §9.3** i nie wolno tak tego raportować: jeden
oceniający, próbka 50 par, brak zamrożonego progu i brak roli bramkowej. Wykonana
po autoryzacji treningu, w trakcie runu ramion; nie wpływa na żaden run.

Arkusz był ślepy: strony losowo przypisane do A/B (24 razy `chosen` jako A, 26 razy
jako B), klucz w osobnym pliku, w arkuszu żadnych etykiet, identyfikatorów ani
śladu głosowania selektora. Seed próbki 20260827.

Artefakty: `artifacts/task07/handoff_v3_bottom/owner_spot_check/`
(`spot_check_answers.txt`, `spot_check_result.json`). `final_tests_used=[]`.

## 1. Wynik

| pomiar | wartość |
|---|---|
| odpowiedzi | 50 / 50 |
| zgodność z selektorem | **42** |
| niezgodność | **2** (pozycje 15, 29) |
| „bez różnicy" | **6** |
| rozstrzygnięte | 44 |
| **zgodność** | **95,45%** |
| dwustronny 95% CI (Clopper-Pearson) | **[84,53%; 99,44%]** |

Progu nie było przed pomiarem i nie zostaje dopisany po nim.

## 2. Czego ten wynik **nie** dowodzi

Zgodność 95,45% jest zgodna z tym, co selektor obiecywał (0,9793 czystości na
etykietach z konstrukcji), ale **nie jest dowodem, że selektor dobrze rozpoznaje
lepsze zapytanie**. Powód jest zmierzony osobno i wcześniej
([diagnostyka kontrastu](task07_pair_contrast_diagnostic_2026-08-28.md)): w 75,3%
par strona `rejected` jest w praktyce nie o tym pasażu. Zgadzanie się z selektorem
na takich parach mierzy głównie to, że oba oceniające podmioty odróżniają temat.

Zbieżność trzech niezależnych liczb jest tu wymowna:

| kto/co ocenia | trafność |
|---|---|
| właściciel, 44 rozstrzygnięte pary | 95,45% |
| punkt startowy SFT (bez treningu), 269 par dev | 93,68% |
| selektor Qwen3.8-27B, etykiety z konstrukcji | 97,93% |

Człowiek i **nietrenowany** model startowy trafiają niemal tak samo. To znaczy, że
zadanie rozstrzygane w tych parach jest zadaniem, które model w dużej mierze już
umie — a nie że wszyscy troje są zgodnie doskonali.

## 3. Gdzie właściciel się nie zgodził albo zawahał

Osiem pozycji bez zgody (2 niezgody + 6 remisów) ma **wyższe** pokrycie pasażu po
stronie odrzuconej niż pozycje zgodne:

| grupa | n | mediana pokrycia `rejected` |
|---|---|---|
| zgoda | 42 | 0,185 |
| „bez różnicy" | 6 | 0,290 |
| niezgoda | 2 | 0,145 |

Czyli wahanie pojawia się dokładnie tam, gdzie **obie** strony są sensownie
związane z pasażem i wybór dotyczy jakości, nie tematu. Przykłady:

- **#20** `co zawiera quesadilla` vs `czy quesadilla jest pikantna` — dwa poprawne
  zapytania o ten sam pasaż; wybór jest kwestią gustu, nie poprawności.
- **#45** `jaki jest zasięg występowania orangutanów` vs `jakie jest środowisko
  orangutana` — jak wyżej.
- **#29** (niezgoda) `jak brzmi imię Tallie dla dziewczynki` vs `jak brzmi imię w
  języku Hindi dla dziewczyny o imieniu Tallie` — właściciel wybrał wersję
  bardziej specyficzną, selektor krótszą.
- **#15** (niezgoda) `definicja czasu standardowego w Lawrenceville` vs `definicja
  czasu standardowego cdt`.

To jest ta sama granica, którą kalibracja selektora zapisała jako zakres ważności:
selektor jest walidowany na ugruntowaniu, kopiowaniu i ogólności, a nie na formie i
focusie.

## 4. Wniosek operacyjny

Kontrola nie wykazała wady par, która blokowałaby trening — pary są spójne z
polityką i czytelne dla człowieka. Wykazała natomiast, że **niosą mało trudnego
sygnału**, co jest wnioskiem o danych, nie o metodzie, i było już zapisane przed
tym pomiarem. Kryterium rozstrzygającym pozostaje probe embedder na naturalnych
zamrożonych zapytaniach.
