# Pomiar: podaż osi A po certyfikacji odpowiadalności

Kontrakt: `task06-axis-a-supply-after-certification-v1`, status
`design_input_measured_no_pairs_built`. Kod:
`src/doc2query/preferences/axis_a_supply.py`,
`scripts/measure_task06_axis_a_supply.py`, 7 testów CPU. Artefakty:
`reports/measurements/task06/axis_a_supply_v1/{authorized,rest}.json`.
Sygnał odpowiadalności: sędzia `task06-answerability-judge-v1` **przyjęty bramką K1–K3**
([raport](task06_answerability_judge_v1_2026-08-20.md)). Wejścia pinowane po SHA-256.
`task07_training_authorized=false`, `final_tests_used=[]`, żadnej pary nie zbudowano,
żadnego progu nie zamrożono.

To jest **wejście projektowe** dla ADR V2-03, liczone tą samą definicją, jaką inwentarz
V2-00 mierzył *bez* sygnału odpowiadalności — z tą jedną różnicą, że rolę round-tripu jako
proxy odpowiadalności przejął zwalidowany sędzia.

## Definicje (te same, co w V2-00, plus werdykt sędziego)

- **certyfikowany `chosen`**: czysty wg zamrożonych warunków polityki (format + guard
  wtrącenia, round-trip @20, brak halucynacji encji, `pool_margin > 0`, brak copy-risk)
  **oraz** werdykt sędziego `yes`;
- **`rejected` osi A**: kandydat dopuszczalny formatem z nazwanym defektem — werdykt `no`
  **albo** brak round-tripu @100;
- **grupa parowalna**: ma obie strony na **różnych** kandydatach, spełniających zamrożone
  ograniczenie różnorodności (`normalized_query_jaccard ≤ 0,85`);
- `uncertain` blokuje rolę `chosen` i **nie jest defektem** — nie może wyprodukować strony
  `rejected`, zgodnie z §3 ADR sędziego.

## Wynik

| zakres | grup `eligible` | certyfikowany `chosen` | `rejected` osi A | obie strony | **pary** |
|---|---|---|---|---|---|
| autoryzowane (v1+v2+v3) | 3 619 | 2 362 (65,3%) | 3 506 (96,9%) | 2 254 | **2 253 (62,3%)** |
| v4–v11 | 22 373 | 14 341 (64,1%) | 21 756 (97,2%) | — | **13 736 (61,4%)** |
| **razem** | **25 992** | 16 703 | 25 262 | — | **15 989 (61,5%)** |

Per kohorta autoryzowana: v1 **199** par z 362 grup (55,0%), v2 **270** z 466 (57,9%),
v3 **1 784** z 2 791 (63,9%).

Werdykty na poziomie kandydata: w kohortach autoryzowanych `yes` 10 804 / `no` 12 804 /
`uncertain` 68 (0,3%); w v4–v11 `yes` 66 683 / `no` 81 502 / `uncertain` 434 (0,3%).
**Zero kandydatów bez werdyktu** — złączenie po `sha256(prompt, zapytanie, pasaż)` pokryło
wszystkie 172 295 reprezentantów bramki.

## Trzy wnioski, które wchodzą wprost do ADR V2-03

**1. Podaż nie jest już wąskim gardłem — i to zmienia arytmetykę całego zadania.**
Polityka v1.1 po bramce audytu dawała **122 pary akceptowalne** z 500 ocenionych, czyli o
rząd wielkości poniżej progu 1000 par przed finalnym DPO. Oś A z samych kohort
autoryzowanych daje **2 253 pary**, a z wszystkich jedenastu **15 989**. Argument „nie ma
z czego zbudować danych treningowych", który był najmocniejszym zarzutem wobec v1.1,
przestaje obowiązywać dla v2.

**2. Koszt konserwatywności sędziego jest zmierzony i akceptowalny.** Czysty `chosen`
przed filtrem miały 2 971 grupy (82,1%), po filtrze 2 362 (65,3%) — sędzia zachowuje
**79,5%** podaży, a stabilnie: 79,7% / 75,5% / 80,1% w v1/v2/v3 i 79,1% w v4–v11.
Zgadza się to z kierunkiem obciążenia wykrytym w kalibracji (`recall_no` 0,9429 przy
`recall_yes` 0,8328): filtr traci część kandydatów faktycznie odpowiadalnych, ale nie
wpuszcza nieodpowiadalnych. Przy 2 253 dostępnych parach ten koszt nie jest wiążący.

**3. Naturalnych `rejected` jest w nadmiarze, więc konstruowane są niepotrzebne.**
Defektowy kandydat osi A istnieje w **96,9%** grup autoryzowanych i 97,2% w v4–v11. To
niezależnie potwierdza decyzję właściciela o pominięciu V2-04 (konstruowane rejected) w
pierwszym wydaniu polityki v2 — i robi to teraz na mocnym sygnale, a nie na round-tripie,
który audyt v1 zdyskwalifikował jako miernik odpowiadalności.

## Dodatkowo: ograniczenie różnorodności nie kosztuje prawie nic

Grup z obiema stronami było 2 254, parowalnych 2 253 — zamrożona bramka
`normalized_query_jaccard ≤ 0,85` odrzuciła **jedną** grupę. Wynika to z konstrukcji:
strony różnią się defektem odpowiadalności, a nie przestawieniem słów, więc rzadko są
leksykalnie bliźniacze. Dla kontrastu w polityce v1 to samo ograniczenie działało na parach
dobranych marginesem, gdzie bliskie duplikaty były częste (ilustracja z audytu: „definicja
fall branch w tn" vs „definicja fall branch").

## Czego ten pomiar nie robi

Nie buduje par, nie zamraża kwot ani progów, nie wybiera tie-breaków, nie przypisuje osi
grupom i nie autoryzuje budowy par v2 — to wszystko należy do ADR V2-03, który musi
zamrozić predykcje **przed** zbudowaniem pierwszej pary. Nie mierzy osi B (pasmo
overlapu) ani osi C (focus, wypadła z pierwszego wydania decyzją właściciela). Nie zmienia
polityki v1/v1.1 ani jej artefaktów, nie otwiera testów finalnych i nie autoryzuje
treningu.
