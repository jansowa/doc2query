# Diagnostyka post hoc: skąd wzięła się porażka bramki V2-05 (2026-08-23)

**Status dowodowy: analiza eksploracyjna na danych zamkniętej, przegranej
bramki. Nie jest pomiarem konfirmacyjnym i nie zmienia werdyktu.** Bramka V2-05
pozostaje **niedowieziona**
([raport](task06_defect_pairs_v2_audit_2026-08-23.md)), pary v2 nadal nie idą do
żadnego treningu, a żaden próg nie został zmieniony. Jedyną dopuszczalną rolą
tych liczb jest **wejście projektowe** do nowego, prospektywnego ADR (v2.1),
którego predykcje trzeba będzie zweryfikować na **nowych** parach i **nowym**
audycie. Dobieranie przekroju tak, by przegrana bramka „jednak przeszła", jest
wykluczone i nie jest tu robione.

Dane: te same 500 par i 1000 ocen, bez żadnego nowego requestu.

## Wynik: obie porażki pochodzą z osi B

| miara | oś A (n=308) | oś B (n=192) | całość (n=500) | próg |
|---|---|---|---|---|
| P1 nieodpowiadalne `chosen`, `gpt-oss` | **3,90%** [2,03; 6,71] | 6,25% [3,27; 10,66] | 4,80% | ≤ 5% |
| P1 nieodpowiadalne `chosen`, `qwen` | **3,25%** [1,57; 5,89] | 8,33% [4,84; 13,18] | 5,20% ✗ | ≤ 5% |
| P3 sprzeczności konsensusu | **1,30%** [0,35; 3,29] | 6,25% [3,27; 10,66] | 3,20% ✗ | ≤ 3,1% |

Oś B to 38,4% próbki, a wnosi:

- **2,4 z 4,8 pp** (gpt-oss) i **3,2 z 5,2 pp** (qwen) udziału nieodpowiadalnych
  `chosen`;
- **2,4 z 3,2 pp** sprzeczności konsensusu, czyli **trzy czwarte** całego P3.

W obu przypadkach oś A sama w sobie leży wyraźnie pod progiem, a oś B wyraźnie
nad nim. Nie jest to więc rozmyta różnica ogonów, tylko dwie różne populacje
zsumowane w jednej liczbie.

## Interpretacja: oś B nie jest osią defektu

Zbiera się w to trzeci niezależny sygnał przeciw osi B, wszystkie zmierzone:

1. **Sędziowie nie zgadzają się z jej kierunkiem.** Zgodność konsensusu z
   automatem wynosi 0,250 w osi B (n=16) wobec 0,974 w osi A (n=154); etykieta
   `high_lexical_overlap` również 0,250.
2. **Nie dowozi podaży.** 192 pary wobec kwoty 250, mimo prerejestrowanej
   realokacji.
3. **To ona psuje bramkę** — liczby powyżej.

Hipoteza, którą trzeba prospektywnie przetestować, a nie ogłosić: wyższe
pokrycie leksykalne **nie jest** defektem, który człowiek albo sędzia uzna za
powód gorszej jakości zapytania. Cięcie `content_jaccard` jest tu prawdopodobnie
niewinne — problemem jest sama hipoteza osi, a nie jej próg.

Zastrzeżenie: podpróby osi B są małe (n=16 rozstrzygniętych par konsensusu,
12–16 zdarzeń w P1/P3), więc każda z tych liczb osobno jest szumna. Zbieżność
trzech niezależnych sygnałów jest mocniejsza niż każdy z nich.

## Czego ta diagnostyka **nie** ustala

- **Nie ustala, że polityka „tylko oś A" przeszłaby bramkę.** Liczby osi A
  pochodzą z tej samej próbki, na której bramka przegrała, i mają własne
  przedziały (górna granica CI dla P1 sięga 6,71%, czyli powyżej progu 5%).
  Dopiero prospektywny test na nowych parach może cokolwiek orzec.
- **Nie usuwa wady konstrukcyjnej predykcji.** Progi punktowe 5% i 3,1% przy
  n=500 mają przedziały szerokości ~±2 pp, więc rozstrzygnięcie „o jedną parę"
  było wpisane w projekt bramki. v2.1 musi ustalać próg **razem** z liczebnością
  próby — przez wymóg na granicę CI albo przez jawny rachunek mocy — inaczej
  powtórzy ten sam błąd przy innej liczbie.
- Nie jest human evidence i nie autoryzuje niczego.

## Wejście projektowe do ADR v2.1

Do rozstrzygnięcia przez właściciela, prospektywnie i przed jakimkolwiek
kolejnym odczytem:

1. czy oś B wypada z polityki, czy dostaje **inną definicję defektu**
   (obecna, oparta na `content_jaccard`, ma trzy niezależne sygnały przeciw);
2. jaka reguła decyzyjna zastępuje progi punktowe — wymóg na granicę CI,
   rachunek mocy albo większa próbka (2000 par to ~4× koszt, rząd 8–12 okien
   dziennych budżetów Groq);
3. czy przy polityce jednoosiowej podaż osi A (2086 par zbudowanych, 15 989 par
   dostępnych w puli) pozwala na próbkę audytową większą niż 500.

`task07_training_authorized=false`. `final_tests_used=[]`.
