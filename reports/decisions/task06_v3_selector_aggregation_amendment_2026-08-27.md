# Amendment: próg agregacji selektora v3 i jego zakres ważności (2026-08-27)

## Status

**Amendment do ADR** [`task06_judge_selected_pair_policy_v3.md`](task06_judge_selected_pair_policy_v3.md),
przewidziany w jego §6 i spisany **po odczycie kalibracji, a przed zbudowaniem
pierwszej pary v3**. Podstawa liczbowa:
[`task06_v3_selector_calibration_2026-08-27.md`](../measurements/task06_v3_selector_calibration_2026-08-27.md).

ADR świadomie nie zamrażał tego progu, żeby nie powtórzyć błędu P3 z v2.1, gdzie próg
wziął się z arytmetyki („połowa spadku wobec v1"), a nie z pomiaru. Ten dokument
zamyka tę lukę jedną decyzją i jednym ograniczeniem zakresu.

## 1. Zamrożona reguła agregacji: jednomyślność 6/6

Para wchodzi do zbioru treningowego v3 **tylko wtedy**, gdy wszystkie trzy rubryki
wskazują tę samą stronę **w obu kolejnościach** — sześć zgodnych głosów z sześciu.
Każda inna konfiguracja (`position_flip` w którejkolwiek rubryce, rozbieżność między
rubrykami, remis) oznacza, że para **nie powstaje**.

Wyprowadzenie z krzywej czystość/wydajność na klasach rozstrzygalnych (1 080 par,
ramię bez rozumowania):

| reguła | wydajność | czystość |
|---|---|---|
| ≥2 głosy | 0,981 | 0,9547 |
| ≥4 głosy | 0,956 | 0,9584 |
| **≥6 głosów** | **0,717** | **0,9793** |

Wybieram 6/6, a nie 4/6, mimo że 4/6 zachowuje o jedną trzecią więcej par. Powód jest
ilościowy: 4/6 zostawia **4,2% par o odwróconym kierunku**, a 6/6 zostawia **2,1%**.
Przy DPO szum preferencji wchodzi wprost w gradient, a **objętość jest w tym programie
rozwiązywalna inaczej** — pula osi A ma 15 989 kandydatów, z czego 13 736 w kohortach
v4–v11 — natomiast czystość odzyskuje się wyłącznie surowszym filtrem. Zamiana
„więcej danych" na „mniej szumu" jest tu tania, a odwrotna nie jest.

Osiągalne progi są parzyste (2, 4, 6), bo rubryka wnosi dwa głosy tylko przy zgodzie
obu kolejności; nieparzyste dawały identyczne zbiory.

## 2. Zamrożony zakres ważności: trzy defekty, nie pięć

Selektor v3 jest **zwalidowany wyłącznie** dla defektów, które ślepy sędzia widzi:

- **brak ugruntowania** (`ungrounded`) — czystość 1,0000 na 265 parach;
- **kopiowanie** (`copy_verbatim`) — 1,0000 na 227 parach;
- **nadmierna ogólność** (`too_general`) — 0,9433 na 282 parach.

Selektor v3 **nie jest** zwalidowany i nie wolno go używać ani opisywać jako sędziego:

- **zgodności z żądaną formą** — czystość 0,8099;
- **zgodności z żądanym focusem** — czystość **0,3704**, czyli **poniżej losowego**.

Powód jest strukturalny, nie jakościowy: obie te klasy są zepsute *względem kontrolki,
której sędzia z założenia nie widzi* (§3 ADR nakłada ślepość). Zapytanie o inny
fragment pasażu bywa samo w sobie lepsze i sędzia je wybiera — z własnej perspektywy
słusznie. Zaostrzanie progu nic tu nie daje: czystość na tych klasach jest płaska
(0,619 / 0,628 / 0,617 dla 2, 4 i 6 głosów), co jest sygnaturą błędu systematycznego,
a nie szumu.

**Konsekwencje zapisane wiążąco:**

1. **Oś C (focus) zostaje poza wydaniem**, teraz z pomiarem zamiast argumentu o
   zepsutym etykieciarzu. Gdyby kiedyś miała wrócić, wymaga sędziego **widzącego
   kontrolkę**, czyli zniesienia ślepości w tym jednym wymiarze — a to jest inny
   kontrakt i inny ADR.
2. Raporty par v3 **nie mogą** twierdzić, że polityka kontroluje formę albo focus.
   Te pola pozostają etykietami raportowymi, tak jak w v2.1.
3. Rubryki pozostają **wszystkie trzy** — R1 i R2 nie są usuwane, mimo że R3 jest
   najczystsza (0,936 wobec 0,871 i 0,878 na klasach rozstrzygalnych). Usunięcie ich
   po zobaczeniu liczb byłoby dobieraniem rubryk pod wynik, czego §10 ADR zakazuje.
   Jednomyślność trzech rubryk jest zresztą tym, co daje 0,979 — pojedyncza R3 dawała
   0,936.

## 3. Rozumowanie: odrzucone

Ramię z rozumowaniem daje przy 6/6 czystość 0,9871 wobec 0,9793 — **+0,8 pp** — za cenę
**13× dłuższego runu** (405 min wobec 31 min), 2,8% porażek i 232 par utraconych na
obcięciu budżetu 1 024 tokenów. Do produkcji idzie ramię **bez rozumowania**
(`enable_thinking=false`). Jedyna zmierzona przewaga rozumowania — dwukrotnie mniejsze
obciążenie pozycyjne w R3 (0,059 wobec 0,097) — jest już zaadresowana zamianą pozycji,
która jest obowiązkowa niezależnie od trybu.

## 4. Co pozostaje niezamrożone i wymaga osobnych decyzji

- **Predykcje audytu Groq dla par v3** — zamraża je osobny ADR po zbudowaniu par,
  wyprowadzone z tej kalibracji, na parach **nieoglądanych** (zapas 1 453).
- **Otwarcie kohort v4–v11** — nadal zamknięte; jeżeli podaż par v3 przy jednomyślności
  okaże się niewystarczająca, jest to decyzja właściciela, nie automatyczna reakcja na
  niedobór.
- **Turniej** (§5 ADR) pozostaje bez zmian: ranking rubryką R3 z zamianą pozycji, a
  pełny ensemble sześciu głosów wyłącznie na parze finałowej. Kalibracja mierzyła
  właśnie ten drugi etap.

`task07_training_authorized=false`, `final_tests_used=[]`.
