# Pełny audyt puli SFT: 384 576 par, cztery osie (2026-09-01)

## Status

Pomiar wykonany w całości na serwerze inferencji właściciela (Qwen3.8-27B,
temperature 0), zadanie `sft_full_audit` z paczki v3. Journal kompletny:
**388 466 unikalnych kluczy = 384 576 (audyt) + 2 502 (confirm_pairs) +
1 388 (polish_recheck)**, zero duplikatów, 118 werdyktów audytu ze złamanym
schematem (0,03%, pominięte w agregacji). To pomiar wad w istniejącej puli,
nie filtr — żadna para nie została usunięta. `final_tests_used=[]`.

**Zmiana promptu w trakcie pomiaru (na życzenie właściciela, z zapisem w
metadanych):** pierwsze 36 089 werdyktów policzono układem v1 (instrukcja za
danymi), pozostałe 348 487 układem prefix-first (instrukcja w prompcie
systemowym, dane w user; `prompt_version` per rekord). Treść czterech osi
bajt w bajt identyczna w obu układach.

## 1. Wynik (całość, n = 384 458 poprawnych werdyktów)

| oś | v1 (n=36 082) | v2 prefix-first (n=348 376) |
|---|---|---|
| `nieodpowiadalne` | 13,4% | 11,9% |
| `zła polszczyzna` | 4,6% | 8,4% |
| `niesensowne` | 0,5% | 0,7% |
| `zbyt ogólne` | 0,8% | 0,8% |
| **≥1 wada twarda** | **17,8%** | **19,7%** |

## 2. Granica promptu jest widoczna — i dotyczy głównie osi językowej

Rate osi w kolejności przetwarzania (20 kubełków po ~19 tys.): skok
`zła polszczyzna` 4,7→8,7% następuje dokładnie na granicy wersji promptu
(kubełek 1→2) i dalej jest płaski (8,1–8,8%). `nieodpowiadalne` przesuwa się
nieznacznie (13,4→11,9%) i też jest płaskie. Pula była w losowej kolejności,
więc to efekt promptu, nie danych.

**Przegląd próbek (po 12 losowych flag językowych z każdego segmentu):**
precyzja osi `zła polszczyzna` jest niska w OBU segmentach — większość flag to
poprawne polskie zapytania („dlaczego fitness jest ważny", „różne rodzaje
husky", „jaki jest wiek pełnoletności w New Hampshire?") albo słownikowe
zapytania o angielskie terminy („co oznacza dispensation", „definicja plait"),
których ścisła definicja wprost każe nie liczyć jako wadę. Szacunkowa precyzja
z próbki: ~10–25%. Segment v2 flaguje ~2× więcej, ale nie lepiej.

**Wniosek: oś `zła polszczyzna` z tego pomiaru NIE nadaje się na filtr** bez
osobnej, dokładniejszej procedury (sędzia konsekwentnie ignoruje ścisłą
definicję). Nośna jest oś `nieodpowiadalne`: 11,9–13,4% spójne z pilotażem
12 tys. (15,3%, prompt v1 nieostry) i z odrzutami answerability `chosen`
w Task 06 (13,5%).

## 2a. Przegląd jakości ocen per oś (po 10 losowych flag z segmentu v2 + 10 par czystych, seed 20260901)

- **`nieodpowiadalne` — precyzja ~60–80%, oś nośna.** Sędzia łapie subtelne,
  prawdziwe wady: pasaż z numerem telefonu *innej organizacji* niż pytany
  senator; pytanie o socjologa łączącego stratyfikację z technologią (Lenski)
  przy pasażu o Weberze; brak lokalizacji czujnika w pasażu o jego
  właściwościach. Błądzi w dwóch trybach: (a) surowość przy odpowiedziach
  częściowych (małe hipopotamy vs hipopotamy w ogóle), (b) artefakty
  tłumaczenia — „ile furlongów w mili" oflagowane, bo pasaż mówi
  „stadiów/stajnia" (tłumaczenie rozjechało termin, semantycznie odpowiada).
- **`niesensowne` — precyzja ~85–90%, najlepsza oś.** Niemal same prawdziwe
  wady: testy z lukami („_______ teoria jest nazywana trzecią siłą"), urwane
  zapytania („jak długo smażyć hamburgery na"), nieprzetłumaczone zdania
  angielskie, zmielone skróty („oddział Wells Fargo w Monument, co" — CO to
  stan Kolorado).
- **`zbyt ogólne` — precyzja ~70%.** Trafne flagi to zapytania bez kluczowego
  wyróżnika pasażu („o której godzinie można kupić piwo" przy pasażu o Missouri,
  „różnica czasu z Koreą" bez drugiego kraju); fałszywe to zapytania zwyczajnie
  krótkie, ale specyficzne („rodzaje drzew w południowej szwecji").
- **`zła polszczyzna` — precyzja ~10–25%, NIE nadaje się na filtr** (szczegóły
  w §2): sędzia ignoruje ścisłą definicję i flaguje poprawne polskie zapytania
  oraz słownikowe zapytania o angielskie terminy.
- **Pary czyste** — w próbce 10/10 faktycznie czystych (zapytanie sensowne,
  odpowiedź w pasażu); brak sygnału, by audyt masowo przepuszczał wady.

**Wniosek praktyczny:** ewentualny filtr puli powinien opierać się na
`nieodpowiadalne` + `niesensowne` (razem ~12,6% puli), traktować `zbyt ogólne`
jako sygnał miękki, a oś językową zastąpić wąską regułą mechaniczną
(mojibake/regex) o wysokiej precyzji.

## 3. Co dalej (decyzje właściciela, bez zmian względem raportu pilotażowego)

Opcje z `task06_sft_data_audit_2026-08-31.md` §3 pozostają aktualne; ten pomiar
daje mapę wad **per para** dla całej puli (klucz `sft_full_audit::<id>`
w journalu), więc ewentualny filtr nie wymaga już żadnych obliczeń — tylko
decyzji. Dla osi językowej rekomendacja: nie filtrować tym sygnałem;
ewentualnie użyć wyłącznie podzbioru `glowny_problem=tlumaczenie` z dodatkową
weryfikacją mechaniczną (mojibake/regex), która ma wysoką precyzję.

`final_tests_used=[]`
