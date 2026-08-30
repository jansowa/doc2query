# Ramię odniesienia probe: zapytania teachera (2026-08-30)

## Status

**Prospektywny ADR spisany przed generacją.** Autoryzacja: właściciel udostępnił
serwer inferencji na noc i poprosił o wykorzystanie jego potencjału. Dokument
zamraża, czym to ramię jest i — ważniejsze — czym **nie** jest.
`final_tests_used=[]`.

## 1. Po co

Task 07 mierzy, czy DPO poprawia generator. Wiemy już, ile jest do ugrania na
metrykach powierzchniowych, ale **nie wiemy, ile w ogóle da się ugrać na probe**.
Bez tej liczby wynik „DPO poprawiło probe o X" nie ma skali odniesienia: nie
wiadomo, czy X to połowa dostępnego zapasu, czy jego setna część.

Ramię odniesienia odpowiada na to wprost: probe embedder trenowany na
zapytaniach napisanych przez **duży model** (Qwen3.8-27B, inference-only) na tej
samej kohorcie pasaży, tym samym kontraktem promptu i tym samym budżetem.

## 2. Czym to ramię NIE jest

- **Nie jest kandydatem na finalistę.** Procedura produkcyjna to lokalny
  generator; zapytania teachera nie wchodzą do żadnej kohorty par, do żadnego
  continued SFT ani do selekcji finalistów.
- **Nie jest przesłanką za destylacją.** Ablacja teachera z 2026-08-16 wykazała,
  że teacher **nie** bije lokalnego D01 na zamrożonym sygnale budującym
  (34,7% / 41,6%). To ramię mierzy co innego — sufit na probe — i nie unieważnia
  tamtego pomiaru.
- **Nie jest bramką.** Nie ma progu, którego przekroczenie coś przesądza.

## 3. Zamrożone parametry

- **Kohorta**: 1 984 pasaże z `artifacts/task05/d01b_prospective_1_5b_v3/probe_inputs/`,
  zmierzone jako **rozłączne** z kohortami treningowymi Task 07: 0 wspólnych
  pasaży z v3 bottom (2 461 par) i 0 z defect (1 794 pary).
- **Kontrolki**: te same cztery co w designie D01 (`full_question`/`fact_lookup`/
  `beginning`, `keyword_query`/`definition`/`middle`, `full_question`/`procedure`/
  `end`, `keyword_query`/`entity_lookup`/`middle`), po jednym zapytaniu na
  kontrolkę — 4 zapytania na pasaż, czyli ten sam budżet 7 936 par co ramiona
  studenckie.
- **Prompt**: skrócony kontrakt doc2query (ugruntowanie, zakaz kopiowania
  długich fragmentów, zachowanie nazw i liczb, zakaz zdradzania odpowiedzi),
  `prompt_version` w journalu, temperature 0.
- **Dalsza obróbka**: identyczna jak dla ramion studenckich — ten sam
  `probe_recipe`, ta sama receptura negatywów, ten sam frozen manifest do
  ewaluacji. Jakiekolwiek odstępstwo unieważnia porównanie.

## 4. Jak wolno raportować wynik

Jako **sufit odniesienia**, zawsze razem z liczbą ramienia startowego (SFT bez
DPO). Zdanie dopuszczalne: „DPO odzyskało X% dystansu między startem a ramieniem
teachera". Zdanie niedopuszczalne: „teacher jest lepszym generatorem" — do tego
potrzeba porównania w trybie produkcyjnym, którego to ramię nie wykonuje.

Jeśli teacher wypadnie **gorzej** od startu, wynik zostaje zaraportowany tak
samo — jest wtedy informacją o kohorcie probe albo o kontrakcie promptu, nie
powodem do powtórki z innym promptem.
