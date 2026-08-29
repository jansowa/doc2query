# Amendment: kotwica encji zamiast progu udziałowego „na temat" (2026-08-29)

## Status

Amendment do ADR [`task06_defect_pair_pipeline_v1.md`](task06_defect_pair_pipeline_v1.md)
§2, spisany **przed złożeniem jakiejkolwiek pary** — verdicty z serwera nie były
jeszcze przeniesione, żadna para nie powstała, żaden wynik nie był oglądany.
Powód jest konstrukcyjny, wykryty przez test jednostkowy składania, a nie
wynikowy. `final_tests_used=[]`.

## 1. Co było zapisane i dlaczego nie działa

ADR §2 wymagał dla klasy `not_answerable`, by negatyw był „na temat", co
operacjonalizowano jako **pokrycie powierzchniowe ≥0,34**. Ten próg pochodził z
kubełków raportowych
[diagnostyki kontrastu](../measurements/task07_pair_contrast_diagnostic_2026-08-28.md),
gdzie służył do **porównywania stron pary**, a nie do bramkowania pojedynczego
zapytania.

Miara ma dwie wady w roli bramki:

1. **Ziarnistość.** Zapytania mają 2-4 słowa treściowe, więc pokrycie przyjmuje
   wartości 1/4, 1/3, 1/2, 2/3. Kanoniczny przypadek klasy — „ta sama encja,
   atrybut, którego pasaż nie podaje" — ma dokładnie jedną wspólną encję, czyli
   ląduje na **0,333 < 0,34**. Przykład zmierzony w teście: `jaka jest mediana
   dochodu w Sandpoint` przy pasażu o Sandpoint → 1/3, odrzucone.
2. **Słowa funkcyjne.** Filtr „słowo ≥4 znaki" przepuszcza polskie `jaka`,
   `jest`, `które`, `kiedy`, które nigdy nie trafiają do pasażu i dodatkowo
   zaniżają udział — asymetrycznie dla formy `full_question`.

Bramka odrzucałaby więc systematycznie **najlepszą** klasę pipeline'u (w
eksploracji `not_answerable` była jedyną z 10/10 wiarygodnych próbek), i to tym
częściej, im krótsze i bardziej pytające zapytanie.

## 2. Decyzja

Wymóg „na temat" dla `not_answerable` brzmi teraz: **zapytanie dzieli z pasażem
co najmniej jedną kotwicę encji** — wspólne słowo treściowe o długości ≥5 znaków
albo zawierające cyfrę. To wyraża wprost intencję zapisaną w ADR („nazwy własne i
terminy zostają te z pasażu, brakuje wyłącznie odpowiedzi"), zamiast przybliżać
ją ułamkiem.

Dodatkowo pokrycie powierzchniowe liczy się po odrzuceniu zamrożonej listy
polskich słów funkcyjnych i **zostaje wyłącznie jako pomiar** w
`rejected_measurements.passage_coverage_surface`; nie bramkuje niczego.

Pozostałe progi ADR §4 (LCS 5, Jaccard 0,6, długości 2-24 i 0,4-2,5, kontrakt
formy) — **bez zmian**. Definicje klas, kolejność kopalnia→mutacja, limit jednej
pary na (grupę, klasę), wymogi answerability i jednomyślne potwierdzenie w obu
kolejnościach — **bez zmian**.

## 3. Czego ten amendment nie robi

- Nie łagodzi żadnej innej bramki i nie zmienia bramek przedtreningowych §7
  (pass-rate, audyt anty-skrótowy AUC 0,80, spot-check, osobna autoryzacja).
- Nie dotyka par v3 ani wytrenowanych ramion.
- Nie zmienia miary pokrycia w diagnostyce z 28 sierpnia; tamten pomiar był
  porównawczy i pozostaje ważny w swojej roli.
