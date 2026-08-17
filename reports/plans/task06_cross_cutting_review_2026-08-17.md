# Specyfikacja przeglądu przekrojowego po pomiarach 2026-08-16/17

Ten plik jest **planem pracy dla kolejnej sesji**, nie decyzją i nie pomiarem.
Nie autoryzuje żadnego runu, żadnej zmiany progu ani otwarcia testów finalnych.

## Kontekst: trzy niezależne pomiary zbiegły się w jeden wniosek

W ciągu dwóch dni powstały trzy wyniki, każdy z innego kierunku, i wszystkie trzy
mówią, że **część pozytywnych wyników tego programu może pochodzić z selekcji i
próbkowania, a nie z jakości modelu**:

1. **Sędziowie nagradzają kopiowanie**
   ([`task06_reward_validation_corpus_p8_2026-08-16.md`](../measurements/task06_reward_validation_corpus_p8_2026-08-16.md)):
   `copy_verbatim` bije poprawne zapytania na primary, marginesie i round-tripie;
   w 52,8% grup dosłowna kopia pasażu ma najwyższy `pool_margin`. Na produkcyjnej
   kohorcie 24,8–28,8% wskazań czystego argmaxa łamie guard `copy_risk`.
2. **Przewaga studenta nad teacherem to efekt best-of-N**
   ([`task06_teacher_vs_student_v3_2026-08-16.md`](../measurements/task06_teacher_vs_student_v3_2026-08-16.md)):
   teacher ma wyższy **średni** margines (4,19 vs 2,88), student wyższe
   **maksimum**. Przy tym kierunek primary i shadow rozchodzi się w 22,0% grup.
3. **Guardrail zbieżności probe (M-03) obniża efekt Task 05**
   (raport M-03 z 2026-08-16, druga sesja): po zastosowaniu zamrożonego
   guardraila para seeda 43 wypada, zostają 4 zbieżne pary
   (`insufficient_converged_seeds`), a średnia spada z **+0,0319** do **+0,0143**
   (CI `[+0,0025, +0,0238]`).

Żaden z tych wyników nie unieważnia zamkniętego pomiaru i żaden nie był podstawą
do zmiany czegokolwiek zamrożonego. Razem uzasadniają jednak **jednorazowy
przegląd przekrojowy**, bo dotykają tej samej osi: ile z mierzonych zysków to
własność generatora, a ile własność procedury wyboru.

## Rekomendowana kolejność (nie przerywać trwającej sesji)

Sesja realizująca „polityka par → tentative pairs → audyt dual-LLM” **nie
powinna być przerywana**. Jej zamrożona polityka już respektuje wnioski korpusu
walidacyjnego nagrody (`entity_preservation` wykluczone, guard wtrącenia dodany,
`focus_accuracy` tylko jako słaby filtr), a pomiar kopiowania **potwierdza
nośność** jej guarda `copy_risk`, zamiast go podważać. Przerwanie zostawiłoby
runner audytu niezacommitowany i audyt nieuruchomiony, czyli stan gorszy.

1. **Domknąć audyt dual-LLM** (trwająca sesja), po rozstrzygnięciu blokady 447 vs
   500 poniżej.
2. **Przegląd przekrojowy** (nowa sesja) — zakres w tym pliku.
3. Dopiero potem jakakolwiek decyzja o Task 07.

## Ustalenia faktyczne, które trzeba przekazać dalej

### Klucz Groq jest w `.env` i runner go czyta

`doc2query.evaluation.groq_audits.load_api_key` czyta pole **`api_key`** z pliku
`.env`, a `scripts/run_task06_groq_preference_audit.py` przekazuje je przez
`load_api_key(args.env_file)`. Pole `api_key` w `.env` jest niepuste.
**Audyt nie jest zablokowany brakiem klucza** — wcześniejsza notatka o braku
klucza była nieaktualna. Jedyną blokadą pozostaje pin `pair_count`.

### Blokada 447 vs 500: rekomendacja

`configs/preferences/task06_groq_preference_audit_v1.json` pinuje
`pair_count: 500`, a polityka dała 447 par z v1+v2.

Rekomendacja: **amendment obniżający rozwojową bramkę do 447**, bez dopuszczania
kohorty v3 do uzupełnienia próbki. Argumenty:

- dopuszczenie v3 odwraca prerejestrowaną kolejność „v3 dopiero po pozytywnym
  audycie” i pozwala potrzebie ex post kształtować użycie kohort — to jest
  dokładnie ten rodzaj swobody, którego reszta programu sobie zabrania;
- różnica precyzji audytu jest znikoma: przy obserwowanej zgodności ok. 50%
  półszerokość 95% CI to ±4,6 pp przy 447 parach i ±4,4 pp przy 500. Bramka 500
  była liczbą okrągłą, nie wynikiem rachunku mocy;
- 447 par nadal przekracza każdy próg, który uzasadniałby *odrzucenie* polityki.

Amendment musi jawnie zapisać, że próg **1000 par przed finalnym DPO** pozostaje
nietknięty.

### Zmierzona skaza formy w danych par (potwierdzona na 447 parach)

Przewidziana z korpusu walidacyjnego nagrody (guard `copy_risk` odrzuca 29,4%
poprawnych krótkich zapytań `keyword_query`) i **potwierdzona** na rzeczywistych
grupach v1+v2:

| forma | grup `eligible` | pary | yield | `no_admissible_chosen` |
|---|---|---|---|---|
| `full_question` | 456 | 265 | **0,581** | 71 (0,156) |
| `keyword_query` | 372 | 182 | **0,489** | 94 (**0,253**) |

Udział `keyword_query` spada z 44,9% grup `eligible` do **40,7%** par.
`no_admissible_chosen` jest najczęstszą przyczyną braku pary (165 z 381
niesparowanych grup `eligible`) i uderza w formę keyword **1,6× częściej**.

Konsekwencja: **zbiór treningowy DPO będzie przechylony w stronę
`full_question`, i to z powodu artefaktu filtra, nie różnicy jakości.** Audyt
dual-LLM powinien raportować zgodność **w rozbiciu na `requested_form`**, inaczej
ta skaza pozostanie niewidoczna. Nie jest to podstawa do zmiany guarda —
`copy_risk` jest odziedziczonym, zamrożonym kontraktem Task 05 i łapie 180/180
rzeczywistych kopii.

## Zakres przeglądu przekrojowego (nowa sesja)

Wszystko poniżej jest **tanie i CPU-only**; żaden punkt nie wymaga GPU ani nie
otwiera testów finalnych.

1. **Inwentarz twierdzeń wrażliwych na selekcję.** Przejść zamknięte pomiary
   Tasków 03–06 i wypisać, które wnioski opierają się na maksimum z próbkowania
   albo na sygnale primary, a które na średniej i na probe-embedderze. Dla
   każdego zapisać, czy wynik przetrwałby, gdyby kopiowanie było karane w
   selekcji. **Nie zmieniać żadnego statusu** — produktem jest tabela.
2. **Interakcja M-03 z zachowaniem hybrydy Task 05.** Hybryda jest zachowana do
   finalist-freeze review na podstawie efektu +0,0479 (97,5% CI
   `[+0,0450, +0,0508]`). Po guardrailu M-03 obraz to +0,0143 (CI
   `[+0,0025, +0,0238]`) przy `insufficient_converged_seeds`. Ustalić i opisać,
   co to znaczy dla statusu „zachowana do review” — **jako rekomendację dla
   właściciela**, nie jako zmianę statusu. To jest decyzja właściciela.
3. **Projekt nagrody wielokryterialnej pod Task 08.** Na podstawie zmierzonych
   własności komponentów napisać *prospektywną* propozycję składu nagrody:
   kara za kopiowanie jako składnik pierwszej kategorii, `entity_preservation`
   wyłącznie jako detektor halucynacji encji, sygnał ogólności oparty na
   `content_jaccard` albo statystyce korpusowej, `focus` tylko jako słaby filtr
   z abstencją. Task 08 pozostaje `BLOCKED` — to materiał do
   `reports/decisions/enable_grpo.md`, nie jego zamiennik.
4. **Dwie ślepe plamki kodu — decyzja, nie poprawka.** `_PREFIX` w
   `src/doc2query/evaluation/format.py` wymaga dwukropka, więc wtrącenie
   „Oto …” przechodzi jako format poprawny; `split_sentences` rozcina skróty
   i produkuje pseudo-zdania, co psuje `focus_buckets` w Taskach 05–06.
   Przygotować prospektywny ADR z rachunkiem kosztów: co trzeba przeliczyć,
   które zamrożone pomiary tracą porównywalność. **Nie zmieniać kodu bez tego
   ADR.**
5. **Kontrolki D01 bez pokrycia w danych.** `procedure`@end jest napięta w 60,6%
   przypadków (`intent_fit=strained`), a round-robin przypisuje ją co czwartemu
   pasażowi. Zaproponować prospektywnie warunkowe przypisanie intencji do pasaży
   mających na nią materiał. Zamrożonych kohort v1–v11 **nie ruszać**.

## Czego nowa sesja nie robi

- nie przerywa ani nie powtarza żadnego zamkniętego runu;
- nie zmienia progów, `format.py`, splittera zdań ani polityki par;
- nie buduje par z kohorty teachera (ablacja wypadła negatywnie);
- nie otwiera testów finalnych, `final_tests_used=[]`;
- nie dotyka `artifacts/task06/teacher_claude_v1/` poza czytaniem;
- przy edycji `tasks/README.md` sprawdza stan pliku przed zmianą, bo nad tym
  wierszem pracują równolegle inne sesje.
