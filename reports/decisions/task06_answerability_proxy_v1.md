# ADR: proxy odpowiadalności v1 (kalibracja na etykietach audytu Groq)

Status: **prospektywny, zamrożony przed odczytem jakiegokolwiek związku cechy z
etykietą**. Kontrakt: `task06-answerability-proxy-v1`.
`human_evidence_claimed=false`, `task07_training_authorized=false`,
`final_tests_used=[]`.

Data zamrożenia: 2026-08-17. Ten ADR jest zamykany (commit) **przed**
uruchomieniem jakiegokolwiek dopasowania reguły; kryteria akceptacji z §6 nie
wolno zmieniać po odczycie wyników.

## 1. Po co proxy i czym on nie jest

Oś A polityki par v2 wymaga sygnału odpowiadalności po stronie `chosen`.
Zamierzonym sygnałem jest **przypięty lokalny sędzia** `qwen3.6-27b` Q4
(harness V2-01 już istnieje i jest fail-closed:
`src/doc2query/preferences/answerability_judge.py`). Sprzęt bazowy ma 8 GB VRAM
i ollama bez modeli, a wagi 27B czekają na inną maszynę w nieokreślonym
terminie, więc pierwsze wydanie polityki v2 użyłoby albo osi A bez kontroli
odpowiadalności (dokładnie ta luka, którą audyt zmierzył: ~18% `chosen`
nieodpowiadalnych), albo **taniego proxy z już policzonych pól scoringu**.

Decyzja właściciela (2026-08-17): pierwsze wydanie osi A używa proxy
skalibrowanego na etykietach `answerable_a/b` audytu Groq.

**Klauzula zastąpienia (wiążąca).** Proxy jest rozwiązaniem tymczasowym.
Przypięty sędzia lokalny (harness V2-01) zastąpi proxy **osobnym ADR-em**, gdy
będzie sprzęt; ten ADR nie autoryzuje pozostawienia proxy w wersji finalnej ani
nie zwalnia z kalibracji sędziego lokalnego przewidzianej w V2-01. Do momentu
zastąpienia każdy artefakt par v2 nosi jawne pole
`answerability_signal="proxy_v1"`.

Proxy **nie jest** sędzią, nie jest human evidence, nie jest sygnałem selekcji
audytowej i nie wchodzi do żadnej metryki finalnej. Jego jedyna dopuszczona rola
to **filtr strony `chosen`** w osi A (i warunek „odpowiadalny” w osi B, jeśli ADR
V2-03 tak zdecyduje).

## 2. Źródło etykiet

Etykieta dotyczy **strony pary** (jedna z dwóch: `chosen` albo `rejected`), nie
pary. Dla każdej strony bierzemy pola `answerable_a` / `answerable_b` obu sędziów
Groq z zamrożonego audytu v1 i mapujemy opcję A/B na rolę przez
`automatic_chosen_option` z `machine_key.jsonl`.

- etykieta `yes` — **oba** modele mówią `answerable=true` dla tej strony;
- etykieta `no` — **oba** modele mówią `answerable=false`;
- **brak etykiety** (strona wykluczona) — modele się różnią albo któryś model nie
  ocenił pary.

Konsensus dwóch niezależnych sędziów jest tu jedynym dopuszczonym źródłem,
ponieważ pojedynczy sędzia byłby szumem nie do odróżnienia od sygnału.

**Zgodność między sędziami jest raportowana jako sufit szumu** zadania:
udział stron, na których obaj sędziowie mówią to samo o odpowiadalności,
liczony na **wszystkich** stronach z dwiema ocenami (nie tylko na konsensusowych).

### Migawka etykiet (pinowana, dzień 1 audytu)

Audyt v1 jest niedokończony z powodu dziennych budżetów tokenów Groq (status
`incomplete_quota_deferred`; wznowienie po 00:00 UTC należy do operatora).
Ten ADR **nie czeka** na jego domknięcie i kalibruje się na migawce dnia 1,
pinowanej po SHA-256 plików:

- `artifacts/task06/preference_audit_v2/sample.jsonl` (cechy per strona),
- `artifacts/task06/preference_audit_v2/machine_key.jsonl` (mapowanie roli),
- `artifacts/task06/preference_audit_v2/groq_dual_llm/pair_verdicts.jsonl`
  (etykiety).

Uzasadnienie: pokrycie audytu jest w praktyce losowe względem kohort i strat
(requesty idą w porządku po haszu `audit_id`), a etykieta odpowiadalności jest
własnością **strony**, niezależną od porządku marginesowego, którego dotyczy
niedokończona analiza par. Kalibracja proxy nie zależy więc od tego, ile par ma
komplet ocen preferencji.

**Klauzula drugiego, w pełni prospektywnego holdoutu.** Gdy audyt v1 zostanie
domknięty, strony par ocenionych **dopiero po tej migawce** (≈256 par, tj. strony
nieobecne w niniejszej kalibracji) tworzą drugi holdout. Zamrożona reguła jest na
nim oceniana **bez żadnego dopasowania**, tymi samymi kryteriami z §6, i
raportowana osobno. Jest to mocniejszy test niż podział wewnątrz migawki i nie
wolno go użyć do wyboru reguły.

## 3. Co było znane przed zamrożeniem tego ADR (jawne wyliczenie)

Przed zamrożeniem odczytano wyłącznie **własności zbioru etykiet**, nigdy związku
cechy z etykietą:

- par z oceną obu modeli: **244** (z 500);
- stron z dwiema ocenami: **488**; z etykietą konsensusu: **392**
  (`yes` 306, `no` 86), stron rozjechanych: 96;
- zgodność sędziów co do odpowiadalności (sufit szumu): **0,8033** (488 stron);
- bazowy udział klasy większościowej `yes` w zbiorze etykiet: **0,7806**;
- rozkład etykiet per rola: `chosen` 181 `yes` / 26 `no` (12,6% `no`),
  `rejected` 125 `yes` / 60 `no` (32,4% `no`).

**Żadnej cechy nie zestawiono z etykietą** przed zamrożeniem: nie policzono
żadnej korelacji, żadnego progu, żadnej dokładności. Wartości wyżej służą
wyłącznie do uczciwego doboru progów akceptacji z §6 (bez nich kryterium
„accuracy” byłoby wobec bazy 0,78 puste) i są tu zapisane, żeby nie dało się
później udawać, że nie były znane.

## 4. Podział fit/holdout

Deterministyczny, bez losowania:

```
h = sha256(audit_id.encode()).hexdigest()
half = "fit" if int(h[:2], 16) < 0x80 else "holdout"
```

**Obie strony jednej pary trafiają do tej samej połowy** — inaczej pasaż
wyciekałby między połowami (obie strony dzielą ten sam pasaż i tę samą kontrolkę).
Podział jest po `audit_id`, więc jest niezależny od etykiet i od cech.

Holdout jest odczytywany **dokładnie raz**.

## 5. Przestrzeń reguł (zamrożona)

Bez uczenia modeli. Cechy — wyłącznie pola per-kandydat już policzone w
zamrożonym scoringu i obecne w `sample.jsonl` (dokładnie ta lista, 13 pozycji;
boolean mapowany na 0/1):

```
corpus_margin_to_best_nonpositive, corpus_round_trip_at_5,
corpus_possibly_ambiguous_query, content_jaccard, natural_content_jaccard,
passage_recall, query_precision, copy_density, normalized_lcs,
longest_copied_ngram, word_length, pool_margin, pool_positive_score
```

Postać reguły — atom `cecha >= t` albo `cecha <= t`, gdzie `t` należy do
**decyli (0,1…0,9) rozkładu tej cechy na połowie fit**; reguła to pojedynczy atom
albo koniunkcja **dokładnie dwóch** atomów na **różnych** cechach. Reguła
przewiduje `yes` (strona odpowiadalna) wtedy i tylko wtedy, gdy jest spełniona.

Wybór na połowie fit: **maksymalizuj `recall_yes` pod warunkiem
`precision_yes >= 0,88` i `recall_yes >= 0,50`**. Remisy rozstrzyga
deterministycznie: wyższe `precision_yes`, potem mniejsza liczba atomów, potem
porządek leksykograficzny (nazwa cechy, kierunek, próg). Jeśli **żadna** reguła
z przestrzeni nie spełnia warunku na połowie fit, konstrukcja jest **nieudana** i
holdoutu **nie odczytujemy wcale** (patrz §7).

## 6. Kryterium akceptacji na holdoucie (zamrożone przed odczytem)

Proxy jest przyjęty jako filtr strony `chosen` wtedy i tylko wtedy, gdy na
holdoucie **oba** warunki są spełnione:

- **(P1) czystość**: `precision_yes >= 0,88`;
- **(P2) podaż**: `recall_yes >= 0,50`.

Uzasadnienie progów, spisane **przed** odczytem:

- P1 jest sformułowane jako czystość, nie jako accuracy, bo proxy ma jedną
  funkcję: podnieść udział odpowiadalnych po stronie `chosen`. Przy bazie 0,7806
  wartość 0,88 oznacza zejście z 21,9% do ≤12,0% nieodpowiadalnych wśród
  zatrzymanych, czyli **redukcję masy defektu o ≥45% względnie**. Accuracy jako
  kryterium byłaby pusta (reguła stała `yes` daje 0,78).
- 0,88 leży **powyżej** sufitu szumu 0,8033 mierzonego na wszystkich stronach i
  to nie jest sprzeczność, tylko konsekwencja doboru etykiet: holdout zawiera
  wyłącznie strony **konsensusowe**, czyli podzbiór łatwiejszy niż pełne zadanie
  sędziego, na którym sufit 0,8033 był mierzony. Dlatego z przejścia P1
  **nie wolno wnosić**, że proxy jest lepsze od sędziego — na stronach spornych
  (96 z 488, 19,7%) proxy nie jest w ogóle oceniane. To jest wprost powód, dla
  którego proxy sędziego nie zastępuje, tylko go poprzedza, i dlaczego klauzula
  zastąpienia z §1 jest wiążąca.
- P2 chroni podaż: filtr, który wycina ponad połowę dobrych `chosen`, zabiłby
  osie par (inwentarz V2-00: czysty `chosen` w 21102 grupach).

Raportowane obowiązkowo obok kryterium (nie są kryterium): `accuracy`,
`balanced_accuracy`, macierz pomyłek, bootstrapowe 95% CI dla `precision_yes` i
`recall_yes`, liczności obu połówek, zgodność sędziów, baza klasy
większościowej, oraz wynik per rola (`chosen` / `rejected`).

Predykcja zapisana przed odczytem: **spodziewam się, że P1 nie przejdzie** albo
przejdzie z CI obejmującym 0,88. Cechy z listy są leksykalno-rankingowe, a audyt
v1 zmierzył, że round-trip praktycznie nie różnicuje odpowiadalności
(69,8% vs 65,5%; 61,6% vs 62,5%) — jeśli round-trip nie różnicuje, to szansa, że
inne pole tej samej rodziny różnicuje mocno, jest mała. Ta predykcja jest zapisana
po to, żeby ewentualne przejście proxy było wynikiem, a nie potwierdzeniem
oczekiwań.

## 7. Konsekwencje niedowiezienia (zamrożone)

Jeśli konstrukcja na fit jest nieudana (§5) albo holdout nie spełnia P1∧P2:

1. raportujemy to **wprost** jako wynik negatywny, z liczbami;
2. proxy **nie jest** używane jako filtr strony `chosen`; oś A powstaje bez
   kontroli odpowiadalności, wyłącznie na round-tripie i pozostałych warunkach
   czystości;
3. ADR V2-03 zapisuje wtedy predykcje **ostrożnie**: nie wolno przewidywać
   spadku udziału nieodpowiadalnych `chosen` do 5%, bo żaden zmierzony mechanizm
   tego nie uzasadnia; dopuszczalna predykcja to brak pogorszenia względem v1
   (≤ zmierzonej wartości v1 z pełnego audytu), a poprawa odpowiadalności wraca
   dopiero z sędzią lokalnym;
4. luka odpowiadalności pozostaje **otwartym, nazwanym długiem** osi A do
   ADR-u sędziego lokalnego.

Nawet przy przejściu P1∧P2 proxy **nie uprawnia** do predykcji 5%: górna granica
tego, co filtr o czystości `p` może dać, to `1 − p` nieodpowiadalnych po stronie
`chosen`, więc ADR V2-03 wyprowadza swoją predykcję z **zmierzonej** czystości na
holdoucie, nie z ambicji.

## 8. Czego ten ADR nie zmienia

Nie zmienia `format.py`, bramki różnorodności, polityki par v1/v1.1 i jej
artefaktów, kontraktu audytu Groq, rubryki sędziów, progu
`source_en_score >= 23,50`, splitów ani statusu Task 07/08. Nie buduje żadnej
pary. Nie dotyka `artifacts/task06/teacher_claude_v1/`. Nie otwiera testów
finalnych.
