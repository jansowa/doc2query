# Task 06 — polityka par v3: lokalny sędzia jako selektor preferencji (ADR V3-01, 2026-08-25)

## Status i charakter dokumentu

**Prospektywny ADR.** Zamraża role, ślepość, rubryki, procedurę selekcji, protokół
kalibracji i granice polityki `task06-judge-selected-pair-policy-v3` **przed pierwszym
wywołaniem selektora i przed zbudowaniem pierwszej pary v3**. Autoryzacja: decyzja
właściciela z 2026-08-25.

**Czego ten ADR świadomie NIE zamraża:** progu agregacji głosów. Zostanie zamrożony
osobnym amendmentem **po kalibracji na etykietach znanych z konstrukcji i przed
zbudowaniem pierwszej pary** (§6). Ustalanie go teraz byłoby zgadywaniem — a to jest
dokładnie błąd, który pogrzebał P3 w v2.1, gdzie próg wziął się z arytmetyki
„połowa spadku wobec v1", a nie z pomiaru.

Nie zmienia: progu `source_en_score >= 23.50`, splitów, wag rerankerów, `format.py`,
splittera zdań, progów bramki różnorodności, zamrożonego kontraktu Groq, digestu wag
sędziego ani żadnego zamkniętego artefaktu (`tentative_pairs_v1_1`,
`preference_audit_v1/v2/v3/v4`, pary v2 i v2.1, wynik bramki V2.1-05).
Nie autoryzuje treningu (`task07_training_authorized=false`) i nie otwiera testów
finalnych (`final_tests_used=[]`).

## 1. Dlaczego v3 istnieje

Bramka V2.1-05 zakończyła się `INCONCLUSIVE` na P3 (17/800, górna granica CP 3,1704%
wobec progu 3,1%), przy czym P2 i P4' przeszły, a guardrail P1 nie zapalił się.
Zapisane w tym samym pomiarze wnioski są dwa i oba prowadzą tutaj:

1. **mechanizm działa** — kontrast wewnątrz pary +38,2 i +55,5 pp, a zgodność
   konsensusu z automatem wzrosła do 45,12% wobec 30,80% (v2.0) i 24,40% (v1);
2. **progi v2.1 nie były wyprowadzone z pomiaru** — założenia planistyczne wzięte z
   podpróby osi A przegranej bramki v2.0 okazały się optymistyczne na obu wymiarach
   (P3 1,30% → 2,12%, P1 3,90%/3,25% → 5,12%/5,75%), a próg P1 leży dziś po
   niewłaściwej stronie prawdy, więc jest **definitywnie** nierozstrzygalny
   konfirmacyjnie.

Decyzja właściciela: zamiast trzeci raz przestawiać poprzeczkę, zmienić **pytanie** —
o lepszej stronie pary ma orzekać sędzia LLM porównujący zapytania wprost, a nie
złożenie sygnałów pośrednich (werdykt odpowiadalności + round-trip + margines).
Filtrowanie danych preferencyjnych sędzią LLM jest praktyką standardową (RLAIF,
UltraFeedback, self-rewarding models), a wielokrotne głosy per para to zwykła
self-consistency.

## 2. Zmiana roli lokalnego modelu — i reguła, która tego wymaga

`Qwen3.8-27B` FP8 na endpoincie vLLM operatora przestaje być wyłącznie **etykieciarzem
odpowiadalności** (jedno pytanie: „czy da się odpowiedzieć z pasażu?") i staje się
**selektorem preferencji** (pytanie: „które z dwóch zapytań jest lepsze?").

AGENTS.md §7 wymienia to wprost jako zakazane „bez prospektywnego ADR" — ten dokument
jest tym ADR-em. Config sędziego v1 nosi `used_for_pair_building: false`; kontrakt v3
jest **osobny** i nosi `used_for_pair_selection: true`, a config v1 zostaje bez zmian,
żeby zamknięte werdykty odpowiadalności pozostały odtwarzalne.

**Amendment wykonawczy do zapisu o przepustowości.** AGENTS.md §7 zakazuje „masowego
scoringu kandydatów (poza budżetem przepustowości)". Ten zapis powstał, gdy lokalny
model działał jako 27B w Q4 przez ollamę z offloadem do RAM na maszynie 16 GB. Stan
faktyczny jest inny: operator ma serwer **FP8 vLLM bez limitów dobowych**, o
zmierzonej przepustowości **19,1 itemu/s**. Masowy scoring przestaje być poza
budżetem i zapis §7 należy czytać jako ograniczenie sprzętowe, które przestało
obowiązywać — nie jako zakaz metodologiczny. Adres endpointu pozostaje **wyłącznie
parametrem CLI**, nigdy w repozytorium; wagi pozostają przypięte digestem, a brak
zgodności digestu przerywa run.

## 3. Ślepość: co sędzia widzi, a czego nie

Widzi: **pasaż i dwa zapytania** oznaczone A i B.

Nie widzi: który kandydat pochodzi z którego slotu dekodowania, jakie ma score
primary/shadow, jaki ma round-trip, jaki ma `content_jaccard`, ani czego chciałby
którykolwiek sygnał automatyczny. Nie widzi też własnych wcześniejszych werdyktów
odpowiadalności.

**Zamiana pozycji jest obowiązkowa.** Każde porównanie wykonuje się w obu
kolejnościach (A/B i B/A). Sędziowie LLM mają silne obciążenie pozycyjne, więc zgoda
w obu kolejnościach jest minimalnym warunkiem uznania wyniku za rozstrzygnięty;
niezgoda między kolejnościami jest zapisywana jako `position_flip` i **nie tworzy**
pary. Wielkość obciążenia pozycyjnego jest jedną z mierzonych wielkości kalibracji.

**Powtórzenie tego samego promptu nie jest głosem.** Przy `temperature=0` to ten sam
rachunek; zmierzony niedeterminizm serwera (powtarzalność 0,9909 przy continuous
batchingu) daje ~1% wahań i nie jest źródłem informacji. Wiele głosów oznacza
**różne rubryki**, nie powtórzenia.

## 4. Trzy rubryki z definicjami i jawną hierarchią konfliktu

Rubryka audytu Groq jest jednym akapitem, wymienia pięć kryteriów bez definicji i
**zakazuje toku rozumowania** — a udział remisów 47,5% u `gpt-oss` i 401 abstencji na
800 par są prawdopodobnie tego skutkiem. Tu nie ma powodu się ograniczać: serwer jest
bez limitów, więc rubryki są pełne, a **rozumowanie jest dozwolone** przed wydaniem
werdyktu (nie jest zapisywane jako sygnał selekcji, wyłącznie do dziennika).

- **R1 `grounding`** — czy na zapytanie można odpowiedzieć wyłącznie z pasażu; czy nie
  wymaga wiedzy zewnętrznej; czy nie przekręca faktów, liczb i nazw własnych.
- **R2 `retrieval_usefulness`** — czy zapytanie ma sens jako realne zapytanie
  wyszukiwawcze; czy nie jest tak ogólne, że pasowałoby do tysiąca pasaży; czy nie
  jest tak wąskie, że jest przepisanym zdaniem; czy nie zdradza odpowiedzi w treści.
- **R3 `holistic`** — całościowa ocena z jawną hierarchią: **ugruntowanie przed
  użytecznością, użyteczność przed naturalnością, naturalność przed długością**.
  Kopiowanie długich fragmentów pasażu dyskwalifikuje niezależnie od reszty.

Każda rubryka zwraca `better ∈ {A, B, tie}` i `confidence ∈ [0,1]` w czystym JSON.
Hierarchia z R3 jest wiążąca dla interpretacji konfliktów między R1 i R2 — nie
uśredniamy sprzecznych kryteriów bez reguły.

## 5. Procedura selekcji

**Etap 1 — dopuszczalność (tanie, mechaniczne, bez sędziego).** Zostają guardy, które
są obiektywne i nie wymagają oceny: `format_valid`, brak prefiksu, metakomentarza,
wielu zapytań i pustego wyjścia, guard wtrącenia `task06_lead_in_guard_v1`. Strona
`chosen` dodatkowo wymaga `corpus_round_trip_at_20 == 1.0` i braku `copy_risk`.
Powód zachowania tych dwóch: są tanie, mierzalne i chronią przed pytaniem sędziego o
oczywisty śmieć. **Werdykt odpowiadalności przestaje decydować o rolach** — zostaje
zapisany jako etykieta raportowa.

**Etap 2 — turniej (sędzia rankinguje).** Wśród dopuszczalnych kandydatów grupy
(ta sama grupa same-prompt, 8 kandydatów) rozgrywa się turniej pojedynczych
porównań: najpierw wyłania się najlepszego (`k−1` porównań), potem najgorszego z
pozostałych (`k−2` porównań). Ranking używa **R3 z zamianą pozycji** (2 wywołania na
porównanie) — pełny ensemble na tym etapie byłby marnotrawstwem.

**Etap 3 — potwierdzenie finalnej pary (pełny ensemble).** Wybrana para przechodzi
**wszystkie trzy rubryki w obu kolejnościach** — sześć głosów. Para wchodzi do
zbioru treningowego tylko wtedy, gdy głosy spełniają regułę agregacji zamrożoną
amendmentem po kalibracji (§6).

**Arytmetyka kosztu** (3 619 grup, średnio 6 dopuszczalnych z 8, czyli 9 porównań na
grupę, 19,1 itemu/s):

| wariant | wywołania | bez rozumowania | z rozumowaniem |
|---|---|---|---|
| pełny ensemble na każdym porównaniu | 195 426 | 2,8 h | 8,5 h |
| **ranking R3 + ensemble na finale (wybrany)** | **86 856** | **1,3 h** | **3,8 h** |

**Strona `rejected` — jedna prerejestrowana ablacja.** Budujemy **dwa** warianty par
na tych samych grupach: `bottom` (najgorszy z turnieju, maksymalny margines) i
`near_miss` (kandydat o randze drugiej od końca wśród dopuszczalnych). Powód: pary
maksymalnego marginesu bywają banalne, a Task 07 ma `top-vs-bottom vs top-vs-near-miss`
na liście ablacji. Oba warianty są zapisywane osobno; **żaden nie jest promowany bez
pomiaru**.

## 6. Kalibracja przed zamrożeniem progu agregacji

Kolejność jest wiążąca: **kalibracja → amendment z progiem → budowa par → audyt Groq**.
Nie wolno zbudować ani jednej pary v3 przed zamrożeniem progu.

**Zbiór podstawowy: korpus walidacyjny nagrody** — 180 pasaży × 8 klas = 1 440 zapytań
o etykietach **znanych z konstrukcji** (m.in. `copy_verbatim`, `ungrounded`,
`wrong_focus`, `wrong_form`, klasy `good_*`). Z tych klas buduje się pary o znanym
kierunku (klasa dobra vs klasa zepsuta) i mierzy się:

- **czystość per rubryka** — jak często R1/R2/R3 wskazują stronę dobrą;
- **obciążenie pozycyjne** — udział `position_flip`, osobno per rubryka;
- **krzywa czystość/wydajność** — dla każdej reguły agregacji (jednomyślność 6/6,
  większość 5/6, 4/6) jednocześnie czystość i odsetek par, które przeżywają;
- **udział remisów i abstencji**, żeby porównać z patologią rubryki Groq (47,5%).

**Zbiór wtórny: 1 600 ocen Groq z 800 audytowanych par v2.1** — zgodność nowego
selektora z niezależnymi sędziami na parach, które już widzieliśmy. Rola wyłącznie
kontrolna: te pary są **spalone** dla przyszłego audytu i nie wejdą do próbki Groq v3.

Próg agregacji zamraża amendment na podstawie krzywej czystość/wydajność, z jawnym
kryterium wyboru zapisanym w tym amendmencie, nie tutaj.

## 7. Obowiązkowe raportowanie efektu ubocznego filtrowania

Filtrowanie do par, w których sędzia jest pewny i spójny, typowo **podnosi precyzję i
obniża trudność** — zostają pary oczywiste. Raport budowy par v3 **musi** zawierać:
liczbę par przeżywających każdy poziom surowości, rozkład `content_jaccard` i długości
obu stron, udział par, w których `rejected` jest nieodpowiadalny (czyli banalnie
gorszy), oraz porównanie tych rozkładów z parami v2.1. Bez tego nie wolno twierdzić,
że filtr „poprawił dane" — mógł je tylko uprościć.

## 8. Nowa rola audytu Groq

Groq **nie znika i nie zmienia kontraktu**. Zmienia się przedmiot: audytuje teraz
selektor v3, a nie złożenie sygnałów v2.1. Warunki:

- próbka wyłącznie z **par nieoglądanych** — zapas to 1 453 pary v2.1, a pary v3
  powstaną na tych samych grupach, więc audytowane mogą być tylko te, których żaden
  wcześniejszy eksport nie pokazał;
- **predykcje wyprowadzone z kalibracji**, nie z ambicji ani z „połowy spadku wobec
  poprzedniej wersji"; zamraża je osobny ADR po kalibracji i przed audytem;
- reguła decyzyjna pozostaje przedziałowa (Clopper–Pearson, werdykt trójwartościowy,
  `INCONCLUSIVE` fail-closed) — ta część v2.1 sprawdziła się i zostaje;
- audyt Groq nadal **nie jest human evidence** i nie zastępuje panelu ludzkiego §9.3.

## 9. Konsekwencje dla tego, co już jest

- **Pary v2.1 przestają być kandydatem na dane treningowe.** Stają się materiałem
  kalibracyjnym i punktem odniesienia. Wynik bramki V2.1-05 (`INCONCLUSIVE` na P3)
  pozostaje zapisany jak jest — **nie jest reinterpretowany, przeliczany ani
  unieważniany**, a progi v2.1 nie są zmieniane.
- Polityka v2.1 i jej moduły pozostają w repozytorium nietknięte; v3 to nowy moduł,
  nowy kontrakt i nowe katalogi wyjściowe.
- Kohorty v4–v11 pozostają zamknięte. Jeżeli podaż par v3 okaże się niewystarczająca,
  ich otwarcie wymaga osobnej decyzji — nie jest domyślną reakcją na niedobór.
- Oś C (focus) nadal jest poza wydaniem, ale przestaje być zablokowana z powodu
  zepsutego etykieciarza: mocny lokalny sędzia może focus oznaczyć. To osobne zadanie
  i osobny ADR; ten dokument go nie autoryzuje.

## 10. Granice i zakazy

- Sędzia nie należy do rodziny generatora (Bielik) i nie jest teacherem ablacji —
  guard `self_preference_guard` zostaje egzekwowany.
- Brak digestu wag albo jego niezgodność **przerywa run**; werdykty idą do journala
  z resume, tak jak w harnessie v1.
- Rozumowanie sędziego jest zapisywane do dziennika, ale **nie jest sygnałem
  selekcji** ani materiałem treningowym.
- Zabronione po odczycie kalibracji: dobieranie rubryk pod wynik, zmiana hierarchii
  konfliktu, powtarzanie kalibracji na innym korpusie dla lepszej liczby.
- `task07_training_authorized=false`, `final_tests_used=[]`.
