# Na jakiej osi stoi kontrast w parach v3 (2026-08-28)

## Status

Diagnostyka **danych**, nie wyników. Spisana **przed** zakończeniem pierwszego
ramienia treningu (w trakcie pomiaru ramię DPO było na kroku ~100/154) i przed
obejrzeniem jakichkolwiek metryk dev ramion — świadomie, żeby nie mogła zostać
użyta jako wyjaśnienie post factum. Nie zmienia ani jednej pary
(`pairs_modified=0`), nie filtruje kohorty i nie wpływa na trwający run.

Powód: właściciel, przeglądając pary w ślepym spot-checku, zgłosił, że część par
wygląda „zbyt banalnie" — przy pasażu o kowadle `chosen` to „definicja kowadła",
a `rejected` to „definicja krawędzi ślusarskiej", czyli zapytanie niepowiązane z
pasażem. Obserwacja jest sprawdzalna, więc została sprawdzona.

Producent: `scripts/measure_task07_pair_contrast.py`; wyniki
`reports/measurements/task07/pair_contrast_v1/{bottom,near_miss}.json`.
`final_tests_used=[]`.

## 1. Miara i jej ograniczenie

Udział słów treściowych zapytania (≥4 znaki), które **dokładnie** występują w
pasażu; bez lematyzacji, model-free. Polska fleksja sprawia, że miara **zaniża**
pokrycie — ale zaniża je **symetrycznie po obu stronach pary**, więc nadaje się do
porównania stron, a nie do orzekania o ugruntowaniu pojedynczego zapytania. Progi
0,34 i 0,60 są arbitralnymi punktami raportowania, nie bramkami; nic od nich nie
zależy.

## 2. Wynik

| | `bottom` (trenowany) | `near_miss` (wariant) |
|---|---|---|
| pary | 2 730 | 2 428 |
| pokrycie `chosen`, mediana | 0,500 | 0,500 |
| pokrycie `rejected`, mediana | **0,200** | **0,333** |
| `rejected` poniżej 0,34 | **75,3%** | **62,6%** |
| obie strony ≥ 0,60 | **7,1%** | **10,9%** |
| `chosen` w całości zawarte w pasażu | 9,3% | 9,3% |

**Kontrast w większości par nie dotyczy jakości zapytania, a tematu.** W trzech
czwartych par `bottom` strona odrzucona jest w praktyce nie o tym pasażu. Par, w
których obie strony są ugruntowane — czyli takich, gdzie wybór faktycznie dotyczy
jakości — jest 7,1%, czyli **194**.

**Prefiks „definicja" nie jest dyskryminatorem.** Ma go 17,4% par po stronie
`chosen` i 17,1% po stronie `rejected` — praktycznie tyle samo. Bierze się z formy,
o którą prosi prompt: 1 195 z 2 730 promptów to `keyword_query`, gdzie „definicja
X" jest formą **oczekiwaną**. Zarzut „zbyt proste" trafia więc w kontrakt formy, nie
w politykę par; formy nie zmieniamy tym dokumentem.

**Wariant `near_miss` przesuwa oś, ale jej nie naprawia.** `rejected` jest lepiej
ugruntowany (0,200 → 0,333), a par z obiema stronami ugruntowanymi jest półtora raza
więcej, ale 62,6% wciąż jest poniżej 0,34. Przyczyna leży poniżej polityki par: w
puli kandydatów dla tego samego promptu drugi najlepszy też bywa nie na temat.

## 3. Zbieżność z niezależnym pomiarem runu

Punkt startowy (adapter D01) stawia `chosen` wyżej niż `rejected` w **93,68%** par
dev — zmierzone przy starcie ramienia DPO, na 269 parach, bez związku z powyższą
analizą leksykalną. Dwa niezależne pomiary mówią to samo: łatwość par i mały zapas
na metryce marginesu to jedno zjawisko.

## 4. Co z tego wynika dla interpretacji ramion

Zapisane **przed** wynikami, żeby wnioskowanie nie było dopasowywane do liczb:

1. Jeśli DPO wygra z continued SFT, oznacza to poprawę na kontraście, który jest w
   trzech czwartych „na temat vs nie na temat". Nie wolno tego raportować jako
   dowodu, że model uczy się **lepszych** zapytań; dowodem może być tylko probe
   embedder na naturalnych zamrożonych zapytaniach (AGENTS.md §9.2).
2. Jeśli różnicy nie będzie, brak efektu **nie** dowodzi bezużyteczności DPO —
   dowodzi, że przy 2 461 parach o takim kontraście i punkcie startowym z 93,68%
   trafności zapasu nie było. Oba warunki są zmierzone, nie zakładane.
3. Wynik `bottom` vs `near_miss` staje się przez to pomiarem informatywnym samym w
   sobie: dwie kohorty różnią się dokładnie tym, ile kontrastu jest tematycznego.

## 5. Czego ten dokument nie robi

- Nie odfiltrowuje 194 par „obie strony ugruntowane" do treningu. Trening na
  podzbiorze wybranym **po** obejrzeniu rozkładu wymagałby prospektywnego ADR
  spisanego przed wynikami; inaczej byłby dobieraniem danych pod rezultat.
- Nie zmienia polityki v3, kontraktu formy ani progu jednomyślności 6/6.
- Nie przerywa trwającego runu top-vs-bottom: jest zarejestrowany i ma zostać
  zmierzony do końca, także jeśli wynik będzie nudny.
- Nie jest panelem AGENTS.md §9.3 ani oceną par przez człowieka — spot-check
  właściciela toczy się osobno i osobno zostanie zaraportowany.
