# Pomiar: charakterystyka detektora zapadnięć probe na 20 seedach (2026-08-23)

Amendment (zamrożony **przed** serią):
[`task04_m03_in_run_collapse_shadow_mode_amendment_2026-08-22.md`](../decisions/task04_m03_in_run_collapse_shadow_mode_amendment_2026-08-22.md).
Kontrakt: `configs/evaluation/task04_m03_in_run_collapse_detection_shadow_v1.yaml`.
Kolejka: `configs/probe_inrun_collapse_shadow_queue_2026-08-22.tsv`.
Artefakty: `runs/task04_probe_shadow_collapse_v1/SHADOW-W06-4.5B-S52…S71`,
`reports/measurements/task04/shadow_collapse_detector_v1/summary.json`.
Skrypt: `scripts/measure_probe_shadow_collapse_detector.py`.

Dwadzieścia runów jednego ramienia (baseline W06), identyczne wejście,
hiperparametry i budżet co seria wariancji S47–S51; różni je wyłącznie seed
(52–71) i tryb obserwacyjny, w którym detektor liczy i zapisuje kontrole
pośrednie, ale **nigdy nie przerywa runu**. Dzięki temu każdy run dostał werdykt
zamrożonego guardraila M-03 i istnieje prawda odniesienia. Żadna różnica między
ramionami nie powstała, progów nie kalibrowano, `final_tests_used=[]`.

## Wynik główny: detektor trafił bezbłędnie na tej próbie

| reguła | TP | FP | FN | TN |
|---|---|---|---|---|
| podłoga retrievalowa (2 kolejne trafienia) | **1** | **0** | 0 | 19 |
| kierunek straty (2 kolejne trafienia) | 0 | 0 | **1** | 19 |
| alternatywa obu — reguła faktycznie wdrożona | **1** | **0** | 0 | 19 |
| kontrfaktycznie: podłoga po **jednym** trafieniu | 1 | **2** | 0 | 17 |

- **Czułość 1/1**, 95% CI [0,025; 1,000] — jedno zapadnięcie to za mało, by
  mówić o czułości; liczba jest podana, żeby nie udawać, że jej nie ma.
- **Fałszywe alarmy 0/19**, 95% CI [0,000; 0,176]. To istotnie mocniejszy wynik
  niż 0/3 z serii walidacyjnej (górna granica spadła z 0,63 do 0,18), ale nadal
  **nie wyklucza** kilkuprocentowego odsetka fałszywych alarmów.

### Wymóg dwóch kolejnych trafień zarobił na siebie natychmiast

Dwa **zdrowe** runy (seedy 55 i 57) miały pojedyncze trafienie podłogi na kroku
256 — recall pośredni 0,344 i 0,469 przy podłodze 0,521 — po czym na kroku 512
były już na 0,914 i 0,938 i skończyły całkowicie normalnie (`ndcg@10` 0,0448 i
0,0449, oba zbieżne). Wariant reguły z jednym trafieniem przerwałby oba, czyli
dałby **2/19 = 10,5% fałszywych alarmów** zamiast zera. Wymóg dwóch kolejnych
trafień, zapisany w ADR jako konserwatywny bufor kosztujący 256 kroków, okazał
się jedynym powodem, dla którego swoistość wyszła czysta.

### Zapadnięcie było totalne, nie graniczne

Seed 64: `corpus_recall_at_100` = **0,000000** i `corpus_ndcg_at_10` =
**0,000000** — model nie odzyskał ani jednego dokumentu z 139 782. Sygnał
pośredni był poniżej podłogi na wszystkich trzech kontrolach (0,133 / 0,055 /
0,070), więc reguła trafiłaby na kroku **512** z 1024 i odcięłaby całą
ewaluację, czyli ~1250 s z ~1460 s runu.

### Kierunek straty nie wniósł nic

Reguła straty nie trafiła **ani razu** w całej serii — także na zapadniętym
seedzie 64, którego strata malała. Łącznie z serią walidacyjną kierunek straty
wykrył 1 z 3 zapadnięć (tylko seed 50). Potwierdza to podział ról zapisany w
ADR: to tani fail-fast, nie guardrail, a `loss_based_guardrail_permitted: false`
z ADR M-03 pozostaje w mocy. Gdyby detekcja opierała się na stracie, dwa z
trzech zapadnięć przeszłyby dalej.

## Odsetek zapadnięć: 1/20 przy tej konfiguracji

**5,0%**, 95% CI Cloppera-Pearsona **[0,1%; 24,9%]**. Wcześniejszy szacunek
18,5% (5/27) pochodzi z **mieszanki** konfiguracji: dziesięciu runów confirmu
TriviaQA, dwunastu runów sweepu budżetu 1024/2048 i pięciu runów serii
wariancji. Ta seria jest pierwszym oszacowaniem przy jednej, ustalonej
konfiguracji, ale jej przedział jest tak szeroki, że **nie zaprzecza** tamtej
liczbie i jej nie zastępuje. Praktyczny wniosek jest niezmieniony: zapadnięcia
zdarzają się dostatecznie często, by zatruć porównanie trzysiedowe.

## Nieselekcjonowany rozkład metryki: przyrząd ma mniejszą rozdzielczość, niż sądziliśmy

Dziewiętnaście zbieżnych runów jednego ramienia, bez reseedu, więc bez selekcji:

| statystyka | wartość |
|---|---|
| średnia `corpus_ndcg_at_10` | **0,057722** |
| odchylenie standardowe | **0,014178** |
| zakres | 0,038828 – 0,089692 |

Półszerokości 95% CI: **0,0160** (n=3), **0,0124** (n=5), **0,0088** (n=10),
**0,0062** (n=20), wobec niezmienionego progu wyższości `+0,01`.

**To koryguje wcześniejszą korektę.** Raport z 2026-08-17 na czterech zbieżnych
runach oszacował sd na 0,0046 i uznał wcześniejsze podejrzenie o braku
rozdzielczości za przedwczesne. Przy n=19 sd jest **trzykrotnie większe** i
zakres jednego ramienia rozciąga się od 0,039 do 0,090 — czyli rozstęp
przekracza mierzony efekt pięciokrotnie. Estymacja z n=4 była po prostu szumna,
i to w optymistyczną stronę.

Zastrzeżenie, bez którego ta liczba wprowadzałaby w błąd: statystyką decyzyjną
programu jest **sparowana różnica per-seed** między ramionami, nie średnia
jednego ramienia. Sparowanie kasuje część wariancji wspólnej dla obu ramion —
zmierzone sd par to 0,0126 przy sd pojedynczego ramienia 0,0142, co oznacza
dodatnią korelację ramion na tym samym seedzie. Planowanie mocy należy więc
robić na 0,0126: półszerokość 0,0110 przy 5 parach i 0,0078 przy 10 parach.
Minimum 5 par seedów z ADR M-03 leży dokładnie **na granicy** rozdzielczości
progu `+0,01`; komfortowy zapas zaczyna się przy ~10 parach. To jest twarda
przesłanka kosztowa dla każdego przyszłego porównania, w tym dla Task 07.

## Koszt

Kontrole pośrednie kosztowały średnio **10,1 s na run** (trzy kontrole po ~3,4 s)
przy treningu ~195 s i budowie indeksu ~730 s, czyli **~0,7% czasu runu**.
Całe okno: 20 runów, ~8,2 h GPU, bez awarii i bez interwencji.

## Uwaga porządkowa o artefaktach

Po policzeniu pomiaru skasowano 20 katalogów `model/` tej serii (~9,5 GB), bo
zamrożony preflight D01b wymaga 30 GB wolnego miejsca. Skasowano **wyłącznie**
wagi probe: `result.json`, `corpus_retrieval_summary.json`, `train_summary.json`,
krzywe straty i dzienniki kontroli pośrednich zostały nietknięte, więc każda
liczba tego raportu jest nadal przeliczalna ze skryptu pomiarowego. Wagi nie są
przez nic referencjonowane (seria jest jednoramienna i nic nie promuje) i są
odtwarzalne z zamrożonej receptury i seeda. Decyzja właściciela, 2026-08-23.

## Czego ten pomiar nie robi

- Nie kalibruje żadnego progu; wszystkie liczby policzono regułą zamrożoną
  przed serią.
- Nie jest porównaniem ramion — wszystkie 20 runów to jedno ramię W06 — i nie
  promuje ani nie degraduje niczego.
- Nie unieważnia i nie przelicza żadnego zamkniętego pomiaru. Runy obserwacyjne
  nie są też porównywalne z runami w trybie przerywającym jako składniki jednego
  porównania.
- Nie autoryzuje treningu Task 07 (`task07_training_authorized=false`) ani
  kampanii Task 09; `final_tests_used=[]`.
- Czułość na jednym zapadnięciu jest praktycznie niezmierzona. Zwiększanie
  pewności co do czułości wymagałoby kilkudziesięciu runów, co przy 5% odsetku
  zapadnięć jest kosztem rzędu dni GPU i nie jest tym ADR-em zlecone.
