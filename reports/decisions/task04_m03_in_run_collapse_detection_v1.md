# Task 04 / M-03 — detekcja zapadnięcia probe w trakcie runu i automatyczny reseed (ADR v1, 2026-08-21)

Status: **zamrożony prospektywnie, przed uruchomieniem pierwszego nowego runu.**
Kontrakt: `task04-m03-in-run-collapse-detection-v1`.
Config progów: `configs/evaluation/task04_m03_in_run_collapse_detection_v1.yaml`.
Autoryzacja właściciela: 2026-08-21 (zakres: detekcja w trakcie runu i reseed).

## 1. Kontekst i po co ten ADR

Pomiar metrologiczny
[`task04_probe_within_arm_variance_2026-08-17.md`](../measurements/task04_probe_within_arm_variance_2026-08-17.md)
ustalił, że **zapadnięcia probe są prawidłowością, nie pechem**: 1/5 replikatów
tego samego ramienia (różny wyłącznie seed) i 5/27 = 18,5% łącznie z kalibracją
M-03. Porównanie dwóch ramion na trzech seedach ma ~71% szans, że zawiera co
najmniej jeden zapadnięty run, a pojedynczy zapadnięty run (W06-S43) wytworzył
różnicę per-seed `+0,1024`, czyli o rząd wielkości większą od mierzonego efektu.

Wnioskiem operacyjnym tamtego raportu nie było dokładanie seedów (koszt rośnie
liniowo, a zapadnięte runy nadal wchodzą do agregatów), lecz **wykrywanie
zapadnięcia w trakcie runu i automatyczny reseed**. Dziś jest to niemożliwe:
`embedder_probe.py` utrwala z całego treningu wyłącznie `losses[0]` i
`losses[-1]`, nie robi żadnej pośredniej ewaluacji, a guardrail M-03 działa
dopiero na artefaktach końcowych.

Rachunek kosztu, który uzasadnia miejsce detekcji: w serii wariancji trening
trwał ~188 s, a samo zbudowanie indeksu korpusu ewaluacyjnego (139 782
dokumenty) ~739 s, przy całości runu rzędu 22 minut. Zapadnięcie wykryte w
trakcie treningu pozwala **pominąć całą ewaluację**, czyli ~90% kosztu runu.

Ten ADR zamraża kontrakt **przed** uruchomieniem czegokolwiek, bo zmienia
`embedder_probe.py` — narzędzie użyte w zamkniętych pomiarach.

## 2. Decyzja

Do `embedder_probe.py` wchodzi wersjonowana, **domyślnie wyłączona** ścieżka
detekcji zapadnięcia w trakcie treningu z automatycznym reseedem. Implementacja:
`src/doc2query/evaluation/probe_in_run_collapse.py` (kontrakt, reguły, budowa
zbioru pośredniego) plus punkty zaczepienia w `embedder_probe.py`; włączana
wyłącznie przez `scripts/train_probe_embedder.py --collapse-detection-config`.

**Bez tej flagi kod probe zachowuje się identycznie jak dotąd**: żadnego
dodatkowego pliku, żadnego dodatkowego klucza w `train_summary.json` ani
`result.json`, żadnej dodatkowej ewaluacji, żadnego zużycia RNG. Odtwarzalność
istniejących artefaktów jest kryterium akceptacji A1 i A4 (§6).

### 2.1. Co jest utrwalane (tylko przy włączonej fladze)

| Plik w katalogu runu | Zawartość |
|---|---|
| `training_loss_curve.jsonl` | **pełna krzywa straty**: jeden wiersz na krok (`step`, `loss`) |
| `training_interim_evaluation.jsonl` | jeden wiersz na pośrednią ewaluację (`step`, `train_holdin_recall_at_100`, `chance_level`, `floor`, `below_floor`, `loss_window_first`, `loss_window_last`, `loss_non_decreasing`, `seconds`) |
| `collapse_detection_journal.jsonl` | jeden wiersz na **próbę** (`attempt_index`, `seed`, `outcome`, `rule`, `detected_at_step`, wartości sygnałów, znaczniki czasu) |

Krzywa straty i dziennik pośrednich ewaluacji są zapisywane przyrostowo i
odtwarzane z checkpointu przy wznowieniu (checkpoint już dziś trzyma pełną listę
strat, więc wznowienie nie gubi ani nie duplikuje wierszy). Próby zakończone
zapadnięciem **zachowują** swoje `training_loss_curve.jsonl` i
`training_interim_evaluation.jsonl` w podkatalogu próby — dowód zapadnięcia nie
znika przy reseedzie; kasowany jest wyłącznie `training_checkpoint.pt`.

### 2.2. Pośrednia ewaluacja retrievalowa: co i na czym

Co `interval_steps = 256` kroków (pierwsza kontrola na kroku 256) liczony jest
`train_holdin_recall_at_100` na **zbiorze zbudowanym wyłącznie z par
treningowych runu**, nie na zbiorze ewaluacyjnym:

- korpus pośredni: **2048** pierwszych unikalnych pozytywów po posortowanym
  `positive_doc_id` (albo wszystkie, jeśli jest ich mniej);
- zapytania: **128** pierwszych wierszy po `example_id`, których pozytyw leży w
  tym korpusie;
- metryka: Recall@100 pozytywu w korpusie pośrednim, dokładny iloczyn skalarny
  na znormalizowanych wektorach.

Powód użycia danych treningowych, a nie ewaluacyjnych, jest dokładnie ten sam,
dla którego ADR M-03 nie filtruje po metryce decyzyjnej: **żadna decyzja o
przerwaniu runu nie może być podjęta na zmiennej wynikowej ani na zbiorze, na
którym potem zapada decyzja o ramieniu.** Pytanie zadawane pośrednio brzmi „czy
ten embedder odzyskuje cokolwiek", a nie „czy jest lepszy" — na danych
treningowych zdrowy run odpowiada trywialnie, a zapadnięty nie odpowiada wcale.

Koszt: 2176 zakodowanych tekstów na kontrolę, przy zmierzonych ~189 tekstach/s
to ~12 s, czyli ~35 s na run o budżecie 1024 kroków (~19% czasu treningu, ~3%
czasu całego runu). Rzeczywisty narzut jest raportowany w pomiarze.

Stan RNG (CPU i CUDA) jest zapisywany przed pośrednią ewaluacją i przywracany po
niej, a model wraca do trybu `train`. Trajektoria treningu runu z włączoną
detekcją, w którym nic nie wykryto, ma więc być **identyczna** z trajektorią tego
samego seeda bez detekcji; sprawdza to kryterium A4.

### 2.3. Reguła detekcji i progi

Kontrola na każdym pośrednim kroku liczy dwa niezależne sygnały:

1. **Podłoga retrievalowa** (`interim_recall_below_chance_floor`) — sygnał
   właściwy. Trafienie, gdy
   `train_holdin_recall_at_100 < min_chance_multiple × (retrieval_depth / rozmiar korpusu pośredniego)`,
   czyli `< 4,0 × 100/2048 = 0,1953`. Mnożnik `4,0` **nie jest nową kalibracją**:
   to ta sama stała `min_chance_multiple`, którą zamroził ADR M-03, przeniesiona
   na mniejszy korpus. Podłoga medianowa M-03 jest tu niedostępna z definicji —
   mediana porównania nie istnieje w trakcie pojedynczego runu — więc używana
   jest wyłącznie część niezależna od ramienia i od wyniku.
2. **Kierunek straty** (`loss_direction_non_decreasing`) — tani fail-fast.
   Trafienie, gdy `mean(ostatnie 64 straty) >= mean(pierwsze 64 straty)`.

**Run jest przerywany, gdy którakolwiek z reguł trafi na dwóch kolejnych
kontrolach** (`consecutive_hits_required = 2`). Najwcześniejsze możliwe
przerwanie to więc krok 512 z 1024. Wymóg dwóch kolejnych trafień jest
konserwatywny celowo: kosztuje 256 kroków (~47 s), a chroni przed przerwaniem
zdrowego, wolno startującego runu.

#### Czym te reguły nie są

- Kierunek straty jest **fail-fastem operacyjnym**, nie guardrailem zbieżności.
  `loss_based_guardrail_permitted: false` z ADR M-03 **zostaje w mocy** i jest
  powtórzone wprost w nowym kontrakcie: poziom straty ma z retrievalem
  `r = −0,199`, więc nie orzeka o zbieżności.
- Zmierzona charakterystyka (czułość 2/5, fałszywe alarmy 0/22) dotyczy reguły
  **końcowej** `last_loss >= first_loss`. Wariant okienkowy użyty tutaj jest jej
  uogólnieniem i jego czułość ani swoistość **nie są zmierzone**; dlatego nie
  wolno mu przypisywać tamtych liczb, a jego zachowanie na nowych runach jest
  jednym z mierzonych wyników (§6, A3).
- **Żadna z tych reguł nie orzeka o zbieżności ukończonego runu.** Jedynym
  organem orzekającym pozostaje zamrożony guardrail M-03 na artefaktach
  końcowych. Reguły tego ADR decydują wyłącznie o tym, czy dany *przebieg*
  zostaje przerwany i powtórzony z innym seedem.

### 2.4. Polityka reseedu

- **Maksymalnie 3 próby** na jeden run (`max_attempts: 3`): pierwotna plus dwa
  reseedy.
- **Wybór seeda jest deterministyczny i bezkolizyjny**:
  `seed(attempt_index) = requested_seed + 1000 × attempt_index`. Żadnego losowego
  ani ręcznego doboru; skok 1000 nie koliduje z żadnym seedem użytym dotąd w
  programie (42–51).
- **Liczba prób nigdy nie jest ukryta.** Po sukcesie `train_summary.json` i
  `result.json` dostają blok `collapse_detection` z: kontraktem, `requested_seed`,
  `effective_seed`, `attempt_count`, `detection_count` oraz pełną listą prób
  (seed, reguła, krok wykrycia, wartości sygnałów). Blok jest obowiązkowy także
  wtedy, gdy nic nie wykryto (`detection_count: 0`). Dziennik prób zostaje na
  dysku niezależnie od bloku.
- **Wyczerpanie prób jest porażką runu, nie cichym sukcesem.** Po trzeciej
  wykrytej próbie run kończy się statusem `collapse_unresolved`, nie zapisuje
  `train_summary.json` ani `result.json` i zwraca niezerowy kod wyjścia. Nigdy
  nie powstaje artefakt `status: "measured"` z ukrytą historią prób.
- **Reseed jest selekcją i tak jest nazwany.** Rozkład seedów runu z włączoną
  detekcją jest warunkowy („nie wykryto zapadnięcia"), więc średnia ramienia jest
  policzona po lepszej stronie rozkładu. Dlatego: (a) detekcja musi być włączona
  **symetrycznie w obu ramionach** porównania, (b) w artefakcie porównania
  zapisywana jest liczba prób obu ramion, (c) guardrail M-03 ze sparowanym
  odrzucaniem seedów obowiązuje **nadal**, bo reguła podłogi 4× ze zbioru
  treningowego nie łapie każdego zapadnięcia.
- Wznowienie po przerwaniu procesu kontynuuje ostatnią niedokończoną próbę z jej
  checkpointu; próby już zapisane w dzienniku nie są powtarzane.

## 3. Rachunek porównywalności: co przestaje być porównywalne

Zmiana dotyka narzędzia, więc rachunek jest jawny.

**Nie przestaje być porównywalne nic, co zostało zmierzone dotąd.** Wszystkie 27
zakończonych runów (10 confirmu TriviaQA S42–46, 12 sweepu budżetu, 5 serii
wariancji S47–51) powstało bez tej ścieżki, a ścieżka jest domyślnie wyłączona.
Run bez flagi wykonuje tę samą pętlę treningową, te same operacje RNG i zapisuje
te same pliki co przed zmianą.

**Przestaje być porównywalne zestawienie runów o różnym ustawieniu detekcji.**
Konkretnie:

1. Run z włączoną detekcją **nie jest** porównywalny z runem bez detekcji jako
   składnik tego samego porównania ramion, bo jego seed jest warunkowany. Nowy
   kontrakt zabrania mieszania: oba ramiona muszą mieć identyczne ustawienie
   detekcji, a artefakt porównania musi je odnotować.
2. Zamknięte pomiary, których **nie wolno** przeliczać, powtarzać ani
   unieważniać tym ADR-em:
   - kalibracja i diagnostyka M-03
     (`reports/measurements/task04_m03_probe_convergence_calibration_2026-08-16.md`);
   - TriviaQA `dev_confirm` Task 05 wraz z nagłówkiem `+0,0478666` i jego
     zapisanym zastrzeżeniem o niezbieżnym W06-S43;
   - sweep budżetu probe 1024/2048;
   - seria wariancji wewnątrz ramienia S47–S51.
   Ich artefakty pozostają bitowo nietknięte. Fakt, że przyszłe runy będą
   powstawać z detekcją, **nie** czyni tamtych wyników błędnymi — czyni je
   pomiarami innego, nieselekcjonowanego reżimu i tak mają być cytowane.
3. Odsetek zapadnięć 18,5% jest własnością reżimu **bez** detekcji. Nie wolno go
   raportować dla runów z detekcją; te mają własną, osobno raportowaną liczbę
   prób i wykryć.

Progów M-03 ten ADR nie zmienia, splitów, progu `source_en_score >= 23.50`, wag
rerankerów ani żadnego zamrożonego configu Tasków 04–06 nie dotyka.

## 4. Czego ten ADR nie robi

- **Nie autoryzuje treningu Task 07** — `task07_training_authorized=false`
  pozostaje bez zmian. Nie autoryzuje też kampanii Task 09 ani żadnego nowego
  porównania ramion.
- **Nie promuje i nie degraduje** żadnego ramienia. Runy walidacyjne §5 używają
  jednego ramienia (W06) i nie tworzą różnicy Hybrid−W06.
- **Nie otwiera** testów finalnych: `final_tests_used=[]`.
- Nie zmienia guardraila M-03 ani jego progów i nie tworzy drugiego organu
  orzekającego o zbieżności.
- Nie dotyka niczego w `artifacts/task06/` ani biegnącego audytu dual-LLM.

## 5. Runy walidacyjne (zamrożone przed uruchomieniem)

Tanie korpusy, dokładnie te same wejścia, hiperparametry i budżet co seria
wariancji S47–S51 (`baseline.jsonl` W06, TriviaQA dev, 1024 kroki, batch 2,
prefix 3072, checkpoint co 64 kroki). Nowy katalog
`runs/task04_probe_inrun_collapse_v1/`; katalogi S47–S51 pozostają nietknięte.

| Run | Seed | Detekcja | Cel |
|---|---|---|---|
| `INRUN-CONTROL-S47-OFF` | 47 | wyłączona | odtwarzalność i baseline niedeterminizmu GPU (A1, A4) |
| `INRUN-S47-ON` | 47 | włączona | brak fałszywego alarmu, identyczność trajektorii (A3, A4) |
| `INRUN-S48-ON` | 48 | włączona | brak fałszywego alarmu (A3) |
| `INRUN-S51-ON` | 51 | włączona | brak fałszywego alarmu (A3) |
| `INRUN-S50-ON` | 50 | włączona | znane zapadnięcie: wykrycie + reseed (A2, A5) |

Pięć zadań, poniżej limitu ośmiu na partię. Kolejka jest wznawialna i pamięta
ukończone zadania; przerwanie kosztuje najwyżej jeden checkpoint.

## 6. Kryteria akceptacji (zamrożone przed odczytem wyników)

- **A1 — brak zmiany przy fladze wyłączonej.** `INRUN-CONTROL-S47-OFF` zapisuje
  dokładnie ten sam zestaw plików i ten sam zestaw kluczy `train_summary.json` /
  `result.json` co zamrożony `PROBE-VAR-W06-4.5B-S47`; żadnego pliku detekcji.
  Test CPU na tanim fixture wymusza równość bajtową podsumowania z wersją sprzed
  zmiany.
- **A2 — prawdziwie dodatni.** `INRUN-S50-ON` wykrywa zapadnięcie najpóźniej na
  kroku 768, wykonuje reseed, a run finalnie przyjęty przechodzi zamrożony
  guardrail M-03 zastosowany post hoc do nowej serii. Raportowany jest
  oszczędzony czas.
- **A3 — brak fałszywych alarmów.** Na trzech znanych zdrowych seedach (47, 48,
  51) liczba wykryć wynosi **0**.
- **A4 — identyczność trajektorii.** Dla seeda 47 różnica `first_loss`,
  `last_loss` i `corpus_ndcg_at_10` między wariantem z detekcją a wariantem bez
  detekcji nie przekracza różnicy między `INRUN-CONTROL-S47-OFF` a zamrożonym
  `PROBE-VAR-W06-4.5B-S47` (czyli samego niedeterminizmu GPU). Test CPU żąda
  równości dokładnej krzywej straty na deterministycznym fixture.
- **A5 — provenance.** Każdy run z włączoną detekcją ma w `result.json` blok
  `collapse_detection` z liczbą prób i seedami; run z wyczerpanymi próbami nie
  produkuje artefaktu `measured`.

**Progi i reguły są zamrożone tym dokumentem. Jeżeli którekolwiek kryterium nie
przejdzie, wynik jest raportowany jako negatywny, a progów nie wolno stroić po
fakcie** — poluzowanie wymaga nowego, prospektywnego ADR. Niedowiezione
kryterium blokuje używanie detekcji w porównaniach ramion, ale nie unieważnia
niczego zamkniętego.

`final_tests_used=[]`. `task07_training_authorized=false`.
