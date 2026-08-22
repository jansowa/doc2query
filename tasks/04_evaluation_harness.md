# Task 04 — Kompletny harness ewaluacyjny

> [Centralny rejestr zadań i statusów](README.md). Każda zmiana statusu lub zakresu tego zadania musi aktualizować rejestr w tym samym commicie.

## Status

`IMPLEMENTED`

Aktualizacja 2026-08-22 (detekcja zapadnięcia probe w trakcie runu i automatyczny
reseed): prospektywny ADR
[`task04_m03_in_run_collapse_detection_v1.md`](../reports/decisions/task04_m03_in_run_collapse_detection_v1.md)
zamrożono **przed pierwszym nowym runem**; zamraża on utrwalanie pełnej krzywej
straty i pośredniej ewaluacji retrievalowej co 256 kroków, regułę detekcji,
politykę reseedu, rachunek porównywalności oraz kryteria akceptacji A1–A5.
Implementacja (`src/doc2query/evaluation/probe_in_run_collapse.py`, haki w
`embedder_probe.py`, flaga `--collapse-detection-config`, 16 testów CPU) jest
**domyślnie wyłączona**. Raport:
[`task04_m03_in_run_collapse_detection_2026-08-21.md`](../reports/measurements/task04_m03_in_run_collapse_detection_2026-08-21.md).

- **Wszystkie pięć kryteriów przeszło.** Run kontrolny z wyłączoną flagą
  odtworzył zamrożony S47 **bit w bit**, więc niedeterminizm GPU okazał się
  zerowy i kryterium identyczności trajektorii (A4) rozstrzygnęło się przy
  dopuszczalnej różnicy 0 — run z detekcją dał te same cyfry. S48 i S51 z
  detekcją również odtworzyły co do cyfry zamrożone wartości serii wariancji.
- **Detekcja zadziałała dwa razy pod rząd na seedzie 50:** próby 50 i 1050
  przerwano na kroku 512 (recall pośredni 0,422 i 0,141 przy podłodze 0,521),
  przyjęto dopiero seed 2050. Separacja jest czysta: zapadnięte 0,141–0,422,
  zdrowe 0,844–0,977, bez przypadku pośredniego na 15 kontrolach. Zapadnięty
  przebieg kosztuje 99 s zamiast ~1500 s, a koszt własny mechanizmu to ~10 s na
  run (~0,7% runu).
- **Kierunek straty wypadł słabiej niż reguła retrievalowa:** przeoczył
  zapadnięcie seeda 1050 (strata malejąca przy recallu 0,141). Potwierdza to
  podział ról z ADR — `loss_based_guardrail_permitted=false` zostaje w mocy.
- **Zero fałszywych alarmów na trzech zdrowych seedach**, ale to 0/3 runów i
  0/9 kontroli: górna granica 95% CI odsetka fałszywych alarmów to ~0,63, więc
  swoistość **nie jest** wykazana.
- **Cena reseedu jest już widoczna:** przyjęty seed 2050 dał `corpus_ndcg_at_10`
  0,0832, powyżej całego zakresu serii zamrożonej (0,0475–0,0582), a sd serii
  wzrosło do 0,0152 wobec 0,0046. Reseed warunkuje rozkład seedów, więc średnich
  tej serii nie wolno czytać jako oszacowania ramienia W06, a detekcja musi być
  włączana symetrycznie w obu ramionach porównania.

Podłoga wyszła 0,521 zamiast zakładanych 0,195, bo pula pośrednia ma 768, a nie
2048 unikalnych pasaży (K=4 zapytania na pasaż); zamrożona reguła „2048 albo
wszystkie, jeśli jest ich mniej" zadziałała dokładnie jak zapisano, a
trudniejsze kryterium i tak przeszło. Zamknięte pomiary (M-03, confirm TriviaQA,
sweep budżetu, seria wariancji) **nie są przeliczane ani unieważniane**, ich
artefakty pozostają nietknięte, a mieszanie runów o różnym ustawieniu detekcji w
jednym porównaniu jest zabronione. Guardrail M-03 pozostaje jedynym organem
orzekającym o zbieżności ukończonego runu. `task07_training_authorized=false`,
`final_tests_used=[]`.

Aktualizacja 2026-08-17 (metrologia przyrządu probe, diagnostyka): pięć
replikatów **tego samego** ramienia W06 (identyczne wejście i hiperparametry co
zadania M-03, różny tylko seed 47–51) zmierzyło własności samego przyrządu, bez
tworzenia jakiejkolwiek różnicy między ramionami. Raport:
[`task04_probe_within_arm_variance_2026-08-17.md`](../reports/measurements/task04_probe_within_arm_variance_2026-08-17.md).

- **Zapadnięcia są prawidłowością:** 1/5 w tej serii, łącznie z kalibracją M-03
  **5/27 = 18,5%**. Przy różnicy wyłącznie w seedzie zapadnięcie nie jest
  własnością ramienia, danych ani budżetu, a procedury treningu probe. Porównanie
  dwóch ramion na trzech seedach ma ~71% szans, że zawiera zapadnięty run.
- **Wariancja wewnątrz ramienia jest znośna:** sd `corpus_ndcg_at_10` = 0,0046
  (CV 8,8%), półszerokość 95% CI 0,0052 przy n=3. Podejrzenie, że przyrząd
  zasadniczo nie ma rozdzielczości na próg `+0,01`, było **przedwczesne** —
  opierało się na sd 0,0126 z par; obie estymacje mają n=4 i są szumne. Problemem
  są jednostronne odchyłki od zapadnięć, nie szum wokół średniej.
- **Kierunek zmiany straty uzupełnia ustalenie `r = −0,199`:** reguła
  `last_loss >= first_loss` wykrywa zapadnięcia z czułością 2/5 przy **zerowych**
  fałszywych alarmach na 22 zbieżnych runach, i wychwytuje m.in. W06-S43, czyli
  run, który zniekształcił nagłówek TriviaQA confirm. Poziom straty pozostaje
  słabym sygnałem, kierunek jest sygnałem swoistym, ale niedoczułym — nadaje się
  na tani fail-fast, nie na zamiennik guardraila retrievalowego.

Wniosek operacyjny to **detekcja zapadnięcia w trakcie runu i automatyczny
reseed**, nie dokładanie seedów; wymaga prospektywnego ADR, bo dotyka
`embedder_probe.py` użytego w zamkniętych pomiarach. Zakres rozpisany w
[`reports/plans/task06_cross_cutting_review_2026-08-17.md`](../reports/plans/task06_cross_cutting_review_2026-08-17.md).
Progów M-03 nie kalibrowano, kodu probe nie zmieniano, nic nie zostało
wypromowane. `final_tests_used=[]`.

Aktualizacja 2026-08-16 (M-03: guardrail zamrożony prospektywnie i skalibrowany):
prerejestrowany ADR
[`task04_m03_probe_convergence_guardrail_v1.md`](../reports/decisions/task04_m03_probe_convergence_guardrail_v1.md)
zamraża kontrakt `task04-m03-probe-convergence-guardrail-v1`
(`configs/evaluation/task04_m03_probe_convergence_guardrail_v1.yaml`,
`src/doc2query/evaluation/probe_convergence.py`,
`scripts/apply_task04_m03_probe_convergence_guardrail.py`, 12 testów CPU):

- sygnał zbieżności jest **retrievalowy** (`corpus_recall_at_100`), a
  `loss_based_guardrail_permitted: false` jest wpisane w kontrakt wprost, bo
  zmierzone `r = −0,199` (n=12) wyklucza stratę treningową jako detektor;
  metryki decyzyjnej (`corpus_ndcg_at_10`) świadomie **nie** używa się jako
  filtru, żeby nie selekcjonować na zmiennej wynikowej;
- podłoga jest **niezależna od ramienia**:
  `max(4 × poziom losowy, ½ mediany sygnału wspólnej dla obu ramion)`;
- seed odrzucany jest **jako para** obu ramion, a wynik bez filtra jest liczony i
  raportowany obowiązkowo;
- decyzja wymaga łącznie ≥5 zbieżnych par seedów, dolnej granicy 95% CI
  sparowanego bootstrapu **po seedach** nie niżej niż niezmieniony próg `+0.01`
  oraz dokładnego jednostronnego testu znakowego `p ≤ 0.05`. Minimum pięciu
  seedów nie jest liczbą z wygody: przy czterech parach najmniejsze osiągalne
  `p` to `1/16 = 0,0625 > 0,05`, więc reguła byłaby konstrukcyjnie
  nierozstrzygalna.

Progi skalibrowano retrospektywnie na 22 zakończonych runach (10 confirmu
TriviaQA S42–46 i 12 sweepu budżetu) i zamrożono dla przyszłych porównań.
Guardrail oznaczył **4/22 runy** jako niezbieżne — dokładnie cztery najniższe po
sygnale, przy separacji 2,3× do pierwszego runu zdrowego. Diagnostyka pięciu
seedów confirmu TriviaQA (osobny artefakt, zgodnie z polityką „nie
reinterpretować zamkniętego confirmu”): po odrzuceniu pary seeda 43, w której
ramię kontrolne W06 nie zbiegło (`corpus_recall_at_100 = 0,000452`, poniżej
poziomu losowego), zostają **4 zbieżne pary**, czyli status
`insufficient_converged_seeds`. Średnia różnica bez filtra to `+0,0319`
(CI `[+0,0065,+0,0680]`), po filtrze `+0,0143` (sd 0,0126,
CI `[+0,0025,+0,0238]`) — dodatnia, ale z dolną granicą poniżej progu `+0,01`.
Zamknięty confirm **nie** jest unieważniony, nie jest powtarzany i nie jest
zastępowany; nic nie zostało wypromowane (`promotion_authorized=false`
w każdym porównaniu). Raport i artefakt:
[`task04_m03_probe_convergence_calibration_2026-08-16.md`](../reports/measurements/task04_m03_probe_convergence_calibration_2026-08-16.md),
[`task04/m03_probe_convergence_v1/summary.json`](../reports/measurements/task04/m03_probe_convergence_v1/summary.json).
Walidacja: Ruff, `mypy src`, pełny pytest. `final_tests_used=[]`.

Aktualizacja 2026-08-16 (M-03: treningi wykonane, guardrail wciąż do
zdefiniowania): cztery treningi seedów 45/46 confirmu TriviaQA są ukończone, a
towarzyszący sweep budżetu probe (12 runów) dostarczył twardej przesłanki dla
tego wymogu: przy stałym ramieniu i budżecie `corpus_ndcg_at_10` waha się od
0,0011 do 0,0826, a korelacja straty treningowej z wynikiem retrievalu wynosi
`r = −0,199` (n=12). Guardrail zbieżności musi więc opierać się na sygnale
retrievalowym, nie na stracie. Agregacji pięciu seedów nie wykonano. Wynik:
[`task06_unattended_compute_window_result_2026-08-16.md`](../reports/measurements/task06_unattended_compute_window_result_2026-08-16.md).

Aktualizacja 2026-08-14 (M-03 w toku): w oknie bezobsługowym 2026-08-14
uruchomiono brakujące seedy 45 i 46 confirmu TriviaQA dla obu ramion (W06 i
Hybrid), wyłącznie jako treningi probe. Agregacja pięciu seedów oraz
jakakolwiek reinterpretacja wyniku confirmu wymagają obecności właściciela i nie
zostały wykonane. ADR:
[`task06_unattended_compute_window_2026-08-14.md`](../reports/decisions/task06_unattended_compute_window_2026-08-14.md).

Aktualizacja 2026-08-13 (rozszerzenie zakresu, M-03): harness probe otrzymuje
obowiązkowy guardrail zbieżności treningu — run probe raportuje trajektorię
eval loss oraz minimalny sanity wynik retrieval, a seed niezbiegnięty (jak
W06 seed 43 w confirmie TriviaQA) jest jawnie flagowany i nie wchodzi do
agregatów bez symetrycznej analizy wrażliwości w obu ramionach. Przy tanich
korpusach ewaluacyjnych (rząd ≤200 tys. dokumentów) domyślna liczba seedów
confirmu wzrasta z 3 do 5. Definicje M-01–M-05: AGENTS.md §9.2. Wymóg jest
prospektywny i nie unieważnia zakończonych pomiarów.

Generyczna optymalizacja Harnessu po pomiarze S07 jest gotowa i zmierzona:
batchowana generacja causal/encoder-decoder, jawne umieszczanie modelu
inferencyjnego na CUDA, niezależne urządzenia sędziów, trwała pula BM25,
crash-safe resume i odzyskiwalne archiwizowanie niezgodnej trajektorii.
64-rekordowy frozen-dev benchmark obu sędziów CUDA osiągnął `3,417` rekordu/s
przy dwóch workerach BM25 i peak reserved `4,152 GiB`; pełny run zakończył
5 tys. przykładów i 25 tys. generacji. Raport:
[`runtime_optimization_2026-07-26.md`](../reports/measurements/task03_s07/runtime_optimization_2026-07-26.md).

Pełnokorpusowy retrieval probe został następnie przebudowany z 6598 osobnych
skanów macierzy na persistent exact sharded index i batched GEMM. Bieżące 100
shardów S07 (2 404 263 dokumenty) jest zgodne i nie wymaga konwersji. Query ma
osobny cache, journal jest crash-safe, model oraz primary judge są pomijane,
gdy ich etap jest kompletny, a postęp zawiera throughput i ETA. Backend zapisuje
`approximate=false`; CUDA audytuje rank/win na ośmiu query względem CPU exact i
automatycznie przechodzi na CPU przed zapisem przy niezgodności. Lekki benchmark
CPU potwierdził identyczne rangi (w tym remisy). Ograniczony benchmark na
rzeczywistych shardach 2 404 263×768 osiągnął 225,68 query/s, czyli 85,06×
wobec obserwowanej starej fazy 2,653 query/s; rank i hard-negative win były
identyczne na 8-query audycie CPU exact. Pełny probe S07 został następnie
ukończony na 6598 query i zapisany jako `development_complete`. Ze względu na
budżet `864000` tokenów wobec `1152000` w P-05 pozostaje jednak
`comparison_eligible=false` i nie może rozstrzygać architektury. Raport:
[`probe_retrieval_optimization_2026-07-26.md`](../reports/measurements/task03_s07/probe_retrieval_optimization_2026-07-26.md).

ADR zamyka S07 jako wynik diagnostyczny bez matched-budget rerunu, promocji do
`dev_confirm` i otwierania testów finalnych:
[`task03_s07_diagnostic_closure_2026-07-26.md`](../reports/decisions/task03_s07_diagnostic_closure_2026-07-26.md).

P-06 mass rescoring naturalnego train został później oznaczony `SUPERSEDED`:
źródłowe etykiety mają kompletne provenance silniejszego rerankera, próg
`23.50` jest już wyegzekwowany, a minimalny source margin wynosi `6.0`.
Lokalne primary/shadow służą do syntetycznych query; dla naturalnych tłumaczeń
wyłącznie do małego P06-T disagreement/manual audit. Nie kończyć pełnego
scoringu train ani nie wyprowadzać z niego drop/weights. ADR:
[`task03_p06_source_provenance_2026-07-26.md`](../docs/decisions/task03_p06_source_provenance_2026-07-26.md).

P06-T ma zamrożony train-only panel 300 rekordów (seed 42), wersjonowany
manifest oraz ślepy formularz z polami intent preservation, answerability,
semantic damage, encoding/text error i opcjonalną klasą powtarzalnego błędu.
Osobny plik triage zawiera wyłącznie lekkie heurystyki tekstowe; pola
primary/shadow/disagreement są `null`, nie zdefiniowano progu drop ani wag i
nie użyto testów finalnych. Ręczne oceny nie zostały wykonane, a właściciel
projektu później świadomie je anulował i zaakceptował resztkowe ryzyko
tłumaczeń. P06-T nie jest już bramką; frozen train pozostaje bez zmian. ADR:
[`task03_p06_t_waiver_2026-07-26.md`](../docs/decisions/task03_p06_t_waiver_2026-07-26.md).
Raport materializacji:
[`task03_p06_t_freeze_2026-07-26.md`](../docs/experiments/task03_p06_t_freeze_2026-07-26.md).

Harness v1.1 P-01–P-04 jest zaimplementowany i ma
testy CPU. P-03 został rzeczywiście zmierzony na 1 000 rekordów zamrożonego
dev; wynik `statistically_separated` nie otworzył żadnego testu finalnego.
HN1 BM25 było istotnie gorsze od HN0 i HN0+filter, natomiast HN0+filter nie
było odróżnialne od HN0. ADR
`reports/decisions/task04_p03_w05_negative_recipe.md` wybiera HN0+filter z
polityką `drop` dla pierwszych porównań probe. Nie jest to wybór generatora
ani otwarcie testów finalnych.

26 lipca pełna, inference-only bramka HN0/HN0+filter/HN1/HN2/HN3 zakończyła
się na wspólnej kohorcie 775/1000 frozen-dev query (`22.5%` common-legal drop).
HN1 nie odróżniło się od HN0+filter według primary; HN2 było łatwiejsze także
według shadow. Primary-perfect HN3 wynika z positive-aware veto tego samego
sędziego i nie jest niezależnym dowodem jakości; shadow dał kierunek przeciwny
oraz `9.81%` winner disagreement. Żaden nowy miner nie uzyskał zgodnej podstawy
do promocji, więc utrzymano HN0+filter/drop. `final_tests_used=[]`,
`training_runs=[]`. ADR:
[`task04_hn_full_gate_v1.md`](../reports/decisions/task04_hn_full_gate_v1.md).

P-04 ma wersjonowany ADR `1.0.0` i kontrakt `task04-p04-v1`: główną metrykę
finalną, guardraile non-inferiority, minimalny efekt `0.01`, seedy i successive
halving, rozdzielone raportowanie wariancji treningowej i bootstrapu po query,
czterowymiarowy budżet oraz jednorazowe otwarcie testów. Manifesty probe
zapisują fingerprint ADR i budżet, a comparatory fail-closed odrzucają braki
lub różnice wersji i budżetu.

Task pozostaje `IMPLEMENTED`, nie `DONE`: nie wykonano pełnych porównywalnych
probe natural/copy/generator, pełnego benchmarku primary/shadow, embeddingowych
miar diversity, ocen ludzi dla co najmniej 300 przypadków, pełnych indeksów
i finalnego rank10/embedder. D00–D12, Task 06 i finalne testy nie zostały
uruchomione. Poniższe akapity zachowują chronologię wcześniejszej implementacji
i blockerów; rozstrzygający bieżący stan P-03/P-04 opisują ich sekcje niżej.

P-05 preflight przeszedł audyt kampanii (9 wymaganych ramion ukończonych,
3 jawnie odroczone). GPU eligibility scoring przypiętym primary zachował
9965 naturalnych i 9973 syntetyczne rekordy. Po wspólnym przecięciu,
deterministycznym K=1 per dokument i wyrównaniu pod 25%/50:50 powstały trzy
ramiona po 9944 unikalne pary/dokumenty, fingerprint
`d89b799a…df67b5c`. Mix ma dokładnie 1243+1243 rekordy w `dev_screen` oraz
4972+4972 w pełnym prefiksie. Planner przechodzi bez blockerów i wskazuje
wyłącznie zamrożony `dev_intrinsic_rank10`; `final_tests_used=[]`.

Trzy probe `dev_screen` seed 42 zakończyły się na identycznym budżecie i 6598
query `dev_intrinsic_rank10`. Mixed 50/50 ma nDCG@10 `0.057568`, gold
`0.052762`, a synthetic-only `0.048618`. Bootstrap mixed-minus-gold wynosi
`+0.004806`, 95% CI `[+0.000876, +0.008692]`; synthetic-minus-gold
`-0.004143`, 95% CI `[-0.007937, -0.000361]`. Żaden wariant nie spełnia
progu praktycznego P-04 `+0.01`, a brak intrinsic guardraili blokuje
`dev_confirm`. Nie ma decyzji finalisty ani otwarcia testów. Audyt:
[`dev_screen_audit_2026-07-23.md`](../reports/measurements/task04_p05_dev_screen/dev_screen_audit_2026-07-23.md).

Follow-up CPU domierzył exact `corpus_round_trip_at_20` dla wszystkich ramion:
gold `0.116854`, mixed `0.130797`, synthetic-only `0.112307`; wspólny frozen
dev ma `format_valid_rate=1.0`, a pinned-primary sentence hit wynosi
`0.894665`. Pełny assembler wykonał 10 000 paired-query bootstrapów per
metryka, a fail-closed engine zwrócił bez błędów `non_inferior_only` dla mixed
i synthetic-only. Wszystkie guardraile przechodzą; primary CI obu wariantów
nie osiąga progu `+0.01`, więc `dev_confirm_authorized_arms=[]`. Testy finalne
pozostają zamknięte. Szczegóły:
[`p04_gate_decision_2026-07-23.md`](../reports/measurements/task04_p05_dev_screen/p04_gate_decision_2026-07-23.md).

Runner przypina
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, pokazuje postęp i uruchamia tylko trzy
25-procentowe runy seed 42. Jest wznawialny: waliduje i pomija ukończone
ramiona, a po zapisaniu modelu i `train_summary.json` pomija również scoring
`filter_negatives` oraz trening. Kodowanie korpusu zapisuje atomowe shardy co
około 1%, a ocenę query dopisuje po każdym rekordzie; oba etapy wykorzystują
przy wznowieniu wyłącznie artefakty o zgodnej tożsamości. Wcześniejszy
niepełny trening archiwizuje bez usuwania i ponawia wyłącznie bieżące ramię.
Logi są dopisywane, więc historia kolejnych uruchomień zostaje zachowana.
Szczegóły:
[`task04_p05_dev_screen_2026-07-21.md`](../reports/blockers/task04_p05_dev_screen_2026-07-21.md).

Po dwóch przerwaniach S07 probe trening zapisuje także atomowy, kroczący
checkpoint modelu, optymalizatora, schedulera, RNG i kroku. Wznowienie
odtwarza identyczną kolejność microbatchy i odrzuca niezgodny fingerprint.
Runner włącza postęp filtrowania, treningu, shardów korpusu i zapytań oraz
domyślnie wybiera najwcześniejszy nieukończony etap.

Centralny harness, zamrożone manifesty/ID, metryki i slice’y, raporty
HTML/Markdown, ślepy eksport A/B, bootstrap oraz zamrożona recepta probe
embeddera są zaimplementowane i przetestowane. P-03 ma gotowy kod i testy
kontraktu negatywów. Dev-only kalibracja progu i train-corpus BM25 są już
zmierzone, zweryfikowane i przypięte. Dodano kompletny one-command runner
P-03, który zamraża train ID, wznawia generację i trening, tworzy wspólną
kohortę HN0/HN0+filter/HN1, zrównuje jawny budżet tokenów, ocenia wyłącznie
zamrożony dev i wykonuje paired-query bootstrap przez dedykowany comparator.
Na lokalnej karcie 8 GB rzeczywiście oceniono W03, W05 i W06 w trybie
deterministic oraz diverse na tym samym zamrożonym panelu 100 rekordów z co
najmniej 10 hard negative’ami; wykonano też nieporównywalny 2-step smoke
probe’a.

Pozostają pełne, porównywalne runy probe natural/copy/W03/W05/W06, pełny
benchmark primary/shadow, embeddingowe miary diversity, oceny ludzi dla co najmniej
300 przypadków i pełny test rank10/embedder. S00 i diagnostyczny S07 są już
zmierzone. Bez pozostałych
pomiarów bramka Fazy B i główny ranking generatorów nie są zamknięte.
Dotychczasowy zakres W03/W05 opisuje
`docs/experiments/task04_8gb_evaluation_2026-07-18.md`, a porównanie z W06
`docs/experiments/task03_w06_vs_1_5b_2026-07-19.md`.

Shortlista nowych baz probe'a (`mmlw-roberta-base`,
`polish-distilroberta`, Ettin 32M/17M), ryzyka oraz redukowana procedura wyboru
recepty v2 są zapisane w
`docs/decisions/probe_embedder_candidates.md`. P-03 zachował model i budżet
recepty v1, ale jawnie podbił wersję kontraktu do `probe-v1.1-p03`; nie wpisano
niewykonanych wyników.

19 lipca ten sam kontrakt intrinsic zastosowano do finalnego checkpointu W06
4.5B/50k. Powstały komplet 500 generacji, raporty i sparowane bootstrapy W06
vs W03/W05. W06 ma potwierdzoną przewagę greedy MRR/nDCG nad W05, ale diverse
retrieval pozostaje nierozróżnialny, a miary różnorodności są gorsze. Nie
zastępuje to nadal niewykonanych pełnych probe embedderów.

Audyt z 18 lipca wykazał, że dotychczasowy kontrakt nie rozdziela rankingu
w małej puli od retrievalu korpusowego, nie ma natywnego polskiego holdoutu
i nie definiuje polityki fałszywych negatywów. Wyniki W03/W05/W06 pozostają
diagnostyczne i nie wolno na ich podstawie wybierać finalisty; wykryte braki
kontraktu zostały następnie zaadresowane przez P-01–P-04.

19 lipca zaimplementowano P-01 Harness v1.1. Kontrakty
`candidate_pool_ranking` i `corpus_retrieval` mają rozłączne prefiksy
`pool_`/`corpus_`, każdy rekord i blok metryk podaje rozmiar puli, a wspólna
walidacja odrzuca Recall@K dla puli mniejszej niż K. Probe korzysta teraz z
jawnego pełnego pliku dokumentów zamiast puli złożonej z rekordów testowych.
Dodano dyskowy BM25 na cachowanej analizie tekstu i brute-force zamrożonego
bi-encodera z revision, licencją i fingerprintami, deterministyczny backfill
puli diagnostycznej oraz korpusowy round-trip@1/5/20/100 ze specyficznością,
marginesem i korelacją z marginesem rerankera. Testy tanie przeszły; nie
zbudowano jeszcze pełnoskalowych indeksów korpusu i nie uruchomiono probe.

19 lipca zaimplementowano kontrakt i kod P-02. Audyt źródeł pierwotnych
wybrał test PolQA jako natywny kandydat oraz odrzucił całe PIRB/MAUPQA jako
jednorodny „native” holdout. Zamrożono rzeczywisty
`test_translated_msmarco_pl` (16 272 rekordy) z profilami `quick=100`,
`medium=500`, `full=16 272`, hashami list ID i fingerprintami. Importer PolQA,
profile kosztu, weryfikacja immutable manifestu, osobne raportowanie
native/translated oraz jawny model-free sygnał `translationese-surface-v1`
są gotowe i przetestowane bez modeli/GPU.

Po usunięciu przejściowego problemu sieciowego domknięto artefakty P-02:
zamrożono 956 pytań `test_native_pl`, pełny korpus PolQA z 7 097 288
dokumentami oraz trzy profile z rzeczywistymi hashami ID i fingerprintami.
Audyt exact-match nie znalazł wspólnych query ani dokumentów z translated
MS MARCO-PL; near-duplicate pozostaje jawnie `NOT MEASURED`. Manifest przeszedł
pełne `--verify` i nie ma blockerów. Nie zbudowano indeksu ani nie uruchomiono
probe. Stan ten był punktem wejścia do implementacji P-03.

19 lipca zaimplementowano bezpieczną część P-03. Recepta
`probe-v1.1-p03`/`probe-negatives-v1` definiuje deterministyczne HN0,
HN0+filter i HN1 BM25, polityki `drop | demote | keep+log` z domyślnym
`drop`, scoring naturalnych i syntetycznych query przez zamrożony primary,
raport flag per query source/generator oraz komplet provenance w manifestach.
Porównania odrzucają różne wersje recepty, strategie, polityki, progi,
identyfikatory/fingerprinty kalibracji i fingerprinty BM25. Testy jednostkowe
i smoke korzystają wyłącznie z mockowanego rerankera, bez modeli i GPU.

19 lipca domknięto oba pierwotne artefakty P-03. Pełny frozen dev (16 272
query) scored primary dał query-macro próg Youdena `8.617486953735352`;
artefakt ma fingerprint `9ee4280f…3b3f4` i nie używa żadnego testu. Zamrożony
`train-corpus-v1` zawiera 2 211 463 dokumenty; BM25 spaCy ma integrity check
`ok` i fingerprint `e5df2432…2119`. Oba są przypięte w recepcie.

Historyczny preflight wykrył brak train-query W05 i bazowych wag Bielik 1.5B.
Runner odtworzył brakujące query legalnie z przypiętego checkpointu, bez użycia
testu ani naturalnych query jako substytutu; późniejsze wznowienie domknęło
HN0/HN0+filter/HN1, jak opisano w sekcji P-03.

## Harness v1.1 — blokery po audycie

Poniższy pakiet jest następnym zadaniem projektu. Kolejność wykonania:
`P-01 → P-02 → P-03 → P-04`. Uzasadnienie historyczne znajduje się w
[`docs/plan_poprawek_po_audytach.md`](../docs/plan_poprawek_po_audytach.md);
operacyjny zakres i status są utrzymywane tutaj oraz w `tasks/README.md`.

### P-01 — rozdzielone protokoły retrieval — `IMPLEMENTED`

- `candidate_pool_ranking`: pozytyw(y) i odziedziczone lub deterministycznie
  uzupełnione negatywy; metryki z prefiksem `pool_`, diagnostyka generatora;
- `corpus_retrieval`: pełny zamrożony `documents.parquet`; metryki z prefiksem
  `corpus_`, główna ocena probe i round-trip generatora;
- indeks korpusowy BM25 oraz zamrożony pomocniczy bi-encoder
  (FAISS albo brute-force), z revision, licencją i fingerprintem;
- `effective_candidate_count`, margines do najlepszego niepozytywnego
  dokumentu i `possibly_ambiguous_query`;
- każda metryka raportuje rozmiar puli; pipeline odrzuca `recall@K`, gdy
  pula ma mniej niż K dokumentów;
- raportuje `corpus_round_trip@1/5/20/100` i jego korelację z marginesem
  rerankera.

Implementacja i testy jednostkowe/smoke są gotowe. Zbudowano diagnostyczny
train-corpus BM25 wymagany przez P-03. Nadal nie zbudowano pełnego indeksu
porównawczego nad całym korpusem ani pomocniczego bi-encodera; ich throughput
oraz round-trip W03/W05/W06 nie zostały zmierzone.

### P-02 — natywny polski holdout — `IMPLEMENTED`

- audyt PIRB, PolQA i ewentualnie MAUPQA w
  `docs/datasets/native_pl_holdout.md`: licencja, pochodzenie języka,
  kontaminacja i overlap z `msmarco_pl`;
- zamrożone `test_native_pl` oraz `test_translated_msmarco_pl`, opcjonalnie
  `test_transfer_ood`, wraz z fingerprintami i hashami ID;
- `evaluate embedder` i raport pokazują native i translated osobno;
- native nie jest używany do strojenia; brak wyniku native oznacza raport
  niekompletny;
- dodać tani, jawnie opisany sygnał „translationese”.

Gotowe: audyt PIRB/PolQA/MAUPQA, przypięte revisions i licencje, bezpieczny
importer test-only, trzy deterministyczne profile kosztu, frozen native i
translated split z fingerprintami/hashami ID, pełny korpus PolQA, weryfikacja
manifestu, osobne sloty native/translated w `evaluate embedder` i raporcie,
status `incomplete` bez zmierzonego native oraz jawny sygnał translationese.
Szczegóły:
[`docs/datasets/native_pl_holdout.md`](../docs/datasets/native_pl_holdout.md).

Zmierzono exact overlap z MS MARCO-PL (zero identycznych query i dokumentów);
near-duplicate pozostaje jawnie niezmierzony i nie jest zastępowany założeniem.
Nie uruchomiono probe, benchmarku PIRB, pełnego indeksu ani żadnego wyniku
eksperymentalnego. Kolejną bramką Harness v1.1 jest P-03.

### P-03 — probe recipe v1 i false negatives — `MEASURED / ADR ACCEPTED`

- dla naturalnych i syntetycznych query primary reranker flaguje odziedziczony
  negatyw jako `possible_false_negative` według progu kalibracyjnego z Task 02;
- polityka `drop | demote | keep+log`, domyślnie `drop`, identyczna dla
  wszystkich wariantów; raportuje odsetek flag per generator;
- wersja recepty jest częścią manifestu, a porównanie odmawia pracy dla
  różnych wersji;
- jednorazowy sensitivity check W05: HN0, HN0+filter i HN1 BM25. Istotna
  różnica wymaga ADR przed dalszymi porównaniami;
- pełne HN0/HN0+filter/HN1/HN2/HN3 jest wymagane przed Task 09.

Kod, konfiguracja, manifesty, walidacja porównań i testy są gotowe. Domyślna
recepta działa fail-closed: wymaga przypiętego artefaktu Task 02 utworzonego
wyłącznie na dev i weryfikuje jego ID, fingerprint, fingerprint danych,
SHA-256 score’ów, rewizję primary, przestrzeń score’u, operator, próg oraz
metodę jego wyboru. HN1 dodatkowo wymaga przypiętego fingerprintu indeksu BM25.

Dev-only kalibracja oraz train-corpus BM25 są gotowe i przypięte. Właściwy
runner zakończył się kodem 0 i zapisał trzy ramiona oraz sparowany bootstrap
w `reports/measurements/task04_p03_w05_sensitivity/`. Na 1 000 query HN1 było
istotnie gorsze od obu pozostałych recept dla nDCG@10, MRR i hard-negative
win rate. HN0+filter względem HN0 miało nDCG@10 `+0.00682`, 95% CI
`[-0.00348, 0.01659]`, więc nie było statystycznie odróżnialne.

ADR `reports/decisions/task04_p03_w05_negative_recipe.md` wybiera HN0+filter
z polityką `drop` dla pierwszych porównań: zachowuje jakość dev w granicach
niepewności i stosuje przypiętą ochronę przed false negative. HN1 jest w tym
zakresie odrzucone. `final_tests_used=[]`; pomiar nie wybiera generatora.
Pełna bramka HN0/HN0+filter/HN1/HN2/HN3 została zmierzona 26 lipca na
zamrożonym dev; decyzja utrzymuje HN0+filter/drop z powodu braku zgodnej
primary/shadow podstawy do promocji nowego minera i nie otwiera testów finalnych.
Historia wznowienia:
[`task04_p03_runtime_recovery_2026-07-20.md`](../docs/experiments/task04_p03_runtime_recovery_2026-07-20.md).

### P-04 — kontrakt statystyczny i budżetowy — `IMPLEMENTED`

Przed pierwszym porównaniem utworzyć ADR z:

- główną metryką (propozycja: `corpus_ndcg@10` probe na `test_native_pl`);
- metrykami i marginesami non-inferiority dla grounding, answerability
  i formatu;
- minimalnym praktycznym efektem, seedami successive halving oraz osobnym
  raportowaniem wariancji między treningami i bootstrapu po query;
- budżetem liczonym równocześnie w tokenach, parach, unikalnych pasażach
  i K query/pasaż;
- regułami dev oraz jednorazowego otwarcia finalnych testów.

Pipeline porównań musi cytować wersję ADR i odmawiać porównania niezgodnych
definicji budżetu. Dopiero spełnienie P-01…P-04 zezwala na porównawcze probe,
eksperymenty D00–D12 oraz Task 06.

Gotowe są ADR
`reports/decisions/task04_p04_statistical_budget_contract.md`, maszynowy
kontrakt `configs/evaluation/comparison_contract_v1.yaml`, zapis wersji i
fingerprintu ADR oraz czterowymiarowego budżetu w manifestach probe i wspólna
walidacja fail-closed w comparatorach. Brak metadanych, różna wersja ADR,
różna definicja budżetu lub różnica w tokenach, parach, unikalnych pasażach
albo K blokuje porównanie. Nie uruchomiono przy tym żadnego probe ani testu.

21 lipca dodano CPU-only decision/preflight engine dla `dev_screen` i
`dev_confirm`. Weryfikuje plik oraz fingerprint ADR, komplet seedów, osobne
wartości/mean/sample SD/range, paired-query bootstrap bez mieszania wariancji,
minimalny efekt i trzy guardraile. Braki zwracają `incomplete`; pozostałe
statusy to `eligible`, `non_inferior_only` albo `rejected`. Engine nie czyta
finalnych testów i nie deklaruje niewykonanych wyników. Jest to wyłącznie
tooling: kampania, P-05/P-06 i porównawcze probe nadal są niewykonane, więc
Task pozostaje `IMPLEMENTED`, nie `DONE`.

Dodano CPU-only materializację wejść pierwszej macierzy P-05 zgodną z
`task04-p04-v1`. Trzy ramiona dzielą jedną kohortę i wszystkie cztery wymiary
budżetu, a mix ma dokładne 50/50 również w prefiksie 25%. Manifest zapisuje
SHA-256, provenance W05, HN0+filter/drop i `final_tests_used=[]`; planner usuwa
komendy przy braku lub driftcie manifestu albo pliku. Nie zmaterializowano
rzeczywistych danych i nie uruchomiono kampanii, P-05/P-06 ani probe.

## Cel

Zbudować centralny system ewaluacji generatora oraz end-to-end wpływu syntetycznych query na embedder.

## Zależności

Taski 01–03. Część infrastrukturalna może być rozwijana równolegle z Taskiem 03.

## Zbiory ewaluacyjne

Utwórz i zamroź:

1. `dev_intrinsic` — do strojenia promptów i progów;
2. `test_intrinsic` — naturalne query, niewidziane dokumenty;
3. `test_adversarial` — ręczne przypadki z Tasku 02;
4. `test_human_panel` — próbka do ocen A/B;
5. `test_embedder` — pełny retrieval test, niewykorzystywany do strojenia generatora.

Zapisz wersje, fingerprint i hash listy ID.

## Generacja ewaluacyjna

Każdy model ma być oceniany w co najmniej dwóch trybach:

- deterministic: greedy lub niska temperatura;
- diverse: ustalona temperatura/top-p i K próbek.

Parametry generacji są częścią identyfikatora runu.

## Intrinsic metrics

### Retrieval/grounding

- rank pozytywnego pasażu wśród co najmniej 10 hard negative’ów;
- Recall@1, Recall@5;
- MRR;
- nDCG@10;
- średni i percentylowy reranker margin;
- sentence-level source hit.

### Lexical copying

- content lemma Jaccard;
- query lemma precision/recall względem passage;
- longest common n-gram;
- normalized LCS;
- copy density;
- entity/number preservation;
- rozkład, nie tylko średnia.

### Diversity

Dla K query na dokument:

- distinct-1/2;
- Self-BLEU;
- mean/max pairwise lemma Jaccard;
- mean/max pairwise embedding cosine;
- duplicate rate;
- semantic cluster count;
- style entropy;
- focus entropy.

### Format i język

- empty/multiple query rate;
- prefiks/metakomentarz;
- długość;
- language ID;
- znaki niedozwolone;
- JSON validity w trybie multi-query.

### Focus

- predicted sentence index;
- bucket distribution;
- accuracy względem kontrolki;
- first-sentence concentration;
- Gini/entropy.

## Slice’y

Wszystkie kluczowe metryki rozbij według:

- overlap kwantyla naturalnego query;
- długości passage;
- liczby zdań;
- target sentence position;
- domeny;
- query style;
- obecności encji/liczb;
- liczby pozytywów;
- trudności rerankera;
- doc near-duplicate cluster size.

## Probe embedder

Zaimplementuj `train_probe_embedder.py` z zamrożoną receptą. Celem nie jest stworzenie najlepszego embeddera, tylko porównanie generatorów.

Wymagania:

- ten sam model bazowy i tokenizer dla wszystkich wariantów;
- ten sam budżet kroków/tokenów;
- identyczny sampling pozytywów i hard negative’ów;
- te same seedy;
- zapis pełnego configu;
- trening na naturalnych query jako gold-data control;
- trening na prostych kopiach/heurystycznych query jako negative control;
- trening na syntetycznych query każdego generatora.

Loss może być MultipleNegativesRankingLoss, CachedMNRL, contrastive margin lub recepta zgodna z docelowym embedderem, ale raz wybrana musi być zamrożona na czas porównań.

## Retrieval evaluation embeddera

Raportuj:

- Recall@1/5/10/100;
- MRR@10;
- nDCG@10;
- MAP;
- hard-negative win rate;
- latency i rozmiar indeksu, jeśli istotne.

Dla każdego porównania wykonaj bootstrap po query:

- różnica metryki;
- 95% CI;
- odsetek bootstrapów, w których wariant wygrywa.

## Raport HTML/Markdown

`build_report.py` tworzy:

- executive summary;
- tabelę eksperymentów;
- wykresy rozkładów;
- Pareto grounding–copying–diversity–embedder score;
- slice’y;
- statystykę istotności;
- co najmniej 100 przykładów side-by-side;
- sekcję „reward hacking / failure modes”.

Raport ma wyraźnie oznaczać metryki niewykonane.

## Human evaluation export

Eksport CSV/JSONL bez nazwy modelu:

- passage;
- query A/B;
- kolejność losowa;
- pytania oceniające;
- hidden experiment IDs.

Importuj oceny i licz Cohen/Fleiss kappa albo Krippendorff alpha zależnie od liczby oceniających.

## Testy

- ręcznie znane rankingi dają poprawne MRR/nDCG;
- duplikaty dają oczekiwaną karę diversity;
- bootstrap jest deterministyczny dla seed;
- slice’y sumują się do całości;
- brak danych nie jest zamieniany na zero;
- porównanie runów sprawdza zgodność test fingerprintu.
- protokoły `pool_*` i `corpus_*` mają rozłączne nazwy;
- `recall@K` jest odrzucane dla puli mniejszej niż K;
- porównanie runów sprawdza wersję recepty probe i kontraktu budżetowego;
- manifest native/translated odrzuca zmianę źródła, rekordów lub profilowej
  listy ID;
- importer native przyjmuje wyłącznie test i nie wymaga sieci/modelu/GPU;
- raport embeddera bez zmierzonego `test_native_pl` ma status `incomplete`;
- sygnał translationese ujawnia składowe i nie deklaruje dowodu tłumaczenia;

## Kryteria akceptacji

- jeden command ocenia checkpoint i generuje komplet artefaktów;
- raport generatora i probe embeddera można odtworzyć z manifestu;
- pipeline odrzuca porównanie runów na różnych wersjach testu;
- raport bez wyniku `test_native_pl` jest oznaczany jako niekompletny;
- główny ranking wariantów może używać wyniku probe embeddera, nie tylko rewardu.
