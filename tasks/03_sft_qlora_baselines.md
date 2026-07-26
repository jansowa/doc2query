# Task 03 — Baseline’y SFT/QLoRA dla Bielika

> [Centralny rejestr zadań i statusów](README.md). Każda zmiana statusu lub zakresu tego zadania musi aktualizować rejestr w tym samym commicie.

## Status

`IMPLEMENTED`

Implementacja pipeline'u, konfiguracji S00–S05, skryptów, testów i wznowienia
jest gotowa. Na RTX 3060 Ti 8 GB wykonano memory probe 512, 20-krokowy smoke,
cztery runy Bielika 1.5B na 10 tys. par oraz jeden run na 50 tys. par. Wszystkie
zakończyły się bez OOM i zachowały końcowy adapter oraz dwa pełne checkpointy.
Wśród runów 10k najniższy eval loss uzyskał LR `2e-4` (1.2505); run 50k z LR
`1e-4` uzyskał 1.1457. Pełny raport i ograniczenia znajdują się w
[`docs/experiments/task03_8gb_weekend_2026-07-17.md`](../docs/experiments/task03_8gb_weekend_2026-07-17.md).

Zadanie pozostaje `IMPLEMENTED`, ponieważ loss i panel nie otwierają bramki
jakościowej. Pierwszy zredukowany probe P-05 oraz pełna bramka P-04
`dev_screen` są zmierzone, ale oba warianty mają tylko
`non_inferior_only`; `dev_confirm_authorized_arms=[]`. S07 jest kompletnym
wynikiem diagnostycznym, ale nieporównywalny budżet probe uniemożliwia wybór
architektury i nie będzie wykonywany matched-budget rerun. Pozostały również
P06-T, pełniejsze porównywalne probe, 4.5B base vs instruct oraz
ordinary/balanced/weighted.
Do czasu prospektywnej decyzji o dalszej ścieżce nie ma podstaw do przejścia
do DPO.

26 lipca kompletny artefakt S07 `dev_screen` osiągnął
`report_status=development_complete`: trening probe, 100 shardów embeddingów
2 404 263 dokumentów i exact retrieval 6598 query są zakończone. nDCG@10 to
`0,046142`, MRR@10 `0,043947`, MAP `0,040625`, a Recall@100 `0,159604`.
Probe S07 ma jednak budżet `2485 par / 864000 tokenów / batch 6`, podczas gdy
P-05 ma `2486 / 1152000 / batch 8`; `comparison_eligible=false`. S07 jest więc
zamkniętym wynikiem diagnostycznym, nie rozstrzyga plT5 kontra Bielik, nie
będzie powtarzany w matched budget i nie jest promowany do `dev_confirm` ani
testów finalnych. ADR:
[`task03_s07_diagnostic_closure_2026-07-26.md`](../reports/decisions/task03_s07_diagnostic_closure_2026-07-26.md).

Późniejszy audyt provenance unieważnił założenie P-06 o potrzebie masowego
rescoringu naturalnego train. Wszystkie 384 576 pozytywne pary mają
`source_en_score >= 23.50`, wszystkie dokumenty mają źródłowe score'y, a
źródłowy margin ma minimum `6.0`. Został on policzony podczas budowy danych
silniejszym rerankerem niż lokalnie dostępny. Nie kończyć pełnego scoringu
`artifacts/task03/p06/train_margins_v1` i nie używać jego niekompletnego
journalu do `drop` ani wag. P-06 mass rescoring jest `SUPERSEDED`; zastępuje go
mały, ślepy audyt integralności tłumaczeń P06-T, bez automatycznej zmiany
etykiet. ADR:
[`task03_p06_source_provenance_2026-07-26.md`](../docs/decisions/task03_p06_source_provenance_2026-07-26.md).

18 lipca uruchomiono nocną kolejkę W06 dla Bielika 4.5B Instruct na 8 GB.
Po wstępnym potwierdzeniu, że BS1/L512 wykonuje backward bez OOM, kolejkę
rozszerzono o wybór pierwszego udanego wariantu `BS2/L512 → BS2/L384 →
BS1/L512`, zawsze z efektywnym batchem 16. Krótki smoke używa 128/32 przykładów
train/eval. Wybrany wariant uruchamia ordinary QLoRA na 50 tys. przykładów
(3125 optimizer steps), z checkpointem co 50 kroków, ewaluacją co 100,
retencją 80 checkpointów i automatycznym wznowieniem pełnego stanu. Cache
Hugging Face i pliki tymczasowe są kierowane na partycję projektu. Samo
uruchomienie kolejki nie jest wynikiem eksperymentalnym; wynik, throughput i
peak VRAM należy dopisać dopiero po zakończeniu smoke/runu.

Systematyczny probe przy efektywnym batchu 16 porównał microbatch
`2/4/8/16`. Throughput wyniósł odpowiednio 1,177/1,493/1,546 przykładu/s,
a peak reserved 4,06/4,85/6,43 GB; BS16 zakończył się OOM. Wybrano
`W06-4.5B-INSTRUCT-50K-8GB-BS8-L512` z gradient accumulation 2. Szczegóły
zapisano w
[`docs/experiments/task03_4_5b_batch_probe_2026-07-18.md`](../docs/experiments/task03_4_5b_batch_probe_2026-07-18.md).
Pojedynczy pełny checkpoint zajmuje około 156,7 MB. Około 63 checkpointy
zajmą łącznie około 9,9 GB, dlatego przy ponad 260 GB wolnego miejsca
zachowywana jest cała krzywa, a nie tylko ostatnie punkty.

W06 zakończył się kodem 0 po 3125 krokach i 8 h 14 min. Finalny checkpoint
oceniono na tym samym zamrożonym panelu co W03/W05. Względem W05 greedy MRR
wzrósł z 0,9433 do 0,9850, a nDCG@10 z 0,9568 do 0,9889; sparowane 95% CI obu
różnic nie obejmują zera. Dla diverse nie ma potwierdzonej przewagi retrieval,
a duplicate rate, Self-BLEU i pairwise lemma Jaccard są gorsze. Pełne wyniki,
koszt i ograniczenia opisuje
[`docs/experiments/task03_w06_vs_1_5b_2026-07-19.md`](../docs/experiments/task03_w06_vs_1_5b_2026-07-19.md).
Po audycie wynik ten należy traktować wyłącznie jako diagnostykę i dowód
wykonalności QLoRA 4.5B na 8 GB. Kolejna kampania 4.5B czeka na Harness v1.1
oraz bramkę P-05/P-06 poniżej.

Przygotowano jedną wznawialną kolejkę techniczną dla bazowego Bielika 1.5B:
`scripts/run_base_1_5b_campaign.sh`. Najpierw domyka ona wyłącznie dev-only
P-03 W05, następnie mierzy brakujące memory probe 768/1024 i uruchamia siedem
jednoczynnikowych runów 10k: długość 768/1024, LoRA rank 16/32,
attention-only, efektywny batch 32 i dropout 0. Istniejące W01–W05 są
automatycznie wykorzystywane jako pokrycie LR/baseline i nie są powtarzane.
Kolejka, konfiguracje i testy wznowienia są gotowe, ale nowych runów nie
uruchomiono w tej sesji. Nie wybiera ona finalisty ani nie zastępuje P-04
i porównywalnego probe. Plan:
[`task03_base_1_5b_campaign_queue.md`](../docs/experiments/task03_base_1_5b_campaign_queue.md).
Runner nie używa CPU-only `.venv`: automatycznie tworzy lub naprawia
projektowe `.venv-gpu` z przypiętym, odtworzonym z udanych runów stosem
CUDA 12.4. Można też jawnie podać `DOC2QUERY_PYTHON`. Preflight raportuje
osobno CPU build Torch i brak CUDA w wybranym procesie.
Kolejkę rozszerzono o pięć przypiętych, porównywalnych ramion
`Bielik-1.5B-v3.0-Instruct`: trzy LR na 10k, replikację seed 43 i baseline
50k. Wszystkie zachowują B1, dane, QLoRA i budżety odpowiadających im ramion
base. To przygotowuje pomiary, ale nie deklaruje zwycięzcy bez P-04.
Wyjście każdego etapu jest jednocześnie widoczne w terminalu i zapisywane do
logu.
Loader SFT odrzuca i raportuje niepuste query zawierające `LF`/`CR`, zamiast
przerywać całą kolejkę. Audyt zamrożonego train v1 znalazł jeden taki rekord
(`65018::1129729`) i zero w dev; rekord nie jest normalizowany ani używany do
treningu, a licznik i identyfikator trafiają do `example_weights.json`.

21 lipca siedem technicznych ramion base oraz I01 Instruct/10k zostało
ukończonych. Ablacje base potwierdziły malejącą wartość dalszej pełnej
macierzy: LR `5e-5` dał eval loss 1,2914 wobec 1,2640 dla `1e-4` i 1,2505
dla `2e-4`; attention-only, efektywny batch 32 i dropout 0 również pogorszyły
loss, a długości 768/1024 nie poprawiły W03. I01 Instruct przy `1e-4`
uzyskał 1,2225 na tym samym zbiorze i kontrakcie completion-only, ale ten
sygnał nie zastępuje intrinsic retrieval ani probe.

ADR early-stop redukuje pozostałą kolejkę do I03 Instruct/10k/LR `2e-4`.
I02/LR `5e-5` ma zostać przerwany, I04/seed 43 odroczony do wyboru kandydata,
a I05/50k do wyboru LR i przejścia dev screen P-04. Nie jest to wybór modelu
ani zgoda na finalne testy. Uzasadnienie:
[`task03_instruct_campaign_early_stop_2026-07-21.md`](../docs/decisions/task03_instruct_campaign_early_stop_2026-07-21.md).

21 lipca przygotowano wyłącznie CPU/read-only tooling preflight po kampanii:
audytor append-only `status.tsv` poprawnie rozstrzyga retry, sprawdza komplet
B01–B07/I01–I05 oraz kontrakty configów i artefaktów, ale nie rankuje ramion
i nie używa eval loss. Planner pierwszej macierzy P-05 obejmuje gold natural,
W05 synthetic-only i mieszankę 50/50, wszystkie z HN0+filter i wspólnym
budżetem P-04. To nie jest wykonanie P-05/P-06: S00 i S07 są wymagane,
lecz niewykonane, a porównawcze probe i testy finalne pozostają nieotwarte.
Status zadania pozostaje `IMPLEMENTED`.

I03 zakończył się kodem 0 po 625 krokach. Z istniejących artefaktów
train/dev odczytano eval loss `1.2006736994`, czas `7541.0831 s`, throughput
`1.326 przykładu/s` oraz peak VRAM allocated/reserved `1.575/2.117 GiB`.
Panel greedy ma 100/100 poprawnych formatów, 100 unikalnych outputów i 8/100
normalized exact match. I03 ma niższy techniczny loss niż I01 i dopasowane
W01/W03, ale nie wykonano retrieval ani probe, więc nie wybrano generatora.
Pełne zestawienie:
[`task03_i03_result_2026-07-21.md`](../docs/experiments/task03_i03_result_2026-07-21.md).
I02/I04/I05 pozostają `DEFERRED`, S00/S07 `required_unexecuted`, a testy
finalne są nieotwarte.

23 lipca zakończyły się trzy zredukowane probe P-05 seed 42 na wspólnym
budżecie i zamrożonym `dev_intrinsic_rank10`. Mixed 50/50 uzyskał nDCG@10
`0.057568` wobec `0.052762` gold i `0.048618` synthetic-only. Paired bootstrap
mixed-minus-gold dał `+0.004806`, 95% CI `[+0.000876, +0.008692]`, więc nie
spełnia progu praktycznego `+0.01`. Brak trzech guardraili P-04 blokuje
promocję do `dev_confirm`; nie wybrano finalisty i nie otwarto testów.
Audyt: [`dev_screen_audit_2026-07-23.md`](../reports/measurements/task04_p05_dev_screen/dev_screen_audit_2026-07-23.md).

Follow-up zmierzył na tym samym frozen dev exact `corpus_round_trip_at_20`
gold/mixed/synthetic `0.116854/0.130797/0.112307` oraz wspólny
`format_valid_rate=1.0` oraz `sentence_level_source_hit=0.894665`. Pełny engine
z 10 000 paired-query bootstrapów zwraca `non_inferior_only` dla mixed i
synthetic-only: wszystkie guardraile przechodzą, lecz dolny CI primary effect
żadnego wariantu nie osiąga `+0.01`. `dev_confirm_authorized_arms=[]`; nie
wybrano finalisty i nie otwarto testów. Raport decyzji:
[`p04_gate_decision_2026-07-23.md`](../reports/measurements/task04_p05_dev_screen/p04_gate_decision_2026-07-23.md).

Do preflightu dodano deterministyczny materializator wspólnej kohorty P-05.
Weryfikuje jawne fingerprinty naturalnych par i generacji W05, odrzuca finalne
testy, buduje jeden seedowany porządek `doc_id/pair_id` i zapisuje osobne
artefakty gold, W05 oraz 50/50. Prefiks `dev_screen` i pełny `dev_confirm`
zachowują dokładne proporcje bez duplikowania rekordów. Planner ufa wyłącznie
manifestowi z pasującymi SHA-256. Jest to nadal wyłącznie tooling/preflight:
nie wykonano rzeczywistej materializacji, P-05/P-06, S00/S07 ani probe.

Po zamknięciu P-05 zapisano prospektywny ADR dalszej kolejności:
[`task03_s00_after_p05_2026-07-23.md`](../reports/decisions/task03_s00_after_p05_2026-07-23.md).
Najpierw należy domknąć S00, a dopiero potem podjąć decyzję o S07/P-06. Nie
autoryzuje to Task 05/06, DPO, nowej kampanii 4.5B, `dev_confirm` ani testów
finalnych.

Audyt wykazał, że dotychczasowy config i panel 100 rekordów nie realizowały
S00. Dodano kontrakt `task03-s00-prompting-v1` oraz runner
`scripts/run_s00_prompting.sh`: prospektywnie wybiera 5 000 rekordów z
zamrożonego `dev_intrinsic_rank10`, przygotowuje 6 rozłącznych demonstracji
per forma, uruchamia zero/few-shot w greedy i sampling, zapisuje postęp w
SQLite i ocenia oba ramiona Harnessem v1.1. CPU preflight potwierdził
fingerprint kohorty `93313d5a…db284`, brak przecięcia demonstracji i
`final_tests_used=[]`. Pełny indeks corpus BM25 nie był obecny i jest
pierwszym etapem właściwego runnera. Raport:
[`task03_s00_preflight_2026-07-23.md`](../reports/measurements/task03_s00_preflight_2026-07-23.md).

Właściwa generacja S00 zakończyła się kompletem 50 000/50 000 completionów:
po 25 000 zero-shot i few-shot. Pierwszy scoring zero-shot został przerwany po
około dziesięciu godzinach, zanim stary evaluator zapisał jakikolwiek trwały
wiersz. Nie ma kompletnych wyników zero-shot/few-shot ani decyzji jakościowej.
S00 i S07 pozostają
`required_unexecuted`, P-06 niewykonane, a status Task 03 pozostaje
`IMPLEMENTED`.

Pierwszy start S00 ujawnił ETA około 20 godzin przy batch 1 promptu. Runner
zoptymalizowano do batcha 32 promptów dla greedy i 8 dla sampling (efektywnie
32 sekwencje), z lewym paddingiem oraz niezależnym automatycznym fallbackiem
CUDA OOM przez dzielenie do 1. Skuteczny batch i sufit OOM są zapisywane per
strategia/tryb w SQLite; `S00_GREEDY_BATCH_SIZE` i
`S00_SAMPLING_BATCH_SIZE` mogą je ograniczyć. Legacy journal z commita
`e6ecfb3` jest zgodny, więc ukończone wyniki nie są kasowane.

Scoring S00 zoptymalizowano po przerwanym przebiegu. Primary, focus,
referencje i shadow są teraz batchowane, a ośmiu read-only workerów BM25
pracuje równolegle z GPU. Jedno zapytanie SQL materializuje score BM25 raz i
wyprowadza z niego top-100, licznik progu oraz score pozytywu. Po każdym batchu
64 rekordów evaluator zapisuje i `fsync`uje journal, pokazuje throughput/ETA,
odrzuca zmianę pełnej tożsamości wejścia/sędziów/indeksu i odzyskuje się po
uciętym ostatnim wierszu. Benchmark 64 prawdziwych outputów dev dał 25,57 s
(`2,50/s`), czyli liniową projekcję 2,77 h na ramię 25 tys.; nie jest to wynik
pełnego runu. `ruff`, `mypy` i 181 testów przechodzą. Raport:
[`task03_s00_scoring_optimization_2026-07-24.md`](../reports/measurements/task03_s00_scoring_optimization_2026-07-24.md).
Pełny scoring obu ramion nadal pozostaje niewykonany, podobnie decyzja
S07/P-06; finalne testy są nieotwarte.

25 lipca wznowiony runner domknął S00. Oba ramiona mają po 25 000 ocenionych
completionów, wspólny fingerprint frozen dev i pełny corpus retrieval.
Zero-shot uzyskał corpus round-trip@20 `0,6492` i format-valid `0,1507`, a
few-shot odpowiednio `0,3824` i `0,0376`; candidate-pool nDCG@10 wyniósł
`0,9131/0,9165`. Wynik jest intrinsic baseline'em i nie zawiera probe ani
oceny człowieka. Raport:
[`task03_s00_result_2026-07-25.md`](../reports/measurements/task03_s00_result_2026-07-25.md).

Osobny ADR daje `GO` dla przygotowania prospektywnego kontraktu S07 i `HOLD`
dla P-06 do zamrożenia planu S07 oraz potwierdzenia offline marginów train.
Nie autoryzuje `dev_confirm`, finalnych testów ani dalszej kampanii 4.5B:
[`task03_s07_p06_after_s00_2026-07-25.md`](../reports/decisions/task03_s07_p06_after_s00_2026-07-25.md).
S00 ma teraz stan `measured`; S07 i P-06 pozostają niewykonane, dlatego status
Task 03 pozostaje `IMPLEMENTED`.

25 lipca zamrożono kompletny prospektywny kontrakt S07 przed pobraniem wag:
`allegro/plt5-base` revision `56379680948ce8b42d3d48df86569cfc210d3060`,
pełny fine-tuning na dokładnie 50 tys. par W05, 1 tys. dev, seed 42, efektywny
batch 16, 3125 kroków, source/target 512/64. mT5-base został jawnie odrzucony,
ale jego alternatywny revision również przypięto. Dodano obsługę seq2seq,
atomowe pełne checkpointy, wznowienie, osobną kolację source/target, poprawne
odcinanie outputu encoder–decoder, memory probe, wznawialną generację kohorty
probe oraz jeden runner egzekwujący kolejność preflight → smoke → memory probe
→ 50k → Harness v1.1 dev → probe `dev_screen`. Kontrakt i wynik tanich bramek:
[`task03_s07_contract_2026-07-25.md`](../reports/decisions/task03_s07_contract_2026-07-25.md).

Preflight przeszedł, `ruff`, strict `mypy` dla 119 plików i 184 testy są
zielone. Lokalny mini-T5 ukończył 20 kroków na 128/32 rekordach, zapisał dwa
pełne checkpointy i model, wznowił się jako `already_complete`, a osobna
generacja smoke zapisała 100/100 rekordów. Następnie poza sandboxem wykonano
wyłącznie dwukrokowy memory probe rzeczywistego plT5 na RTX 3060 Ti. Probe
odtworzył fingerprint W05, zakończył się kodem 0 i zmierzył peak VRAM
allocated/reserved `2,683/2,717 GiB` oraz throughput `0,500 przykładu/s`.
Pełnego S07, Harness v1.1 ani downstream probe nie uruchomiono. S07 ma stan
`contract_frozen_memory_probe_passed_full_run_pending`, P-06 pozostaje `HOLD`,
a `dev_confirm` i finalne testy są nieotwarte. Status Task 03 pozostaje
`IMPLEMENTED`.

Po przerwaniu pierwszego startu pełnego treningu wykonano sweep microbatch
`1/2/4/8/16` przy stałym effective batch 16 i 6 optimizer steps, a BS8/BS16
potwierdzono na 30 krokach. BS16 osiągnął `28,684` examples/s wobec `16,468`
dla BS8 (+74,2%), peak reserved `3,256 GiB` i nie wykazał OOM ani
niefinitywnych wartości. Przypięto BS16/GA1 z pozostawionym gradient
checkpointingiem. Stary checkpoint-50 BS1 zachowano w katalogu interrupted i
nie będzie mieszany z nową trajectory identity. Raport:
[`s07_batch_probe_2026-07-25.md`](../reports/measurements/task03_s07/s07_batch_probe_2026-07-25.md).
Pełny run BS16 nadal jest niewykonany i musi zacząć się od kroku 0; Harness,
probe `dev_screen`, P-06, `dev_confirm` i testy finalne pozostają zamknięte.
Harness S07 ma konserwatywne ustawienia wykonawcze po wcześniejszych problemach
ze stabilnością: scoring/primary batch 16, shadow batch 8 i dwa workery BM25.

26 lipca pełny S07 BS16 zakończył 3125 kroków w `1761,997 s`, osiągając
`28,402` przykładu/s, last eval loss `2,35868` i peak allocated/reserved
`3,256/4,532 GB`. Wygenerowano komplet 25 tys. completionów frozen dev, lecz
pierwotna ścieżka plT5 działała omyłkowo na CPU i zajęła `9145,048 s`.
Scoring primary-GPU/shadow-CPU przerwano po 10 144/25 000 wierszach przy
projekcji około 15 godzin.

Wspólny Harness zoptymalizowano bez zmiany kohorty, modeli ani metryk.
Niekwantyzowana inferencja trafia teraz jawnie na CUDA, oba zamrożone sądy
mieszczą się na GPU, generacja jest batchowana, a BM25 używa trwałych pul i
połączeń. Frozen-dev benchmark 64 rekordów wybrał dwa workery i osiągnął
`3,417` rekordu/s przy peak reserved `4,152 GiB`, co daje wyłącznie liniową
projekcję `2,03 h` dla pełnego scoringu. Starego CPU shadow journalu nie wolno
mieszać z CUDA; runner archiwizuje go odzyskiwalnie i restartuje jednorodny
scoring. Pełny Harness zakończył później 5 tys. przykładów i 25 tys. generacji;
downstream `dev_screen` został następnie ukończony, lecz jego odmienny budżet
probe czyni wynik wyłącznie diagnostycznym.
Raport:
[`runtime_optimization_2026-07-26.md`](../reports/measurements/task03_s07/runtime_optimization_2026-07-26.md).

Pierwsze dwa uruchomienia probe przerwało wyłączenie hosta. Log wskazuje, że
GPU wykonywał filtrowanie lub trening probe, lecz wcześniejsza implementacja
zapisywała pierwszy trwały artefakt dopiero po wszystkich 250 krokach; katalog
wynikowy pozostał pusty. Probe ma teraz jawne logi podfaz, domyślny dla S07
batch 6 zamiast 8, atomowy kroczący checkpoint treningu co 25 kroków oraz
istniejące już wznawianie shardów kodowania korpusu i prefiksu zapytań.
Zmiana batcha jest częścią rozwiązanego recipe fingerprintu, dlatego wszystkie
późniejsze porównywane ramiona muszą użyć tego samego batcha 6.

## Cel

Zaimplementować stabilny trening passage→query i uruchomić serię tanich baseline’ów, zanim projekt przejdzie do DPO lub RL.

## Zależności

Taski 01–02.

## Modele

Przygotuj konfiguracje:

- `bielik_1_5b_base_or_instruct.yaml` — szybkie eksperymenty;
- `bielik_4_5b_base.yaml`;
- `bielik_4_5b_instruct.yaml`;
- `bielik_minitron_7b_instruct.yaml`;
- `bielik_pl_minitron_7b_instruct.yaml`.

Nie uruchamiaj 7B w pełnej skali w tym tasku. Zaimplementuj tylko smoke test i config.

## Ładowanie modelu

Wspieraj:

- 4-bit NF4;
- double quant;
- BF16, jeśli sprzęt wspiera, w przeciwnym razie FP16;
- `prepare_model_for_kbit_training`;
- gradient checkpointing;
- `use_cache=false`;
- automatyczne wykrywanie modułów linear do LoRA;
- jawne logowanie listy target modules i liczby parametrów trainable.

Nie zakładaj nazw modułów bez odczytania architektury. Test powinien przerwać run, jeśli LoRA nie objęła oczekiwanych warstw.

## Format danych

Użyj prompt-completion. Loss tylko na completion. Prompt i output muszą być oddzielne w dataset.

Baseline B0:

```text
Wygeneruj jedno polskie zapytanie wyszukiwawcze, na które można odpowiedzieć na podstawie pasażu.

Pasaż:
{passage}

Zapytanie:
```

Completion: dokładnie naturalne query.

Baseline B1 dodaje instrukcję o niekopiowaniu długich fragmentów, ale bez styl/focus controls.

## Weighted/balanced SFT

Zaimplementuj dwie opcje:

1. `BalancedBatchSampler` wyrównujący buckety:
   - style;
   - focus position;
   - overlap quantile;
   - długość passage;
2. `WeightedSFTTrainer`, który skaluje loss completion na poziomie przykładu.

Wagi muszą być znormalizowane, ograniczone `min/max` i logowane. Zwykły SFT pozostaje domyślnym kontrolnym baseline’em.

## Konfiguracje pamięci

Start dla 4.5B/16 GB:

```yaml
max_length: 768
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
gradient_checkpointing: true
packing: false
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
learning_rate: 1.0e-4
num_train_epochs: 1
```

Wykonaj memory probe dla 512, 768 i 1024. Raportuj realny peak VRAM, nie tylko estymację.

## Eksperymenty

### S00 — prompting bez treningu

- 5 tys. passage dev;
- zero-shot i few-shot z 3–8 naturalnymi przykładami dev dobieranymi per styl;
- greedy i sampling;
- stały, wersjonowany prompt;
- pełna ewaluacja Harness v1.1.

### S01 — tiny smoke

- tiny model lub 1.5B;
- 100–1000 rekordów;
- 20 kroków;
- sprawdzenie spadku loss i zapisu adaptera.

### S02 — 1.5B 10k

Cel: strojenie techniczne, nie wynik finalny.

### S03 — 1.5B 50k

Porównaj LR, rank i max length na małej macierzy.

### S04 — 4.5B 50k base vs instruct

Identyczne dane, seed i budżet tokenów.

### S05 — 4.5B 50k ordinary vs balanced vs weighted

Sprawdź, czy kontrola rozkładu danych poprawia overlap/style bez utraty grounding.

### S06/P-06 — source provenance i integralność tłumaczeń

Masowe porównanie `ordinary/drop/weighted` według lokalnego primary marginu
jest `SUPERSEDED`. Adapter już egzekwuje `source_en_score >= 23.50`, a
źródłowe etykiety i margin pochodzą z silniejszego rerankera użytego przed
kopaniem negatywów. Nie nadpisuj ich słabszym lokalnym sędzią.

Następnym krokiem jest P06-T z prospektywnego ADR: ślepa, deterministyczna
próbka 300 train obejmująca niski source score, niski source margin, flagi
jakości/translation-risk i losową kontrolę. Lokalne primary/shadow mogą służyć
wyłącznie do disagreement/triage. Przygotuj ręczny formularz answerability i
integralności tłumaczenia. Bez powtarzalnej ręcznie potwierdzonej klasy błędu
nie zmieniaj danych, nie trenuj wariantów drop/weighted i nie ustalaj progów.

### S07 — polski baseline seq2seq

Dostrój plT5-base/large albo mT5 na tym samym splicie, liczbie par i budżecie
co Bielik 1.5B. Porównaj koszt generacji, filtering i downstream probe.
Angielski docT5query pozostaje kontekstem historycznym, nie polskim baseline'em.

### Bramka P-05/P06-T przed kolejną kampanią 4.5B

Po ukończeniu Harness v1.1 porównaj S00 zero/few-shot, S03/W05, S07 oraz
gold-data control na candidate-pool, corpus, translated i native. Pierwsza
mała macierz probe obejmuje gold-data control, W05 synthetic-only i jedną
budżetowo dopasowaną mieszankę natural+synthetic. Następnie wykonaj mały audyt
P06-T; nie wykonuj masowego rescoringu ani SFT drop/weighted bez nowego ADR.
Decyzja o dalszej skali musi opierać się na retrieval i ADR, nie eval loss.

## Checkpointing i wznowienie

- atomowy zapis;
- możliwość resume;
- zapis adaptera i tokenizer config;
- zapis próbki generacji na stałym panelu 100 passage;
- walidacja po ustalonej liczbie kroków;
- early stopping wyłącznie na predefiniowanej metryce dev.

## Wymagane skrypty

- `scripts/train_sft.py`
- `scripts/run_memory_probe.py`
- `scripts/generate_panel.py`
- `scripts/compare_sft_runs.py`

## Testy

- completion-only masking;
- truncation nie usuwa completion;
- prompt nie jest liczony do loss;
- LoRA target modules nie są puste;
- liczba trainable params jest rozsądna;
- save/load adapter daje te same logits w tolerancji;
- resume nie resetuje schedulera;
- weighted loss odpowiada ręcznemu obliczeniu na toy batchu.

## Kryteria akceptacji

- smoke test przechodzi na małym modelu;
- 1.5B generuje poprawny format i reaguje na trening;
- 4.5B QLoRA mieści się w 16 GB dla co najmniej jednego użytecznego configu;
- base vs instruct są porównane na identycznym pipeline;
- powstaje tabela z loss, intrinsic metrics, peak VRAM, throughput i probe-embedder score, jeżeli Task 04 jest już gotowy;
- nie ma podstaw do przejścia do DPO, dopóki SFT baseline nie jest stabilny.
