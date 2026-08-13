# Task 07 — DPO i kontrola continued SFT

> [Centralny rejestr zadań i statusów](README.md). Każda zmiana statusu lub zakresu tego zadania musi aktualizować rejestr w tym samym commicie.

## Status

`IN PROGRESS`

Aktualizacja 2026-08-13 (rozszerzenie specyfikacji, decyzja właściciela):
Task 07 jest jawnie także destylacją procedury dwumodelowej z selektorem do
pojedynczego generatora; dla finalistów obowiązuje ewaluacja trybu
produkcyjnego bez selektora (M-05, sekcja poniżej; definicje M-01–M-05 w
AGENTS.md §9.2). Zakres modelowy: 4.5B jako główny finalista i opcjonalnie
1.5B na sprzęcie lokalnym 8/16 GB; pipeline musi pozostać przenośny dla
większych modeli na sprzęcie zewnętrznym.

Aktualizacja 2026-08-12 (Task 06 execution design): legalna pula Task 06,
macierz kandydatów i evidence zostały zaprojektowane model-free; wybory są
rozstrzygnięte, ale preflight nadal czeka na jawną komendę operatorską. Nie
istnieją jeszcze rzeczywiste pary `chosen/rejected`, zamrożona polityka ich
budowy, kalibracja ani dual-LLM evidence. Właściciel zastąpił ręczny panel 500
par audytem `gpt-oss-120b` + `qwen3.6-27b`; nie wolno nazywać go human evidence.
D01 controlled 4.5B pozostaje jedynym przyszłym startem, W06 wyłącznie anchor
i source; `task07_training_authorized=false`, `final_tests_used=[]`.

Aktualizacja 2026-08-12: decyzja właściciela po pozytywnym Task 05 confirmie
przypina adapter `D01-4.5B-STYLE-50K-S42` jako przyszły pojedynczy start DPO.
W06 pozostaje safety anchor i źródłem kandydatów procedury Task 06; jego wag
nie łączy się z D01. Handoff preflight zweryfikował tożsamości i nadal zapisuje
`task07_training_authorized=false`. Continued-SFT oraz score-weighted
continued-SFT pozostają obowiązkowymi kontrolami. Rzeczywisty Task 07 czeka na
dane i bramki Task 06 oraz osobną autoryzację wykonania; `final_tests_used=[]`.

Gotowy jest wyłącznie model-free, przedeksperymentalny fundament. Obejmuje
ścisłe kontrakty preference/continued-SFT/weighted-SFT, provenance Task 06,
tożsamości modelu, adaptera i tokenizera, wcześniej policzone długości oraz
reference logprobs, fail-closed walidatory, deterministyczny matched-budget
planner trzech obowiązkowych ramion i czystą funkcję sigmoid DPO loss.
Manifest planu ma status `planned_not_trained`, hashe wszystkich wejść oraz
jawne flagi potwierdzające brak ładowania modeli, liczenia logprobów i treningu.

Dodano deterministyczny, fail-closed handoff danych z Task 06. Cienki skrypt
`scripts/package_task07_inputs.py` przyjmuje wyłącznie zamrożone preference i
continued-SFT train/dev oraz osobny artefakt gotowych przypisań wag. Packager
nie przyjmuje ścieżki testowej, nie wylicza wag, nie filtruje i nie relabeluje.
Wymaga dokładnego pokrycia 1:1, zgodności completion z `chosen`, zachowuje
prompt i candidate IDs, wykrywa passage/cluster leakage, waliduje fingerprinty
datasetu/selekcji/polityki wag i przed atomową publikacją sprawdza wynik przez
istniejący `validate_dpo_dataset`. Istniejący output nie jest nadpisywany, a
przerwany staging jest usuwany.

Dodano również model-free launch preflight `scripts/prepare_task07_launch.py`.
Konsumuje wyłącznie manifest i sześć artefaktów handoffu, zamrożony plan oraz
wcześniej policzone manifesty i rekordy token lengths i reference logprobs.
Ponownie używa walidatorów datasetu, token-length evidence, planu i logprobów;
nie tokenizuje, nie liczy logprobów, nie wylicza wag i nie ładuje stosu
modelowego. Fail-closed sprawdza SHA-256, liczniki, fingerprinty, dokładne
pokrycie i kolejność `preference_id`, dataset/selection provenance, wspólną
politykę wag, identyczny start/reference model stack z tokenizatorem,
fingerprint planu i kohorty, seedy, matched budget oraz train/dev leakage.
Interfejs nie przyjmuje artefaktu testowego.

Po poprawnej walidacji preflight atomowo publikuje osobny, wersjonowany
kontrakt `task07-model-free-launch-bundle-v1` dla DPO, continued SFT i
score-weighted continued SFT. Status to wyłącznie
`ready_for_model_smoke_not_trained`, a manifest zapisuje hashe wszystkich 12
wejść, liczniki rekordów i jawne flagi
`model_loading_performed=false`, `tokenizer_loading_performed=false`,
`reference_logprobs_computed=false`, `training_started=false`,
`evaluation_started=false` i `final_tests_used=[]`. Output nie jest
nadpisywany, katalog jest publikowany atomowo, a staging usuwany po błędzie.
Bundle nie zawiera komend treningowych i nie stanowi zgody na eksperyment.

Dodano także wyłącznie model-free, post-run evidence/comparison preflight.
Wersjonowany `Task07ComparisonProtocolManifest` ma status
`protocol_frozen_not_applied`, dokładnie trzy obowiązkowe ramiona i co
najmniej dwa jawnie przypięte seedy. Wiąże SHA-256 i fingerprinty handoffu
Task 06, selection preflightu Task 06 i launch bundle Task 07 oraz przypina
dataset, selection/weight policy, kohortę, plan, model stack, tokenizer,
matched-budget definitions, wymagane metryki z kierunkami i definicjami,
definicje CI, minimalne liczebności, role artefaktów i jawnie dostarczone
guardraile. Preflight nie wylicza progów ani ich nie stosuje.

Wersjonowany `Task07ArmOutcomeEvidenceManifest` opisuje jedno ramię i seed,
status runu i fingerprint configu, pełne tożsamości porównania oraz osobne
sekcje intrinsic primary, niezależnego shadow, probe/extrinsic, human i cost.
Każdy artefakt ma SHA-256, record count i provenance; każda metryka ma jawnie
dostarczone CI i sample size. Manifest nie dopuszcza final-test paths i wymaga
`final_tests_used=[]`.

`Task07ComparisonPreflight` ponownie używa walidatora handoffu Task 06,
kontraktów launch Task 07, deskryptorów evidence Task 09 oraz wspólnych hash
helpers. Przed odczytem odrzuca final-test paths, sprawdza integralność
wszystkich wejść, dokładne pokrycie arm × seed i drift datasetu, kohorty,
planu, polityk, stosu modelowego, tokenizera, budżetu, configu, provenance i
definicji metryk. Brak CI, liczebności albo wymaganych human/shadow/probe
evidence zamyka preflight. Kod nie agreguje ani nie porównuje wartości
eksperymentalnych i nie tworzy rankingu, winnera, decyzji continue/stop ani
promocji.

Cienki skrypt `scripts/prepare_task07_comparison_preflight.py` publikuje
deterministyczny bundle przez staging i `os.replace`, odmawia nadpisania i
sprząta staging po błędzie. Maksymalny status to
`ready_for_future_task07_comparison_not_compared`; flagi
`comparison_started`, `selection_performed`, `promotion_performed`,
`model_loading_performed`, `training_started` i `evaluation_started` są
zawsze `false`, a `final_tests_used=[]`.

Nowy comparison preflight ma 30 przechodzących syntetycznych testów CPU.
Po zakończeniu runu Task 05 ukierunkowany mypy dla nowych plików i pełny CPU
pytest (`444 passed`) również przeszły. Nie uruchomiono żadnego procesu
modelowego.

Nie dostarczono ani nie policzono rzeczywistych token lengths i reference
logprobs. Nie wykonano treningów, modelowych smoke testów, tokenizacji modelem,
właściwego precompute reference logprobs, QLoRA/PEFT save-load, kalibracji ani
wyliczenia polityki wag, wyboru beta/LR, ewaluacji, wielu seedów ani żadnej
bramki promocji. Nie wykonano również generacji/scoringu i audytu człowieka
wymaganych do wytworzenia rzeczywistych wejść Task 06.
Nie zmaterializowano rzeczywistych manifestów outcome, nie zebrano evidence
intrinsic primary, shadow, probe/extrinsic, human ani cost, nie uruchomiono
comparison preflightu na rzeczywistych runach i nie wykonano porównania
ramion. Nie wybrano zwycięzcy i nie wykonano decyzji continue/stop ani
promocji.
Wszystkie testy finalne pozostają zamknięte (`final_tests_used=[]`).

## Cel

Sprawdzić, czy preference optimization poprawia konkretne wady generatora ponad to, co daje zwykłe dalsze SFT na wybranych dobrych przykładach.

## Zależności

Task 06.

## Implementacja

Użyj TRL `DPOTrainer` lub równoważnej, dobrze przetestowanej implementacji. Obsłuż QLoRA i start z adaptera SFT.

Wymagania:

- prompt/chosen/rejected;
- adapter SFT jako punkt startowy;
- completion truncation bez utraty ważnych tokenów;
- `precompute_ref_log_probs` jako opcja oszczędzająca pamięć;
- walidacja, że model referencyjny odpowiada dokładnie punktowi startowemu;
- logowanie chosen/rejected rewards, margins, accuracies i length statistics;
- zapis adaptera i manifestu.

## Eksperyment kontrolny

Dla każdej konfiguracji DPO uruchom `continued SFT`:

- te same prompty;
- tylko `chosen` jako completion;
- ten sam przybliżony budżet tokenów i kroków;
- analogiczny LR search.

Bez tej kontroli nie wolno przypisać poprawy samemu DPO.
Dodatkową obowiązkową kontrolą jest `score-weighted continued SFT` przy tym
samym przybliżonym budżecie. Co najwyżej jedną metodę listwise (LiPO albo PRO)
wolno dopuścić dopiero po stabilnym DPO i tylko przy potwierdzonej jakości
rankingu kandydatów.

## Konfiguracja startowa

```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
gradient_checkpointing: true
max_length: 768
max_prompt_length: derived_from_data
beta: 0.1
loss_type: sigmoid
learning_rate: 1.0e-5
precompute_ref_log_probs: true
```

Parametry zależne od wersji TRL mają być sprawdzone z aktualną dokumentacją i smoke testem.

## Ablacje

- beta `0.05, 0.1, 0.2`;
- LR `5e-6, 1e-5, 2e-5`;
- top-vs-bottom vs top-vs-near-miss preferences;
- bez overlap component w scorerze;
- bez focus component;
- bez diversity component;
- DPO na 20k vs 50k/100k par;
- classical sigmoid vs co najwyżej jeden alternatywny stabilny loss po baseline.

Nie wykonuj pełnej siatki. Użyj sekwencyjnego wyboru i przerwij słabe runy.

## Ewaluacja

Po każdym runie:

1. intrinsic generator test;
2. preference test accuracy;
3. stały panel przykładów;
4. probe embedder;
5. bootstrap względem SFT i continued SFT;
6. failure slices.

Monitoruj szczególnie:

- czy DPO nie skraca nadmiernie query;
- czy poprawa overlapu nie pogarsza grounding;
- czy model nie uczy się artefaktów scorerów;
- czy diversity wynika z jakości, a nie losowości;
- czy preferencje na naturalnych style’ach się utrzymują.

## Tryb produkcyjny i destylacja selektora (M-05)

Zaakceptowana procedura danych (dwa generatory + pełny scoring + safe-anchor
selector) jest zbyt kosztowna jako docelowy sposób generowania korpusu przy
500 tys. pasaży na lokalnym sprzęcie. Celem DPO jest między innymi
internalizacja preferencji selektora przez jeden model. Dlatego każdy
finalista Task 07 jest oceniany dodatkowo w trybie produkcyjnym: pojedynczy
generator, bez scoringu i selekcji, z zaraportowanym kosztem (passage/s,
VRAM) na 1000 pasaży. Porównanie obejmuje co najmniej: (a) pełną procedurę z
selektorem, (b) finalistę DPO w trybie produkcyjnym, (c) continued SFT w
trybie produkcyjnym. Jeżeli offline selekcja zachowuje zysk probe, którego
DPO nie internalizuje, jest to jawna przesłanka bramki awansu wyników GRPO
do selekcji finalistów (Task 08, AGENTS.md Faza E); sam run GRPO jest
planowany po Task 07 niezależnie od tego wyniku (cel edukacyjny,
AGENTS.md §2).

## Memory strategy 8–16 GB

Kolejność oszczędzania pamięci:

1. precompute ref logprobs;
2. batch 1;
3. krótszy max length;
4. gradient checkpointing;
5. QLoRA;
6. niższy rank LoRA;
7. activation offloading tylko po benchmarku kosztu.

Nie duplikuj niepotrzebnie pełnych wag base modelu.

## Testy

- toy DPO loss: chosen wyżej niż rejected zmniejsza loss;
- ref logprobs mają właściwy shape i dataset order;
- restart nie miesza precomputed logprobs;
- PEFT save/load;
- DPO dataset nie zawiera par bez marginesu;
- długości prompt/chosen/rejected po tokenizacji są raportowane.

## Kryteria akceptacji

DPO przechodzi do finalnej macierzy tylko, jeżeli:

- wygrywa z bazowym SFT i continued SFT na predefiniowanej metryce;
- poprawa nie jest ograniczona do score modelu użytego do tworzenia preferencji;
- probe embedder potwierdza brak szkody lub poprawę;
- efekt utrzymuje się w co najmniej dwóch seedach redukowanego runu;
- owner-approved dual-LLM audit potwierdza kierunek zmiany; nie jest to human evidence.
