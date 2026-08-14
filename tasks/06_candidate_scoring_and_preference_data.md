# Task 06 — Generacja kandydatów, scoring i dane preferencyjne

> [Centralny rejestr zadań i statusów](README.md). Każda zmiana statusu lub zakresu tego zadania musi aktualizować rejestr w tym samym commicie.

## Status

`IN PROGRESS`

Aktualizacja 2026-08-14 (przerwany run v2 i amendment): pierwszy run generacji
v2 przerwał się po 3559/4000 kandydatach na `ValueError: query completion must
be a single line` — w slocie 7 (temperatura 1.2) model zwrócił completion z
znakiem nowej linii, a runner nie obsługiwał tego wyjątku. Bezobsługowe
uruchomienie z `; systemctl poweroff` wyłączyło komputer zgodnie z projektem
także po błędzie; journal zachował 3559 wierszy. Amendment
[`task06_same_prompt_v2_invalid_completion_amendment_2026-08-14.md`](../reports/decisions/task06_same_prompt_v2_invalid_completion_amendment_2026-08-14.md)
przenosi na ten etap politykę zamrożonego pipeline'u D01: niepoprawny completion
jest resamplowany na nowym, deterministycznym seedzie (do 4 prób), a dopiero po
ich wyczerpaniu zachowywana jest pierwsza niepusta linia z jawną flagą
`format_repair`. Pierwsza próba zachowuje dotychczasowy seed, więc
`identity_sha256` się nie zmienia i run wznawia się bez utraty ~23 min GPU.
Kohorta, prompty, decoding i progi bramki pozostają nietknięte. Walidacja: Ruff,
`mypy src`, pełny pytest `516 passed`. `final_tests_used=[]`.

Aktualizacja 2026-08-13 (decyzja o kohorcie v2, delegowana właścicielem):
właściciel delegował wybór ścieżki naprawy deficytu par. ADR
[`task06_same_prompt_expansion_v2_2026-08-13.md`](../reports/decisions/task06_same_prompt_expansion_v2_2026-08-13.md)
zamraża nową kohortę 500 pasaży rozłączną klastrowo z całą dotychczasową pracą
Task 06, z tym samym kontraktem „jeden prompt, osiem odpowiedzi”, ale szerszym
rozkładem decodingu (temperatury 0.6–1.2, top_p 0.92/0.97, osiem nowych
seedów). Etap 1 (CPU) jest wykonany: quality-blind ID freeze i materializacja
500/500 unikalnych klastrów z legalnej puli 291463 par, zero nakładania z 544
klastrami smoke/pilot i 49352 klastrami selekcji 50k SFT
(`src/doc2query/preferences/same_prompt_cohort.py`,
`scripts/freeze_task06_same_prompt_expansion_v2.py`, artefakt
`artifacts/task06/same_prompt_expansion_v2`). Ścieżkę generacji v2 zweryfikowano
na prawdziwych artefaktach do momentu ładowania modelu. Etap 2 czeka wyłącznie
na wolne GPU: `bash scripts/run_task06_same_prompt_expansion_v2.sh` (~50 min:
generacja, scoring, bramka o niezmienionych progach). Runner jest w pełni
wznawialny tą samą komendą — oba kosztowne etapy mają fsyncowane journale z
granulacją jednego batcha, a gotowa bramka nie jest nadpisywana; wznawianie po
przerwaniu jest sprawdzone testem, nie tylko lekturą kodu. `generation_batch_size`
i `scoring.max_batch_size` są teraz faktycznie respektowane (wcześniej kod
używał literału 8) z walidacją 1–8; efektywny batch pozostaje 8, więc identity
zakończonego runu v1 się nie zmienia.
`tentative_pair_build_authorized=false` — budowa par wymaga osobnego ADR
zamrażającego politykę `chosen/rejected` i kalibrację komponentów. Raport:
[`task06_same_prompt_expansion_v2_cohort_2026-08-13.md`](../reports/measurements/task06_same_prompt_expansion_v2_cohort_2026-08-13.md).
Walidacja: Ruff, `mypy src`, pełny pytest `512 passed`. `final_tests_used=[]`.

Aktualizacja 2026-08-13 (wynik expansion + bramka różnorodności): run
`same_prompt_expansion_v1` jest zakończony — 4000/4000 wygenerowanych i
4000/4000 ocenionych kandydatów dla 500 promptów × 8 odpowiedzi, bez resume,
peak VRAM 3.43 GB. Prospektywny ADR
[`task06_same_prompt_diversity_gate_v1.md`](../reports/decisions/task06_same_prompt_diversity_gate_v1.md)
zamroził progi bramki przed odczytem jakichkolwiek par, a zaimplementowana,
quality-blind bramka (`src/doc2query/preferences/diversity_gate.py`,
`scripts/apply_task06_same_prompt_diversity_gate.py`, polityka w
`configs/preferences/task06_same_prompt_diversity_gate_v1.yaml`) została
zastosowana na CPU: **362/500 grup `eligible` (72.4%), 138 odrzuconych
(27.6%)**; przyczyny: 133 `duplicate_rate`, 72 `insufficient_effective_candidates`,
38 `self_bleu`, 24 `no_pairable_candidate_pair`. Bramka czyta wyłącznie
`generations.jsonl`; manifest zapisuje `judge_scores_read=false`,
`candidates_ranked=false`, `pairs_built=false`,
`model_loading_performed=false`. Ta kohorta daje najwyżej 362 pary, czyli
mniej niż wymagane 500 par rozwojowej bramki dual-LLM — uzupełnienie wymaga
nowej generacji na GPU i osobnej decyzji właściciela (większe K z deduplikacją
albo nowe grupy same-prompt). Par nie zbudowano, Groq nie uruchomiono, Task 07
zamknięty. Wynik:
[`task06_same_prompt_expansion_result_2026-08-13.md`](../reports/measurements/task06_same_prompt_expansion_result_2026-08-13.md).
Walidacja: Ruff, `mypy src`, pełny pytest `503 passed`. `final_tests_used=[]`.

Aktualizacja 2026-08-13 (rozszerzenie specyfikacji, decyzja właściciela):
dodano obowiązkową bramkę różnorodności same-prompt przed budową par (sekcja
poniżej), dopuszczono prospektywną ablację teachera na lokalnym
`Qwen3.6-27B` Q4 oraz preferencję lokalnego, przypiętego sędziego
`qwen3.6-27b` w przyszłych audytach dual-LLM. Zamrożony kontrakt trwającego
expansion runu 500×8 pozostaje bez zmian; bramka dotyczy dopiero budowy par.

Aktualizacja 2026-08-13: pilot 512 zakończył wszystkie fazy: 4096 kandydatów,
4096 scoringów, 512 natural diagnostics i 2048 safe-selected. Selector zmienił
anchor w 482/512 grup i wybrał 1164 W06 + 884 D01. Audyt wykrył błędny prefiks
`task06-smoke` w provenance pilota. Wykonano udokumentowaną, mechaniczną
migrację pełnego łańcucha fingerprintów i odbudowano selekcję; teksty, score'y
i kolejność nie zmieniły się. Aktywne artefakty nie zawierają starej etykiety.

Nie zbudowano par z istniejącej macierzy: W06 i cztery kontrolowane sloty D01
nie mają tego samego promptu, więc takie pary naruszałyby kontrakt DPO. Nowy ADR
zamraża poprawny etap: quality-blind 500 passage'y, jedna zbalansowana kontrolka
D01 na passage i osiem odpowiedzi na dokładnie ten sam prompt. Przygotowano
resumowalny runner generacji/scoringu i uruchomiono go w odłączonej sesji.
Pary, Groq i Task 07 pozostają niewykonane do zakończenia tego runu.
Walidacja: Ruff, `mypy src`, `git diff --check` i pełny pytest `483 passed`.
`final_tests_used=[]`.

Aktualizacja 2026-08-12 (autoryzowany smoke): właściciel polecił przygotować i
uruchomić wyłącznie smoke Task 06. Dodano wykonawczy, fail-closed i resumowalny
runner dla 32 pasaży × 8 kandydatów (4 W06 + 4 D01), scoringu
primary/shadow/corpus, małej diagnostyki naturalnego marginu oraz zastosowania
niezmienionego safe-anchor selectora. Wszystkie GPU batch oraz semantic encode
mają cap 8; shadow nie jest sygnałem selekcji, a pilot 512 nadal jest jawnie
zablokowany. Prospektywnie zamrożono ID i dopiero potem zmaterializowano 32
rekordy train z 32 różnych legalnych klastrów; wybór nie użył pól jakości.

Pierwsza próba w sandboxie zatrzymała się bezpiecznie przed modelami z powodu
braku urządzeń GPU. Po informacji właściciela runner uruchomiono poza
sandboxem, wskazując istniejący projektowy cache Hugging Face. Smoke zakończył
się pełnym sukcesem technicznym: wygenerowano i oceniono 256/256 kandydatów,
primary/shadow/corpus scoring ukończył oba ramiona, diagnostyka naturalnych
marginów objęła 32/32 rekordy, a safe-anchor selector wybrał 128/128 zapytań.
W 29/32 grup selekcja różniła się od czystego W06; wybrano 73 W06 i 55 D01.
Nie zbudowano jeszcze par `chosen/rejected` i nie wykonano Groq. Wynik oraz
artefakty opisuje
[`task06_candidate_smoke_preparation_2026-08-12.md`](../reports/measurements/task06_candidate_smoke_preparation_2026-08-12.md).
`final_tests_used=[]`.
Końcowa walidacja po runie: Ruff, `mypy src`, `git diff --check` oraz pełny
pytest (`482 passed`, 16 ostrzeżeń zależności) przeszły.

Aktualizacja 2026-08-12 (prospektywny execution design, fail-closed): wykonano
ID-only audyt wyłącznie frozen train, bez odczytu pól jakości, emisji surowych
ID, otwierania testów lub TriviaQA. Z 356856 unikalnych pasaży train
wykluczono klastrowo 49367 pasaży reprezentowanych we wspólnej 50k selekcji
SFT W06/D01; legalna pula ma 307309 pasaży w 306903 klastrach. Zamrożony
config projektuje K=8 (4×W06 + 4×D01), dwa seedy, jawne kontrolki D01,
decoding, primary/shadow/corpus evidence, batch cap 8 i rozdział safe-anchor
od przyszłego selektora chosen/rejected. Preflight ma status
`verified_design_pending_explicit_operator_command`: właściciel wybrał 512
pasaży i kalibrację na prospektywnie zamrożonym natural dev. Ręczne 500 par
zostało świadomie zastąpione ślepym audytem dwóch modeli Groq:
`openai/gpt-oss-120b` i `qwen/qwen3.6-27b`, po 500 ocen każdego. To nie jest
human evidence. Kontrakt wymaga globalnej przerwy ≥4 s między requestami bez
równoległych wywołań, limitów minutowych i dziennych, retry, przełączenia na
drugi model oraz czystego resumable stopu po
wyczerpaniu obu. Runner nadal czeka na osobną komendę operatorską; modeli,
generacji, scoringu, Groq ani selekcji nie uruchomiono, `final_tests_used=[]`.
ADR:
[`task06_candidate_generation_and_scoring_design_v1.md`](../reports/decisions/task06_candidate_generation_and_scoring_design_v1.md).
Pełny CPU pytest zakończył się `480 passed`; Ruff, `mypy src`, ukierunkowany
mypy nowych plików i `git diff --check` przeszły. Rozszerzony
`mypy src tests scripts` zachowuje 19 wcześniejszych błędów w sześciu
niezmienianych plikach testowych.

Aktualizacja 2026-08-12: właściciel zatwierdził handoff potwierdzonego D01b
Hybrid. Zamrożono dwumodelową procedurę danych W06+D01+safe-anchor selector i
osobno D01 controlled 4.5B jako pojedynczy start przyszłego Task 07. Config
przypina base revision, oba adaptery i manifesty, selektor oraz pozytywny
confirm. Rzeczywisty model-free preflight zwrócił
`verified_ready_for_task06_execution_design_not_generation`, bez ładowania
modelu. Nie wybrano jeszcze kohorty Task 06, K/request matrix, seedów, budżetu,
kalibracji ani human panelu; dlatego generacja, scoring i selekcja nadal są
`false`. Raport:
[`task06_d01b_hybrid_handoff_2026-08-12.md`](../reports/measurements/task06_d01b_hybrid_handoff_2026-08-12.md).
Po zmianie pełny pytest ma wynik `475 passed`; Ruff, mypy i
`git diff --check` przeszły.

Zaimplementowano niezależny od wyniku Task 05 fundament: ścisłe kontrakty
scored-candidate/preference z pełnymi składowymi i provenance,
deterministyczną selekcję `top-vs-near-miss`/`top-vs-bottom`, kontrolę leakage
passage i near-duplicate cluster, eksport TRL wraz z obowiązkowym zbiorem
continued-SFT oraz eksport/import ślepego audytu A/B.

Gotowy jest również quality-blind planner przyszłej generacji. Konsoliduje
wiele naturalnych par jednego dokumentu, dziedziczy split i cluster z dedup
mapy, konstrukcyjnie odrzuca test oraz leakage klastra, a następnie wybiera
K=4–8 requestów metodą coverage-first po osiach form/intent/focus,
temperature i seed. Każdy request ma stabilne ID, pełny prompt i fingerprint
planu; atomowy manifest jawnie zapisuje `planned_not_generated`,
`generation_started=false`, `scoring_started=false` i `final_tests_used=[]`.
Szablon pozostaje planning-only i nie autoryzuje modelu ani runu.

Zaimplementowano kolejny model-free etap: ścisły `GeneratedCandidate` związany
z `CandidateGenerationRequest`, pełne provenance generatora i decoding oraz
oddzielne kontrakty primary, shadow, corpus retrieval, lexical/copy, focus,
style i format evidence. Fail-closed assembler wymaga dokładnego pokrycia 1:1,
sprawdza ID, plan, checkpoint/adapter, passage/split/cluster, przypiętych
sędziów i ich revisions, ponownie liczy oba marginy, rozdziela surowe skale
primary/shadow, odrzuca test i duplikaty po normalizacji. Zapisuje kanoniczny
`CandidateEvidenceBundle` oraz manifest z hashami wejść, licznikami i statusem
`evidence_assembled_not_ranked`. Assembler nie ma pola ani logiki
`total_score`, nie kalibruje, nie ustala wag/progów i nie wybiera par.

Gotowy jest także wyłącznie przedeksperymentalny handoff Task 06 → Task 07.
Deterministyczny packager konsumuje wcześniej zmaterializowane preference oraz
continued-SFT train/dev i osobny, wcześniej policzony artefakt przypisań wag.
Wymaga dokładnego pokrycia i kolejności `preference_id`, dodatnich skończonych
wag, przypiętych fingerprintów datasetu, selekcji i polityki wag oraz własnych
hashy artefaktu wag. Odrzuca duplikaty, orphan/missing ID, drift provenance,
test oraz leakage passage/near-duplicate cluster. Zachowuje prompt,
`chosen/rejected` i candidate IDs znak w znak, tworzy po jednym continued-SFT
i weighted-SFT na parę, po czym atomowo publikuje manifest
`task06-preference-data-for-task07-v1` z SHA-256, licznikami i jawnymi polami
`automatic_thresholds_created=false`, `relabeling_performed=false` oraz
`final_tests_used=[]`. Nie wylicza wag i nie czyta osobnego artefaktu testowego.

Dodano model-free, przedselekcyjny preflight przyszłej selekcji. Wersjonowany
`CandidateSelectionPolicyManifest` wiąże hash i fingerprint istniejącego
`CandidateEvidenceBundle` oraz jego manifestu z dataset/split/cohort i dokładną
listą candidate IDs. Wymaga pełnego zestawu `primary`, `shadow`,
`corpus_retrieval`, `lexical_copy`, `focus`, `style` i `format`; dla każdej
metryki zapisuje jawnie dostarczony kierunek, definicję normalizacji z
parametrami, wagę, nazwane progi i fingerprint kalibracji. Osobno przypina
minimalny margin, definicje near-miss/bottom i limity per passage. Status
polityki to wyłącznie `policy_frozen_not_applied`.

Osobne wersjonowane manifesty kalibracji komponentów zapisują artefakt z
SHA-256, licznikiem i provenance oraz porównywalne definicje metryk.
`HumanPreferenceCalibrationEvidenceManifest` wymaga zamrożonego ślepego panelu,
SHA-256 i liczby rekordów, fingerprintów kohorty, protokołu anotatorów i
kryteriów, jawnie dostarczonej liczebności, agreement oraz CI. Wszystkie
manifesty wymagają `final_tests_used=[]`; żaden nie wylicza wartości
eksperymentalnych.

`PreferenceSelectionPreflight` konsumuje wyłącznie jawnie wskazane pliki,
odrzuca final-test paths przed odczytem i ponownie używa kontraktów
`CandidateEvidenceBundle`, `EvidenceArtifact` z Task 09 oraz hash helpers Task
07. Sprawdza integralność, record counts, provenance, dokładne pokrycie
komponentów/candidate IDs, dataset/split/cohort drift i porównywalność definicji
metryk. Wagi i progi wyłącznie waliduje jako przypięte i skończone. Atomowy
bundle używa stagingu, `os.replace`, cleanupu po błędzie i odmowy nadpisania;
ma najwyżej status `ready_for_future_preference_selection_not_selected` oraz
flagi `generation_started=false`, `scoring_started=false`,
`calibration_computed=false`, `selection_started=false`,
`preferences_built=false`, `model_loading_performed=false`. Payload nie zawiera
`total_score`, rankingu ani `chosen/rejected`. Cienki skrypt
`scripts/prepare_task06_selection_preflight.py` jedynie uruchamia tę walidację.

Dla nowego preflightu przechodzi 20 syntetycznych testów CPU; cały ukierunkowany
zestaw nowych i bezpośrednio powiązanych kontraktów Task 06/07/09 ma wynik
132 passed. Pełny pytest był wówczas odłożony, aby nie dotykać aktywnego
`dev_confirm` Task 05; po jego zakończeniu kontrola repozytorium uzyskała
`425 passed`. Zsynchronizowano wyłącznie ścieżkę logów i execution batch w
preflight fixture Task 05 z zamrożonym configiem; kontrakty Task 06 pozostały
bez zmian. Po domknięciu pilota pełna kontrola repozytorium została ponowiona
i uzyskała `444 passed`.

Nie uruchomiono generacji, scoringu modeli, materializacji właściwych
preferencji ani audytu człowieka. Nadal nie wykonano kalibracji, zamrożenia wag
i progów na rzeczywistych evidence, wyliczenia funkcji przypisującej wagi,
rankingu ani wyboru
`chosen/rejected`. Handoff nie autoryzuje żadnego z tych etapów.
`generate_candidates.py` i
`score_candidates.py` pozostają celowo niewdrożone do czasu decyzji
właściciela zapisanych przez nowy execution design; checkpointy i sędziowie są
już przypięci, ale kohorta, kalibracja oraz budżet nie są autoryzowane. Nowe skrypty
`validate_generated_candidates.py` i `assemble_candidate_evidence.py` jedynie
walidują lub składają wcześniej policzone rekordy. Nie należy uruchamiać
kampanii przed osobną konfiguracją.

D01b `dev_confirm` zakończył się `non_inferior_only` i nie wypromował hybrydy
do finalist freeze. Task 06 nie może zatem domniemywać jej jako generatora;
rzeczywista generacja nadal wymaga osobnej decyzji przypinającej stabilny
checkpoint wejściowy. Późniejszy jednoseedowy 4.5B scale-interaction screen
ma status `eligible`, ale jawnie nie ma selection claim. ID-only audyt znalazł
tylko 591 legalnych nieoglądanych rekordów dev, niewystarczających dla
prospektywnego 97.5% confirmu wobec niezmiennego progu `+0.01`. Confirm i
promocja są fail-closed `BLOCKED` do decyzji właściciela oraz ewentualnego
dostarczenia nowej nietestowej kohorty. Nie odblokowuje to Task 06.

Zewnętrzny TriviaQA dev-confirm na 8000 query zakończył się `rc=0` i przeszedł
prerejestrowaną bramkę: Hybrid-minus-W06 `corpus_ndcg_at_10` wynosi
`+0.04786661287844578`, 97.5% CI
`[0.045011840373656756, 0.05082630534799233]`, a wszystkie guardraile
przeszły. Artefakt zachowuje Hybrid do finalist-freeze review, ale zgodnie z
ADR nadal zapisuje `task06_or_task09_promotion_authorized=false`. W06 seed 43
nie zbiegł, jednak post-hoc wynik seedów 42+44 pozostaje dodatni ponad próg;
jest to caveat stabilności, nie zastępcza bramka. Właściciel następnie
zaakceptował dwumodelową procedurę W06+D01+selector dla danych oraz D01 jako
pojedynczy adapter startowy przyszłego Task 07. Potwierdzony probe ocenia
wartość wybranych danych, a wybór startu DPO jest osobno zapisaną decyzją.
Generacja i scoring nadal wymagają prospektywnego execution ADR i nie zostały
autoryzowane ani uruchomione; `final_tests_used=[]`.

## Cel

Zbudować wysokiej jakości pary `chosen/rejected` dla DPO, nie myląc preferencji dokumentów z preferencjami odpowiedzi generatora.

## Zależności

Taski 02, ukończony Harness v1.1 z Task 04 i Task 05 oraz stabilny checkpoint
SFT.

## Generacja kandydatów

Dla każdego wybranego passage wygeneruj 4–8 kandydatów przez kombinację:

- 2–4 stylów adekwatnych do passage;
- 2–3 focusów;
- temperatur `0.3, 0.7, 1.0`;
- co najmniej dwóch seedów;
- opcjonalnie baseline’u bez treningu.

Zapisz pełne logprobs, parametry generacji, kontrolki i checkpoint ID, jeżeli biblioteka to wspiera.

Nie generuj preferencji na testach.

## Kandydaci negatywni

Źródła rejected:

1. gorszy, ale poprawnie sformatowany kandydat tego samego SFT;
2. kandydat z wysokim overlapem/kopiowaniem;
3. kandydat z niskim grounding margin;
4. kandydat dotyczący niepożądanego focusu;
5. kandydat duplikujący inne query;
6. ostrożnie: query powiązane z hard-negatywnym dokumentem.

Ostatnia kategoria nie może dominować, bo zbyt łatwe rejected uczą tylko tematyczności.

## Composite score

Zapisuj osobno każdy komponent oraz total. Przykładowe pola:

```json
{
  "ground_score": 0.81,
  "negative_margin": 0.42,
  "corpus_round_trip": 1.0,
  "effective_candidate_count": 3,
  "possible_false_negative": false,
  "overlap_reward": 0.65,
  "focus_accuracy": 1.0,
  "style_accuracy": 1.0,
  "format_score": 1.0,
  "copy_penalty": 0.1,
  "answerability_flag": true,
  "total_score": 2.97
}
```

Nie usuwaj składowych po zsumowaniu.

Primary jest builder judge, shadow sędzią potwierdzającym, corpus retrieval
niezależnym sygnałem, a panel ludzki kalibracją. Raportuj niezgodność. Kandydat
z wysokim primary score, ale słabym round-trip jest wartościowym rejected typu
„zbyt ogólne”. Dla zaakceptowanych kandydatów wykonuj re-mining zgodny
z wersjonowaną polityką Task 04 i zapisuj provenance minera.

Większy, zamrożony model inference-only może być dodatkowym źródłem kandydatów
wyłącznie jako jawna ablacja teachera. Opcjonalny zamrożony answerability judge
może rozstrzygać disagreement i wskazywać evidence; żadnego z sędziów nie
wolno dostrajać na outputach generatora.

## Bramka różnorodności same-prompt (obowiązkowa przed budową par)

Pomiar expansion 500×8 wykazał kolaps różnorodności przy identycznym
promptcie: `duplicate_rate` średnio 0.399 (pilot: 0.0049), self-BLEU 0.603,
mediana max pairwise lemma Jaccard 1.0. Pary budowane z niemal identycznych
kandydatów kodują szum sędziów, nie różnicę jakości, i uczą DPO artefaktów.

Wymagania przed materializacją par z dowolnej kohorty same-prompt:

- grupa wchodzi do budowy par tylko wtedy, gdy po normalizacji i deduplikacji
  ma co najmniej 3 efektywnie różne kandydatury oraz spełnia prospektywnie
  zamrożony próg grupowej różnorodności (duplicate_rate, self-BLEU lub
  odpowiednik); próg ustala osobny ADR przed odczytem par;
- dozwolone osie naprawy w ramach *tego samego* promptu: rozkład decodingu
  (temperatury, min-p/top-p, seedy) oraz większe K z deduplikacją — kontrakt
  DPO wymaga wspólnego promptu, nie wspólnych parametrów samplingu;
- odsetek grup odrzuconych przez bramkę jest raportowany, nie ukrywany.

Stan realizacji bramki: progi są zamrożone ADR
[`task06_same_prompt_diversity_gate_v1.md`](../reports/decisions/task06_same_prompt_diversity_gate_v1.md)
(min. 3 efektywne kandydatury po deduplikacji near-duplicate przy lemma Jaccard
0.90, `duplicate_rate <= 0.50`, `effective_self_bleu <= 0.75`, minimalny Jaccard
reprezentantów `<= 0.85` spójnie z `SelectionPolicy`). Pierwsze zastosowanie na
kohorcie `same_prompt_expansion_v1` dało 362/500 grup `eligible`; progów nie
wolno zmieniać po zobaczeniu tego wyniku.

Dopuszczalna jest również prospektywna ablacja teachera (osobny ADR): lokalny,
przypięty `Qwen3.6-27B` Q4 generuje kandydatów na dokładnie te same prompty
jako dodatkowe źródło `chosen`. Provenance teachera jest oddzielne, model
pozostaje zamrożony, a budżet musi mieścić się w przepustowości kilku tysięcy
promptów na dobę. Par zawierających kandydatów teachera nie może oceniać
sędzia tożsamy z teacherem (self-preference bias); audytuje je drugi model.

## Budowa par

Preferowana metoda:

- wybierz top candidate jako `chosen`;
- wybierz `rejected` z dolnej części, ale o poprawnym formacie i minimalnej relewancji;
- wymagaj minimalnego `score_margin`;
- nie paruj identycznych lub niemal identycznych query;
- zachowaj rozkład typów błędów rejected;
- ogranicz liczbę par z jednego passage;
- nie pozwalaj, aby ten sam tekst query był zawsze chosen lub zawsze rejected bez analizy.

Warianty:

- pairwise top-vs-bottom;
- top-vs-near-miss;
- kilka rejected na chosen;
- listwise dane zachowane do przyszłych metod, nawet jeśli DPO używa par.

## Kontrola jakości preferencji

Automatycznie odrzuć:

- brak wyraźnego marginesu;
- na oba query nie można odpowiedzieć z pasażu;
- oba query identyczne po normalizacji;
- chosen z invalid format;
- chosen skrajnie ogólne;
- konflikt między reranker margin a answerability checks;
- wysoce niepewny focus.

Walidacja par (owner waiver 2026-08-12: dual-LLM zamiast ludzi):

- min. 500 par na etapie rozwoju, każda oceniona przez oba przypięte LLM;
- min. 1000 par przed finalnym DPO, jeśli właściciel nie zmieni osobno tej bramki;
- ślepa kolejność;
- preferencja każdego LLM i kod przyczyny;
- zgodność automatycznego rankingu z każdym LLM i consensus obu;
- analiza według źródła rejected;
- przyszłe audyty preferują lokalny, przypięty checkpoint `qwen3.6-27b` Q4
  zamiast wariantu API tego samego modelu (przypięte wagi, brak dryfu wersji
  i limitów quota); dla wywołań API zapisuj wersję/datę modelu; zmiana
  transportu sędziego wymaga własnego ADR i nie dotyczy zamrożonych kontraktów.

## Leakage i splity

Preference train/dev/test muszą dziedziczyć split passage. Żaden passage/near-duplicate z preference dev/test nie może wejść do preference train.

## Wymagane skrypty

- `scripts/apply_task06_same_prompt_diversity_gate.py`
- `scripts/freeze_task06_same_prompt_expansion_v2.py`
- `scripts/run_task06_same_prompt_expansion_v2.sh`
- `scripts/generate_candidates.py`
- `scripts/score_candidates.py`
- `scripts/select_candidate_sets.py`
- `scripts/build_preferences.py`
- `scripts/export_preference_audit.py`
- `scripts/import_preference_audit.py`

## Artefakty

- `candidates/<run_id>/*.parquet`
- `preferences/<version>/train.parquet`
- `preferences/<version>/dev.parquet`
- `preferences/<version>/test.parquet`
- `preferences/<version>/manifest.json`
- raport jakości i rozkładów.

## Kryteria akceptacji

- format zgodny z TRL DPO: prompt/chosen/rejected;
- każdy rekord ma wszystkie składowe score i provenance;
- zgodność automatu z oboma LLM i ich wzajemna zgodność są raportowane;
- rejected nie są wyłącznie nonsensowne;
- score margin i typ rejected są zbalansowane;
- preference test jest zamrożony;
- continued-SFT dataset z samymi chosen jest generowany jako obowiązkowa kontrola.
