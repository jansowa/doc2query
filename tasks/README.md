# Rejestr zadań

Ten plik jest centralnym spisem treści i źródłem prawdy o stanie realizacji
programu. Szczegółowe wymagania oraz kryteria akceptacji pozostają w plikach
poszczególnych zadań.

## Zasady aktualizacji

Każda sesja realizująca zadanie musi w tym samym commicie:

1. zaktualizować jego status w poniższej tabeli;
2. zaktualizować sekcję `Status` w pliku zadania;
3. opisać w kolumnie „Stan i następny krok” faktycznie wykonany zakres oraz
   niewykonane pomiary, kosztowne runy lub bramki;
4. nie oznaczać zadania jako ukończonego na podstawie samej implementacji, jeśli
   jego kryteria wymagają wyników eksperymentalnych.

Dozwolone statusy:

- `TODO` — prace nie zostały rozpoczęte;
- `IN PROGRESS` — trwa implementacja lub walidacja;
- `IMPLEMENTED` — kod i tanie testy są gotowe, ale pozostały jawnie wskazane
  runy, pomiary albo bramki;
- `DONE` — wszystkie kryteria akceptacji zostały spełnione;
- `BLOCKED` — dalsza praca zależy od wskazanej bramki lub decyzji;
- `OPTIONAL` — eksperyment nie należy do domyślnej ścieżki.

Statusy muszą opisywać stan potwierdzony artefaktami. Nie wolno wpisywać
wyników treningu, benchmarku ani eksperymentu, którego faktycznie nie
uruchomiono.

## Spis i stan realizacji

| Task | Zakres | Status | Stan i następny krok |
|---|---|---|---|
| [00](00_repository_bootstrap.md) | Bootstrap repozytorium i odtwarzalność | `DONE` | Szkielet projektu, środowisko, CLI, testy i rejestrowanie są gotowe. |
| [01](01_data_contract_audit_and_splits.md) | Kontrakt danych, audyt, deduplikacja i splity | `IMPLEMENTED` | Pełny `msmarco_pl` przetworzono do zamrożonych splitów v1 i par doc2query bez leakage pozytywów. Dla rekordów z <10 negatywami przyjęto corpus retrieval oraz oznaczone, deterministyczne backfillowanie tylko w diagnostycznej puli. Pozostał raport tokenowych percentyli/HTML. |
| [02](02_reranker_and_reward_proxies.md) | Zamrożone rerankery i proxy nagrody | `IMPLEMENTED` | Integracja, reward proxies i testy są gotowe; primary zmierzył pełny frozen dev, a query-macro próg Youdena `possible_false_negative` jest przypięty bez użycia testu. Bramka HN domierzyła primary/shadow na 775 wspólnych dev query z 10 negatywami; nadal pozostał pełny benchmark obu sędziów na całym dev/test i wymaganych slice'ach. |
| [03](03_sft_qlora_baselines.md) | Baseline'y SFT/QLoRA | `IMPLEMENTED` | S07 jest kompletną, nieporównywalną budżetowo diagnostyką bez promocji. P-06 mass rescoring jest `SUPERSEDED`. Właściciel świadomie anulował ręczne P06-T i zaakceptował resztkowe ryzyko tłumaczeń na podstawie wcześniejszego udanego treningu embeddera; frozen train i próg `>=23.50` pozostają bez zmian. Nie kończyć lokalnego scoringu ani trenować drop/weighted. |
| [04](04_evaluation_harness.md) | Harness ewaluacyjny | `IMPLEMENTED` | P-01–P-05 i pełny S07 Harness/probe są gotowe. Pełna dev-only bramka HN0–HN3 ukończyła 775 wspólnych legalnych query i utrzymała HN0+filter/drop z powodu braku zgodnej primary/shadow podstawy do promocji nowego minera; nie otwarto testów finalnych. Artefakt S07 pozostaje `comparison_eligible=false`; pozostały porównywalne probe i testy finalistów. Rozszerzenie 2026-08-13: do zakresu dodano M-03 (guardrail zbieżności probe, 5 seedów na tanich korpusach). |
| [05](05_controlled_diversity_and_multiquery.md) | Kontrolowany styl, focus i multi-query | `IMPLEMENTED` | TriviaQA 3-seed confirm przeszedł (`nDCG@10 +0.0478666`, 97.5% CI `[+0.0450118,+0.0508263]`); analiza 42+44 utrzymuje efekt mimo niestabilnego W06 S43. Właściciel zaakceptował W06+D01+selector jako procedurę danych Task 06 i D01 controlled jako przyszły start Task 07. Handoff preflight jest verified, ale pełne 4.5B, Task 09 i final test pozostają nieautoryzowane. |
| [06](06_candidate_scoring_and_preference_data.md) | Scoring kandydatów i dane preferencyjne | `IN PROGRESS` | Pilot 512×8 ukończony: 4096/4096 scoringów i 2048 safe-selected (1164 W06 + 884 D01), selector zmienił 482/512 grup. Naprawiono wyłącznie błędne etykiety `smoke` w provenance i odbudowano selekcję; teksty/score’y bez zmian. Pilot nie wystarcza do DPO, bo sloty D01 mają różne prompty. Zamrożono więc same-prompt expansion: 500 promptów D01 × 8 odpowiedzi, potem scoring, tentative pairs i dual-LLM Groq. Run expansion jest ukończony (4000/4000 generacji i scoringów, bez resume, peak VRAM 3.43 GB), a jego grupowe metryki pokazały kolaps różnorodności przy identycznym promptcie. Progi bramki różnorodności same-prompt zamrożono prospektywnym ADR i zastosowano bramkę na CPU: 362/500 grup `eligible` (72.4%), 138 odrzuconych. Kohorta daje najwyżej 362 pary, czyli mniej niż 500 wymaganych do rozwojowej bramki dual-LLM. Właściciel delegował wybór ścieżki naprawy; ADR v2 zamraża nową, rozłączną klastrowo kohortę 500 pasaży z szerszym decodingiem. Etap CPU jest wykonany (500/500 unikalnych klastrów, zero nakładania z 544 klastrami smoke/pilot i 49352 klastrami 50k SFT), kohorta v2 jest ukończona (4000/4000 generacji i scoringów; jeden run przerwał się na niepoprawnym completion w slocie 1.2, po amendmencie resamplującym wznowił się bez utraty pracy), a bramka dała 466/500 grup `eligible` (93.2%) wobec 362/500 w v1 — razem 828 grup, powyżej progu 500 par. Na okno bezobsługowe 2-3 dób zamrożono rozłączne kohorty v3/v4/v5 po 3000 pasaży i uruchomiono nadzorowaną kolejkę (`scripts/run_unattended_queue.sh`, watchdog w cronie co 20 min i po restarcie) na ~53 h: M-03 (seedy 45/46 confirmu TriviaQA, wyłącznie treningi), dziewięć rozłącznych kohort v3-v11 po 3000 pasaży oraz sweep budżetu probe 1024/2048 jako diagnostyka M-01/M-03. Po wyczerpaniu kolejki osobny strażnik (`scripts/poweroff_when_queue_drained.sh`, cron 5/25/45) wyłącza maszynę po godzinie stabilnego stanu wyczerpania. Kolejka produkuje wyłącznie artefakty; agregacje i decyzje czekają na właściciela. Par nie zbudowano, Groq nie uruchomiono, Task 07 nadal zamknięty, `final_tests_used=[]`. Rozszerzenie 2026-08-13: dopuszczono prospektywną ablację teachera (lokalny Qwen3.6-27B Q4) i lokalnego sędziego qwen3.6-27b za osobnym ADR. Okno tokenowe 2026-08-14/16 (komenda właściciela: generować dane bezpośrednio tokenami modelu asystującego): zamrożono prospektywnie dwa ADR-y przed generacją. Korpus walidacyjny nagrody (180 pasaży × 8 klas = 1440 zapytań o etykietach z konstrukcji) jest **ukończony i zmierzony na CPU**: PASS dla P1/P3-wrong-focus/P4-dobre/P5/P6/P7 (bramka różnorodności o niezmienionych progach przepuściła 180/180 grup), FAIL dla P2 (`entity_preservation` to detektor halucynowanych encji, nie sygnał specyficzności — remis 1.0 w 180/180), P3-good-specific (0.622; 46 focusów nierozstrzygniętych) i P4-wrong-form (0.75; `_PREFIX` wymaga dwukropka, więc „Oto …” przechodzi). Progów nie kalibrowano, `format.py` i splittera zdań nie zmieniono. Ablacja teachera na modelu API (600 pasaży kohorty v3 × 4 kontrolki × 4 kandydatury = 9600 zapytań, osobne provenance, wagi nieprzypięte) jest **w toku**: 11/24 shardów kompletnych (4400 rekordów). Scoring GPU obu zbiorów, budowa par i Groq pozostają nieautoryzowane; kolejka bezobsługowa zakończyła się 2026-08-16 wynikiem 25/25 zadań bez awarii w 49,18 h GPU: kohorty v3-v11 dały 25164/27000 grup `eligible` (93,2%), co z v1 i v2 daje 25992 grupy do budowy par (powyżej progu 1000), a sweep budżetu probe pokazał, że wariancja seedowa (0,0011-0,0826) bywa większa od mierzonych efektów i że strata treningowa nie jest guardrailem zbieżności (`r = -0,199`). GPU jest wolne i czeka na decyzję właściciela o polityce `chosen/rejected`. |
| [07](07_dpo_training.md) | DPO i continued-SFT control | `IN PROGRESS` | D01 controlled 4.5B pozostaje jedynym przyszłym startem DPO, W06 anchor/source bez łączenia wag. Task 06 ma dopiero fail-closed execution design, nie dane ani osobną politykę chosen/rejected; `task07_training_authorized=false`, a smoke/PEFT, trzy ramiona, seedy i ewaluacja pozostają niewykonane; `final_tests_used=[]`. Rozszerzenie 2026-08-13: Task 07 jest jawnie także destylacją selektora; obowiązkowa ewaluacja trybu produkcyjnego bez selektora (M-05) dla finalistów. |
| [08](08_grpo_multiobjective_rl.md) | Wielokryterialny GRPO/RL | `BLOCKED` | Od 2026-08-13 planowany etap programu (cel edukacyjny właściciela, AGENTS.md §2): ścieżka 1.5B/8 GB po Task 07, R06 4.5B preferencyjnie na 16 GB; wymagana decyzja `reports/decisions/enable_grpo.md`. Porównanie DPO vs continued SFT vs best-of-N jest kontekstem interpretacyjnym; twarde przesłanki Fazy E są bramką awansu do selekcji finalistów, nie startu. Materiał przygotowawczy 2026-08-16: zmierzony w Task 06 korpus walidacyjny komponentów nagrody daje pierwsze twarde przesłanki do składu wielokryterialnej nagrody (`entity_preservation` nie karze ogólności; `format_valid` obchodzony wtrąceniem bez interpunkcji; ogólność rozdzielił `content_jaccard` w 85.6% grup). Status bez zmian — decyzja `enable_grpo.md` nadal wymagana. |
| [09](09_experiment_campaign.md) | Kampania eksperymentalna | `BLOCKED` | Model-free scaffold i fail-closed projekt pilota Task 06 są gotowe, lecz brakuje decyzji właściciela oraz rzeczywistych wyników Task 06–07, Pareto review i decyzji finalistów. Task 09 i finalne testy pozostają zamknięte. Rozszerzenie 2026-08-13: kampanię poprzedza Etap 0 — walidacja predykcyjna probe (M-01/E15); probe mieszany (M-02) i tryb produkcyjny (M-05) obowiązkowe dla finalistów; 7B+ wyłącznie na sprzęcie zewnętrznym. |
| [10](10_final_scaleup_inference_release.md) | Finalny trening, inference i release | `BLOCKED` | Wymaga wyników Task 09 i zatwierdzonego finalnego ADR. Rozszerzenie 2026-08-13: wymagana prerejestrowana, świadoma mocy reguła decyzyjna testów finalnych (M-04) oraz przenośny pipeline treningowy dla większych modeli na sprzęcie zewnętrznym. |
| [11](11_judge_robustness_audit.md) | Audyt odporności sędziego i fallbacki | `OPTIONAL` | Późny eksperyment badawczy; nie obejmuje treningu ani dostrajania rerankera. |

## Operacyjna kolejność po audycie

Ten rejestr jest jedynym operacyjnym źródłem kolejności i statusów.
[`docs/plan_poprawek_po_audytach.md`](../docs/plan_poprawek_po_audytach.md)
pozostaje zapisem przesłanek i identyfikatorów P-xx, ale nie jest równoległym
backlogiem. Zakres P-xx został przeniesiony do wskazanych plików zadań.

1. **Bramka P-05 dev screen rozstrzygnięta:** oba warianty są
   `non_inferior_only`; `dev_confirm_authorized_arms=[]`. Nie uruchamiać
   `dev_confirm` dla gold/mixed/W05 synthetic z tej macierzy.
2. **Następna kolejka baseline:** S00 i S07 są kompletne na frozen dev. S07
   pozostaje diagnostyką: jego probe użył `2485 par / 864000 tokenów / batch
   6`, a P-05 `2486 / 1152000 / batch 8`, dlatego nie wolno porównywać efektów
   ani wybierać architektury. Nie wykonywać matched-budget rerunu S07 i nie
   promować go do `dev_confirm` lub testów finalnych. P-06 mass rescoring jest
   `SUPERSEDED`: source scores pochodzą z silniejszego rerankera użytego przed
   kopaniem negatywów, próg `23.50` jest już wyegzekwowany, a minimalny margin
   frozen train wynosi `6.0`. Nie kończyć
   `artifacts/task03/p06/train_margins_v1` i nie używać go do drop/weighted.
   P06-T zmaterializował prospektywną próbkę 300 train, ale właściciel
   świadomie anulował ręczne kodowanie i zaakceptował resztkowe ryzyko
   tłumaczeń, wskazując wcześniejszy udany trening embeddera jako praktyczną
   przesłankę. Nie jest to nowy pomiar repozytorium. Frozen train pozostaje
   bez zmian, a P06-T nie jest już bramką.
   Nie zmieniać progu P-04 po obejrzeniu wyników.
   Testy finalne wolno otworzyć raz dopiero po zamrożeniu rzeczywistych
   finalistów. W06 pozostaje eksploracyjnym dowodem wykonalności 4.5B/8 GB,
   a nie ogólną zgodą na kampanię skali; jedynym wyjątkiem jest ograniczony,
   prospektywny pilot interakcji D01b 4.5B opisany w punkcie 4.
3. **Pełna bramka HN ukończona:** na wspólnej kohorcie 775/1000 query HN1
   nie odróżniło się od HN0+filter według primary, HN2 było łatwiejsze także
   według shadow, a HN3 miało konstrukcyjnie perfect primary przy przeciwnym
   kierunku shadow i `9.81%` disagreement. Utrzymano HN0+filter/drop; testów
   finalnych nie otwarto.
4. **Task 05:** D01 style/intent-only 1.5B/50k i 4.5B/50k wraz z matched
   W05/W06, common exact-K, primary/shadow/corpus scoringiem oraz copy/semantic
   `compare` są ukończone. Pełne ramiona kontrolowane zatrzymały guardraile.
   Retrospektywny D01b safe-anchor wskazał wykonalną hybrydę i niezależną
   poprawę shadow Recall@1, ale nie może autoryzować probe na tym samym dev.
   Prospective v3 uruchomiło niezmieniony selektor na niewidzianej kohorcie
   2000 rekordów: wszystkie prerejestrowane bramki przeszły, a decyzja
   `authorize_equal_budget_probe_inputs` doprowadziła do równobudżetowego
   `dev_screen` 1.5B hybrid-vs-W05. Run zakończył się `rc=0`; hybrid uzyskał
   status `eligible`, z dolną granicą 95% CI dla `corpus_ndcg_at_10`
   `0.012093457847827558` przy wymaganym minimum `+0.01`, a wszystkie
   prerejestrowane guardraile non-inferiority przeszły. Pełnobudżetowy
   `dev_confirm` na seedach 42/43/44 zakończył się następnie `rc=0`, lecz
   decyzją `non_inferior_only`: dolna granica CI `0.006927431152133765` nie
   osiągnęła wymaganego `+0.01`. Hybryda nie jest zachowana jako finalista.
   Ten zakończony eksperyment nie otworzył 4.5B ani finalnych testów.
   Późniejsza osobna decyzja właściciela dopuściła wyłącznie jednoseedowy,
   development-only pilot interakcji ze skalą: W06 4.5B vs D01b 4.5B na
   nowych rozłącznych kohortach 1000/2000. Pilot zakończył się `rc=0` i
   screeningowym `eligible`: różnica `corpus_ndcg_at_10` to
   `+0.02073792007878962`, a 95% CI
   `[0.011055484860771694, 0.03017264616376007]`; wszystkie guardraile
   przeszły. Pełny ID-only audyt znalazł tylko 591 legalnych nieoglądanych
   rekordów. Planowany 97.5% CI nie ma wystarczającej czułości wobec
   niezmiennego progu `+0.01`, także przy optymistycznym założeniu
   niezależności trzech seedów. Właściciel dostarczył następnie nowy,
   nieoglądany TriviaQA dev. Prospektywnie zamrożono 8000 query i corpus 139782
   dokumentów; leakage względem treningu wynosi zero. Trzyseedowy confirm
   zakończył się `rc=0`. Hybrid-minus-W06 `corpus_ndcg_at_10` wynosi
   `+0.04786661287844578`, a 97.5% CI
   `[0.045011840373656756, 0.05082630534799233]`; wszystkie guardraile
   przeszły. W06 seed 43 nie zbiegł i jest jawnym caveatem stabilności, ale
   symetryczna post-hoc analiza seedów 42+44 nadal daje `+0.0206102`, CI
   `[+0.0174116,+0.0237756]`. Hybrid jest zachowany do finalist-freeze review.
   Właściciel zaakceptował następnie W06+D01+selector jako procedurę danych i
   D01 jako przyszły start Task 07; model-free handoff preflight przeszedł.
   Execution ADR, pełna kampania 4.5B i finalne testy pozostają zamknięte.
   Pozostałe D00–D12 nadal wymagają własnych pomiarów i bramek.
5. **Po Task 05:** P-08 w Task 06 i Task 07. Bramka różnorodności same-prompt
   jest zamknięta metodologicznie: progi zamrożono ADR
   `reports/decisions/task06_same_prompt_diversity_gate_v1.md` przed odczytem
   par, a jej pierwsze zastosowanie na kohorcie `same_prompt_expansion_v1` dało
   362/500 grup `eligible`. Nie zmieniać progów po zobaczeniu tego wyniku i nie
   powtarzać ani nie nadpisywać zakończonego runu expansion. Deficyt do 500 par
   rozwojowej bramki dual-LLM wolno uzupełnić wyłącznie nową, osobno
   autoryzowaną generacją same-prompt (większe K z deduplikacją albo nowe grupy
   z legalnej puli), nigdy poluzowaniem bramki. Wybraną ścieżką jest ADR
   `reports/decisions/task06_same_prompt_expansion_v2_2026-08-13.md`: nowa,
   rozłączna klastrowo kohorta 500 pasaży z szerszym decodingiem. Jej kohorta
   jest już zamrożona na CPU; pozostaje etap GPU i bramka o niezmienionych
   progach. Budowa par nadal wymaga osobnego ADR zamrażającego politykę
   `chosen/rejected` i kalibrację komponentów.
6. **Kampania:** Task 09 dopiero po wcześniejszych etapach; Task 10 dopiero po
   finalnym ADR.
7. **Opcjonalne:** P-09 i Task 11 wyłącznie po własnych bramkach. Task 08
   jest od 2026-08-13 planowanym etapem po Task 07 (cel edukacyjny; bramka
   awansu do selekcji finalistów opisana w jego pliku i w AGENTS.md Faza E).

Równobudżetowy D01b probe `dev_screen` 1.5B hybrid-vs-W05 jest ukończony.
Końcowy `run-all` zakończył się `rc=0`, zapisał `dev_screen_complete`,
`dev_confirm_authorized=true` i `final_tests_used=[]`. Hybrid uzyskał
`eligible`; dolna granica 95% CI dla `corpus_ndcg_at_10` wynosi
`0.012093457847827558` przy progu `+0.01`, a wszystkie guardraile przeszły.
Nie uruchamiać ponownie dev-screen.

Pełny D01b `dev_confirm` jest ukończony dla 7936 par / 1984 dokumentów / K=4,
batch 2, 4000 kroków i seedów 42/43/44. Końcowy `run-all` zakończył się
`rc=0`; decyzja `non_inferior_only` zatrzymała promocję, ponieważ dolna granica
95% CI głównej metryki wyniosła `0.006927431152133765` przy progu `+0.01`.
Nie uruchamiać ponownie 1.5B dev-screen ani dev-confirm. Spośród nowych runów
4.5B dopuszczony jest tylko prospektywny scale-interaction pilot wskazany w
tabeli; pełna kampania 4.5B i wszystkie finalne testy pozostają zamknięte.
Nie uruchamiać pełnego
`scripts/score_train_margins.py`, nie ustalać lokalnego progu i nie trenować
ordinary/drop/weighted bez nowego prospektywnego ADR opartego na ręcznie
potwierdzonej, powtarzalnej klasie błędu tłumaczenia.
Nie należy ponownie uruchamiać
`scripts/run_p05_dev_screen.sh` ani `scripts/run_p05_guardrails.sh`: wyniki i
decyzje są kompletne. I02/I04/I05 pozostają odroczone, S00 jest zmierzone,
S07 diagnostycznie kompletne, P-06 mass rescoring zamknięte jako
`SUPERSEDED`, ręczne P06-T świadomie anulowano, a
niewykonane D00–D12, Task 06 i wszystkie testy finalne pozostają zamknięte.
D01 `dev_confirm` został zakończony wyłącznie na train/dev bez użycia final test.

## Kolejność bazowa

Po domknięciu pakietu naprawczego kolejność bazowa to
`00 → 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10`.
Task 04 może częściowo powstawać równolegle z Task 03. Task 08 jest od
2026-08-13 planowanym etapem po Task 07 (cel edukacyjny właściciela; bramka
awansu wyników do selekcji finalistów w jego pliku i w AGENTS.md Faza E).
Task 11 pozostaje opcjonalny i wolno go rozpocząć wyłącznie po spełnieniu
warunków opisanych w jego pliku. Nadrzędne bramki badawcze i zasady
bezpieczeństwa znajdują się w [`AGENTS.md`](../AGENTS.md).

Aktualizacja specyfikacji 2026-08-13 (decyzja właściciela): kontrakt
sprzętowy w `AGENTS.md` §12 opisuje odtąd 8 GB (RTX 3060 Ti) jako budżet
bazowy, 16 GB (RTX 5070 Ti, 64 GB RAM) jako rozszerzenie oraz sprzęt
zewnętrzny dla finalistów 7B+ przez przenośny pipeline. Rozszerzenia
metodologiczne M-01–M-05 (walidacja predykcyjna probe, probe mieszany,
stabilność seedów, reguła decyzyjna testów finalnych, tryb produkcyjny) są
zdefiniowane w `AGENTS.md` §9.2 i rozpisane w Taskach 04, 06, 07, 09 i 10.
GRPO (Task 08) jest planowanym etapem programu z jawnym celem edukacyjnym
(AGENTS.md §2) i osobną bramką awansu wyników do selekcji finalistów. Zmiany
są prospektywne i nie unieważniają żadnych zamrożonych kontraktów, kohort
ani zakończonych pomiarów.
