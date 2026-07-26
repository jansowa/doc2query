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
| [04](04_evaluation_harness.md) | Harness ewaluacyjny | `IMPLEMENTED` | P-01–P-05 i pełny S07 Harness/probe są gotowe. Pełna dev-only bramka HN0–HN3 ukończyła 775 wspólnych legalnych query i utrzymała HN0+filter/drop z powodu braku zgodnej primary/shadow podstawy do promocji nowego minera; nie otwarto testów finalnych. Artefakt S07 pozostaje `comparison_eligible=false`; pozostały porównywalne probe i testy finalistów. |
| [05](05_controlled_diversity_and_multiquery.md) | Kontrolowany styl, focus i multi-query | `IMPLEMENTED` | Gotowe są kontrakty i kod CPU, w tym wznawialny post-D01 pipeline oraz niezależny pakiet natural-query calibration/audits. Na pełnym `dev_intrinsic_rank10` zmaterializowano opisową kalibrację reguł i ślepe próbki 500 etykiet oraz 200 pasaży. Kontrakt dopuszcza jednego właściciela-oceniającego, jawnie pozostawiając agreement `NOT MEASURED`; formularze nie mają jeszcze ocen. Nocna kolejka D01 nie została uruchomiona; D00–D12, human check i probe z CI nadal są niewykonane. |
| [06](06_candidate_scoring_and_preference_data.md) | Scoring kandydatów i dane preferencyjne | `TODO` | Wymaga stabilnego checkpointu SFT, ukończonego Harness v1.1 oraz Task 02 i 05. |
| [07](07_dpo_training.md) | DPO i continued-SFT control | `TODO` | Wymaga danych preferencyjnych z Task 06. |
| [08](08_grpo_multiobjective_rl.md) | Wielokryterialny GRPO/RL | `OPTIONAL / BLOCKED` | Uruchamiać wyłącznie po spełnieniu bramki i zapisaniu decyzji `reports/decisions/enable_grpo.md`. |
| [09](09_experiment_campaign.md) | Kampania eksperymentalna | `BLOCKED` | Pełna dev-only bramka HN jest ukończona, a P06-T anulowane. Kampanię nadal blokują eksperymenty Task 05 oraz Task 06–07 w zakresie dopuszczonym przez ich bramki; P-06 mass rescoring nie jest zależnością. |
| [10](10_final_scaleup_inference_release.md) | Finalny trening, inference i release | `BLOCKED` | Wymaga wyników Task 09 i zatwierdzonego finalnego ADR. |
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
   a nie zgodą na dalszą kampanię skali.
3. **Pełna bramka HN ukończona:** na wspólnej kohorcie 775/1000 query HN1
   nie odróżniło się od HN0+filter według primary, HN2 było łatwiejsze także
   według shadow, a HN3 miało konstrukcyjnie perfect primary przy przeciwnym
   kierunku shadow i `9.81%` disagreement. Utrzymano HN0+filter/drop; testów
   finalnych nie otwarto.
4. **Task 05:** następną dopuszczoną kolejką jest D01 style/intent-only,
   dopasowane 1.5B/50k i 4.5B/50k oraz diagnostyczne panele dev. Pozostałe
   D00–D12 nadal wymagają własnych pomiarów i bramek.
5. **Po Task 05:** P-08 w Task 06 i Task 07.
6. **Kampania:** Task 09 dopiero po wcześniejszych etapach; Task 10 dopiero po
   finalnym ADR.
7. **Opcjonalne:** Task 08, P-09 i Task 11 wyłącznie po własnych bramkach.

Najbliższy punkt wejścia to D01 style/intent-only z Task 05 przez
`scripts/run_task05_d01_overnight.sh`.
Nie uruchamiać pełnego
`scripts/score_train_margins.py`, nie ustalać lokalnego progu i nie trenować
ordinary/drop/weighted bez nowego prospektywnego ADR opartego na ręcznie
potwierdzonej, powtarzalnej klasie błędu tłumaczenia.
Nie należy ponownie uruchamiać
`scripts/run_p05_dev_screen.sh` ani `scripts/run_p05_guardrails.sh`: wyniki i
decyzje są kompletne. I02/I04/I05 pozostają odroczone, S00 jest zmierzone,
S07 diagnostycznie kompletne, P-06 mass rescoring zamknięte jako
`SUPERSEDED`, ręczne P06-T świadomie anulowano, a
`dev_confirm`, niewykonane D00–D12, Task 06 i wszystkie testy finalne pozostają
zamknięte. D01 jest dopuszczony wyłącznie na train/dev.

## Kolejność bazowa

Po domknięciu pakietu naprawczego kolejność bazowa pozostaje
`00 → 01 → 02 → 03 → 04 → 05 → 06 → 07 → 09 → 10`.
Task 04 może częściowo powstawać równolegle z Task 03. Taski 08 i 11 są
opcjonalne i wolno je rozpocząć wyłącznie po spełnieniu warunków opisanych w
ich plikach. Nadrzędne bramki badawcze i zasady bezpieczeństwa znajdują się w
[`AGENTS.md`](../AGENTS.md).
