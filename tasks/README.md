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
| [02](02_reranker_and_reward_proxies.md) | Zamrożone rerankery i proxy nagrody | `IMPLEMENTED` | Integracja, reward proxies i testy są gotowe; primary zmierzył pełny frozen dev, a query-macro próg Youdena `possible_false_negative` jest przypięty bez użycia testu. Pozostał pełny benchmark primary/shadow na dev/test z hard negative'ami. |
| [03](03_sft_qlora_baselines.md) | Baseline'y SFT/QLoRA | `IMPLEMENTED` | Ukończono siedem ramion base oraz I01/I03 Instruct/10k. Zredukowany probe P-05: mixed 50/50 `0.057568` nDCG@10, gold `0.052762`, synthetic-only `0.048618`; mixed-minus-gold CI `[+0.000876,+0.008692]` nie osiąga progu `+0.01`, więc żadne ramię nie może być `eligible` do `dev_confirm`. Round-trip@20 i format zmierzono; pozostał sentence-level guardrail i finalny raport engine. I02/I04/I05 są `DEFERRED`, S00/S07 `required_unexecuted`; testy finalne są nieotwarte. |
| [04](04_evaluation_harness.md) | Harness ewaluacyjny | `IMPLEMENTED` | P-01–P-04 są gotowe, a trzy probe P-05 `dev_screen` seed 42 ukończone na wspólnym budżecie i 6598 query. Exact round-trip@20 wynosi gold/mixed/synthetic `0.116854/0.130797/0.112307`, a format valid `1.0`. Primary CI żadnego wariantu nie osiąga `+0.01`, więc `dev_confirm` pozostaje zamknięty. Brakuje kanonicznego `sentence_level_source_hit`; fail-closed engine zwraca `incomplete`. Wznawialny runner GPU i assembler pełnego raportu są gotowe; testów finalnych nie otwarto. |
| [05](05_controlled_diversity_and_multiquery.md) | Kontrolowany styl, focus i multi-query | `IMPLEMENTED` | Gotowe są kontrakty i kod CPU: `form`/`intent`, evidence i F0–F3, controlled SFT/inference, retry/deduplikacja, multi-query JSON, concept coverage oraz top-N/MMR/coverage-aware. D00–D12, audyty 500/200, kalibracja per domena, human check i probe z CI pozostają niewykonane; mogą ruszyć dopiero po bieżącej kolejce i pierwszych probe zgodnych z P-04. |
| [06](06_candidate_scoring_and_preference_data.md) | Scoring kandydatów i dane preferencyjne | `TODO` | Wymaga stabilnego checkpointu SFT, ukończonego Harness v1.1 oraz Task 02 i 05. |
| [07](07_dpo_training.md) | DPO i continued-SFT control | `TODO` | Wymaga danych preferencyjnych z Task 06. |
| [08](08_grpo_multiobjective_rl.md) | Wielokryterialny GRPO/RL | `OPTIONAL / BLOCKED` | Uruchamiać wyłącznie po spełnieniu bramki i zapisaniu decyzji `reports/decisions/enable_grpo.md`. |
| [09](09_experiment_campaign.md) | Kampania eksperymentalna | `BLOCKED` | Wymaga Harness v1.1, baseline'ów P-05/P-06, pełnej bramki hard negative'ów i wcześniejszych etapów dopuszczonych przez kontrakt statystyczny. |
| [10](10_final_scaleup_inference_release.md) | Finalny trening, inference i release | `BLOCKED` | Wymaga wyników Task 09 i zatwierdzonego finalnego ADR. |
| [11](11_judge_robustness_audit.md) | Audyt odporności sędziego i fallbacki | `OPTIONAL` | Późny eksperyment badawczy; nie obejmuje treningu ani dostrajania rerankera. |

## Operacyjna kolejność po audycie

Ten rejestr jest jedynym operacyjnym źródłem kolejności i statusów.
[`docs/plan_poprawek_po_audytach.md`](../docs/plan_poprawek_po_audytach.md)
pozostaje zapisem przesłanek i identyfikatorów P-xx, ale nie jest równoległym
backlogiem. Zakres P-xx został przeniesiony do wskazanych plików zadań.

1. **Teraz — domknięcie P-05 dev screen:** probe seed 42, round-trip@20 i
   format są zmierzone. Pozostał GPU-only `sentence_level_source_hit`, scalenie
   pełnego raportu i ponowne uruchomienie engine. Primary CI już wyklucza status
   `eligible` dla obu wariantów, więc `dev_confirm` pozostaje zamknięty. Pełna
   bramka HN przed Task 09 nadal pozostaje niewykonana.
2. **Po guardrailach — ewentualny dev confirm:** stosować wyłącznie kontrakt
   `task04-p04-v1`, wspólny budżet i dev-only successive halving. Testy finalne
   wolno otworzyć raz dopiero po zamrożeniu finalistów. W06 pozostaje
   eksploracyjnym dowodem wykonalności 4.5B/8 GB, a nie zgodą na dalszą
   kampanię skali.
3. **Task 05:** implementacja CPU jest gotowa; D00–D12 pozostają niewykonane
   do zakończenia kolejki baseline'ów i pierwszych probe zgodnych z P-04.
4. **Po bramce:** eksperymenty Task 05, potem P-08 w Task 06 i Task 07.
5. **Kampania:** Task 09 dopiero po baseline'ach, sensitivity check negatywów
   i pełnej bramce HN; Task 10 dopiero po finalnym ADR.
6. **Opcjonalne:** Task 08, P-09 i Task 11 wyłącznie po własnych bramkach.

Najbliższy jednoznaczny punkt wejścia to uruchomienie wznawialnego
`scripts/run_p05_guardrails.sh`, który domierzy tylko sentence-level guardrail,
zbuduje pełny raport i uruchomi engine. Nie należy ponownie
uruchamiać `scripts/run_p05_dev_screen.sh`: trzy wyniki są kompletne, a runner
je tylko zweryfikuje i pominie. `dev_confirm` pozostaje zamknięty do wyniku
fail-closed decision engine. I02/I04/I05 pozostają odroczone, S00/S07 wymagane i
niewykonane, a D00–D12, Task 06 i wszystkie testy finalne pozostają poza
bieżącym krokiem.

## Kolejność bazowa

Po domknięciu pakietu naprawczego kolejność bazowa pozostaje
`00 → 01 → 02 → 03 → 04 → 05 → 06 → 07 → 09 → 10`.
Task 04 może częściowo powstawać równolegle z Task 03. Taski 08 i 11 są
opcjonalne i wolno je rozpocząć wyłącznie po spełnieniu warunków opisanych w
ich plikach. Nadrzędne bramki badawcze i zasady bezpieczeństwa znajdują się w
[`AGENTS.md`](../AGENTS.md).
