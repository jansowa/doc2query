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
| [02](02_reranker_and_reward_proxies.md) | Zamrożone rerankery i proxy nagrody | `IMPLEMENTED` | Integracja, kalibracja, reward proxies i testy są gotowe; base-ranknet i primary v3 zmierzyły panel 100 generacji W05. Pozostał benchmark primary/shadow na dev/test z hard negative'ami. |
| [03](03_sft_qlora_baselines.md) | Baseline'y SFT/QLoRA | `IMPLEMENTED` | Run W06 4.5B Instruct/50k zakończył 3125 kroków w 8 h 14 min. Jego wyniki są diagnostyczne, nie selekcyjne, dopóki Task 04 nie dostarczy Harness v1.1. Po v1.1 wykonać P-05: S00 zero/few-shot, S07 plT5/mT5 i małą macierz probe, następnie P-06: czyszczenie/ważenie par na 1.5B. Nie rozpoczynać kolejnej kampanii 4.5B przed tą bramką. |
| [04](04_evaluation_harness.md) | Harness ewaluacyjny | `IN PROGRESS` | P-01/P-02 oraz kod P-03 są zaimplementowane. `probe-negatives-v1` ma HN0/HN0+filter/HN1, `drop/demote/keep+log`, mockowane testy, provenance i twardą zgodność porównań. Audyt nie znalazł dev-only artefaktu progu Task 02 ani zbudowanego indeksu BM25, więc konfiguracja blokuje się przed ładowaniem modelu, a diagnostyczny W05 nie został uruchomiony. Nie użyto finalnych testów do strojenia. Następnie: utworzyć i przypiąć oba artefakty, wykonać wyłącznie W05 sensitivity, zapisać ADR przy istotnym lub nierozstrzygalnym wyniku, potem P-04. Do tego czasu zakaz porównawczych probe, eksperymentów D i Task 06. |
| [05](05_controlled_diversity_and_multiquery.md) | Kontrolowany styl, focus i multi-query | `TODO` | Kod taksonomii i kontrolek może powstawać równolegle, ale eksperymenty D00–D12 wymagają ukończonego Harness v1.1 z Task 04. |
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

1. **Teraz — Task 04 / Harness v1.1:** P-01/P-02 i implementacja P-03 są
   gotowe. P-03 pozostaje pomiarowo zablokowane do czasu dev-only kalibracji
   Task 02, indeksu BM25 i diagnostycznego W05. Następnie P-04. Niedomknięte
   W05/P-04 są blokerami pierwszego porównawczego probe.
2. **Brama tanich baseline'ów — Task 03:** P-05 i P-06 na 1.5B. W06 pozostaje
   eksploracyjnym dowodem wykonalności 4.5B/8 GB, a nie zgodą na dalszą
   kampanię skali.
3. **Równolegle tylko implementacja Task 05:** można przygotowywać schematy,
   `form`/`intent`, evidence i selektory z P-07, lecz D00–D12 czekają na v1.1.
4. **Po bramce:** eksperymenty Task 05, potem P-08 w Task 06 i Task 07.
5. **Kampania:** Task 09 dopiero po baseline'ach, sensitivity check negatywów
   i pełnej bramce HN; Task 10 dopiero po finalnym ADR.
6. **Opcjonalne:** Task 08, P-09 i Task 11 wyłącznie po własnych bramkach.

Najbliższy jednoznaczny punkt wejścia dla kolejnej sesji to odblokowanie
pomiarowej części P-03: dev-only artefakt progu Task 02, projektowy indeks BM25
i wyłącznie diagnostyczny W05 HN0/HN0+filter/HN1. Szczegóły i warunki są w
[`Task 04`](04_evaluation_harness.md) oraz blockerze
`reports/blockers/task04_p03_w05_sensitivity.md`. P-04 i porównawcze probe
pozostają po tej bramce.

## Kolejność bazowa

Po domknięciu pakietu naprawczego kolejność bazowa pozostaje
`00 → 01 → 02 → 03 → 04 → 05 → 06 → 07 → 09 → 10`.
Task 04 może częściowo powstawać równolegle z Task 03. Taski 08 i 11 są
opcjonalne i wolno je rozpocząć wyłącznie po spełnieniu warunków opisanych w
ich plikach. Nadrzędne bramki badawcze i zasady bezpieczeństwa znajdują się w
[`AGENTS.md`](../AGENTS.md).
