# Task 09 — Kampania eksperymentalna i wybór strategii

> [Centralny rejestr zadań i statusów](README.md). Każda zmiana statusu lub zakresu tego zadania musi aktualizować rejestr w tym samym commicie.

## Status

`BLOCKED`

Aktualizacja 2026-08-13 (rozszerzenie specyfikacji, decyzja właściciela):
kampanię poprzedza obowiązkowy Etap 0 — walidacja predykcyjna probe
(M-01/E15); probe w reżimie mieszanym (M-02) i ocena trybu produkcyjnego
(M-05) są obowiązkowe dla finalistów. Skala: 4.5B jest głównym finalistą na
sprzęcie lokalnym 8/16 GB; 7B+ wyłącznie na sprzęcie zewnętrznym przez
przenośny pipeline. Definicje M-01–M-05: AGENTS.md §9.2.

Aktualizacja 2026-08-12 (Task 06 execution design): powstał wyłącznie
prospektywny, model-free projekt pilota Task 06. Właściciel rozstrzygnął 512
pasaży, natural-dev calibration i dual-LLM audit 500 par zamiast ręcznego
panelu; Groq ma przypięte limity i resumable quota stop. Preflight czeka na
osobną komendę operatorską; nie wykonano generacji, scoringu, preferencji ani
Task 07. Task 09 nadal jest `BLOCKED`, bez otwierania finalnych testów
(`final_tests_used=[]`).

Aktualizacja 2026-08-12: właściciel zatwierdził handoff D01b Hybrid do
projektowania Task 06 i D01 controlled 4.5B jako przyszły start Task 07.
Model-free preflight przeszedł, ale jawnie nie autoryzuje generacji, scoringu,
DPO ani Task 09. Kampania pozostaje `BLOCKED` do rzeczywistych wyników Task
06–07 i późniejszego Pareto review; finalne testy pozostają zamknięte.

Gotowy jest wyłącznie deterministyczny, model-free fundament przedkampanijny.
Wersjonowany `ExperimentEvidenceManifest` zapisuje identyfikatory eksperymentu,
ramienia i etapu, status i seed runu, commit oraz fingerprint konfiguracji,
dataset/split/cohort, pełne tożsamości modelu, adaptera i tokenizera,
pięciowymiarowy budżet z niezmiennikiem Task 04, fingerprint recepty probe,
hashe/liczniki/provenance artefaktów oraz rozdzielone metryki intrinsic,
probe/extrinsic, human i cost z kierunkiem, CI i liczebnością. Kontrakt wymaga
`final_tests_used=[]`.

`CampaignEvidenceRegistry` konsumuje wyłącznie jawnie przekazane manifesty.
Weryfikuje SHA-256, record counts, fingerprinty konfiguracji, budżetu, stosu
modelowego i deskryptorów artefaktów oraz ich provenance. Odrzuca duplikat
`experiment_id/arm_id/seed`, raportuje brakujące seedy, metryki, CI,
liczebności, human evidence i wymagane role artefaktów. Seedy agreguje tylko
przy zgodnym configu z wyłączeniem pola seed, dataset/split/cohort, budżecie,
probe recipe, stosie modelowym i definicjach metryk. Między ramionami fail-closed
wykrywa drift datasetu, splitu, kohorty, budżetu, probe recipe, commitu i
definicji metryk; nie uśrednia ani nie szereguje nieporównywalnych danych.

Czysta funkcja Pareto respektuje kierunki `min`/`max`, nie scalarizuje metryk,
nie wybiera zwycięzcy i przy niepełnym lub nieporównywalnym evidence zwraca
`evidence_incomplete_not_ranked`. Cienki skrypt
`scripts/build_task09_evidence_registry.py` nie generuje komend treningowych
ani ewaluacyjnych. Bundle ma osobny wersjonowany manifest, deterministyczny
payload, odmawia nadpisania i jest publikowany atomowo przez staging oraz
`os.replace`; staging jest usuwany po błędzie. Maksymalny status to
`registry_ready_for_future_stage_review_no_selection`, a flagi kampanii,
ładowania modelu, treningu, ewaluacji i selekcji pozostają `false`.

Interfejs nie przyjmuje ścieżki finalnego testu. Jawnie zakazane ścieżki
final-test są odrzucane przed jakimkolwiek odczytem; nie otwarto i nie użyto
testów finalnych. Syntetyczne testy CPU obejmują integralność, porównywalność,
drifty, kompletność evidence, Pareto min/max, brak scalar winnera, atomową
publikację oraz brak zależności modelowych.

Task nadal oczekuje na wcześniejsze etapy w zakresie dopuszczonym przez bramki.
Wcześniejszy 1.5B `dev_confirm` zakończył się `non_inferior_only`, lecz osobny
4.5B scale-interaction screen oraz zewnętrzny TriviaQA confirm rozstrzygnęły
interakcję ze skalą. Confirm na 8000 query i seedach 42/43/44 zakończył się
`rc=0`: Hybrid-minus-W06 `corpus_ndcg_at_10` wynosi `+0.0478666`, 97.5% CI
`[+0.0450118, +0.0508263]`, a wszystkie guardraile przeszły. Hybrid ma status
`eligible_for_finalist_freeze_review` i jest zachowany do review. W06 seed 43
nie zbiegł; analiza seedów 42+44 nadal utrzymuje kierunek i próg, ale pozostaje
post-hoc caveatem stabilności. Sam confirm nie autoryzuje Task 09. Właściciel
zaakceptował później handoff do projektowania Task 06; nadal brakuje
prospektywnego execution ADR oraz rzeczywistych wyników Task 06–07.
Nie dołączono żadnych rzeczywistych wyników Task 03–07, nie wykonano kampanii,
successive halving, Pareto review, decyzji continue/stop, promocji, wyboru
finalistów ani finalnego ADR. Harness v1.1 P-01…P-04, P-05 i pełna dev-only
ablacja polityki hard negative'ów z Task 04 są już rozstrzygnięte; nie otwarto
testów finalnych.
P06-T zostało świadomie anulowane decyzją właściciela i nie jest już bramką.
P-06 mass rescoring i warianty
drop/weighted według lokalnego marginu są `SUPERSEDED`, nie są zależnością.

## Cel

Przeprowadzić eksperymenty w kolejności minimalizującej koszt i wybrać procedurę finalną na podstawie dowodów.

## Zależności

Taski 03–08 w zakresie dopuszczonym przez bramki.

## Zasada sekwencyjności

Nie uruchamiaj pełnej macierzy kartezjańskiej. Stosuj successive halving:

1. 10k przykładów / 1 seed;
2. odrzuć warianty wyraźnie słabe;
3. 50k / 2–3 seedy;
4. probe embedder;
5. 100k–500k tylko finalistom.

Budżet porównuj w tokenach i krokach, nie tylko liczbie przykładów.
Kontrakt budżetowy z Task 04 porównuje jednocześnie tokeny, pary, unikalne
pasaże i K query/pasaż oraz wersję recepty probe.

## Minimalna kolejność

### Etap 0 — walidacja predykcyjna probe (M-01/E15)

Przed decyzjami selekcyjnymi kampanii wykonaj jednorazowy trening średniej
skali (rząd 50–100 tys. par syntetycznych, realistyczna receptura embeddera
docelowego, prospektywny ADR) dla dwóch wariantów, które probe wyraźnie
rozdzielił (np. dane W06 vs procedura hybrid). Celem jest kalibracja
transferu rankingu probe na skalę. Wynik nie promuje wariantu; określa
jedynie zaufanie do probe jako instrumentu selekcji i może skorygować próg
praktyczny P-04 wyłącznie nowym prospektywnym ADR.

### Etap 1 — pipeline

- E00 prompting;
- E01 1.5B smoke;
- E02 1.5B 10k.

### Etap 2 — model i SFT

- 1.5B vs 4.5B;
- 4.5B base vs instruct;
- ordinary vs balanced vs weighted;
- wybór max length i LoRA target modules.

### Etap 3 — kontrolki

- style only;
- focus only;
- style + focus;
- K independent vs multi-query JSON;
- coverage-aware selection.

### Etap 4 — preference

- best-of-N offline;
- continued SFT;
- DPO;
- różne typy rejected.

### Etap 5 — RL opcjonalny

- tylko po formalnej bramce.

### Etap 6 — skala

- pełne 500k na najlepszym 4.5B (sprzęt lokalny 8/16 GB);
- 7B standard vs 7B PL na identycznym subset — wyłącznie na sprzęcie
  zewnętrznym, przez przenośny pipeline z jawnymi parametrami sprzętowymi;
- finalny 7B na większych zasobach tylko, gdy zmierzona przewaga uzasadnia
  koszt (wymaga wykonanego M-01 i przewagi w macierzy Pareto).

## Metryka wyboru

Utwórz ranking wielokryterialny, ale nie ukrywaj Pareto frontu. Priorytety:

1. probe embedder nDCG@10/MRR/Recall z CI;
2. ugruntowanie, możliwość odpowiedzi z pasażu i source retrieval;
3. diversity/focus coverage;
4. kopiowanie względem naturalnego rozkładu;
5. human preference;
6. koszt queries/s i VRAM, w tym koszt trybu produkcyjnego bez selektora
   (M-05).

Nie sumuj bezrefleksyjnie wszystkiego do jednego score. Użyj score tylko do wstępnej selekcji, a finalną decyzję opisz w ADR.
Główny wynik musi pochodzić z natywnego polskiego holdoutu; poprawa wyłącznie
na tłumaczonym teście nie wystarcza bez jawnego ADR.

## Eksperymenty opcjonalne po bramkach

- MIX0–MIX4 (100/75/50/25/0% natural) — co najmniej jeden punkt mieszany
  (M-02, np. 50/50) jest obowiązkowy dla każdego finalisty przy dopasowanym
  budżecie; pełna siatka pozostaje opcjonalna;
- probe recipe v2 z GPL/MarginMSE tylko jako osobna, pełna replikacja
  porównań dla 2–3 finalistów;
- kontrfaktyczne negatywy dopiero po stabilnym corpus-mined HN;
- noisy self-training wyłącznie po osobnej bramce;
- drugi backbone probe jako potwierdzenie finalistów.

## Kryteria eliminacji

Odrzuć wariant, gdy:

- invalid rate jest wysoki;
- source Recall@1 istotnie spada;
- overlap poprawia się tylko kosztem ugruntowania lub możliwości odpowiedzi z pasażu;
- diversity wynika z halucynacji;
- focus controls są ignorowane;
- probe embedder przegrywa z prostszym wariantem;
- koszt wzrasta bez korzyści;
- wynik zależy od jednego seeda;
- automatyczny reward nie zgadza się z human panel.

## Raporty

Po każdym etapie twórz:

- `reports/stage_<n>_summary.md`;
- tabelę runów;
- decyzję `continue/stop`;
- listę hipotez potwierdzonych/obalonych;
- rekomendację kolejnego eksperymentu;
- szacowany koszt następnej fazy.

## ADR finalny

Utwórz `docs/adr/00xx-final-training-strategy.md` zawierający:

- wybrany model;
- SFT/DPO/GRPO;
- dane i filtering;
- kontrolki;
- liczbę query na passage;
- selektor kandydatów;
- parametry generacji;
- dowody intrinsic, extrinsic i human;
- odrzucone alternatywy;
- znane ograniczenia.

## Kryteria akceptacji

- każdy finalista ma probe embedder evaluation;
- każdy finalista ma probe w reżimie mieszanym natural+synthetic (M-02);
- każdy finalista ma zmierzony tryb produkcyjny bez selektora (M-05);
- walidacja predykcyjna probe (M-01/E15) jest wykonana i zaraportowana przed
  decyzjami selekcyjnymi;
- porównania używają identycznych testów i fingerprintów;
- co najmniej kluczowe porównania mają CI i wiele seedów;
- istnieje jawna decyzja, czy DPO i RL były warte kosztu;
- wybór 4.5B vs 7B jest oparty na wyniku, nie założeniu.
