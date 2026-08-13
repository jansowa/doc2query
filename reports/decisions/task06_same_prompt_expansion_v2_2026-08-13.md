# Task 06 — druga kohorta same-prompt (ADR v2, 2026-08-13)

## Kontekst

Bramka różnorodności same-prompt (ADR
[`task06_same_prompt_diversity_gate_v1.md`](task06_same_prompt_diversity_gate_v1.md))
przepuściła 362/500 grup kohorty `same_prompt_expansion_v1`. Zamrożona
procedura Task 06 dopuszcza najwyżej jedną tentative preference na prompt, więc
ta kohorta daje maksymalnie 362 pary — mniej niż 500 par wymaganych przez
rozwojową bramkę ślepego audytu dual-LLM i mniej niż 1000 par wymaganych przed
finalnym DPO. Progów bramki nie wolno poluzować po zobaczeniu wyniku, a
zakończonego runu v1 nie wolno powtarzać ani nadpisywać.

Właściciel delegował wybór ścieżki naprawy (komenda 2026-08-13: „Podejmij sam
decyzję”).

## Decyzja

Deficyt uzupełnia **nowa kohorta 500 pasaży** (`same_prompt_expansion_v2`),
rozłączna klastrowo z całą dotychczasową pracą Task 06, z tym samym kontraktem
„jeden prompt, osiem odpowiedzi”, ale z **szerszym rozkładem decodingu**.
Kontrakt zamraża
`configs/preferences/task06_same_prompt_expansion_v2.yaml`.

Uzasadnienie wyboru spośród dwóch dozwolonych osi naprawy:

- nowe grupy zwiększają liczbę **różnych pasaży**, a nie liczbę próbek tego
  samego promptu; dla DPO wartościowa jest różnorodność kontekstów, nie gęstsze
  próbkowanie jednego rozkładu;
- większe K na istniejących 500 promptach podniosłoby jedynie odsetek grup
  przechodzących bramkę (sufit to nadal 500 par) i wymagałoby dopisywania
  kandydatów do zamrożonego artefaktu v1;
- szerszy decoding jest jednak przejęty z osi „większe K” i zastosowany w nowej
  kohorcie: `duplicate_rate` v1 rósł głównie w slotach o temperaturze ≤ 0.5.

### Kohorta (quality-blind, ID freeze przed materializacją tekstu)

- źródło: wyłącznie frozen `train` v1 (`doc2query_train.parquet`,
  `dedup_map.parquet`, `split_manifest.json` z `positive_canonical_leakage=0`);
  ścieżki i SHA-256 są przypięte przez read-only design
  `task06_candidate_execution_design_v1.yaml`
  (`sha256=e332793388a376b461c1469e0f8bbc012433e54d8781fb4adc992ee3100f6f23`,
  którego nie wolno modyfikować — pinuje go identity zakończonego pilota);
- wykluczenia klastrowe: 49367 pasaży wspólnej selekcji 50k SFT W06/D01,
  32 klastry smoke (`cohort.ids.json` sha `f2402cb4…`) oraz 512 klastrów pilota
  (`cohort.ids.json` sha `75f43b9c…`, nadzbiór 500 pasaży v1);
- wybór: `sha256_cluster_first_quality_blind_v2`, seed `20260814`, wyłącznie po
  ID (`pair_id`, `example_id`, `doc_id`, `cluster_id`); żadne pole jakości,
  score, margin ani wynik bramki nie wpływa na wybór;
- wymóg ≥10 hard negatywów na rekord, unikalność klastra i `example_id`;
- kolejność: ID freeze → weryfikacja rozłączności → materializacja tekstu →
  atomowy manifest z licznikami i hashami.

### Decoding (osiem slotów, ten sam prompt)

| slot | temperature | top_p | seed |
|---|---|---|---|
| 0 | 0.6 | 0.97 | 7601 |
| 1 | 0.8 | 0.97 | 7602 |
| 2 | 1.0 | 0.97 | 7603 |
| 3 | 1.2 | 0.92 | 7604 |
| 4 | 0.6 | 0.92 | 7611 |
| 5 | 0.8 | 0.92 | 7612 |
| 6 | 1.0 | 0.92 | 7613 |
| 7 | 1.2 | 0.97 | 7614 |

Cztery kontrolki D01 pozostają identyczne jak w v1 i są przypisywane
round-robin po quality-blind uporządkowaniu, po jednej na pasaż. `max_new_tokens
= 64`, batch generacji i scoringu = 8, adapter i base revision bez zmian
(D01 4.5B, `runs/D01-4.5B-STYLE-50K-S42/adapter`).

Ryzyko: wyższe temperatury mogą obniżyć `format_valid` i jakość kandydatów
(w v1 `valid_rate=1.0`). Jest to akceptowane świadomie — degradacja formatu
odsiewa się w scoringu i w filtrach jakości par, a bramka różnorodności nie jest
metryką jakości. Odsetek nieprawidłowych formatów będzie raportowany.

### Bramka i granice

- Bramka różnorodności działa na v2 z **niezmienionymi** progami polityki
  `task06_same_prompt_diversity_gate_v1.yaml`
  (`sha256=ddbd9c8334e397611da4a639508689089b96ddeacf001cd7966dc5ec96d9f2c7`).
  Nie wolno kalibrować progów pod tę kohortę.
- Pary z v1 i v2 są łączalne, bo kohorty są rozłączne klastrowo; kontrakt
  „żaden passage/near-duplicate z preference dev/test w preference train”
  pozostaje w mocy.
- `tentative_pair_build_authorized=false`. Budowa par wymaga osobnego ADR
  zamrażającego politykę `chosen/rejected` (wagi, progi, kalibracja komponentów,
  preflight selekcji) — `dpo_pair_selector` ma nadal status
  `not_frozen_not_authorized`.
- Zamrożony run v1 pozostaje nietknięty; v2 zapisuje do nowego katalogu
  `artifacts/task06/same_prompt_expansion_v2`.
- `task07_training_authorized=false`, `task09_authorized=false`,
  `final_tests_used=[]`. Testy finalne pozostają zamknięte.

## Kolejność wykonania

1. **CPU, wykonane w tej sesji:** zamrożenie kohorty v2 (ID freeze +
   materializacja + manifest).
2. **GPU, ~1 h, oczekuje na wolne GPU:** generacja 500 × 8 = 4000 kandydatów,
   potem scoring primary/shadow/corpus (v1: 1510 s + 1308 s, peak VRAM 3.43 GB;
   v2 może być nieco dłuższy przy wyższych temperaturach).
3. **CPU:** bramka różnorodności na v2, raport odsetka odrzuceń.
4. **CPU, osobny ADR:** zamrożenie polityki `chosen/rejected` i kalibracji
   komponentów, potem preflight selekcji.
5. **Sieć, bez GPU:** budowa tentative pairs i ślepy audyt dual-LLM (Groq),
   ≥500 par.

Etapy 4–5 nie są tym ADR autoryzowane.
