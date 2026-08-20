# Task 06 — Generacja kandydatów, scoring i dane preferencyjne

> [Centralny rejestr zadań i statusów](README.md). Każda zmiana statusu lub zakresu tego zadania musi aktualizować rejestr w tym samym commicie.

## Status

`IN PROGRESS`

Aktualizacja 2026-08-17 (kierunek polityki par v2, decyzja właściciela):
właściciel — po zapoznaniu się z wynikami dnia 1 audytu dual-LLM — zatwierdził
kierunkowo przejście z porządkowania par marginesem primary na **pary
zakotwiczone w osiach defektów**: `rejected` ma zmierzony, nazwany defekt
(oś A: nieodpowiadalność/halucynacja; oś B: łatwość leksykalna/kopiowanie;
oś C: niezgodność z żądanym focusem), a `chosen` jest od niego wolny; monotonia
(problem populacyjny) jest jawnie poza parami i mieszka w kontrolkach, M-05 i
set-level nagrodzie GRPO. Pełna specyfikacja z zadaniami V2-00…V2-07,
zależnościami, kosztami i prerejestrowanymi predykcjami:
[`task06_defect_anchored_pairs_v2_spec_2026-08-17.md`](../reports/plans/task06_defect_anchored_pairs_v2_spec_2026-08-17.md).
Specyfikacja jest planistyczna: progi zamrożą dopiero prospektywne ADR-y
(V2-01 sędzia odpowiadalności, V2-03 polityka par v2) po pełnym wyniku audytu
v1; polityka v1/v1.1, pary, eksport i audyt pozostają zamrożonym punktem
odniesienia. `task07_training_authorized=false`, `final_tests_used=[]`.

Realizacja rozpoczęta tego samego dnia (raport:
[`task06_defect_inventory_and_focus_v2_2026-08-17.md`](../reports/measurements/task06_defect_inventory_and_focus_v2_2026-08-17.md)):

- **V2-00 wykonane** (`src/doc2query/preferences/defect_inventory.py`,
  `scripts/run_task06_defect_inventory.py`, 10 testów CPU): na 25992 grupach
  `eligible` (reprezentanci bramki, loadery v1 z pinowaniem SHA-256) podaż par
  defektowych wynosi: **oś A 17669 grup** (68,0%; defekt = brak rt@100 —
  naturalna podaż wystarcza bez konstruowanych), **oś B 4857 grup** przy
  kandydującym cięciu p75 `content_jaccard ≥ 0,0857` (1900 przy p90; wybór
  cięcia należy do ADR V2-03), **oś C 0 grup** na starych etykietach focus,
  czysty `chosen` w 21102 grupach (81,2%). Znalezisko:
  `entity_preservation` jest w tych kohortach **stałą 1,0 z konstrukcji** —
  scoring używał `SimplePolishNormalizer`, który zawsze zwraca `entities=()`,
  więc komponent nie jest tam nawet detektorem halucynacji; rola ta wymagałaby
  relabelingu backendem spaCy (decyzja przy V2-03), a oś A opiera się na
  round-tripie i sędzim V2-01.
- **V2-02 zaimplementowane i zmierzone; kryterium akceptacji NIEDOWIEZIONE**
  (`src/doc2query/data/focus_labels_v2.py`, `scripts/validate_task06_focus_v2.py`,
  11 testów CPU): splitter `focus-v2:pl-abbrev-v1` eliminuje wszystkie
  pseudo-zdania (8→0 na 180 pasażach, 716→659 zdań), ale abstencja focusa
  spada tylko z 25,6% do 25,0% (`good_specific`), a sukces minimalnie spada
  (0,622→0,600) — wąskim gardłem jest scorer leksykalny, nie segmentacja.
  Formuła scoringu v2 jest bajtowo zgodna z v1, nic zamrożonego nie zmieniono,
  progu `minimum_confidence` nie dostrajano pod pomiar. Oś C pozostaje
  zablokowana do decyzji właściciela przy V2-03: mocniejszy przypisywacz
  (wariant rerankerowy, GPU) albo rezygnacja z osi C w pierwszym wydaniu v2.
- **V2-01: harness sędziego odpowiadalności zaimplementowany, nie uruchomiony**
  (`src/doc2query/preferences/answerability_judge.py`,
  `scripts/run_task06_answerability_judge.py`,
  `configs/preferences/task06_answerability_judge_v1.json` ze statusem
  `draft_pending_weight_pinning`, 11 testów CPU z fałszywym backendem). Runner
  jest fail-closed: odmawia pracy bez przypiętego digestu wag i przy
  niezgodności digestu z lokalnym backendem; `temperature=0`, deterministyczny
  seed, trwały journal per item z resume, `uncertain` blokuje rolę `chosen`,
  ale nie liczy się jako defekt; walidacja odrzuca sędziego z rodziny
  generatora (Bielik). Kalibracja jest gotowa programowo: itemy per (para,
  strona) z etykiet `answerable_a/b` audytu Groq oraz klasy konstrukcyjne
  korpusu walidacyjnego (`ungrounded` → `no`), z analizą zgodności i CI.
  **Blokada operatorska**: maszyna bazowa ma 8 GB VRAM i ollama bez modeli;
  Q4 27B (~17 GB, 53 GB wolnego na dysku) wymaga pobrania wag i przypięcia
  digestu — decyzja właściciela. Progi akceptacji kalibracji zamrozi ADR V2-01
  razem z digestem, przed uruchomieniem kalibracji.
- Krok 0 (dokończenie audytu v1: 128 + 43 requesty) czeka na odnowienie
  dziennych budżetów Groq o 00:00 UTC; cronów zgodnie z ograniczeniami nie
  zainstalowano.

Aktualizacja 2026-08-17 (wieczór, kolejne decyzje właściciela i proxy
odpowiadalności). Właściciel zawęził pierwsze wydanie polityki v2: **oś C wypada**
(wraca w v2.1 po mocniejszym labelerze focusa), **konstruowanych rejected (V2-04)
nie budujemy** (inwentarz V2-00 pokazał wystarczającą podaż naturalną), kohorty
autoryzowane pozostają v1+v2+v3 jak w v1.1, a rolę niedostępnego sędziego 27B w
osi A miało tymczasowo przejąć **proxy odpowiadalności** skalibrowane na
etykietach `answerable_a/b` audytu Groq.

- **Krok 0 ZAMKNIĘTY 2026-08-19: audyt v1 jest kompletny (500/500 par,
  `status: complete`).** Domknięcie zajęło trzy okna dziennych budżetów Groq
  (dzień 2 dopełnił do 249/250 i 250/250, dzień 3 dorzucił jeden brakujący
  request). Trzy główne wnioski utrzymały się, jeden osłabł ilościowo: zgodność
  automatu z `gpt-oss` **0,7179** (CI [0,656; 0,779]) i z `qwen` **0,7080**
  (CI [0,661; 0,755]) wobec zgodności **między modelami 0,8793**
  (CI [0,833; 0,925]) — CI nie nachodzą, więc niezgodność z porządkiem
  marginesowym nadal nie jest szumem sędziów, ale luka jest węższa niż po dniu 1
  (16 pp zamiast 22 pp). Bramka fail-closed wyklucza **378/500 par (75,6%)**,
  czyli akceptowalnych jest **122** — o rząd wielkości poniżej bramki 1000 par,
  co jest twardym argumentem liczbowym za polityką v2. Remisy dominują
  (`gpt-oss` 51,8% + 9,2% `both_bad`, `qwen` 32,2%) i są **częstsze** w pasmie
  wysokiej pewności u obu modeli. Nieodpowiadalne `chosen`: **16,6%** i **18,8%**;
  round-trip nadal nie różnicuje odpowiadalności (70,6% vs 66,1%; 59,5% vs 56,5%).
  Pasma marginesu: `qwen` płaski (0,693/0,730/0,700), więc podniesienie
  `min_margin_gap` nadal bez uzasadnienia — progów nie zmieniono. Nowy przypis:
  `gpt-oss` uznał 2/500 stron `chosen` za `format_valid=false` (`qwen` 0/1000) —
  0,4% u jednego sędziego, `format.py` bez zmian. Slice'y konsensusu pokazują
  spójny kierunek: im bardziej defektowy `rejected`, tym wyższa zgodność
  (`weak_corpus_round_trip` 0,909, `judge_rank_disagreement` 0,900,
  `lower_content_jaccard` 0,856 vs `lower_primary_margin` 0,797) — niezależne
  wsparcie dla kotwiczenia par w defektach. Baseline dla predykcji ADR V2-03:
  `consensus_supports` 24,4%, `consensus_contradicts` 6,2%, nieodpowiadalne
  `chosen` 16,6%/18,8%, remisy 51,8%/32,2%, wykluczone 75,6%. Raport: sekcja
  „WYNIK PEŁNY” w
  [`task06_dual_llm_pair_audit_2026-08-17.md`](../reports/measurements/task06_dual_llm_pair_audit_2026-08-17.md)
  (liczby dnia 1 zachowane, nie nadpisane).
- **ADR V2-01 zamrożony 2026-08-19 i sędzia przeniesiony na drugą maszynę.**
  Prospektywny ADR
  [`task06_answerability_judge_v1.md`](../reports/decisions/task06_answerability_judge_v1.md)
  przypina sędziego i zamraża kryteria **przed** jakimkolwiek werdyktem: **K1**
  zgodność z konsensusem Groq (`accuracy ≥ 0,85` i `balanced_accuracy ≥ 0,75` na
  817 stronach konsensusowych; baza klasy większościowej 0,7846, sufit szumu
  0,817), **K2** dwustronne sanity na klasach z konstrukcji (`ungrounded` → `no`
  w ≥ 0,80 przy ≤ 0,20 odrzuceń w `good_specific` i `good_alternative`), **K3**
  abstencja ≤ 0,25. Odstępstwo od specyfikacji zapisane jawnie: maszyna bazowa
  (8 GB VRAM, 16 GB RAM) nie serwuje 27B Q4 uczciwie — zmierzone Q3_K_S działa
  (3,4 s/item, 1045 itemów/h, parsowalność 8/8), ale jest gorszym punktem
  jakościowym, więc sędzią jest **Qwen3.8-27B FP8** na endpoincie vLLM operatora
  (adres wyłącznie parametrem CLI, nie w repo). Wagi Q3_K_S
  (digest `418bbc5c98e5…`) **nie wyprodukowały żadnego werdyktu kalibracyjnego**.
  Zmierzono też, że równoległość ollamy nic tu nie daje (1,03× na 3 pasach, 1,00×
  na 4) — serwer obsługuje jeden slot, a wąskim gardłem jest strumieniowanie wag
  warstw na CPU. Ramię kontrastowe z thinkingiem **porzucone** (koszt pomiaru na
  maszynie bazowej nie domknął się; wraca tylko za osobnym ADR-em).
  Implementacja: label-free pakiet (`src/doc2query/preferences/answerability_remote.py`,
  `scripts/export_task06_answerability_packet.py`,
  `scripts/task06_judge_remote.py` — samodzielny, journal z `fsync` i resume,
  itemy jednego pasażu w jednym pasie dla prefix-cache'u,
  `scripts/import_task06_answerability_verdicts.py` liczący K1–K3 maszynowo,
  **11 testów CPU**). Pakiet: **1540 itemów**, 665 pasaży,
  `items_sha256 = 31ac2436c34ef55d…`; etykiety zostają lokalnie, więc zdalny
  sędzia nie ma czego dostroić pod wynik. Status: **kalibracja WYKONANA, bramka K1–K3 PRZESZŁA** (2026-08-20).
- **Sędzia odpowiadalności przyjęty** (`Qwen/Qwen3.8-27B-FP8`, vLLM 0.27.1 operatora,
  prompt `task06-answerability-pl-v1`, dekodowanie `json_schema_enum`, 1540 werdyktów,
  zero `out_of_schema`, wszystko przy pierwszej próbie): **K1** accuracy **0,8566**
  (próg 0,85; n=809) i balanced accuracy **0,8878** (próg 0,75), przy `recall_no` 0,9429;
  **K2** `ungrounded` → `no` w **180/180**, `good_specific` 0,1056 i `good_alternative`
  0,1222 odrzuceń (cap 0,20); **K3** abstencja **0,0065** (cap 0,25). Status artefaktu:
  `accepted_as_axis_a_answerability_signal`. Zapas K1 nad progiem jest mały (0,66 pp),
  ale **przejście jest stabilne**: drugi, niezależny run tego samego przyrządu dał 0,8557.
  Ten sam drugi journal dał przy okazji samodzielny wynik — **powtarzalność serwera to
  0,9909** (14 różnic na 1540 mimo `temperature=0` i przypiętego seeda), co potwierdza
  niedeterminizm continuous batchingu. Przegląd ręczny 34 werdyktów (zobowiązanie §5 ADR,
  mógł kalibrację wyłącznie unieważnić — nie unieważnił) pokazał, że **referencja jest
  słabszą stroną**: w 8 z 10 rozbieżności „sędzia `yes` / konsensus `no`" werdykt sędziego
  jest bardziej obronny, a w 4 z 6 „nadmiernych odrzuceń" klas `good_*` błędna jest
  etykieta z konstrukcji, nie sędzia. Kierunek obciążenia do uwzględnienia w V2-03:
  `recall_no` 0,9429 wobec `recall_yes` 0,8328 znaczy, że filtr osi A będzie
  **konserwatywny** — kosztem podaży, nie czystości. Raport:
  [`task06_answerability_judge_v1_2026-08-20.md`](../reports/measurements/task06_answerability_judge_v1_2026-08-20.md).
- **Paczkowanie zapytań ODRZUCONE bramką A/B** (amendment
  [`..._batching_amendment_2026-08-20.md`](../reports/decisions/task06_answerability_judge_v1_batching_amendment_2026-08-20.md),
  raport `reports/measurements/task06/judge_batching_ab_v1/`): **B1 zgodność 0,9052**
  wobec progu 0,98 i **B2 dryf istotny** — jednostronna migracja w `uncertain`
  (`yes→uncertain` 21 vs 2, p=0,0001; `no→uncertain` 19 vs 2, p=0,0002), abstencja rośnie
  z 0,65% do 2,99%. Przy powtarzalności przyrządu 0,9909 to ~10× poza szumem. Do tego
  **zysku wydajnościowego nie ma żadnego**: zmierzone 16,3 it/s paczkami wobec 19,1 i 17,0
  it/s pojedynczo (pule pojedynczo trzymają 19,2–19,5 it/s) — serwer jest ograniczony
  dekodowaniem, nie prefillem, więc wcześniejszy szacunek 3–5× przyspieszenia był błędny,
  a wariant hybrydowy z wieloma pasażami opiera się na tej samej obalonej premisie.
  `--batch-size` zostaje na 1.
- **Certyfikacja puli WYKONANA i podaż osi A zmierzona** (2026-08-20,
  `src/doc2query/preferences/axis_a_supply.py`, `scripts/measure_task06_axis_a_supply.py`,
  7 testów CPU): 172 295 werdyktów przyrządem pojedynczym, **zero kandydatów bez
  werdyktu**, zero `out_of_schema`. Podaż par osi A: **2 253 pary z kohort autoryzowanych**
  (62,3% z 3 619 grup; v1 199, v2 270, v3 1 784) i **13 736 z v4–v11**, razem **15 989**
  (61,5% z 25 992 grup). Trzy wnioski wchodzące do ADR V2-03: (a) podaż **przestaje być
  wąskim gardłem** — v1.1 dawała po bramce audytu 122 pary akceptowalne wobec progu 1000,
  oś A daje 2 253 z samych kohort autoryzowanych; (b) koszt konserwatywności sędziego jest
  zmierzony i akceptowalny — zachowuje **79,5%** czystych `chosen` (79,7/75,5/80,1% w
  v1/v2/v3, 79,1% w v4–v11), spójnie z kierunkiem `recall_no` 0,9429 vs `recall_yes`
  0,8328; (c) naturalny `rejected` osi A istnieje w **96,9%** grup, co niezależnie
  potwierdza pominięcie V2-04 (konstruowane rejected) — i to na zwalidowanym sygnale, a nie
  na round-tripie, który audyt zdyskwalifikował. Ograniczenie różnorodności odrzuciło
  **jedną** grupę (2 254 → 2 253), bo strony różnią się defektem, nie przestawieniem słów.
  Raport:
  [`task06_axis_a_supply_after_certification_2026-08-20.md`](../reports/measurements/task06_axis_a_supply_after_certification_2026-08-20.md).
- **ADR V2-03 ZAMROŻONY 2026-08-20, przed zbudowaniem pierwszej pary v2**
  ([`task06_defect_pair_policy_v2.md`](../reports/decisions/task06_defect_pair_policy_v2.md),
  config `configs/preferences/task06_defect_pair_policy_v2.yaml`, moduł
  `src/doc2query/preferences/pair_policy_v2.py` + eksport
  `pair_audit_export_v2.py`, 32 testy CPU). Zamrożone: **osie A i B** (oś C poza
  wydaniem — V2-02 nie dowiozło kryterium, więc słaby filtr focusa z v1 **zdjęto**, a
  `focus_accuracy` jest wyłącznie etykietą); definicje bajtowo zgodne ze zmierzoną
  podażą (`chosen` = czysty **i** werdykt `yes`; `rejected` osi A = `no` **albo** brak
  rt@100; `uncertain` blokuje `chosen` i nie jest defektem); **cięcie osi B p75
  `content_jaccard ≥ 0,0857`** dla `rejected` i **p50 ≤ 0,0556** dla `chosen`
  (uzasadnienie: p90 przy dwóch nieznanych mnożnikach zawężających — filtr sędziego
  79,5% i ograniczenie do 13,9% grup — groziło zejściem poniżej kwoty 250 par osi B);
  **kwoty 250/250** w próbce 500 par z realokacją niewykorzystanej kwoty i jawnym
  raportem niedoboru; **deterministyczne przypisanie osi haszem grupy** (kolejność prób,
  nie priorytet, nie licznik), max 1 para na prompt; **tie-break DivPO przyjęty**
  (`chosen` najbardziej odrębny, `rejected` najbardziej typowy, remisy po
  `candidate_index`); margines primary **wyłącznie** sanity `pool_margin > 0`
  (`margin_used_for_ordering=false`, pasma marginesu **usunięte** ze strat próbki);
  **veto shadow zdjęte** z uzasadnieniem (sprawdzało zgodność z porządkiem marginesowym,
  którego v2 nie ma) — z jawnym zapisem, że wzrost liczby par względem v1 **nie jest**
  miarą jakości; konstruowane rejected pominięte (`constructed_rejected=false`, cap 0%);
  werdykty czytane z journala przypiętego po SHA-256 (`fe675f07…`, 23 676 rekordów),
  brak werdyktu = przerwanie runu. **Predykcje wiążące** (bramka fail-closed V2-05):
  P1 nieodpowiadalne `chosen` **≤ 5%** u każdego sędziego (wyprowadzone z czystości
  filtra: 17,7% × (1 − `recall_no` 0,9429) = 1,01%, próg z ~5× zapasem — nie z ambicji);
  P2 `consensus_supports_automatic` **≥ 30%** z dolną granicą CI powyżej 24,4%
  (rozkład baseline'u 0,348 × 0,701 przy drugim czynniku podniesionym do 0,85 ze
  slice'ów defektowych); P3 `consensus_contradicts_automatic` **≤ 3,1%**; P4 kontrast
  osi ≥ +20 pp jako test mechanizmu. Remisy raportowane **bez progu**. Znany dług:
  czytnik audytu Groq stratyfikuje po `primary_margin_gap_band`, którego eksport v2
  celowo nie produkuje — uruchomienie V2-05 wymaga czytnikowej adaptacji osobnym
  amendmentem, bez zmiany promptu, rubryki, modeli ani reguł decyzyjnych.
- **Pary v2 ZBUDOWANE i ślepy eksport zamrożony** (2026-08-20, po zamrożeniu ADR):
  **2 278 par** z kohort autoryzowanych (v1 204, v2 274, v3 1 800; 62,9% z 3 619 grup
  `eligible`), **oś A 2 086 / oś B 192**. Dwie kontrole zgodności wychodzą dokładnie:
  2 278 = zmierzone 2 253 pary osi A + 25 grup parowalnych wyłącznie na osi B, a rozkład
  werdyktów reprezentantów jest identyczny z pomiarem podaży (`yes` 10 804 / `no` 12 804 /
  `uncertain` 68) przy **zero** kandydatach bez werdyktu. Dominująca przyczyna braku pary:
  brak dopuszczalnego `chosen` (1 300 grup) — koszt wymogu „czysty **i** `yes`”; brak
  defektowego `rejected` tylko w 83 grupach. **Wynik negatywny: oś B nie dowiozła kwoty
  250 par — dała 192.** Przyczyną nie jest cięcie overlapu, ale zamrożona reguła
  przypisania osi: hasz daje kolejność prób, więc oś A (parowalna często) absorbuje
  większość grup parowalnych na obu osiach — 954 pary (41,9%) powstały na osi zapasowej, a
  z ~359 grup parowalnych na osi B do B trafiła nieco ponad połowa. Reguły nie zmieniono,
  cięcia nie ruszono, kwoty nie obniżono; zadziałała prerejestrowana realokacja (oś A
  kwota efektywna 308, oś B 192), więc próbka ma **500 par**, `shortfall_pair_count=0`,
  `development_gate_met=true` — ale predykcja P4 będzie mierzona na 192 parach osi B.
  **Degradacja marginesu jest zmierzona, nie tylko zadeklarowana**: 617 par (27,1%) ma
  `primary_margin_delta < 0`, czyli `chosen` o **niższym** marginesie niż `rejected` —
  polityka v1, wymagająca przewagi ≥ 1,0, nie zbudowałaby żadnej z nich; mediana delty
  +1,12…+1,43 pokazuje, że i w parach zgodnych kierunkowo zapas nad progiem v1 był mały.
  Zdjęcie weta shadow też jest wycenione: `shadow_agrees` w 73,1% par, więc v1 unieważniłaby
  **613 par (26,9%)** — wzrostu podaży nie wolno przypisywać samemu sygnałowi defektu.
  Rozkład etykiet: `judge_unanswerable` 68,9%, `weak_corpus_round_trip` 45,0%,
  `high_lexical_overlap` 8,4%, `copy_risk` 4,1%; `possible_ambiguous_query` jest
  praktycznie stałą (99,6%), więc **nie nadaje się** na wymiar slice'owania w audycie
  (zapisane przed audytem). Cięcia osi B trzymają się w danych (`chosen` p75 0,045–0,051
  wobec pułapu 0,0556; `rejected` minimum dokładnie 0,0857). Ślepy eksport w **nowym**
  katalogu `artifacts/task06/preference_audit_v3_defect_pairs/`: 500 par, 12 strat
  (`cohort_id × axis × requested_form`, **bez** pasm marginesu), ziarno 20260820,
  orientacja 250/250, **500/500 zobowiązań zweryfikowanych**, dokładnie pięć dozwolonych
  pól ślepych (ten sam zestaw co v1), zero wycieku `chosen`/`rejected`/osi/werdyktów do
  ślepych rekordów; eksporty v1 i v2 nietknięte. Audytu v2 **nie uruchomiono**, predykcje
  pozostają nieodczytane. Raport:
  [`task06_defect_pairs_v2_2026-08-20.md`](../reports/measurements/task06_defect_pairs_v2_2026-08-20.md).
- **Krok 0 był zablokowany operatorsko do 2026-08-19.** Wznowienie audytu uruchomiono
  2026-08-17 18:20 UTC i wykonało **0 requestów**: dzienne budżety tokenów obu
  modeli są wyczerpane (`gpt-oss` 186 057, `qwen` 228 603 przy limicie 185 000),
  status ponownie `incomplete_quota_deferred`. Ogony ledgerów są jednoznaczne
  (122/122 i 207/207 requestów rozstrzygniętych, zero bez zapisanej odpowiedzi),
  więc `--allow-ambiguous-resend` **nie jest potrzebny** przy wznowieniu.
  W konsekwencji **ADR V2-03 nie został napisany**: jego predykcje wymagają
  baseline'u z pełnych 500 par, a zamrażanie ich na niekompletnym audycie
  łamałoby regułę „predykcje przed odczytem, nigdy dostrajane po fakcie”.
  Budowa par v2 (V2-03/V2-04-bis) i audyt v2 (V2-05) czekają na krok 0.
- **Proxy odpowiadalności v1: ADR zamrożony, kryterium NIEDOWIEZIONE.**
  Prospektywny ADR
  [`task06_answerability_proxy_v1.md`](../reports/decisions/task06_answerability_proxy_v1.md)
  (commit `8bec836`) zamroził **przed** policzeniem jakiegokolwiek związku cechy z
  etykietą: etykietę = konsensus obu sędziów Groq co do odpowiadalności **strony**
  pary (rozjazd sędziów = brak etykiety), deterministyczny podział fit/holdout
  50/50 po `sha256(audit_id)` z obiema stronami pary w tej samej połowie (pasaż
  nie przecieka), przestrzeń 13 już policzonych cech scoringu × 2 kierunki ×
  decyle połowy fit, regułę = atom albo koniunkcja dwóch atomów, cel wyboru na
  fit, kryterium akceptacji `precision_yes >= 0,88` **i** `recall_yes >= 0,50`,
  jednorazowy odczyt holdoutu oraz wiążącą klauzulę zastąpienia proxy przypiętym
  sędzią lokalnym osobnym ADR-em. ADR jawnie wylicza, co było znane przed
  zamrożeniem (392 etykiety konsensusu z 488 stron, sufit szumu = zgodność
  sędziów **0,8033**, baza klasy większościowej **0,7806**) i zapisuje predykcję,
  że P1 raczej nie przejdzie.
  Implementacja: `src/doc2query/preferences/answerability_proxy.py`,
  `scripts/run_task06_answerability_proxy.py`, **10 testów CPU** (m.in. brak
  etykiety bez konsensusu, mapowanie roli po `automatic_chosen_option`,
  determinizm wyboru, fail-closed: nieudana konstrukcja **nie czyta holdoutu**).
  Wynik: z 14 920 reguł 253 przeszły kryterium na fit; zwycięzca
  `longest_copied_ngram <= 3 AND pool_positive_score >= 7,777` ma na fit (n=212)
  czystość 0,9042, a na **holdoucie (n=180) 0,8707** (CI [0,8163; 0,9184]) przy
  `recall_yes` 0,9078 (CI [0,8582; 0,9504]), `accuracy` 0,8222,
  `balanced_accuracy` 0,7103. **Próg 0,88 nie został osiągnięty**, więc status
  artefaktu to `rejected_axis_a_without_answerability_filter`.
  Konsekwencje (wprost z §7 ADR, bez negocjacji): oś A powstanie **bez filtra
  odpowiadalności** po stronie `chosen`; ADR V2-03 **nie może** przewidywać
  spadku udziału nieodpowiadalnych `chosen` do 5% (dopuszczalna predykcja to brak
  pogorszenia względem wartości z pełnego audytu v1); luka odpowiadalności
  pozostaje **nazwanym długiem** do ADR-u sędziego lokalnego (harness V2-01 jest
  gotowy i fail-closed, brakuje wyłącznie wag 27B na maszynie 16 GB).
  Sygnały uboczne: zwycięska reguła używa **absolutnego** `pool_positive_score`,
  nie marginesu (zgodnie z tezą v2, że cross-encoder jest filtrem absolutnego
  score, nie rankingiem dwóch zapytań), a przekroju per rola (czystość 0,9024 na
  stronach `chosen`) **świadomie nie użyto** jako furtki, bo nie był kryterium w
  ADR i byłby wyborem podzbioru po zobaczeniu wyniku. Reguła zostaje jako punkt
  odniesienia dla przyszłej kalibracji sędziego lokalnego na tych samych
  etykietach. Raport:
  [`task06_answerability_proxy_v1_2026-08-17.md`](../reports/measurements/task06_answerability_proxy_v1_2026-08-17.md).
- **Baseline monotonii (oś D) zmierzony** (`src/doc2query/evaluation/query_monotony.py`,
  `scripts/run_task06_monotony_baseline.py`, 9 testów CPU; 224 000 zapytań z 11
  kohort w 19 s CPU, wejścia pinowane po SHA-256). Zadeklarowane wejście
  projektowe: żadnego progu, żadnej bramki, żadnej pary. Wynik przesuwa adresata
  problemu P3: **monotonia słów początkowych jest dyktowana kontrolką, nie
  kolapsem modelu** — `intent=procedure` daje `jak` w **100%** przypadków
  (distinct=1, entropia 0,0000 w 8 z 11 kohort), `intent=definition` daje
  `definicja` w 99,5–100%, więc dwa słowa zbierają **połowę populacji**
  (0,2518 + 0,2503), podczas gdy `entity_lookup` ma 816–885 różnych otwarć.
  Celu parowego ani set-level nagrody nie ma tu co naprawiać — model robi to, o co
  go proszono; do poprawy są **dwie kontrolki intencji napisane jak szablony**.
  Konsekwencja dla M-05: rozkład słów początkowych trzeba raportować **per
  `intent`**, inaczej pomiar „odkryje” szablon i przypisze go modelowi.
  Drugie znalezisko: kontrolka `length` **nigdy nie została użyta** (jedyna
  zaobserwowana wartość we wszystkich 11 kohortach to `medium`), więc ciasnoty
  rozkładu długości (średnia 5,09–5,15 słowa, p05–p95 = 2–9) **nie wolno**
  przypisać modelowi; kontrolka `form` natomiast separuje wyraźnie (6,12 vs 4,06
  słowa), co dowodzi, że kontrolki działają. Baseline set-level dla nagrody GRPO
  (V2-07): distinct-1/distinct-2 per grupa to 0,470–0,477 / 0,659–0,668 stabilnie
  w kohortach v3–v11, przy **v1 wyraźnie niżej** (0,326 / 0,455) — niezależne
  potwierdzenie zmierzonego kolapsu v1 (`duplicate_rate` 0,399) innym przyrządem.
  Reaktywacji `entity_preservation` **nie wykonano**: wymaga backendu spaCy
  (`pl_core_news`), a modele spaCy są hostowane na GitHubie, nieosiągalnym z tej
  maszyny (connect timeout). Raport:
  [`task06_monotony_baseline_2026-08-17.md`](../reports/measurements/task06_monotony_baseline_2026-08-17.md).

Aktualizacja 2026-08-16 (autoryzacja właściciela: zamrożenie polityki par i
budowa tentative par): właściciel autoryzował zamrożenie polityki
`chosen`/`rejected`, zbudowanie tentative par i ślepy audyt dual-LLM.
Prospektywny ADR
[`task06_tentative_pair_policy_v1.md`](../reports/decisions/task06_tentative_pair_policy_v1.md)
zamroził politykę **przed odczytem jakiejkolwiek pary**, z jawnym wyliczeniem,
co było widoczne przed zamrożeniem progów: primary `pool_margin` jest jedynym
sygnałem budującym, shadow wyłącznie veto (nigdy selekcja), corpus round-trip
niezależnym filtrem (`chosen` @20, `rejected` @100), strategia
`top_vs_near_miss`, maksymalnie jedna para na prompt, pary tylko wewnątrz grupy
same-prompt i tylko z reprezentantów klastrów grup `eligible` bramki. Minimalny
margines primary zamrożono na **1.0** jako jedną naturalną jednostkę log-odds
(≈2,72× lepsze szanse) na surowej skali pair-logitu sędziego, sześciokrotnie
poniżej minimalnego marginesu frozen train (6.0) — argument skalowy, nie
wydajnościowy. Wiążące wnioski korpusu walidacyjnego nagrody są respektowane:
`entity_preservation` jest **wykluczone** z polityki (detektor halucynacji, nie
sygnał specyficzności), `focus_accuracy` działa wyłącznie jako słaby filtr
(abstencja nigdy nie karze), a zmierzoną ślepą plamkę `format_valid` domyka
osobny guard wtrącenia `task06_lead_in_guard_v1` działający tylko w polityce par
— `src/doc2query/evaluation/format.py` i progi bramki różnorodności pozostają
**nietknięte**.

Zaimplementowano fail-closed builder (`src/doc2query/preferences/pair_policy.py`,
`scripts/build_task06_tentative_pairs.py`, 18 testów CPU): pinowanie SHA-256
scoringu, summary, cohort records i obu artefaktów bramki, walidacja
fingerprintów manifestu, atomowa publikacja przez staging + `os.replace`, odmowa
nadpisania, odmowa kohorty spoza `authorized_cohorts`, jawne pola
`shadow_used_for_selection=false`, `total_score_computed=false`,
`thresholds_calibrated_here=false`, `audit_completed=false`,
`task07_training_authorized=false`.

Pary zbudowano **wyłącznie z kohort v1+v2** (828 grup `eligible`), zgodnie z
kolejnością z ADR: **202 pary z v1 (55,8%) i 245 par z v2 (52,6%), łącznie 447**.
To **mniej niż 500** par rozwojowej bramki dual-LLM, więc uruchamia się
prerejestrowana ścieżka niedoboru: audyt obejmie wszystkie uzyskane pary,
niedobór jest raportowany, żadnego progu nie wolno poluzować, a rozszerzenie na
kohortę v3 wymaga osobnej decyzji właściciela jako amendment do ADR. Kohorty
v3–v11 dostaną pary tą samą polityką dopiero po pozytywnym audycie. Veto shadow
zadziałało 85 razy (10,3% grup `eligible`) — niezależna, mierzalna niezgodność
sędziów, rzędu wielkości zgodna z 9,81% disagreement bramki HN. Raport:
[`task06_tentative_pairs_v1_v2_2026-08-16.md`](../reports/measurements/task06_tentative_pairs_v1_v2_2026-08-16.md).
Walidacja: Ruff, `mypy src`, pełny pytest. `final_tests_used=[]`.

Ślepy eksport audytowy jest **zamrożony** (`src/doc2query/preferences/pair_audit_export.py`,
`scripts/export_task06_preference_audit.py`, 11 testów CPU, artefakt
`artifacts/task06/preference_audit_v1/`). Zawiera 447 ślepych par w **dokładnie
pięciu** dozwolonych polach (`audit_id`, `passage`, `query_a`, `query_b`,
`orientation_commitment`) — bez `chosen`/`rejected`, bez marginesów i bez typów
błędu; osobny, nieprzekazywany sędziom klucz odślepiający; pełne rekordy próbki
do późniejszej analizy. Orientacja A/B jest **kontrbalansowana** (224/223) i
**zobowiązana przed jakąkolwiek oceną**: każdy ślepy wiersz nosi
`sha256(sól‖pair_id‖orientacja)`, sól jest opublikowana w manifeście, a
447/447 zobowiązań zweryfikowano ponownie z klucza (test regresji psuje się przy
podmianie jednej orientacji). Próbka jest deterministyczna: strata
kohorta × `requested_form` × pasmo marginesu, alokacja metodą największych reszt,
ziarno per-stratum z SHA-256, powtórzony eksport daje identyczne SHA-256.

Aktualizacja 2026-08-17 (decyzja właściciela o niedoborze + audyt dual-LLM):
niedobór 53 par rozstrzygnięto **dopuszczeniem kohorty v3**, nie obniżeniem
bramki. Amendment
[`task06_tentative_pair_policy_v3_topup_amendment_2026-08-17.md`](../reports/decisions/task06_tentative_pair_policy_v3_topup_amendment_2026-08-17.md)
tworzy `configs/preferences/task06_tentative_pair_policy_v1_1.yaml`, którego
**jedyną** różnicą wobec v1 jest lista `authorized_cohorts` rozszerzona o
`same_prompt_expansion_v3` (plus pole `adr`). Zamrożony plik v1 i jego artefakty
pozostają nietknięte jako ślad audytowy, a zamrożonego kontraktu Groq
(`pair_count=500`) **nie zmieniono w ogóle**.

Pary przebudowano pod v1.1 do katalogów `tentative_pairs_v1_1/`: v1 **202**,
v2 **245**, v3 **1565** — razem **2012 par**. Amendment jest zweryfikowany
maszynowo: `pair_ids_fingerprint` dla v1 i v2 jest **identyczny** z tym zbudowanym
pod v1, a jedynym różniącym się polem w rekordach par jest `policy_id`, co dowodzi,
że amendment nie zmienił sposobu budowy pary. Nowy ślepy eksport
`artifacts/task06/preference_audit_v2/` ma **500 par** (`development_gate_met=true`,
niedobór 0), orientację kontrbalansowaną **250/250** i 500/500 zobowiązań
zweryfikowanych. Przy populacji 2012 sampler **faktycznie losuje** proporcjonalnie
w 18 stratach, więc 1512 par pozostaje nieoglądanym zapasem. Eksport
`preference_audit_v1` (447 par) jest **superseded** i nie wolno go użyć do ocen.

Runner audytu dual-LLM (`src/doc2query/preferences/groq_pair_audit.py`,
`scripts/run_task06_groq_preference_audit.py`, 30 testów CPU z wstrzykniętym
transportem i zegarem) realizuje kontrakt: globalna serializacja ≥4 s bez
równoległości, trwały per-model journal z resume, odmowa ponowienia requestu bez
zapisanej odpowiedzi, odroczenie modelu po wyczerpaniu limitu i czysty stop, gdy
odroczone są oba, oraz disagreement wykluczający parę z automatycznej akceptacji.
Klucz Groq jest w lokalnym `.env` pod polem `api_key` (wcześniejszy raport o jego
braku był moim błędem odczytu) i nigdy nie jest logowany.

Audyt **uruchomiono** i jest **niekompletny zgodnie z kontraktem**: miękkie
dzienne budżety tokenów (185 tys. na model) nie pozwalają ocenić 500 par w jednym
dniu, więc oba modele są odroczone jako `daily_token_budget_exhausted`, a run
zatrzymał się czysto ze statusem `incomplete_quota_deferred`. Pokrycie po dniu 1:
`gpt-oss-120b` 244/500 par, `qwen3.6-27b` 414/500, analiza na **244 parach z
oceną obu modeli**. Raport:
[`task06_dual_llm_pair_audit_2026-08-17.md`](../reports/measurements/task06_dual_llm_pair_audit_2026-08-17.md).

Wyniki wstępne, wszystkie z jawnym mianownikiem i CI:

- zgodność automatu z `gpt-oss` **0,701** (n=97, CI [0,608; 0,794]) i z `qwen`
  **0,688** (n=276, CI [0,634; 0,743]), ale zgodność **między modelami 0,915**
  (n=82, CI [0,854; 0,964]). Ta asymetria jest głównym wynikiem: niezgodność z
  porządkiem po marginesie primary nie jest szumem sędziów, bo sędziowie zgadzają
  się ze sobą;
- bramka fail-closed wyklucza **184/244 par (75,4%)** z automatycznej akceptacji
  (60 konsensus za, 15 konsensus przeciw, 7 disagreement, 162 abstencje). Przy
  utrzymaniu uzysku 500 par dałoby ~123 pary akceptowalne, o rząd wielkości
  poniżej bramki 1000 par przed finalnym DPO;
- remisy dominują (`gpt-oss` 50,0% + 10,2% `both_bad`, `qwen` 33,3%) i **nie
  maleją z pewnością sędziego**, czyli są pewnymi deklaracjami równoważności;
- pewność sędziego przewiduje zgodność u obu modeli (0,375→0,730 i 0,577→0,714);
- brak obciążenia pozycyjnego (47,4% i 50,4% wyborów A);
- kontrola krzyżowa formatu: **zero niezgodności** z pipeline'em, żadna nowa
  ślepa plamka `format.py` się nie ujawniła;
- kontrola krzyżowa answerability: **~18% par `chosen` uznano za nieodpowiadalne
  z pasażu** mimo spełnionego round-tripu w top-20, a round-trip praktycznie
  **nie różnicuje** ocenianej odpowiadalności (69,8% vs 65,5% i 61,6% vs 62,5%).
  Specyfikacja wymaga odrzucania par, na które nie da się odpowiedzieć z pasażu,
  a polityka par nie ma takiej kontroli poza round-tripem;
- **korekta wstępnej obserwacji**: pozorny monotoniczny wzrost zgodności z pasmem
  marginesu (0,691→0,711→0,750 przy 228 ocenach) **nie utrzymał się**. Na
  pełniejszym pokryciu `qwen` jest płaski (0,688 / 0,693 / 0,680), więc wielkość
  marginesu primary nie niesie informacji o zgodności z czytającym i
  podniesienie `min_margin_gap` nie ma w tych danych uzasadnienia.

Dwa deterministyczne defekty sędziów przy `temperature=0` (qwen gubi znaki w
środku `audit_id`, gpt-oss zwraca `reason_code` poza listą) blokowały konkretne
pary na stałe i odroczyły modele. Naprawiono je w parserze runnera, nie w
zamrożonym kontrakcie: ID rozpoznawane po unikalnym przedrostku 8 znaków z
wymuszonym pełnym pokryciem requestu (1 naprawa na 658 ocen), a `reason_code`
poza schematem zapisywany dosłownie jako `out_of_schema` (2 wystąpienia), bo jest
metadaną diagnostyczną i twarde odrzucanie requestu za to pole wprowadzałoby
obciążenie pokrycia. Progów nie zmieniono, rubryki sędziego nie zmieniono,
`task07_training_authorized=false`, `final_tests_used=[]`.

Aktualizacja 2026-08-16 (wynik okna bezobsługowego): kolejka zakończyła się
**25/25 zadań bez ani jednej awarii** w 49,18 h GPU; nadzorca nie padł ani razu,
a strażnik wyłączył maszynę 16.08 o 20:45 po godzinie stabilnego stanu
wyczerpania. Kohorty v3–v11 mają komplet 9 × 24000 wygenerowanych i ocenionych
kandydatów; bramka o niezmienionych progach dała 25164/27000 grup `eligible`
(93,2%, rozstęp międzykohortowy 92,3–94,1%). Z v1 (362) i v2 (466) daje to
**25992 grup** do budowy par, czyli powyżej progu 1000 par przed finalnym DPO.
Polityka resamplingu zadziałała 153 razy i **ani razu** nie trzeba było naprawiać
wiersza przez ucięcie. Wykonano też cztery treningi M-03 (seedy 45/46 obu ramion)
oraz dwunastoelementowy sweep budżetu probe, który pokazał, że wariancja między
seedami (0,0011–0,0826 przy stałym ramieniu i budżecie) bywa większa niż mierzone
efekty, a strata treningowa nie jest guardrailem zbieżności (`r = −0,199`).
Agregacji pięciu seedów nie wykonano (zamrożony `compare` pinuje 42/43/44) i
niczego nie promowano. Raport:
[`task06_unattended_compute_window_result_2026-08-16.md`](../reports/measurements/task06_unattended_compute_window_result_2026-08-16.md).
`final_tests_used=[]`.

Aktualizacja 2026-08-14 (okno tokenowe: korpus walidacyjny nagrody i ablacja
teachera na modelu API): właściciel udostępnił budżet tokenów modelu
asystującego przy jednoczesnym pełnym obłożeniu GPU kolejką bezobsługową,
z komendą generowania danych bezpośrednio tokenami modelu. Zamrożono dwa
prospektywne ADR-y **przed** generacją i pomiarem:

1. [`task06_reward_validation_corpus_v1.md`](../reports/decisions/task06_reward_validation_corpus_v1.md)
   — korpus diagnostyczny 180 pasaży × 8 klas błędu (1440 zapytań) o etykietach
   nadanych z konstrukcji, z predykcjami P1–P8 i progami zamrożonymi przed
   odczytem. Generacja i pomiar CPU są **ukończone**: 6/9 predykcji PASS,
   3 FAIL. Szczegóły i diagnozy:
   [`reports/measurements/task06_reward_validation_corpus_v1.md`](../reports/measurements/task06_reward_validation_corpus_v1.md).
   Najważniejsze: bramka różnorodności o niezmienionych progach przepuściła
   180/180 grup o świadomie różnych klasach (P7), a `entity_preservation`
   okazał się detektorem halucynowanych encji, **nie** sygnałem specyficzności
   (remis 1.0 w 180/180 grup — konwencja `empty=1.0`). `format_valid` nie
   wykrywa wtrącenia „Oto …” bez dwukropka (45/45 przypadków tego wariantu
   przechodzi jako poprawne), a `assign_focus` nie rozstrzyga focusu w 46/180
   rekordów klasy `good_specific`. Żadnego progu nie kalibrowano, `format.py`
   ani splittera zdań nie zmieniono — zmiana wymaga własnego ADR, bo dotknęłaby
   interpretacji zamrożonych pomiarów Tasków 04–05.
2. [`task06_claude_teacher_ablation_v1.md`](../reports/decisions/task06_claude_teacher_ablation_v1.md)
   — jawna ablacja teachera przewidziana w tym pliku (sekcja o bramce
   różnorodności): 600 pasaży kohorty v3 × 4 kontrolki D01 × 4 kandydatury
   = 9600 zapytań, osobne provenance, katalog
   `artifacts/task06/teacher_claude_v1/`. Wszystkie cztery kontrolki są
   generowane, więc dla każdego pasażu istnieje kandydat teachera na
   **identyczny** prompt, jaki round-robin przypisał lokalnemu generatorowi.
   Generacja jest **ukończona**: 24/24 shardów, **9600/9600** rekordów,
   walidacja bez błędów (`cohort.validation.json`,
   `candidates.jsonl sha256=40f7a6d6f85bb14b…`). Raport:
   [`task06_claude_teacher_ablation_generation_v1.md`](../reports/measurements/task06_claude_teacher_ablation_generation_v1.md).
   Teacher nie ma przypiętych wag (`claude-opus-5[1m]`, transport API), więc
   kohorta jest nieodtwarzalna bit-exact i pozostaje **osobnym ramieniem
   ablacyjnym**, nie ścieżką główną; lokalny `Qwen3.6-27B` Q4 pozostaje
   preferowanym teacherem programu. Grupy teachera **nie** wchodzą do bramki
   różnorodności same-prompt (bramka mierzy kolaps samplingu, a kandydatury
   teachera są pisane z intencją bycia różnymi) i nie uzupełniają deficytu par
   z v1/v2.

Sygnał jakościowy z ukończonej generacji teachera (raportowany, nie ukrywany):
zamrożone kontrolki D01 często nie mają pokrycia w pasażu. `intent_fit=strained`
wystąpiło w 3132/9600 rekordów (32.6%), z bardzo nierównym rozkładem:
`procedure`@end **60.6%**, `entity_lookup`@middle 28.5%, `definition`@middle
27.6%, `fact_lookup`@beginning 13.8%. Kontrolka `procedure`@end jest w praktyce
niewykonalna na `msmarco_pl` (notki faktograficzne, biogramy, cenniki, hasła
słownikowe), a round-robin przypisuje ją co czwartemu pasażowi. To przesłanka
dla przyszłego, prospektywnego projektu kontrolek (warunkowe przypisanie
intencji zamiast stałej rotacji), a nie decyzja — nic zamrożonego nie zmieniono.
`focus_fit=degenerate` wystąpiło w 1004/9600 rekordów (10.5%).
`passage_quality_note` ustawiono dla 266/600 pasaży (44.3%): duplikaty zdań,
zdania rozcięte na skrótach, mojibake, resztki interfejsu strony i błędy
tłumaczenia nazw własnych (`Georgia`→„Gruzja”, `Lebanon`→„Liban”). Jest to
niezależne potwierdzenie ryzyka translationese zaakceptowanego świadomie w
Task 03 — bez zmiany frozen train i progu `source_en_score >= 23.50`.

Aktualizacja 2026-08-16 (autoryzacja GPU dla kohort tokenowych): właściciel
udostępnił GPU po zakończeniu kolejki bezobsługowej, z warunkiem sensownej
wznawialności. Amendment
[`task06_llm_cohort_gpu_scoring_amendment_2026-08-16.md`](../reports/decisions/task06_llm_cohort_gpu_scoring_amendment_2026-08-16.md)
autoryzuje **wyłącznie scoring** obu kohort tokenowych zamrożonym kontraktem
(primary builder, shadow kontrola, corpus round-trip, batch 8) oraz pomiar
prerejestrowanych kryteriów; nie zmienia żadnej predykcji ani progu.

Wejścia scoringu zmaterializowano w zamrożonym schemacie rekordów generacji
(`scripts/materialize_llm_cohort_scoring_inputs.py`, idempotentnie): 1440
rekordów korpusu nagrody i 9600 rekordów teachera. Weryfikacja kontraktu
same-prompt wypadła czysto: dla **600/600** pasaży `prompt_sha256` teachera jest
identyczny z promptem, jaki dostał lokalny generator w kohorcie v3 — czyli dla
każdego pasażu istnieje para kandydatów na bajtowo tym samym promptcie.

Wznawialność (warunek właściciela): scoring idzie istniejącą ścieżką
`evaluate_intrinsic_records` z fsyncowanym `scoring.journal.jsonl` i
`scoring.resume.json`, runner (`scripts/run_llm_cohort_scoring.sh`) bierze
`flock` i pomija kohortę już ukończoną. Maksymalna strata przy zabiciu procesu
to jeden batch ośmiu rekordów.

Scoring korpusu nagrody jest **ukończony** (1440/1440, 175 s, batch 8) i
**P8 zmierzone**:
[`task06_reward_validation_corpus_p8_2026-08-16.md`](../reports/measurements/task06_reward_validation_corpus_p8_2026-08-16.md).
P8a **PASS**: `ungrounded` ma niższy primary score niż `good_specific` w
**95,6%** grup (próg 0,85, zero remisów), a niezależna kontrola shadow daje
96,7% — kierunek nie jest artefaktem jednego sędziego. P8b była zapisana
**kierunkowo, bez progu**, więc nie orzeczono PASS/FAIL: 124/180 słabszych,
54 remisy (w 43 z nich również `good_specific` nie wróciło w top-20, co mówi o
trudności korpusu, nie o metryce), średnie 0,750 vs 0,072. Wznawianie
potwierdzone w praktyce: pierwszy run przerwał się na 944/1440, a ponowne
uruchomienie tej samej komendy wypisało `[intrinsic resume] 944/1,440 rows
durable` i dokończyło bez powtarzania pracy.

Dwa wyniki mają bezpośrednią konsekwencję dla polityki par. `ungrounded` ma
**ujemny** średni `pool_margin` (−0,48), czyli primary stawia twardy negatyw nad
pozytywem — sygnał gruntowania działa też w wartości bezwzględnej. Zarazem
sygnały sędziowskie **najmocniej nagradzają kopiowanie**: `copy_verbatim` bije
`good_specific` na primary (11,99 vs 10,67), marginesie (7,71 vs 5,21) i
round-tripie, a w **95/180 grup (52,8%)** dosłowna kopia ma najwyższy
`pool_margin` w grupie. Ponieważ zamrożona polityka par używa `pool_margin` jako
jedynego sygnału budującego, bez ochrony przed kopiowaniem ponad połowa par
wybrałaby jako `chosen` kopię pasażu. Ochrona istnieje
(`copy_risk.reject_chosen_on_copy_risk`, odziedziczona z Task 05) i zmierzona
skuteczność to **180/180 (100%)** złapanych kopii, przy mierzalnym koszcie:
29,4% poprawnych, krótkich zapytań w formie `keyword_query` też jest odrzucanych
jako `chosen` (przyczyny: `normalized_lcs` 15, `minimum_query_words` 13,
`copy_density` 9, reszta w kombinacjach). Nic z tego nie zmieniono — to
zamrożony kontrakt; zapis służy jako znana przyczyna, jeśli audyt dual-LLM
pokaże niedoreprezentowanie formy `keyword_query` wśród `chosen`.

Scoring kohorty teachera jest **ukończony** (9600/9600) i porównanie z
prerejestrowanymi kryteriami **policzone**:
[`task06_teacher_vs_student_v3_2026-08-16.md`](../reports/measurements/task06_teacher_vs_student_v3_2026-08-16.md).
Porównanie objęło **600/600** par (pasaż, kontrolka) o identycznym
`prompt_sha256`, zero odrzuceń.

**Odpowiedź ablacji jest negatywna: teacher API nie bije lokalnego D01 4.5B na
zamrożonym sygnale budującym.** Primary wskazuje teachera jako lepszego w 34,7%
pasaży w wersji literalnej z ADR (best-of-8 studenta) i 41,5–41,7% w analizie
post-hoc równobudżetowej (student ograniczony do czterech slotów, oba rozłączne
podzbiory). Shadow daje lekkie wskazanie na teachera (54,7–56,7%), ale kierunek
primary i shadow **rozchodzi się w 22,0% grup** — ponad dwa razy częściej niż
9,81% disagreement bramki HN — więc żaden z tych odsetków nie jest mocnym
dowodem. Corpus round-trip nie różnicuje (76–85% remisów). Wynik jest negatywny
i użyteczny: zdejmuje presję wprowadzania do pipeline'u teachera o nieprzypiętych
wagach. Nie znaczy to, że zapytania teachera są gorsze dla wyszukiwania —
zmierzono zgodność z sędziami, a wiążącą metryką pozostaje probe-embedder na
naturalnych zapytaniach, którego tu nie uruchamiano.

Mechanizm: teacher jest **bardziej równy** (średni `pool_margin` 4,19 vs 2,88),
a student wygrywa **maksimum**, nie średnią — wysokotemperaturowe próbkowanie
produkuje pojedyncze bardzo wysoko oceniane wyniki. To efekt best-of-N.
`format_valid`: teacher 1,0000, student 0,9998.

Znalezisko P8 potwierdzone na danych produkcyjnego pipeline'u: wybór kandydata po
`pool_positive_score` sięga po kandydatów bardziej kopiujących (argmax
`copy_density` 0,3702 vs 0,3061 średnio u teachera i 0,3494 vs 0,2869 u
studenta), a **24,8% (teacher) i 28,8% (student)** kandydatów wskazanych przez
czysty argmax łamie zamrożony guard `copy_risk`. Guard w polityce par jest więc
nośny, nie formalny. Ani guarda, ani polityki nie zmieniono.

Budowa par z udziałem teachera, audyt dual-LLM i trening pozostają
nieautoryzowane; `final_tests_used=[]`.

Aktualizacja 2026-08-14 (kohorta v2 zamknięta, okno bezobsługowe): run v2 jest
ukończony — 4000/4000 wygenerowanych (3 completions resamplowane, zero napraw
przez ucięcie) i 4000/4000 ocenionych. Bramka różnorodności na v2 dała
**466/500 grup `eligible` (93.2%)** wobec 362/500 w v1, co potwierdza skuteczność
szerszego rozkładu decodingu przy niezmienionych progach. Razem obie kohorty dają
**828 grup** kwalifikujących się do budowy par, czyli powyżej rozwojowego progu
500 par. Bramka wymagała jednej poprawki: pinowała wyłącznie kontrakt
`...-expansion-v1`, więc odrzucała artefakt v2; teraz akceptuje zbiór znanych
kontraktów i wymaga zgodności summary z identity (test regresji dodany).
Na okno bezobsługowe 2–3 dób (ADR
[`task06_unattended_compute_window_2026-08-14.md`](../reports/decisions/task06_unattended_compute_window_2026-08-14.md))
zamrożono dziewięć rozłącznych klastrowo kohort v3-v11 po 3000 pasaży (27000 pasaży, 216000 kandydatów) (nowe,
opcjonalne `cohort.partition`; brak partycji jest bajtowo identyczny z
dotychczasowym zachowaniem, co potwierdzono na zamrożonej kohorcie v2) i
uruchomiono nadzorowaną kolejkę `scripts/run_unattended_queue.sh`. Budowa par
pozostaje nieautoryzowana i celowo nie wchodzi do okna: jest tania na CPU i
wymaga decyzji właściciela o polityce `chosen/rejected`.
Walidacja: Ruff, `mypy src`, pełny pytest. `final_tests_used=[]`.

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
- `scripts/build_task06_tentative_pairs.py`
- `scripts/export_task06_preference_audit.py`
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
