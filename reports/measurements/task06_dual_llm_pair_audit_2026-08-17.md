# Pomiar: ślepy audyt dual-LLM par preferencyjnych (2026-08-17, dzień 1 z 2)

Kontrakt: `configs/preferences/task06_groq_preference_audit_v1.json`
(`task06-groq-dual-llm-preference-audit-v1`, **niezmieniony**).
Polityka par: `task06-tentative-pair-policy-v1.1` (amendment
[`task06_tentative_pair_policy_v3_topup_amendment_2026-08-17.md`](../decisions/task06_tentative_pair_policy_v3_topup_amendment_2026-08-17.md)).
Eksport: `artifacts/task06/preference_audit_v2/` (500 par).
Artefakty audytu: `artifacts/task06/preference_audit_v2/groq_dual_llm/`
(`ledgers/`, `pair_verdicts.jsonl`, `analysis.json`, `summary.json`).
`human_evidence_claimed=false`, `safe_anchor_selection_signal=false`,
`task07_training_authorized=false`, `final_tests_used=[]`.

**Audyt jest niekompletny i to jest stan zgodny z kontraktem.** Miękkie dzienne
budżety tokenów (185 tys. na model) nie pozwalają ocenić 500 par w jednym dniu.
Oba modele są odroczone jako `daily_token_budget_exhausted`, run zatrzymał się
czysto ze statusem `incomplete_quota_deferred` i wznawia się jedną komendą.

## Przebieg

| model | requesty | ocenione pary | szac. tokeny dziś | odroczenie |
|---|---|---|---|---|
| `openai/gpt-oss-120b` | 140 | **244/500** | 186 057 | `daily_token_budget_exhausted` |
| `qwen/qwen3.6-27b` | 225 | **414/500** | 184 303 | `daily_token_budget_exhausted` |

Trzy uruchomienia operatorskie, globalna serializacja ≥4 s, zero równoległości.
Pokrycie jest w praktyce losowe względem kohort i strat, bo requesty idą w
porządku po haszu `audit_id`.

Analiza obejmuje **244 pary z oceną obu modeli** (mianownik `rated_pair_count`).

## Dwa deterministyczne defekty sędziów (znalezisko o narzędziu, nie o danych)

Przy `temperature=0` oba modele psuły odpowiedź **powtarzalnie**, więc ponawianie
samo z siebie nie pomagało i każdy defekt blokował konkretną parę na stałe:

1. **`qwen` gubi znaki w środku `audit_id`**: `241b038fc3b311a023a2cf1e` →
   `241b038fc3b023a2cf1e`. Jedna para zabiła 18 prób i odroczyła cały model.
2. **`gpt-oss` zwraca `reason_code` poza zamrożoną listą** (`answerability`),
   również deterministycznie na konkretnych parach.

Obie sytuacje naprawiono w parserze runnera (to własny kod tej sesji, nie
zamrożony kontrakt), z jawnym uzasadnieniem metodologicznym:

- ID rozpoznaje się po **unikalnym przedrostku 8 znaków** w obrębie jednego
  requestu, przy wymuszonym pełnym pokryciu requestu — naprawa nie może scalić
  dwóch par. Każda naprawa jest oznaczona na ocenie i policzona: **1 naprawa**
  na 658 ocen;
- `reason_code` jest metadaną diagnostyczną i **nie wchodzi** do żadnej
  zgodności ani konsensusu. Twarde odrzucanie requestu za ten kod usuwałoby
  systematycznie pary, które sędzia opisuje inaczej, czyli wprowadzałoby
  **obciążenie pokrycia**. Kod poza schematem zapisuje się dosłownie jako
  `out_of_schema`: **2 wystąpienia**, oba `answerability`. Surowość pozostała na
  `preference`, `confidence` i polach boolowskich.

## Wynik główny

### Zgodność

| porównanie | wartość | n | 95% CI (bootstrap) |
|---|---|---|---|
| automat vs `gpt-oss-120b` | **0,701** | 97 | [0,608; 0,794] |
| automat vs `qwen3.6-27b` | **0,688** | 276 | [0,634; 0,743] |
| **`gpt-oss` vs `qwen`** | **0,915** | 82 | [0,854; 0,964] |

Mianownikiem zgodności z automatem są wyłącznie oceny rozstrzygnięte (A albo B).

**To jest najważniejsza liczba tego pomiaru.** Dwa niezależne modele zgadzają
się ze sobą w 91,5%, a z porządkiem po marginesie primary tylko w ~69–70%.
Gdyby niezgodność z automatem była szumem sędziów, modele rozjeżdżałyby się
także między sobą. Skoro nie, to **ok. 30% par jest uporządkowanych odwrotnie
względem spójnego, powtarzalnego kryterium**. Jest to przesłanka przeciw
traktowaniu `pool_margin` jako wystarczającego sygnału jakości, a nie przeciw
sędziom.

### Konsensus i bramka fail-closed

Na 244 parach z dwiema ocenami:

| werdykt | liczba | udział |
|---|---|---|
| `consensus_supports_automatic` | **60** | 24,6% |
| `abstained` (co najmniej jeden model nie rozstrzygnął) | 162 | 66,4% |
| `consensus_contradicts_automatic` | 15 | 6,1% |
| `disagreement` (modele przeciwne) | 7 | 2,9% |

Zgodnie z zamrożoną polityką `disagreement_policy: exclude_from_automatic_acceptance`
**184 z 244 par (75,4%) jest wykluczonych z automatycznej akceptacji.** Gdyby ten
uzysk utrzymał się na pełnych 500 parach, dałoby to ~123 pary akceptowalne — o
rząd wielkości poniżej bramki 1000 par przed finalnym DPO.

### Remisy dominują i są deklarowane z wysoką pewnością

| model | `tie` | `both_bad` | rozstrzygnięte |
|---|---|---|---|
| `gpt-oss-120b` | 122 (50,0%) | 25 (10,2%) | 97 (39,8%) |
| `qwen3.6-27b` | 138 (33,3%) | 0 | 276 (66,7%) |

Odsetek remisów w podziale na pewność sędziego **nie maleje** z pewnością:
`gpt-oss` 0,636 przy `confidence∈[0,7;0,9)` i 0,599 przy `[0,9;1,0)`; `qwen`
0,257 i 0,343. Remisy nie są więc hedgowaniem przy niepewności — sędziowie
**pewnie stwierdzają równoważność**. To przesuwa interpretację w stronę „dane
faktycznie zawierają pary bliskie równoważności”, a nie „rubryka jest zbyt
uboga”, choć rozrzut między modelami (50% vs 33%) pokazuje, że część tego to
styl odpowiedzi modelu.

Ilustracja z próbki (`00ac1e4c317e643b3d3d3d49`, pasaż o miejscowości Fall
Branch w Tennessee): `chosen` = „definicja fall branch w tn”, `rejected` =
„definicja fall branch”, margines primary 1,299, jaccard 0,60. Para przechodzi
bramkę różnorodności, ale różnica to dwa znaki uściślenia — remis jest tu
racjonalny.

### Pewność sędziego przewiduje zgodność

| model | `confidence∈[0,7;0,9)` | `confidence∈[0,9;1,0)` |
|---|---|---|
| `gpt-oss-120b` | 0,375 (n=8) | 0,730 (n=89) |
| `qwen3.6-27b` | 0,577 (n=52) | 0,714 (n=224) |

Kierunek jest ten sam u obu modeli. To argument, że audyt niesie sygnał, a nie
szum: gdy sędzia deklaruje wysoką pewność, częściej zgadza się z automatem.

### Brak obciążenia pozycyjnego

Wśród ocen rozstrzygniętych: `gpt-oss` A 46 / B 51 (47,4% A), `qwen` A 139 /
B 137 (50,4% A). Kontrbalansowanie orientacji 250/250 zadziałało; nie ma śladu
preferencji pozycji.

## Kontrole krzyżowe z pipeline'em (na danych już zebranych)

### Format: brak nowej ślepej plamki

Polityka wymusza `format_valid=True` po obu stronach, więc każde `invalid` od
sędziego byłoby kandydatem na ślepą plamkę `format.py`. Wynik: **zero
niezgodności** — 244+244 oraz 414+414 ocen `format_valid=True`. Zmierzona
wcześniej plamka „Oto …” jest domknięta guardem wtrącenia, a sędziowie nie
wskazali nic ponad to. Zastrzeżenie: sędziowie mogą być na format pobłażliwi, to
jest wynik negatywny, nie dowód poprawności.

### Answerability: `corpus_round_trip_at_20` nie mierzy odpowiadalności

To najmocniejszy wynik kontroli krzyżowej.

| rola | sygnał round-trip@20 | sędzia: odpowiadalne | sędzia: nieodpowiadalne |
|---|---|---|---|
| `chosen`, gpt-oss | trafienie (wymóg polityki) | 201 | **43 (17,6%)** |
| `chosen`, qwen | trafienie (wymóg polityki) | 338 | **76 (18,4%)** |
| `rejected`, gpt-oss | trafienie | 132 (69,8%) | 57 |
| `rejected`, gpt-oss | pudło | 36 (65,5%) | 19 |
| `rejected`, qwen | trafienie | 196 (61,6%) | 122 |
| `rejected`, qwen | pudło | 60 (62,5%) | 36 |

Dwa niezależne wnioski:

1. **~18% par `chosen` jest uznane za nieodpowiadalne z pasażu**, mimo że
   spełniają wymóg round-tripu w top-20 pełnego korpusu 2,4 mln dokumentów.
   Oba modele podają praktycznie tę samą liczbę. Specyfikacja wprost wymienia
   „na oba query nie można odpowiedzieć z pasażu” jako kryterium automatycznego
   odrzucenia, a polityka par **nie ma** żadnej kontroli odpowiadalności poza
   round-tripem.
2. **Round-trip praktycznie nie różnicuje ocenianej odpowiadalności**: dla
   `rejected` udział „odpowiadalne” to 69,8% vs 65,5% (gpt-oss) i 61,6% vs 62,5%
   (qwen). U `qwen` różnica jest zerowa co do kierunku. Round-trip mierzy
   leksykalną odzyskiwalność, nie to, czy pasaż odpowiada na zapytanie — więc
   „niezależny filtr” polityki nie realizuje zadania, które specyfikacja
   przypisuje odpowiadalności.

## Korekta wstępnej obserwacji o pasmach marginesu

Przy mniejszym pokryciu (228 ocen) zgodność wydawała się rosnąć monotonicznie z
pasmem marginesu (0,691 → 0,711 → 0,750). **Ta obserwacja nie utrzymała się.**
Na pełnym dzisiejszym pokryciu, per model, na ocenach rozstrzygniętych:

| pasmo | `gpt-oss` | `qwen` |
|---|---|---|
| [1,0; 2,0) | 0,700 (n=40) | 0,688 (n=125) |
| [2,0; 4,0) | 0,684 (n=38) | 0,693 (n=101) |
| [4,0; ∞) | 0,737 (n=19) | 0,680 (n=50) |

`qwen`, lepiej obsadzony, jest **płaski**. Wniosek jest mocniejszy niż
poprzedni, nie słabszy: **wielkość marginesu primary w zakresie, w którym
budujemy pary, nie niesie informacji o tym, czy porządek zgadza się z
niezależnym czytającym.** Nie jest więc tak, że problemem jest samo najwęższe
pasmo [1,2) — podniesienie progu `min_margin_gap` nie ma w tych danych
uzasadnienia. Progów **nie zmieniono**; zmiana po odczytaniu audytu jest wprost
zakazana przez ADR polityki par.

Slice'y konsensusu (mianownik ograniczony do 75 par, na których oba modele
rozstrzygnęły zgodnie) są w `analysis.json` i mają zbyt mały mianownik na
wnioski: `weak_corpus_round_trip` 1,000 (n=12), `lower_content_jaccard_than_chosen`
0,870 (n=46), pozostałe typy 0,800 (n≈75).

## Granice tego pomiaru

- **To nie jest human evidence** i nie jest panelem kalibracyjnym Task 02.
- **Audyt nie jest sygnałem selekcji.** Nie wolno użyć go do wyboru
  `chosen`/`rejected`; wtedy LLM stałby się drugim sędzią budującym i
  cyrkularność wróciłaby innymi drzwiami.
- Pokrycie jest niekompletne (244/500 par z dwiema ocenami). Liczby mogą się
  zmienić po domknięciu; kierunek trzech głównych wniosków (asymetria 0,92 vs
  0,69, dominacja remisów, luka answerability) jest jednak spójny między
  modelami i między dwoma poziomami pokrycia.
- Rubryka sędziego jest uboga: nazywa pięć osi, ale ich nie definiuje, nie
  ustala hierarchii, nie daje polityki remisu i nie mówi, że są to zapytania
  doc2query mające pomóc embedderowi odzyskać **ten** pasaż. Nie wolno jej
  poprawiać po zobaczeniu tych liczb — to byłoby dostrajanie przyrządu pod
  wynik. Jeśli ma być lepsza, wymaga nowej, prerejestrowanej wersji audytu
  uruchomionej jako jawne A/B rubryk, przy zachowaniu raportu z wersji v1.
- Żadnego progu nie zmieniono, żaden artefakt nie został wypromowany,
  `final_tests_used=[]`.

## Następny krok

Wznowić audyt jutro tą samą komendą (dopełnienie 128 requestów `gpt-oss` i 43
`qwen`), po czym przeliczyć analizę na pełnych 500 parach. Decyzje o polityce par
i o Task 07 pozostają zamknięte do tego momentu.

---

# WYNIK PEŁNY (2026-08-19): 500/500 par, `status: complete`

Sekcja dopisana, **nie nadpisuje** liczb dnia 1 — porównanie obu pokryć jest samo
w sobie informacją o stabilności pomiaru. Audyt domknięto w trzech oknach
dziennych budżetów (dzień 1: 244 pary z dwiema ocenami; dzień 2: dopełnienie do
249/250 `gpt-oss` i 250/250 `qwen`; dzień 3: **1 brakujący request**).
`rated_pair_count = 500`, `development_gate_met = true`,
`safe_anchor_selection_signal = false`, `final_tests_used = []`.
Naprawy ID po przedrostku: `qwen` **2** na 1000 ocen, `gpt-oss` 0.
`reason_code` poza schematem: `gpt-oss` **5** (wszystkie `answerability`).

## Trzy główne wnioski utrzymały się, jeden osłabł ilościowo

| miara | dzień 1 (244 pary) | **pełne 500 par** | 95% CI |
|---|---|---|---|
| automat vs `gpt-oss` | 0,701 (n=97) | **0,7179** (n=195) | [0,656; 0,779] |
| automat vs `qwen` | 0,688 (n=276) | **0,7080** (n=339) | [0,661; 0,755] |
| **`gpt-oss` vs `qwen`** | 0,915 (n=82) | **0,8793** (n=174) | [0,833; 0,925] |

Asymetria — najważniejszy wynik tego audytu — **pozostaje**: dwa niezależne
modele zgadzają się ze sobą w 87,9%, a z porządkiem po marginesie primary w
70,8–71,8%. Przedziały CI zgodności między modelami i z automatem **nie
nachodzą** na siebie, więc niezgodność z porządkiem marginesowym nie jest szumem
sędziów. Uczciwie trzeba jednak zapisać, że zgodność między modelami spadła z
0,915 do 0,879, a zgodność z automatem lekko wzrosła (0,69–0,70 → 0,71–0,72):
przy pełnym pokryciu luka jest **węższa niż wyglądała po dniu 1** (16 pp zamiast
22 pp), choć jakościowo ten sam wniosek.

## Bramka fail-closed: uzysk potwierdzony i gorszy niż prognoza

| werdykt | liczba | udział |
|---|---|---|
| `consensus_supports_automatic` | **122** | 24,4% |
| `abstained` | 326 | 65,2% |
| `consensus_contradicts_automatic` | **31** | 6,2% |
| `disagreement` | 21 | 4,2% |

**378 z 500 par (75,6%) jest wykluczonych z automatycznej akceptacji** — dokładnie
tyle, ile przewidywała ekstrapolacja z dnia 1 (75,4%). Akceptowalnych par jest
**122**, czyli o rząd wielkości poniżej bramki 1000 par przed finalnym DPO. To
jest twardy argument liczbowy za polityką v2: przy tym uzysku sama v1.1 nie
dowiezie materiału treningowego, nawet mając 2012 zbudowanych par.

## Remisy: nadal dominują i nadal nie maleją z pewnością

| model | `tie` | `both_bad` | rozstrzygnięte |
|---|---|---|---|
| `gpt-oss-120b` | 259 (51,8%) | 46 (9,2%) | 195 (39,0%) |
| `qwen3.6-27b` | 161 (32,2%) | 0 | 339 (67,8%) |

Udział remisów w pasmach pewności: `gpt-oss` 0,581 przy `[0,7;0,9)` i 0,613 przy
`[0,9;1,0)`; `qwen` 0,259 i 0,329. U **obu** modeli remisy są **częstsze** w
pasmie wysokiej pewności — czyli to pewne deklaracje równoważności, nie hedging.
Wniosek dnia 1 nie tylko się utrzymał, ale wzmocnił.

## Odpowiadalność: luka potwierdzona na pełnej próbie

| rola | sygnał rt@20 | `gpt-oss`: odpowiadalne / nie | `qwen`: odpowiadalne / nie |
|---|---|---|---|
| `chosen` | trafienie (wymóg polityki) | 417 / **83 (16,6%)** | 406 / **94 (18,8%)** |
| `rejected` | trafienie | 272 / 113 (70,6% odp.) | 229 / 156 (59,5% odp.) |
| `rejected` | pudło | 76 / 39 (66,1% odp.) | 65 / 50 (56,5% odp.) |

1. **16,6% i 18,8% stron `chosen` jest nieodpowiadalne z pasażu** mimo round-tripu
   w top-20 pełnego korpusu 2,4 mln dokumentów. Oba modele podają zbliżoną
   wartość, jak po dniu 1.
2. **Round-trip nadal nie różnicuje odpowiadalności**: 70,6% vs 66,1% (`gpt-oss`)
   i 59,5% vs 56,5% (`qwen`). Różnica 3–4,5 pp przy n≈115 w komórce „pudło” jest
   w granicach szumu i ma **ten sam kierunek** u obu modeli, ale wielkość efektu
   jest bez znaczenia praktycznego.

To jest dokładnie ta luka, którą próbowało zamknąć proxy odpowiadalności
(nieudanie, [`task06_answerability_proxy_v1_2026-08-17.md`](task06_answerability_proxy_v1_2026-08-17.md))
i którą teraz adresuje przypięty sędzia z ADR-u V2-01.

## Format: pierwsza niezgodność, znikoma

Przy pełnym pokryciu `gpt-oss` uznał **2 z 500** stron `chosen` i **2 z 500**
`rejected` za `format_valid=false` (`qwen`: 0 na 1000). Po dniu 1 było zero, więc
to nowa obserwacja — ale 0,4% przy jednym z dwóch sędziów nie jest przesłanką do
ruszania `format.py`, tylko przypisem. Zamrożonego `format.py` **nie zmieniono**.

## Pasma marginesu: `qwen` nadal płaski

Na ocenach rozstrzygniętych, per model:

| pasmo | `gpt-oss` | `qwen` |
|---|---|---|
| [1,0; 2,0) | 0,694 (n=85) | 0,693 (n=153) |
| [2,0; 4,0) | 0,708 (n=72) | 0,730 (n=126) |
| [4,0; ∞) | 0,789 (n=38) | 0,700 (n=60) |

Korekta z dnia 1 **utrzymuje się**: lepiej obsadzony `qwen` jest płaski
(0,693 / 0,730 / 0,700), a wzrost u `gpt-oss` opiera się na 38 ocenach.
Podniesienie `min_margin_gap` nadal nie ma w tych danych uzasadnienia i **progów
nie zmieniono**.

Slice'y konsensusu mają teraz sensowne mianowniki: `weak_corpus_round_trip`
0,909 (n=33), `judge_rank_disagreement` 0,900 (n=30),
`lower_content_jaccard_than_chosen` 0,856 (n=97), `lower_primary_margin`
0,797 (n=153), `possible_ambiguous_query` 0,796 (n=152), `copy_risk` 0,833 (n=6).
Kierunek jest spójny: **im bardziej defektowy `rejected`, tym częściej sędziowie
zgadzają się z automatem** — co jest niezależnym wsparciem dla kotwiczenia par w
defektach zamiast w marginesie.

## Powody werdyktów

| `reason_code` | `gpt-oss` | `qwen` |
|---|---|---|
| `grounding` | 176 | 150 |
| `naturalness` | 143 | 36 |
| `retrieval_usefulness` | 142 | 270 |
| `mixed` | 12 | 44 |
| `answer_leakage` | 11 | 0 |
| `uncertain` | 11 | 0 |
| `out_of_schema` | 5 | 0 |

`grounding` jest u `gpt-oss` najczęstszym powodem, a u `qwen` drugim — przy
rubryce, która nie faworyzuje żadnej osi. Sędziowie sami wskazują grounding jako
główną osiową różnicę, co jest zbieżne z priorytetem 1 osi A polityki v2.

## Brak obciążenia pozycyjnego (potwierdzone)

`gpt-oss` A 99 / B 96 (50,8% A), `qwen` A 169 / B 170 (49,9% A).
Kontrbalansowanie 250/250 zadziałało.

## Co ten pełny wynik zmienia w planie

1. **Krok 0 specyfikacji v2 jest zamknięty** — ADR V2-03 ma teraz baseline z
   pełnych 500 par i może zamrozić predykcje.
2. Baseline dla predykcji v2 (wartości, wobec których v2 będzie mierzone):
   `consensus_supports_automatic` **24,4%**, `consensus_contradicts_automatic`
   **6,2%**, nieodpowiadalne `chosen` **16,6% / 18,8%**, remisy
   **51,8% / 32,2%**, wykluczone z akceptacji **75,6%**.
3. Granice pomiaru z dnia 1 obowiązują bez zmian: to **nie** jest human evidence,
   audyt **nie** jest sygnałem selekcji, rubryki nie wolno poprawiać po zobaczeniu
   liczb, żadnego progu nie zmieniono i `final_tests_used=[]`.
