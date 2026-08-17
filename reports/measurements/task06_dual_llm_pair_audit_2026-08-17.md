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
