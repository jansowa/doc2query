# Task 06 — polityka par zakotwiczonych w defektach (ADR V2-03, 2026-08-20)

## Status i charakter dokumentu

**Prospektywny ADR.** Zamraża definicje osi, cięcia, kwoty, tie-breaki, próbkę
audytową i **predykcje** dla polityki `task06-defect-pair-policy-v2` **przed
zbudowaniem pierwszej pary v2**. Opisuje regułę, nie wynik: w chwili podpisania
nie istnieje żadna para v2, żaden licznik par v2 i żaden odczyt wyniku audytu v2.

Nie unieważnia polityki v1/v1.1, jej par, eksportów ani audytu — te pozostają
zamrożonym pomiarem i **punktem odniesienia predykcji**. Nie autoryzuje treningu
(`task07_training_authorized=false`), nie otwiera testów finalnych
(`final_tests_used=[]`), nie zmienia `format.py`, bramki różnorodności, splitów,
progu `source_en_score >= 23.50`, rubryki sędziów audytu ani zamrożonego
kontraktu Groq.

Realizuje zadanie V2-03 specyfikacji
[`task06_defect_anchored_pairs_v2_spec_2026-08-17.md`](../plans/task06_defect_anchored_pairs_v2_spec_2026-08-17.md).

## 1. Co było widoczne przed zamrożeniem (uczciwe wyliczenie)

Ten ADR jest pisany **po** czterech zamkniętych pomiarach i celowo z nich
korzysta — taki był warunek kolejności w specyfikacji (§8: „ADR V2-03 czeka na
pełny wynik audytu v1 i na kalibrację V2-01”). Widoczne było:

- **pełny audyt dual-LLM par v1** (500/500 par): zgodność automatu z sędziami
  0,7179 i 0,7080 wobec 0,8793 między sędziami; `consensus_supports_automatic`
  24,4%, `abstained` 65,2%, `consensus_contradicts_automatic` 6,2%,
  `disagreement` 4,2%; nieodpowiadalne `chosen` 16,6% (`gpt-oss`) i 18,8%
  (`qwen`); remisy 51,8%+9,2% i 32,2%, częstsze przy wysokiej pewności; pasma
  marginesu płaskie u lepiej obsadzonego `qwen` (0,693/0,730/0,700); slice'y
  konsensusu rosnące z defektowością `rejected` (0,909 `weak_corpus_round_trip`
  vs 0,797 `lower_primary_margin`);
  [raport](../measurements/task06_dual_llm_pair_audit_2026-08-17.md);
- **kalibracja sędziego odpowiadalności V2-01** i jego przyjęcie bramką K1–K3:
  accuracy 0,8566, balanced 0,8878, `recall_no` **0,9429**, `recall_yes` 0,8328,
  `ungrounded` → `no` 180/180, abstencja 0,0065, powtarzalność serwera 0,9909;
  [raport](../measurements/task06_answerability_judge_v1_2026-08-20.md), ADR
  [V2-01](task06_answerability_judge_v1.md);
- **inwentarz podaży defektów V2-00**: oś A 17 669 grup na round-tripie, oś B
  4 857 grup przy cięciu p75 `content_jaccard ≥ 0,0857` (p50 = 0,0556,
  p90 = 0,1212 na 101 146 kandydatach), oś C **0**, `entity_preservation`
  zdiagnozowane jako stała 1,0 z konstrukcji;
  [raport](../measurements/task06_defect_inventory_and_focus_v2_2026-08-17.md);
- **podaż osi A po certyfikacji**: 2 253 pary z kohort autoryzowanych (62,3% z
  3 619 grup), 15 989 ze wszystkich jedenastu, filtr sędziego zachowuje 79,5%
  czystych `chosen`, naturalny `rejected` osi A w 96,9% grup, bramka
  różnorodności odrzuciła 1 grupę na 2 254;
  [raport](../measurements/task06_axis_a_supply_after_certification_2026-08-20.md).

**Nie było widoczne i nie zostało policzone przed podpisaniem:** żadna para v2,
żadna liczba par v2 przy jakimkolwiek kandydacie cięcia osi B **po** dołączeniu
sygnału sędziego, żaden rozkład osi po przypisaniu haszem, żaden wynik audytu v2.
Podaż osi B jest znana **wyłącznie** z pomiaru V2-00, czyli na proxy
round-tripowym, bez filtra sędziego po obu stronach — to jest jawna niepewność
tego ADR-u, nie ukryta.

## 2. Decyzja: zasada budowy

Sygnałem budującym parę jest **kontrast defektowy**: `rejected` ma zmierzony,
nazwany defekt, `chosen` jest od tego defektu wolny. `pool_margin` primary
przestaje być kluczem porządkującym i zostaje **wyłącznie warunkiem sanity** po
stronie `chosen` (`pool_margin > 0`). Każdy rekord pary nosi
`margin_used_for_ordering=false`.

Uzasadnienie jest zmierzone, nie estetyczne: pasma marginesu są płaskie względem
zgodności z sędziami, a slice'y konsensusu rosną z defektowością `rejected`
(0,909 vs 0,797). Porządek marginesowy dawał 122 pary akceptowalne z 500, czyli
o rząd wielkości poniżej bramki 1000 par.

Polityka `task06-defect-pair-policy-v2` jest zaimplementowana w
`src/doc2query/preferences/pair_policy_v2.py`, uruchamiana przez
`scripts/build_task06_defect_pairs.py`, z progami przypiętymi w
`configs/preferences/task06_defect_pair_policy_v2.yaml`.

### 2.1 Zakres — bez zmian wobec v1

Wyłącznie wewnątrz grupy same-prompt (identyczny `prompt_sha256`), wyłącznie
grupy `eligible` bramki różnorodności, wyłącznie **reprezentanci klastrów**
near-duplicate wyznaczeni przez bramkę, **maksymalnie 1 para na grupę** (czyli
na prompt), split `train`, bez powtórzenia `passage_cluster_id`. Ograniczenie
różnorodności pary `normalized_query_jaccard ≤ 0,85` zostaje bajtowo takie samo.

`authorized_cohorts` = `same_prompt_expansion_v1`, `v2`, `v3` — dokładnie ta sama
trójka, którą dopuścił amendment v1.1 i na której zmierzono podaż osi A (2 253
pary). Kohorty v4–v11 dostają pary **dopiero po pozytywnym audycie v2**; builder
jest fail-closed wobec każdej innej kohorty.

### 2.2 Sygnał odpowiadalności

Werdykt przyjętego sędziego `task06-answerability-judge-v1` łączony z kandydatem
po `judge_item_id = sha256(wersja_promptu ‖ zapytanie ‖ pasaż)[:24]`. Źródło
werdyktów jest przypięte po SHA-256:
`artifacts/task06/answerability_verdicts/verdicts_pool_authorized_single.jsonl`,
`sha256 = fe675f07935c4b70e4bc21212ea9eee0b1b3566de44753106bc0841651f339c1`,
23 676 rekordów.

**Fail-closed:** brak werdyktu dla któregokolwiek reprezentanta grupy `eligible`
przerywa cały run. Certyfikacja puli wykazała zero kandydatów bez werdyktu, więc
ten warunek jest egzekwowalny bez utraty podaży; gdyby przestał być spełniony,
budowa par ma się zatrzymać, a nie cicho pominąć kandydata.

`uncertain` **blokuje rolę `chosen` i nie jest defektem** — nie może wyprodukować
strony `rejected`. Bez zmian wobec §3 ADR V2-01.

## 3. Oś A — odpowiadalność i grounding (priorytet 1)

Definicje **dokładnie takie, jakimi zmierzono podaż** 2 253 par:

- **`chosen`**: czysty wg zamrożonych warunków polityki — `format_valid=true`,
  `has_prefix/has_metacomment/multiple_query/empty=false`, guard wtrącenia
  `task06_lead_in_guard_v1`, `corpus_round_trip_at_20 == 1.0`,
  `entity_preservation == 1.0`, `pool_margin > 0`, brak `copy_risk` wg
  odziedziczonego bez zmian kontraktu Task 05 — **oraz** werdykt sędziego `yes`;
- **`rejected`**: kandydat dopuszczalny formatem (te same flagi + guard) z
  nazwanym defektem: werdykt `no` **albo** `corpus_round_trip_at_100 < 1.0`.

`rejected` **może** być `copy_risk` (kopiowanie jest jawnie wymienionym w
specyfikacji źródłem rejected) i nie musi mieć `pool_margin > 0`.

### 3.1 `entity_preservation` jest tu warunkiem bezwładnym — i tak zostaje nazwane

Warunek `entity_preservation == 1.0` zostaje w definicji `chosen` **wyłącznie**
dla bajtowej ciągłości ze zmierzoną podażą. Na tych kohortach jest to **stała z
konstrukcji** (`SimplePolishNormalizer.analyze()` zwraca `entities=()`, konwencja
`empty=1.0`), więc nie odrzuca niczego. Config nosi
`entity_preservation.role: inherited_inert_check` i
`claimed_as_hallucination_filter: false`. **Nie wolno** w żadnym raporcie
twierdzić, że polityka v2 filtruje halucynacje encji. Reaktywacja tego sygnału
(backend spaCy) wymagałaby osobnego, wersjonowanego relabelingu i osobnej
decyzji; ten ADR jej nie podejmuje.

## 4. Oś B — łatwość leksykalna (priorytet 2)

- **`rejected`**: kandydat **odpowiadalny** (werdykt `yes`), z
  `corpus_round_trip_at_100 == 1.0`, dopuszczalny formatem, o
  `content_jaccard >= 0,0857`;
- **`chosen`**: czysty jak w osi A (pełny zestaw warunków + werdykt `yes`) i
  dodatkowo `content_jaccard <= 0,0556`.

Wymóg odpowiadalności i round-tripu po stronie `rejected` jest **twardy** —
inaczej oś B mieszałaby się z osią A i uczyłaby tematyczności zamiast parafrazy.
Wymóg round-tripu @20 i werdyktu `yes` po stronie `chosen` jest strażnikiem przed
pomyleniem parafrazy z ogólnością: niski overlap sam w sobie bywa też cechą
zapytania zbyt ogólnego.

### 4.1 Cięcie osi B: p75 = 0,0857 (górne) i p50 = 0,0556 (dolne)

Wybór spośród **zmierzonych kandydatów inwentarza V2-00** (p50 = 0,0556,
p75 = 0,0857, p90 = 0,1212 na 101 146 kandydatach odpowiadalnych-przez-proxy w
kohortach v1–v11). Zamraża się **p75** jako cięcie górne i **p50** jako górną
granicę `chosen`, z trzech powodów:

1. **podaż z zapasem na dwa nieznane mnożniki**. p75 dało 4 857 grup osiągalnych,
   p90 tylko 1 900 (4×rzadsze). Do tej liczby dochodzą dwa nieznane przed
   pomiarem mnożniki zawężające: filtr sędziego po **obu** stronach (zmierzony
   koszt po stronie `chosen` to 79,5% zachowanej podaży) i ograniczenie kohort do
   v1+v2+v3 (3 619 z 25 992 grup, czyli 13,9%). Przy p90 iloczyn tych mnożników
   grozi zejściem poniżej kwoty 250 par osi B w próbce audytowej, co unieważniłoby
   bramkę V2-05 z powodu niedoboru, a nie z powodu jakości par;
2. **rozdzielność cięć bez luki i bez nakładania**: `chosen ≤ p50 < p75 ≤
   rejected` daje gwarantowany, dodatni odstęp overlapu wewnątrz każdej pary,
   przy czym odstęp nie jest kluczem porządkującym — jest warunkiem
   dopuszczalności;
3. **cięcia są quality-blind wobec par**: pochodzą z rozkładu korpusowego
   opublikowanego przed istnieniem jakiejkolwiek pary v2 i nie były dostrajane
   pod liczbę par (nie policzono jej przy żadnym kandydacie).

Wartości bezwzględne `content_jaccard` są niskie, bo mianownikiem jest unia
lematów treściowych całego pasażu; cięcia względne pozostają dobrze określone.
**Progów nie wolno zmienić po zobaczeniu liczby par ani wyniku audytu.**

## 5. Oś C — poza pierwszym wydaniem

Oś C (zgodność z żądanym focusem) **wypada z wydania v2.0** decyzją właściciela
z 2026-08-17, potwierdzoną tutaj pomiarem: V2-02 dowiózł czystszą segmentację
(0 pseudo-zdań wobec 8), ale **nie dowiózł kryterium akceptacji** — abstencja
focusa spadła o 0,6–1,2 pp, bo wąskim gardłem jest scorer leksykalny, nie
splitter. Podaż osi C na starych etykietach wynosiła 0.

Konsekwencja dla `chosen`: **słaby filtr focusa z polityki v1 zostaje usunięty**
(w v1 `focus_accuracy == 0.0` dyskwalifikowało `chosen`). Powody: (a) etykiety
focusa są zmierzenie zepsute, więc filtr wnosił szum, nie sygnał; (b) usunięcie
zachowuje bajtową ciągłość z definicją `clean_chosen`, którą zmierzono podaż
2 253 par. `focus_accuracy` zostaje **wyłącznie etykietą raportową**. Oś C wraca
w v2.1 po mocniejszym przypisywaczu focusa i po osobnym, prospektywnym ADR.

## 6. Przypisanie osi i kwoty

### 6.1 Przypisanie osi grupie — deterministyczne, po haszu

Dla grupy o identyfikatorze `group_id`:

```
parzystość = int(sha256("task06-defect-pair-policy-v2:axis" ‖ group_id)[:16], 16) % 2
kolejność preferencji = ("A", "B") gdy parzystość == 0, inaczej ("B", "A")
```

Grupa buduje parę na **pierwszej osi z tej kolejności, na której jest
parowalna**; jeśli nie jest parowalna na pierwszej, próbuje drugą. Para należy do
**dokładnie jednej** osi, zapisanej w polu `axis`; rekord nosi też
`axis_preference_order`.

Fallback między osiami jest w kolejności **z hasza, nie z priorytetu**, żeby
przypisanie nie zależało od liczby już zbudowanych par ani od kolejności
przetwarzania. Kwoty **nie** są egzekwowane na etapie budowy — budowa jest
wyczerpująca dla autoryzowanych kohort, a bilansowanie osi należy do próbki
audytowej (§6.2). Dzięki temu budowa jest funkcją czystą od danych, a nie od
stanu licznika.

### 6.2 Kwoty w próbce audytowej: 250 / 250 z fallbackiem do podaży

Próbka rozwojowa ma `target_pair_count = 500`, z kwotą **250 par osi A i 250 par
osi B**. Alokacja przebiega niezależnie w każdej osi (proporcjonalnie metodą
największych reszt po stratach), a **jeżeli któraś oś ma mniej niż 250 par**,
bierze się jej całą podaż, a niewykorzystaną resztę kwoty przenosi się do drugiej
osi. Niedobór i faktyczny bilans osi są raportowane jawnie w manifeście
(`axis_quota_shortfall`). Poluzowanie **jakiegokolwiek** progu w reakcji na
niedobór jest zabronione — dopuszczalną reakcją jest wyłącznie raportowany
niedobór albo nowy, prospektywny ADR.

Strata: `cohort_id × axis × requested_form`. Pasma `primary_margin_gap`
**znikają** ze strat — utrzymanie ich reintrodukowałoby margines jako wymiar
projektu. Ziarno: **20260820**. Porządek: `pair_id`. Alokacja:
`proportional_largest_remainder`. Orientacja A/B: deterministyczna,
kontrbalansowana, z zobowiązaniem `sha256(sól ‖ pair_id ‖ orientacja)` podjętym
**przed** jakąkolwiek oceną; sól publikowana w manifeście. Katalog eksportu:
`artifacts/task06/preference_audit_v3_defect_pairs/` — **nowy**, eksporty v1 i v2
pozostają nietknięte.

## 7. Tie-break: wariant DivPO — **przyjęty**

Spośród dopuszczalnych kandydatów danej roli wybiera się:

- **`chosen`**: najbardziej **odrębny** w grupie — najniższa średnia
  `normalized_query_jaccard` do pozostałych reprezentantów grupy;
- **`rejected`**: najbardziej **typowy** — najwyższa ta sama średnia.

Remisy rozstrzyga `candidate_index`, potem `candidate_id`. Jeżeli wybrany
`rejected` narusza `normalized_query_jaccard(chosen, rejected) ≤ 0,85`, przechodzi
się do kolejnego `rejected` w porządku tie-breaku (`chosen` pozostaje bez zmian);
brak takiego kandydata to porażka grupy z przyczyną `near_duplicate_query_pair`.

Uzasadnienie przyjęcia (Lanchantin i in. 2025): optymalizacja preferencyjna
typowo **zmniejsza** różnorodność (Kirk i in. 2024), a kolaps same-prompt jest u
nas zmierzony (kohorta v1: distinct-1/2 per grupa 0,326/0,455 wobec 0,470/0,477 w
v3–v11). Tie-break jest deterministyczny, darmowy, **nie wprowadza nowego
sygnału** i **nie jest kluczem jakości** — działa dopiero na zbiorze już
dopuszczonym przez defekt. Ryzyko zapisane jawnie: kryterium jest leksykalne,
więc może preferować `chosen` odrębny przez rzadkie słowo, a nie przez lepszą
treść; strażnikami pozostają werdykt `yes`, round-trip @20 i (w osi B) cięcie
overlapu.

Margines primary **nie** wchodzi do tie-breaku w żadnej roli.

## 8. Shadow: veto zdjęte, komponenty nadal zapisywane

Veto shadow z polityki v1 (`shadow_pool_margin`/`shadow_pool_rank` sprzeczne z
primary) **nie obowiązuje w v2**. Powód jest wprost konsekwencją §2: veto v1
sprawdzało, czy drugi sędzia potwierdza **porządek marginesowy**; w v2 nie ma
porządku marginesowego, więc ten sam warunek smuggle'owałby margines z powrotem
jako kryterium selekcji. Defekt osi A i B jest orzekany sygnałami niezależnymi od
obu rerankerów (sędzia odpowiadalności, round-trip korpusowy, overlap
leksykalny), a nie zgodnością sędziów.

Komponenty shadow są nadal **w całości zapisywane** w rekordzie pary, a etykieta
`shadow_agrees` nadal raportowana — do analizy audytu, nie do selekcji. Manifest
nosi `shadow_used_for_selection=false` i `shadow_used_for_veto=false`.

Konsekwencja do zapisania uczciwie: v2 usuwa jeden fail-closed guard, który w v1
unieważnił 10,3% grup `eligible`. Podaż osi A wzrośnie z tego powodu, a nie
tylko z powodu lepszego sygnału — porównanie liczby par v1 vs v2 **nie jest**
miarą jakości polityki i nie będzie tak raportowane.

## 9. Konstruowane `rejected`: pominięte (V2-04 nie wchodzi)

Naturalny `rejected` osi A istnieje w **96,9%** grup autoryzowanych — na
zwalidowanym sygnale sędziego, nie na zdyskwalifikowanym round-tripie. Preferencje
na próbkach z własnego rozkładu modelu działają lepiej (Tajwar i in. 2024), więc
konstruowane rejected nie wchodzą do wydania v2.0. Każdy rekord nosi
`constructed_rejected=false`; udział konstruowanych jest zamrożony na **0%**.

## 10. Etykiety defektu (raportowane, nieselekcyjne)

Wyprowadzane z zamrożonych pól, wyłącznie do rozkładów i do analizy audytu według
źródła `rejected`: `judge_unanswerable`, `weak_corpus_round_trip`,
`high_lexical_overlap`, `copy_risk`, `possible_ambiguous_query`, `wrong_focus`,
`shadow_agrees`, `judge_rank_disagreement`, `lower_primary_margin`. Etykiety
**nie** wpływają na wybór pary. `lower_primary_margin` przestaje być etykietą
zawsze prawdziwą — w v2 pojawia się tylko wtedy, gdy faktycznie zachodzi, i jest
sygnałem diagnostycznym (jak często defekt idzie w parze ze spadkiem marginesu).

## 11. PREDYKCJE — zamrożone przed budową pierwszej pary

Baseline: **pełny audyt v1 na 500 parach** (§1). Audyt v2 obejmie 500 par nowego
eksportu, tą samą zamrożoną maszynerią i rubryką, `pair_count=500`. Mianowniki są
jawne; wszystkie udziały liczone z 500 par, per model tam, gdzie baseline jest per
model.

| # | predykcja | baseline v1 | próg v2 | wiążąca? |
|---|---|---|---|---|
| P1 | udział stron `chosen` uznanych za **nieodpowiadalne** | 16,6% (`gpt-oss`) / 18,8% (`qwen`) | **≤ 5% u każdego sędziego** | **tak** |
| P2 | `consensus_supports_automatic` | 24,4% | **≥ 30%**, przy dolnej granicy 95% CI powyżej 24,4% | **tak** |
| P3 | `consensus_contradicts_automatic` | 6,2% | **≤ 3,1%** (co najmniej połowa spadku) | **tak** |
| P4 | kontrast osi: udział `rejected` nieodpowiadalnych w osi A minus ten udział w osi B | — (v1 nie ma osi) | **≥ +20 pp**, u każdego sędziego | **tak** |
| P5 | udział remisów | 51,8% / 32,2% | **bez progu** — raportowany kierunkowo | nie |

### Skąd biorą się te liczby (wyprowadzenie, nie ambicja)

**P1 ≤ 5%.** Wyprowadzone ze **zmierzonej czystości filtra**, nie z ambicji.
Strona `chosen` wymaga werdyktu `yes`, a sędzia ma zmierzone
`recall_no = 0,9429` wobec referencji, której jednym ze składników jest właśnie
konsensus obu sędziów Groq. Punktowa reszta nieodpowiadalnych po filtrze to
17,7% (średnia baseline'u) × (1 − 0,9429) = **1,01%**. Próg 5% zostawia ~5×
zapasu na dwie nazwane niepewności: przesunięcie populacji (kalibracja liczona na
809 stronach audytu i klasach korpusu, budowa par na pełnej puli) oraz szum przy
n = 500. Zapis wymagany przez ADR proxy jest tu spełniony: przy **zwalidowanym
sędzim** predykcja poprawy odpowiadalności jest dopuszczalna, w przeciwieństwie do
scenariusza po porażce proxy leksykalnego.

**P2 ≥ 30%.** Rozkład baseline'u: `consensus_supports_automatic` = P(oba modele
rozstrzygają) × P(zgoda z automatem | oba rozstrzygają) = (174/500 = 0,348) ×
(122/174 = 0,701) = 0,244. Slice'y konsensusu v1 pokazują, że przy **nazwanym
defekcie** `rejected` drugi czynnik rośnie do 0,856–0,909
(`lower_content_jaccard_than_chosen` 0,856 przy n = 97, `weak_corpus_round_trip`
0,909 przy n = 33) wobec 0,797 dla samego marginesu. Przy **niezmienionym**
pierwszym czynniku (0,348 — predykcja P5 jawnie nie zakłada spadku remisów) i
konserwatywnie wziętym drugim czynniku 0,85 wychodzi 0,348 × 0,85 = **0,296**.
Próg 30% jest tą wartością zaokrągloną w górę, a nie liczbą wybraną dla efektu.
Warunek CI chroni przed „poprawą” mieszczącą się w szumie.

**P3 ≤ 3,1%.** Sprzeczność konsensusu z automatem powstaje, gdy oba modele
rozstrzygają **przeciw** stronie wskazanej przez politykę. Baseline 6,2% (31 par)
pochodzi z porządku marginesowego, który — jak pokazał audyt — nie koreluje z
oceną sędziów. W v2 strona `chosen` jest dodatnio orzeczona sygnałem, którego
zgodność z konsensusem Groq zmierzono na 0,8566, więc oczekiwanie **co najmniej
połowy** spadku jest zachowawcze wobec siły tego sygnału. Progu nie stawiam
niżej, bo część sprzeczności pochodzi z osi wartości, której sędzia
odpowiadalności nie widzi (naturalność, użyteczność retrievalowa).

**P4 ≥ +20 pp.** To **test mechanizmu**, nie jakości: oś A dobiera `rejected`
m.in. po werdykcie `no`, a oś B **wymaga** po tej stronie werdyktu `yes`. Jeżeli
niezależni sędziowie Groq nie zobaczą między tymi populacjami dużej różnicy
odpowiadalności, to znaczy, że oś A nie mierzy tego, co deklaruje, i wynik jest
falsyfikacją polityki, nawet gdyby P1–P3 przeszły. Kierunek jest wymuszony
konstrukcją, więc próg musi być duży: baseline v1 pokazuje 59,5–70,6%
odpowiadalnych `rejected` w populacji bez osi, a różnica 20 pp jest wielokrotnie
większa niż 3–4,5 pp, którymi round-trip różnicował odpowiadalność (czyli niż
zmierzony poziom szumu tego kontrastu).

**P5 bez progu.** Zgodnie z §6 specyfikacji: remis nie jest porażką pary
defektowej, bo defekt bywa subtelny (zwłaszcza w osi B, gdzie **obie** strony są
odpowiadalne). Raportujemy kierunek i pasma pewności, ale nie stawiamy progu,
którego nie potrafię wyprowadzić z pomiaru.

### Reguła bramki (fail-closed)

Bramka V2-05 jest zdana, gdy **wszystkie cztery** wiążące predykcje (P1–P4) są
dowiezione. Niedowiezienie choćby jednej oznacza: pary v2 **nie zastępują**
niczego, nie idą do żadnego treningu, polityka wraca do projektowania, a wynik
negatywny zostaje zapisany jak każdy inny. **Zabronione** jest wtedy: obniżanie
progu predykcji, zmiana cięcia osi B, zmiana kwot, zmiana rubryki sędziów audytu,
dobieranie sędziego i powtórzenie audytu na nowej próbce w celu uzyskania lepszej
liczby.

## 12. Konsekwencje i granice

- Progów, cięć, kwot, tie-breaku i predykcji nie wolno zmieniać po zobaczeniu
  liczby zbudowanych par ani po zobaczeniu wyniku audytu; zmiana wymaga nowego,
  prospektywnego ADR.
- Liczba par v2 **nie jest** miarą jakości polityki (§8) i nie będzie tak
  raportowana.
- Audyt dual-LLM pozostaje evidence kalibracyjnym, **nie** sygnałem selekcji i
  **nie** human evidence.
- Artefakty v1/v1.1 (`tentative_pairs_v1_1/`, `preference_audit_v1/`,
  `preference_audit_v2/`) pozostają nietknięte; v2 pisze wyłącznie do nowych
  katalogów i odmawia nadpisania istniejącego wyjścia.
- Etap jest w całości CPU; polityka nie ładuje modelu i nie uruchamia GPU.
  Werdykty sędziego są odczytywane z zamrożonych journali, nie generowane tutaj.
- **Znany dług wobec V2-05**: czytnik audytu Groq (`groq_pair_audit.py`)
  stratyfikuje analizę po `primary_margin_gap_band`, którego eksport v2 celowo nie
  produkuje. Uruchomienie audytu v2 wymaga więc **czytnikowej** adaptacji
  (podmiana wymiaru stratyfikacji na `axis`) zapisanej osobnym amendmentem — bez
  zmiany promptu, rubryki, modeli, limitów ani reguł decyzyjnych. Ten ADR takiej
  adaptacji **nie** wprowadza i jej nie autoryzuje.
- `task07_training_authorized=false`, `final_tests_used=[]`.
