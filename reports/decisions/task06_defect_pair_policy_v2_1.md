# Task 06 — polityka par v2.1: jedna oś, reguła przedziałowa, moc policzona z próbą (ADR V2.1-03, 2026-08-23)

## Status i charakter dokumentu

**Prospektywny ADR.** Zamraża zakres osi, definicje stron pary, regułę decyzyjną,
liczebność próby audytowej i predykcje polityki `task06-defect-pair-policy-v2.1`
**przed zbudowaniem pierwszej pary v2.1 i przed pierwszym nowym requestem Groq**.
W chwili podpisania nie istnieje żadna para v2.1, żaden licznik par v2.1, żaden
nowy werdykt sędziów audytu i żadna nowa liczba, którą ten dokument mógłby
opisywać.

Nie unieważnia niczego. Polityka v1/v1.1 i v2, ich pary, ślepe eksporty i oba
zakończone audyty pozostają zamrożonym pomiarem i punktem odniesienia. Bramka
V2-05 pozostaje **niedowieziona**; pary v2 nie idą do żadnego treningu, a jej
progów nikt tu nie obniża, nie przelicza ani nie reinterpretuje.

Nie zmienia: promptu, rubryki, modeli, limitów ani kontraktu
`task06-groq-dual-llm-preference-audit-v1`; wag i digestu sędziego
`task06-answerability-judge-v1`; progów bramki różnorodności same-prompt;
`format.py` ani splittera zdań; splitów; progu `source_en_score >= 23.50`; wag
rerankerów; artefaktów `preference_audit_v1/v2/v3`, `tentative_pairs_v1_1` ani
kohort v1–v11.

Nie autoryzuje: treningu (`task07_training_authorized=false`), testów finalnych
(`final_tests_used=[]`), kohort v4–v11 (pozostają zamknięte tak, jak zapisał ADR
v2 §2.1), ani żadnego odczytu par v2.1 przed wykonaniem audytu.

Realizuje zadanie V2-03 specyfikacji
[`task06_defect_anchored_pairs_v2_spec_2026-08-17.md`](../plans/task06_defect_anchored_pairs_v2_spec_2026-08-17.md)
w wydaniu drugim, po porażce bramki V2-05.

## 1. Co było widoczne przed zamrożeniem (uczciwe wyliczenie)

Ten ADR jest pisany po pełnym, zamkniętym audycie v2 i celowo z niego korzysta.
Widoczne było:

- **pełny audyt dual-LLM par v2** (500/500 par, 250/250 requestów u obu sędziów):
  P1 4,80% (`gpt-oss`) / **5,20%** (`qwen`) wobec progu 5% → FAIL; P2 30,80%
  (CI [26,78%; 35,05%]) → PASS; P3 **3,20%** wobec progu 3,1% → FAIL; P4 +45,9 /
  +56,7 pp wobec +20 pp → PASS; remisy 63,2% / 34,2%; zgodność między modelami
  0,983; abstencja konsensusu 65,4%;
  [raport](../measurements/task06_defect_pairs_v2_audit_2026-08-23.md);
- **diagnostyka post hoc porażki** (jawnie **eksploracyjna**): P1 w osi A
  3,90% / 3,25% wobec 6,25% / 8,33% w osi B; P3 w osi A **1,30%** wobec 6,25% w
  osi B; oś B to 38,4% próbki i wnosi trzy czwarte sprzeczności;
  [raport](../measurements/task06_v2_gate_failure_decomposition_2026-08-23.md);
- **przekroje z `analysis.json` audytu v2**: zgodność konsensusu z automatem
  0,974 (oś A, n=154) wobec 0,250 (oś B, n=16); po etykietach defektu
  `judge_rank_disagreement` 1,000 (n=48), `judge_unanswerable` 0,993 (n=142),
  `lower_primary_margin` 0,984 (n=126), `weak_corpus_round_trip` 0,957 (n=69),
  `high_lexical_overlap` **0,250** (n=16), `copy_risk` 0,667 (n=6); rozkład
  konsensusu w osi A: 154 par rozstrzygniętych z 308, z tego 150 wspiera automat
  i 4 mu przeczą; nieodpowiadalne `rejected` w osi A 49,0% / 63,0%;
- **koszt tokenowy audytu**, odczytany z ledgerów zamkniętego runu v2
  (`artifacts/task06/preference_audit_v3_defect_pairs/groq_dual_llm/ledgers/`):
  `gpt-oss` 235 348 tokenów na 500 par = **470,7 tok/para**, `qwen` 186 634 =
  373,3 tok/para, przy bezpiecznym dziennym limicie 185 000 tok/model, czyli
  **393 pary na okno** (model wiążący: `gpt-oss`; limit requestowy 950 req/dobę
  przy `batch_size=2` daje 1900 par i **nie** wiąże);
- wcześniejsze zamrożone pomiary cytowane w ADR v2 §1 (audyt v1, kalibracja
  sędziego V2-01 z `recall_no` 0,9429, inwentarz V2-00, podaż osi A po
  certyfikacji: 2 253 pary w kohortach autoryzowanych i 15 989 w całej puli) oraz
  wynik budowy par v2: **2 278 par, oś A 2 086 / oś B 192**.

**Status dowodowy tych liczb w tym ADR.** Wartości z osi A pochodzą z podpróby
próbki, na której bramka przegrała, więc **nie są** wynikiem konfirmacyjnym i
**nie są** tu deklarowane jako prawda o polityce. Wchodzą wyłącznie jako
**założenia planistyczne** do rachunku mocy — dokładnie w roli, na którą
diagnostyka post hoc sama zezwala („wejście projektowe"). Każda predykcja tego
ADR musi być zweryfikowana na **nowych** parach i w **nowym** audycie. Założenia
planistyczne mają własne przedziały (np. P1 osi A u `gpt-oss`: 12/308, górna
granica CP 6,24%), co jest jawnie uwzględnione w §4.6 (analiza wrażliwości).

**Nie było widoczne i nie zostało policzone przed podpisaniem:** żadna para v2.1,
żadna liczba par v2.1, żaden rozkład etykiet defektu po zdjęciu osi B, żaden
werdykt nowego audytu, żadna podłoga sędziów na złotym zapytaniu.

## 2. Diagnoza: co dokładnie zawiodło w v2

Trzeba rozdzielić dwie porażki, bo mają różne przyczyny i różne naprawy.

1. **Porażka hipotezy osi B.** Trzy niezależne, zmierzone sygnały: zgodność
   konsensusu 0,250 wobec 0,974 w osi A, niedowieziona podaż (192 wobec kwoty
   250) i trzy czwarte sprzeczności P3 przy 38,4% próbki. Naprawa: usunąć oś
   (§3), a nie przestawić jej cięcie.
2. **Porażka projektu bramki.** Progi punktowe 5% i 3,1% przy n=500 mają
   przedziały szerokości ~±2 pp, więc rozstrzygnięcie „o jedną parę" było wpisane
   w konstrukcję. Naprawa: reguła przedziałowa z próbą dobraną razem z progiem
   (§4), a nie inny próg przy tej samej liczebności.

Do tego dochodzi **trzecia rzecz, która nie jest porażką, ale wnioskiem
metrologicznym**: wyprowadzenie progu P1 („reszta punktowa 1,01% z
`recall_no = 0,9429`, próg 5% zostawia ~5× zapasu") zostało sfalsyfikowane
pomiarem — zmierzono 4,80% / 5,20% na całości i 3,90% / 3,25% w osi A, czyli
3–5× powyżej modelu reszty. Model reszty był więc zły, a próg 5% okazał się nie
progiem z zapasem, lecz progiem **na wartości oczekiwanej**. Konsekwencja jest
statystyczna, nie retoryczna: próg leżący ~1 pp od prawdy jest nierozstrzygalny
przy każdej osiągalnej próbie (§4.3).

## 3. Decyzja 1: oś B wypada z polityki — v2.1 stoi na jednej osi

**Oś B (łatwość leksykalna, cięcie `content_jaccard`) zostaje usunięta z
polityki.** Nie dostaje nowej definicji defektu w tym wydaniu.

Uzasadnienie jest w §2 punkt 1 i jest liczbowe: sędziowie nie potwierdzają
kierunku osi (0,250 przy n=16, ta sama wartość na etykiecie
`high_lexical_overlap`), oś nie dowozi podaży i to ona psuje bramkę. Cięcie
`content_jaccard` jest przy tym prawdopodobnie **niewinne** — dlatego naprawą nie
jest jego przestawienie: podejrzenie pada na samą hipotezę, że wyższe pokrycie
leksykalne jest defektem, który sędzia (a docelowo człowiek) uzna za powód
gorszej jakości zapytania.

### 3.1 Co to znaczy dla różnorodności par — zapisane wprost

**v2.1 jest polityką jednoosiową.** Wszystkie pary uczą jednej rzeczy:
zapytanie odpowiadalne i ugruntowane w pasażu jest lepsze od zapytania
nieodpowiadalnego lub bez round-tripu korpusowego. To trzeba zapisać jako
ograniczenie, nie jako uproszczenie:

- **czego v2.1 nie naprawi**: łatwości leksykalnej, monotonii otwarć, zgodności z
  focusem, naturalności i użyteczności wyszukiwawczej. Jeżeli DPO na tych parach
  poprawi grounding i nie ruszy niczego innego, będzie to **oczekiwany** wynik, a
  nie niespodzianka — i tak ma być raportowany;
- **ryzyko kierunkowe**: optymalizacja preferencyjna typowo zmniejsza
  różnorodność (Kirk i in. 2024), a jednoosiowy sygnał nie zawiera żadnej presji
  przeciwnej. Jedyną presją różnorodnościową wewnątrz przestrzeni par pozostaje
  tie-break DivPO (§6.3), zachowany bez zmian;
- **różnorodność wewnątrz osi A jest realna i mierzona**: etykiety defektu
  `judge_unanswerable` (n=142 w audycie v2, zgodność 0,993),
  `weak_corpus_round_trip` (69, 0,957) i `judge_rank_disagreement` (48, 1,000) to
  rozłączne populacje o różnej trudności. Dlatego etykieta defektu wchodzi do
  stratyfikacji próbki audytowej (§5.3) i do raportowania per etykieta — żeby
  „jedna oś" nie oznaczała „jedna populacja bez wewnętrznej struktury";
- **gdzie wraca łatwość leksykalna**: do kontrolek generacji, M-05 i set-level
  nagrody GRPO — czyli tam, gdzie zmierzony baseline monotonii z 2026-08-17
  wskazał jej prawdziwe źródło. Ten baseline mówi wprost, że monotonia otwarć
  jest **dyktowana kontrolką** (`intent=procedure` → `jak` w 100%, `definition` →
  `definicja` w 99,5–100%), a kontrolka `length` nigdy nie została użyta. To jest
  problem promptu i kontrolek, nie problem, który para preferencyjna może
  rozwiązać — co jest niezależnym argumentem, że oś B była od początku
  skierowana w złe miejsce.

### 3.2 Powrót osi B to nazwany dług, z warunkiem wstępnym

Oś B (albo szerzej: oś „kopiowanie / zbytnia łatwość") może wrócić w przyszłym
wydaniu **wyłącznie** po tym, jak jej etykieta defektu zostanie zwalidowana na
poziomie etykiety, tak jak zwalidowano sędziego odpowiadalności bramką K1–K3 —
czyli po wykazaniu, że niezależny sędzia lub człowiek uznaje tak oznaczone
zapytania za gorsze. **Nowe cięcie tej samej cechy nie jest walidacją.**
`copy_risk` (n=6, zgodność 0,667) nie jest przy tym dowodem w żadną stronę.

### 3.3 Mechaniczne konsekwencje zdjęcia osi B

Wszystkie wynikają z §3 i żadna nie jest osobną decyzją jakościową:

- **znika przypisanie osi haszem grupy**, `axis_preference_order`, kwoty 250/250,
  realokacja i `axis_quota_shortfall`. Budowa jest wyczerpująca dla
  autoryzowanych kohort na jedynej osi;
- **znika warunek `content_jaccard <= 0,0556` po stronie `chosen`** — istniał
  wyłącznie po to, by oś B była rozdzielna. Definicja `chosen` wraca **bajtowo**
  do definicji osi A z ADR v2 §3, czyli do tej, na której zmierzono podaż 2 253
  par;
- **znika wymiar `axis`** ze strat próbki i z przekrojów analizy; wraca wymiar
  etykiety defektu (§5.3). Adaptacja czytnika z amendmentu z 2026-08-21 pozostaje
  w mocy co do zasady (klucze idą za kontraktem eksportu), a nowy kontrakt
  eksportu deklaruje własny wymiar przekroju;
- pole `axis` w rekordzie pary zostaje, ze stałą wartością `"A"`, dla ciągłości
  schematu i dla jawności, że polityka jest jednoosiowa.

### 3.4 Definicje stron pary — bez zmian wobec zmierzonej podaży

- **`chosen`**: `format_valid=true`;
  `has_prefix/has_metacomment/multiple_query/empty=false`; guard wtrącenia
  `task06_lead_in_guard_v1`; `corpus_round_trip_at_20 == 1.0`;
  `entity_preservation == 1.0` (nadal **warunek bezwładny**,
  `role: inherited_inert_check`, `claimed_as_hallucination_filter: false` — nie
  wolno twierdzić, że polityka filtruje halucynacje encji); `pool_margin > 0`
  jako sanity; brak `copy_risk` wg odziedziczonego kontraktu Task 05; **werdykt
  sędziego `yes`**;
- **`rejected`**: kandydat dopuszczalny formatem (te same flagi + guard) z
  nazwanym defektem: werdykt `no` **albo** `corpus_round_trip_at_100 < 1.0`;
  może być `copy_risk` i nie musi mieć `pool_margin > 0`;
- `uncertain` blokuje rolę `chosen` i **nie jest** defektem;
- werdykty czytane z przypiętego po SHA-256 journala
  `verdicts_pool_authorized_single.jsonl`
  (`fe675f07935c4b70e4bc21212ea9eee0b1b3566de44753106bc0841651f339c1`, 23 676
  rekordów), fail-closed: brak werdyktu dla reprezentanta grupy `eligible`
  przerywa run;
- `margin_used_for_ordering=false`, `shadow_used_for_selection=false`,
  `shadow_used_for_veto=false`, `constructed_rejected=false` (cap 0%), oś C poza
  wydaniem, `focus_accuracy` wyłącznie etykietą raportową, maksymalnie **1 para
  na grupę**, `normalized_query_jaccard(chosen, rejected) <= 0,85`,
  `authorized_cohorts = {same_prompt_expansion_v1, v2, v3}`.

Liczba par v2.1 **nie jest** miarą jakości polityki i nie będzie tak raportowana.

## 4. Decyzja 2: reguła przedziałowa zamiast progów punktowych

### 4.1 Kształt reguły

Dla każdej predykcji udziałowej liczony jest **dokładny jednostronny 95%
przedział Cloppera–Pearsona** i porównywany z **niezmienionym** progiem. Werdykt
jest trójwartościowy:

| werdykt | warunek (predykcja typu „≤ próg") | znaczenie |
|---|---|---|
| **PASS** | górna granica CP 95% ≤ próg | próg dowieziony |
| **FAIL** | dolna granica CP 95% > próg | naruszenie dowiedzione |
| **INCONCLUSIVE** | przedział zawiera próg | próba nie rozstrzyga |

Dla predykcji typu „≥ próg" granice są zamienione. `INCONCLUSIVE` jest
**fail-closed**: bramka nie jest zdana, pary nie idą do treningu. Różni się od
`FAIL` tylko w opisie przyczyny — i to jest cała jego rola: uczciwie rozdzielić
„polityka jest zła" od „przyrząd nie miał rozdzielczości".

Progi **nie są zmieniane**. Reguła jest **surowsza** od reguły v2 (wymaga całego
przedziału pod progiem, nie punktu), a ceną jest liczebność próby (§5).

**Wielokrotność.** Bramka wymaga koniunkcji wszystkich predykcji konfirmacyjnych,
także per sędzia. Test przekroju–unii nie wymaga korekty na wielokrotność:
błąd I rodzaju koniunkcji jest ograniczony przez najmniejsze α składowe, czyli
0,05. Korekty **nie** stosujemy i nie wolno jej dopisać po wyniku.

**Brak eskalacji.** Nie ma drugiego etapu, dolewania par ani analizy pośredniej.
Próba jest jedna, ustalona z góry (§5), a prawdopodobieństwo `INCONCLUSIVE`
zostało w niej **wycenione** (§4.5). Reakcją na `INCONCLUSIVE` jest wyłącznie
nowy prospektywny ADR — nigdy dodatkowy odczyt tych samych danych.

### 4.2 Predykcje v2.1

Wszystkie mianowniki jawne: `n = 800` par komórki bramkowej (§5), udziały liczone
per sędzia tam, gdzie baseline jest per sędzia.

| # | predykcja | próg (bez zmian) | rola | punkt decyzyjny przy n=800 | założenie planistyczne | moc |
|---|---|---|---|---|---|---|
| **P1** | `chosen` uznane za nieodpowiadalne, u każdego sędziego | 5% | **guardrail** | FAIL dopiero od **51/800** (6,375%) | 3,90% / 3,25% (oś A) | 0,39 jako test konfirmacyjny — patrz §4.3 |
| **P2** | `consensus_supports_automatic` | 30% | konfirmacyjna | PASS od **262/800** (32,75%) | 48,7% (150/308) | **1,000** |
| **P3** | `consensus_contradicts_automatic` | 3,1% | konfirmacyjna | PASS do **16/800** (2,00%) | 1,30% (4/308) | **0,964** |
| **P4'** | kontrast wewnątrz pary: nieodpowiadalne `rejected` − nieodpowiadalne `chosen`, u każdego sędziego | +20 pp | konfirmacyjna | dolna granica bootstrapu ≥ +20 pp | +45,1 / +59,8 pp | ~1,000 (13,2 i 21,9 sd zapasu) |
| **P5** | udział remisów | — | raportowana kierunkowo, bez progu | — | 63,2% / 34,2% | n/d |

Uwagi do tabeli:

- **P2 jest surowsze niż w v2**, gdzie wymagano punktu ≥ 30% i dolnej granicy
  powyżej baseline'u 24,4%. Teraz wymagana jest dolna granica ≥ **30%**, czyli
  próg jest ten sam, a burden of proof wyższy. Wolno tak zaostrzyć, bo założenie
  planistyczne osi A (48,7%) daje przy n=800 moc 1,000 — zaostrzenie jest
  darmowe. Warunek „CI powyżej 24,4%" jest tym samym automatycznie spełniony i
  zostaje jako kontrola opisowa;
- **P4 musiało zostać przedefiniowane**, bo kontrast **między osiami** nie
  istnieje w polityce jednoosiowej. P4' mierzy ten sam mechanizm w wersji
  **sparowanej**: czy niezależni sędziowie widzą różnicę odpowiadalności między
  stroną, którą polityka nazwała defektową, a stroną, którą nazwała czystą — w
  obrębie tej samej pary. Próg +20 pp zostaje bez zmian. Przedział: percentylowy
  bootstrap 95% po **parach** (10 000 replikacji, ziarno 20260823), zgodnie z
  idiomem bootstrapu po jednostkach obowiązującym w tym repozytorium; statystyką
  jest różnica udziałów w tej samej próbce par, więc bootstrap uwzględnia ich
  zależność. Zapas nad progiem jest rzędu 13–22 odchyleń, więc wybór metody
  przedziału nie jest tu sporny;
- **P5 nadal bez progu**, bo nadal nie potrafię go wyprowadzić z pomiaru. Remisy
  raportowane per sędzia i per pasmo pewności.

### 4.3 P1 schodzi z roli konfirmacyjnej do guardraila — z policzoną ceną

To jest jedyna zmiana w tym ADR, która **osłabia** ciężar dowodu, dlatego zostaje
opisana wprost, z liczbami i z odrzuconą alternatywą.

**Fakt statystyczny.** Przy regule „górna granica CP ≤ 5%" i prawdzie równej
zmierzonej wartości osi A (3,90% u `gpt-oss` — sędzia wiążący, bo słabszy)
potrzebna próba wynosi:

| moc | n | koszt (okna dziennych budżetów Groq) | mieści się w podaży 2 086? |
|---|---|---|---|
| 0,50 | 1 103 | 3 token-doby → 4 okna | tak |
| 0,80 | **2 310** | 5,9 token-doby → 6–7 okien | ledwo, bez zapasu |
| 0,90 | 3 147 | 8,0 token-doby → 9–10 okien | **nie** (wymaga kohort v4–v11) |

Przy prawdzie równej zmierzonej wartości **całej** próbki v2 (4,80%) próg 5% jest
nieosiągalny przy **żadnym** n (n dla mocy 0,50 to ~32 tys. par). Różnica
1,1 pp między prawdą i progiem nie jest rozstrzygalna w tym programie.

**Decyzja.** P1 zostaje **guardrailem**: bramka jest przerwana wtedy i tylko
wtedy, gdy naruszenie jest **dowiedzione** (dolna granica CP 95% > 5%, czyli
≥ 51/800 = 6,375% u któregokolwiek sędziego). Próg 5% pozostaje w dokumencie
bez zmiany; zmienia się kierunek ciężaru dowodu.

**Dlaczego to jest dopuszczalne, a nie obniżeniem bramki po fakcie:**

1. **próg nie został ruszony** — ruszona została rola predykcji, jawnie i przed
   jakimkolwiek nowym odczytem;
2. **wyprowadzenie progu zostało sfalsyfikowane** (§2): 5% miało być progiem z
   pięciokrotnym zapasem nad modelową resztą 1,01%, a jest progiem na wartości
   oczekiwanej. Utrzymywanie predykcji konfirmacyjnej przy progu, o którym
   **wiemy**, że leży na prawdzie, to deklarowanie bramki, której nie da się
   zdać ani obalić;
3. **rzecz, którą P1 miało potwierdzać, ma niezależne potwierdzenie**: sędzia
   odpowiadalności przeszedł własną, prospektywną bramkę K1–K3 (accuracy 0,8566,
   balanced 0,8878, `ungrounded` → `no` 180/180, abstencja 0,0065), a audyt v2
   zmierzył **3,5× spadek** nieodpowiadalnych `chosen` (16,6%/18,8% → 4,8%/5,2%)
   przy zgodności między sędziami 0,983. Filtr odpowiadalności działa — to jest
   już zmierzone i nie potrzebuje trzeciego potwierdzenia;
4. **cena alternatywy jest nieproporcjonalna**: 2 310 par to 6–7 okien Groq,
   wyczerpanie praktycznie całej autoryzowanej podaży i konieczność
   autoryzowania kohort v4–v11 — za 80% szansy rozstrzygnięcia różnicy, która
   nie zmienia żadnej decyzji treningowej (4% wobec 5% szumu w `chosen` przy
   ~2 tys. par DPO).

**Czego guardrail nie wykrywa — policzone.** Prawdopodobieństwo zapalenia
guardraila przy n=800:

| prawdziwy udział | 3,90% | 5,0% | 5,5% | 6,5% | 8,0% | 10,0% | 16,6% (poziom v1) |
|---|---|---|---|---|---|---|---|
| P(FAIL) | 0,001 | 0,048 | 0,157 | 0,577 | 0,964 | 1,000 | 1,000 |

Czyli: guardrail rzetelnie łapie **regresję materialną** (≥8%) i powrót do
poziomu v1, a nie łapie pogorszenia o 1–1,5 pp. To jest dokładnie to, co ma
robić, i tak ma być raportowane. Fałszywy alarm przy prawdzie równej progowi
wynosi 0,048, zgodnie z α.

**Zapis wiążący.** W raporcie z audytu v2.1 P1 musi być podane jako liczba z
przedziałem CP i z jawną adnotacją, że jako test konfirmacyjny ma przy n=800 moc
0,39 (`gpt-oss`) i 0,76 (`qwen`). **Zabronione** jest raportowanie „P1 przeszła"
w znaczeniu konfirmacyjnym, jeżeli guardrail się nie zapalił. Poprawne
sformułowanie: „guardrail P1 nie wykrył naruszenia; próba nie rozstrzyga progu
5%".

### 4.4 Reguła bramki V2.1-05 (fail-closed)

Bramka jest zdana, gdy **jednocześnie**:

1. **P2 = PASS** (dolna granica CP ≥ 30%);
2. **P3 = PASS** (górna granica CP ≤ 3,1%);
3. **P4' = PASS** u **każdego** sędziego (dolna granica bootstrapu ≥ +20 pp);
4. **guardrail P1 nie zapalił się** u żadnego sędziego (dolna granica CP ≤ 5%);
5. komórka bramkowa ma **n = 800** ocenionych par, audyt ma status `complete`,
   zero par bez werdyktu i zero `out_of_schema` po stronie decyzyjnej.

Każde `FAIL` lub `INCONCLUSIVE` w punktach 1–3, zapalony guardrail albo
niespełniony punkt 5 oznaczają: pary v2.1 **nie zastępują niczego, nie idą do
żadnego treningu, polityka wraca do projektowania**, a wynik negatywny zostaje
zapisany jak każdy inny.

**Zabronione po odczycie wyniku:** obniżanie lub podnoszenie progów, zmiana
reguły przedziałowej, zmiana α, dopisanie korekty na wielokrotność, zmiana
definicji stron pary, zmiana rubryki lub modeli sędziów, dobieranie sędziego,
dolewanie par, powtórzenie audytu na nowej próbce dla lepszej liczby,
raportowanie przekroju (kohorta, etykieta defektu, pasmo pewności) jako wyniku
bramki. Przekroje są opisowe — ich rolą jest projektowanie następnego wydania,
nie ratowanie tego.

### 4.5 Moc łączna i wycena ryzyka nierozstrzygnięcia

Przy założeniach planistycznych §4.2 i n=800 moc konfirmacyjnej części bramki
(koniunkcja P2, P3, P4') wynosi **0,964** — dominuje P3. Prawdopodobieństwo
`INCONCLUSIVE` gdziekolwiek w bramce to zatem ~3,6%, i to jest cena, za którą
kupujemy rezygnację z mechanizmu eskalacji. Dla porównania: przy zachowaniu P1
jako predykcji konfirmacyjnej moc łączna wynosiłaby 0,285 przy n=800, 0,50 przy
n=1 200 i 0,82 przy n=2 500.

### 4.6 Analiza wrażliwości — zapisana przed odczytem

Założenia planistyczne pochodzą z podpróby n=308 i są niepewne. Jeżeli prawda
jest gorsza, moc spada w sposób policzony **z góry**, żeby późniejsza porażka nie
była tłumaczona ad hoc:

| predykcja | założenie | moc | pesymistyczny wariant | moc |
|---|---|---|---|---|
| P3 | 1,30% | 0,964 | 2,00% | **0,566** |
| P2 | 48,7% | 1,000 | 35,0% | 0,915 |
| P4' | +45,1 pp | ~1,000 | +30 pp | ~1,000 |

Wniosek zapisany prospektywnie: **P3 jest jedyną predykcją, która może
zakończyć się `INCONCLUSIVE` z realnym prawdopodobieństwem**, i to wyłącznie
wtedy, gdy prawdziwy udział sprzeczności jest istotnie wyższy niż w osi A audytu
v2. Taki wynik będzie sygnałem, że wartości osi A były optymistyczne przez
selekcję podpróby — i tak zostanie opisany, bez zmiany progu.

### 4.7 Koszt w oknach dziennych budżetów Groq — policzony z ledgerów

Podstawa: **470,7 tok/para** (`gpt-oss`, model wiążący), bezpieczny limit
**185 000 tok/dobę/model**. Kontrola metody: zamknięty audyt v2 (n=500) to
1,27 token-doby i zużył faktycznie **2 okna**, więc realistyczny plan to
`ceil(token-doby)` z jednym oknem zapasu na narzut szeregowania i wznowień.

| n par | tokeny `gpt-oss` | token-doby | okna (plan) |
|---|---|---|---|
| 500 (v2, kontrola) | 235 348 | 1,27 | 2 (zmierzone) |
| **800** (komórka bramkowa) | 376 557 | 2,04 | **3–4** |
| **1 100** (bramka + kotwice) | 517 766 | 2,80 | **3–4** |
| 1 200 | 564 835 | 3,05 | 4–5 |
| 2 086 (cała podaż) | 981 872 | 5,31 | 6–7 |
| 2 500 | 1 176 740 | 6,36 | 7–8 |

Wybrany wariant (1 100 par) kosztuje **3–4 okna**, czyli 1–2 okna więcej niż
audyt v2, i nie wymaga autoryzowania nowych kohort.

## 5. Decyzja 3: liczebność i skład próby audytowej

### 5.1 Rachunek podaży

Podaż osi A w kohortach autoryzowanych: **2 086 par zbudowanych** pod v2 (przy
15 989 dostępnych w całej puli v1–v11, z czego kohorty v4–v11 pozostają
zamknięte). Po zdjęciu osi B liczba par v2.1 z tych samych kohort będzie
**co najmniej** 2 086, bo grupy, które pod v2 trafiły na oś B, są kandydatami do
sparowania na osi A — ale **tego nie zakładam i nie przewiduję**: builder
raportuje faktyczną liczbę, a próbka jest z niej losowana.

### 5.2 Wybrana próba: 800 par bramki + 300 par kotwicy = 1 100

- **komórka bramkowa: n = 800.** Ustalona **razem z progami** przez rachunek mocy
  §4.2: P3 osiąga 0,964, P2 1,000, P4' ~1,000, a moc łączna 0,964. To jest
  odpowiedź na wadę konstrukcyjną v2: liczebność nie jest okrągła ani
  odziedziczona, wynika z progu i z założenia planistycznego;
- **komórka kotwic złotych: 300 par** (§6), **niebramkowa**;
- **razem 1 100 par**, czyli ~53% podaży 2 086. **Co najmniej 900 par zostaje
  nieoglądanych** jako zapas — świadomie, bo próbka wyczerpująca populację nie
  zostawia materiału na jakikolwiek przyszły test prospektywny, a jednego
  spalonego zapasu nie da się odtworzyć;
- **rozwojowa bramka 500 par** jest z nadmiarem spełniona (`development_gate_met`);
- **kohorty v4–v11 pozostają zamknięte.** Wybrana próba nie wymaga ich
  autoryzowania — to był jeden z powodów jej wyboru.

### 5.3 Ścieżka niedoboru (prerejestrowana)

Jeżeli builder da mniej niż 1 100 par: komórka bramkowa ma priorytet do 800,
kotwice biorą resztę (minimum 200, inaczej komórka kotwic **nie powstaje** i
zostaje odroczona). Jeżeli par jest mniej niż 800, audyt idzie na całej podaży,
a **rachunek mocy dla faktycznego n jest przeliczony i zaraportowany przed
odczytem** wyników. Jeżeli par jest mniej niż 500, bramka nie startuje.
Poluzowanie jakiegokolwiek progu w reakcji na niedobór jest zabronione.

### 5.4 Straty, ziarno, ślepość

- straty: `cohort_id × rejected_defect_label × requested_form`; wymiar `axis`
  **znika** (jedna oś), pasma marginesu pozostają usunięte;
- alokacja: `proportional_largest_remainder`; porządek: `pair_id`; ziarno:
  **20260823**;
- orientacja A/B deterministyczna, kontrbalansowana 400/400 w komórce bramkowej i
  150/150 w komórce kotwic, z zobowiązaniem `sha256(sól ‖ pair_id ‖ orientacja)`
  podjętym **przed** jakąkolwiek oceną; sól publikowana w manifeście;
- dokładnie pięć dozwolonych pól ślepych, zero wycieku `chosen`/`rejected`/
  marginesów/etykiet, rozdzielony klucz odślepiający — kontrakt eksportu bez
  zmian merytorycznych;
- katalog eksportu: `artifacts/task06/preference_audit_v4_defect_pairs_v2_1/` —
  **nowy**; eksporty v1, v2 i v3 pozostają nietknięte, builder odmawia nadpisania
  istniejącego wyjścia;
- **obie komórki są rozłączne**: żadna grupa, żaden `passage_cluster_id` i żaden
  `chosen` nie występuje w obu.

## 6. Komórka kotwic złotych — pomiar kalibracyjny, nie bramka

### 6.1 Po co ona jest

Porażka P1 pokazała, że nie potrafimy wyprowadzić **absolutnego** progu
odpowiadalności, bo nie znamy podłogi przyrządu: nie wiemy, jaki odsetek
**naturalnych** zapytań `msmarco_pl` ci sami sędziowie Groq uznają za
nieodpowiadalne z ich własnego pozytywnego pasażu. Bez tej liczby każdy próg
absolutny jest modelem reszty — a jeden taki model właśnie się rozsypał.

### 6.2 Konstrukcja

300 par, w których jedna strona to **naturalne zapytanie** ze splitu `train`
związane z tym samym pasażem, a druga to `chosen` polityki v2.1 z grupy
**nienależącej** do komórki bramkowej. Ta sama rubryka, ten sam prompt, ci sami
sędziowie, ten sam kontrakt — audyt zwraca per stronę `answerable_a`/
`answerable_b`, więc podłoga jest odczytywalna bezpośrednio.

Rozdzielczość przy n=300: obserwowane 1% → CP [0,27%; 2,56%]; 4% → CP
[2,32%; 6,40%]; 10% → CP [7,29%; 13,32%]. To wystarcza, by rozstrzygnąć pytanie
projektowe „czy podłoga jest bliska zeru, czy porównywalna z udziałem w
`chosen`", i nie wystarcza na nic więcej — tak ma być raportowane.

### 6.3 Ograniczenia, zapisane wiążąco

- **niebramkowa**: żaden wynik kotwic nie wchodzi do reguły §4.4, w żadnym
  kierunku. Nie może uratować ani obalić bramki;
- **nie relabeluje i nie filtruje żadnej pary naturalnej**, nie dotyka progu
  `source_en_score >= 23.50`, nie jest sygnałem selekcji ani drop/weighted, nie
  narusza §5.1 AGENTS.md — jest diagnostyką na zamrożonym splicie `train`;
- **nie jest human evidence** i nie zastępuje panelu ludzkiego §9.3 AGENTS.md;
- head-to-head „złote wobec syntetycznego" (preferencje, naturalność,
  użyteczność) jest raportowany **kierunkowo, bez progu** — jest ciekawy i
  niebezpieczny, bo łatwo go nadinterpretować przy jednym pasażu i jednej rubryce;
- jej jedyna dopuszczalna rola przyszła: wyprowadzenie progu absolutnego w
  kolejnym prospektywnym ADR jako **podłoga + tolerancja**, zamiast modelu
  reszty.

## 7. Co zostaje bez zmian (lista zamknięta)

Tie-break DivPO (§7 ADR v2, z tym samym zapisanym ryzykiem leksykalności),
shadow zapisywany ale nieselekcyjny, konstruowane `rejected` na 0%, oś C poza
wydaniem, `entity_preservation` jako warunek bezwładny, maksymalnie 1 para na
grupę, `normalized_query_jaccard ≤ 0,85`, `authorized_cohorts = v1+v2+v3`,
kontrakt Groq (prompt, rubryka, modele, `batch_size=2`, limity, retry, resume),
digest i wagi sędziego odpowiadalności, progi bramki różnorodności, `format.py`,
splitter zdań, splity, progi P-04/M-03, wagi rerankerów. Etap budowy par jest w
całości CPU; polityka nie ładuje modelu i nie uruchamia GPU.

## 8. Kolejność wykonania po zamrożeniu tego ADR

Nic z poniższych **nie** zostało wykonane w sesji, która ten dokument zamroziła:

1. implementacja polityki v2.1 i eksportu (CPU, testy) — bez zmian w
   `pair_policy_v2.py` używanym przez zamknięty pomiar v2; nowy moduł/wersja;
2. budowa par v2.1 na kohortach v1+v2+v3 i raport liczby par oraz rozkładu
   etykiet defektu (liczba par nie jest miarą jakości);
3. ślepy eksport dwóch rozłącznych komórek (800 + 300) z weryfikacją zobowiązań;
4. audyt dual-LLM na niezmienionym kontrakcie, 3–4 okna budżetów Groq;
5. pomiar bramki V2.1-05 dokładnie regułą §4.4 i raport, w tym wynik kotwic.

Dopiero pozytywna bramka otwiera rozmowę o kohortach v4–v11 i o danych do Task 07.

## 9. Granice

- Ten ADR nie jest wynikiem. Nie zawiera żadnej liczby o parach v2.1, bo żadna
  nie istnieje.
- Wartości planistyczne z osi A audytu v2 są jawnie eksploracyjne i nie mogą być
  raportowane jako potwierdzenie polityki v2.1 ani jako wynik bramki.
- Audyt dual-LLM pozostaje evidence kalibracyjnym, **nie** sygnałem selekcji i
  **nie** human evidence.
- Bramka V2-05 pozostaje przegrana; ten dokument jej nie przelicza.
- `task07_training_authorized=false`, `final_tests_used=[]`.
