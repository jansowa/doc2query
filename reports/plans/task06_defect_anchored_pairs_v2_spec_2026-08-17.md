# Specyfikacja: pary preferencyjne zakotwiczone w defektach (pair policy v2)

Status dokumentu: **specyfikacja planistyczna**, kierunkowo zatwierdzona przez
właściciela 2026-08-17. Nie jest ADR-em i niczego nie zamraża — każdy próg
liczbowy wskazany niżej zamraża dopiero właściwy, prospektywny ADR **przed**
odczytem wyników etapu, którego dotyczy. Dokument nie autoryzuje treningu
(`task07_training_authorized=false`), nie otwiera testów finalnych
(`final_tests_used=[]` wszędzie) i nie unieważnia żadnego zamrożonego artefaktu.
Implementacja w kolejnych sesjach, zadaniami V2-00…V2-07.

## 1. Diagnoza: dlaczego polityka v1/v1.1 wymaga następczyni

Polityka v1 porządkuje pary **marginesem primary** (`pool_margin`), strategią
`top_vs_near_miss`. Zebrane dowody wskazują, że ten sygnał jest źle wycelowany
względem czterech problemów nazwanych przez właściciela:

| problem właściciela | co robi v1 | zmierzony dowód niedopasowania |
|---|---|---|
| P1: zapytanie zbyt proste, za dużo słów kluczowych z pasażu | `copy_risk` odcina tylko ekstremum; margines primary **nagradza** pokrycie leksykalne | pasma marginesu płaskie względem zgodności z sędziami (qwen: 0,688/0,693/0,680); wartość doc2query to ekspansja słownictwa (Nogueira i Lin, doc2query/docTTTTTquery), nie kopiowanie |
| P2: pytania zawsze o początek pasażu | kontrolki focus w promptcie; `focus_accuracy` tylko słaby filtr | `split_sentences` tnie na skrótach, `focus_buckets` wskazuje nagłówki; `assign_focus` abstynuje w 26% (korpus walidacyjny nagrody) — przyrząd pomiarowy focusa jest zepsuty |
| P3: monotonia długości i słów początkowych | nieadresowane przez pary (własność populacji, nie pary) | preference optimization redukuje różnorodność (Kirk et al. 2024); kolaps same-prompt zmierzony u nas (duplicate_rate 0,399) |
| P4: pasaż nie zawiera odpowiedzi | jedyny filtr to `corpus_round_trip` | audyt dual-LLM (dzień 1): **~18% par `chosen` nieodpowiadalne z pasażu** wg obu sędziów niezależnie; round-trip **nie różnicuje** odpowiadalności (61,6% vs 62,5% u qwen); zbieżne z doc2query-- (Gospodinov, MacAvaney, Macdonald 2023: filtrowanie halucynacji poprawia retrieval) |

Dowód zbiorczy: zgodność sędziów **między sobą 0,915** przy zgodności z
automatem **0,69–0,70** — niezgodność z porządkiem marginesowym nie jest szumem
sędziów, tylko własnością sygnału. Remisy dominują (50% + 10% `both_bad` u
gpt-oss) i **nie maleją z pewnością sędziego**, czyli są pewnymi deklaracjami
równoważności: większość par marginesowych nie niesie kontrastu jakości.
Liczby pochodzą z dnia 1 audytu (244/500 par); przed zamrożeniem ADR V2-03
należy je zastąpić wynikiem pełnych 500 par.

## 2. Zasada przewodnia

**DPO ma tłumić zmierzone tryby błędu generatora, nie maksymalizować „jakość”.**
Sygnałem budującym parę jest **kontrast defektowy**: `rejected` ma zmierzony,
nazwany defekt, `chosen` jest od tego defektu wolny. Margines primary zostaje
zdegradowany do warunku sanity po stronie `chosen` (`pool_margin > 0` — pasaż
wygrywa z twardymi negatywami), a **przestaje być kluczem porządkującym**.

To jest powrót do litery specyfikacji Task 06, która od początku wymienia
źródła rejected (kopiowanie, słaby grounding, zły focus, duplikacja) i wymaga
zbalansowania ich rozkładu — polityka v1 spłaszczyła te osie do jednego
marginesu.

Uzasadnienie z literatury: cross-encodery działają jako *etykiety par
(zapytanie, dokument)* (GPL, Wang et al. 2022) albo *filtr absolutnego score*
(InPars, Bonifacio et al. 2022; Promptagator, Dai et al.), a nie jako ranking
dwóch zapytań między sobą; nadoptymalizacja proxy to znany tryb awarii (Gao,
Schulman, Hilton 2023).

## 3. Osie par

Każda para należy do dokładnie jednej osi; oś jest przypisywana grupie
deterministycznie (hash grupy → oś z niedomkniętą kwotą), maksymalnie **1 para
na prompt** (bez zmian). Kwoty osi i progi zamraża ADR V2-03.

### Oś A — odpowiadalność i grounding (problem P4, priorytet 1)

- `rejected` (naturalny): kandydat z grupy z co najmniej jednym z defektów:
  `entity_preservation < 1.0` (zmierzony detektor halucynowanych encji — to
  jest jego właściwa rola, w przeciwieństwie do wykluczonej roli „sygnał
  specyficzności”), brak round-tripu @100, werdykt `no` sędziego
  odpowiadalności (V2-01).
- `chosen`: format + guard wtrącenia + round-trip @20 + `entity_preservation
  == 1.0` + werdykt `yes` sędziego odpowiadalności + `pool_margin > 0` +
  `copy_risk = false`.
- To wprost domyka zmierzoną lukę: v1 nie miała żadnej kontroli
  odpowiadalności poza round-tripem, a 18% `chosen` jest nieodpowiadalne.

### Oś B — łatwość leksykalna / kopiowanie (problem P1, priorytet 2)

- `rejected` (naturalny): kandydat **odpowiadalny** (żeby oś nie mieszała się z
  A), ale o wysokim pokryciu leksykalnym — pasmo górne `content_jaccard` /
  `copy_density` (kandydat progu: górny kwartyl rozkładu korpusowego z już
  opublikowanych summary; wartość zamrozi ADR V2-03, quality-blind wobec par).
- `chosen`: odpowiadalny + round-trip @20 + **niskie/średnie** pokrycie
  leksykalne. Round-trip i sędzia odpowiadalności pełnią rolę strażnika przed
  pomyleniem parafrazy z ogólnością (niski overlap sam w sobie bywa też cechą
  zapytań zbyt ogólnych — dlatego oba warunki są twarde).
- Para uczy: „nie przepisuj pasażu — parafrazuj”, co jest dokładnie osią
  wartości treningowej dla embeddera (ekspansja słownictwa).

### Oś C — zgodność z żądanym focusem (problem P2, priorytet 3)

- **Zablokowana na V2-02**: obecne etykiety focus są zepsute przez splitter
  zdań, więc najpierw powstaje `focus_v2` (nowy, wersjonowany komponent), a
  stare artefakty pozostają nietknięte.
- `rejected` (naturalny): kandydat, którego focus wg `focus_v2` (z pewnością
  ≥ progu z ADR) jest niezgodny z żądanym w promptcie — typowo `beginning`
  przy żądanym `middle`/`end` (lead bias).
- `chosen`: odpowiadalny + zgodny z żądanym focusem.
- DPO uczy tu **posłuszeństwa kontrolce**, nie rozkładu focusów — rozkład
  zapewniają kontrolki w generacji (jak dotąd). To także wzmacnia tryb
  produkcyjny (M-05), bo model lepiej słuchający kontrolek mniej potrzebuje
  selektora.

### Oś D — monotonia (problem P3): **jawnie poza parami**

Zgodnie z intuicją właściciela: monotonia to własność **populacji** zapytań,
której cel parowy nie wyraża; co gorsza, DPO różnorodność typowo zmniejsza.
Miejsca właściwe:

1. kontrolki form/intent/length w generacji (istnieją);
2. obowiązkowa ewaluacja trybu produkcyjnego bez selektora (M-05, Task 07) z
   raportem rozkładu długości i słów początkowych („jak”, „kiedy”, …);
3. set-level komponent nagrody w GRPO (Task 08): distinct-n / self-BLEU liczone
   na **zbiorze** zapytań per pasaż;
4. opcjonalnie, wewnątrz osi A–C: tie-break w stylu DivPO (Lanchantin et al.
   2025) — spośród dopuszczalnych `chosen` wybierz najbardziej odrębnego w
   grupie (najniższy średni lemma-Jaccard do pozostałych), spośród
   dopuszczalnych `rejected` najbardziej typowego. Deterministyczne, tanie i
   nie wymaga nowego sygnału; decyzję podejmuje ADR V2-03.

## 4. Źródła rejected: naturalne przed konstruowanymi

1. **Naturalne (on-policy) mają pierwszeństwo** — preferencje na próbkach z
   własnego rozkładu modelu działają lepiej (Tajwar et al. 2024). Podaż jest
   duża: 25992 grupy `eligible` z pełnym scoringiem primary/shadow/corpus.
2. **Konstruowane** wyłącznie do dopełnienia kwot: perturbacje dobrego
   zapytania wstrzykujące nazwany defekt (podmiana encji, generalizacja przez
   usunięcie encji, wklejenie n-gramu pasażu, przesunięcie focusa). Maszyneria
   już istnieje — korpus walidacyjny nagrody powstał dokładnie tak (8 klas
   defektów z etykietami z konstrukcji). Wymagania: twardy cap udziału
   (kandydat: ≤30%, zamraża ADR), flaga `constructed_rejected=true` w
   provenance każdej pary, raport per oś.
3. Ostrzeżenie ze specyfikacji obowiązuje: zbyt łatwe rejected uczą tylko
   tematyczności — dlatego osie B i C wymagają rejected **odpowiadalnych**
   (defekt jest subtelny, nie nonsensowny), a oś A ma mieć rozkład od
   halucynacji encji po miękki brak odpowiedzi.

## 5. Zadania do implementacji

### V2-00 — inwentarz podaży defektów (CPU, tanie)

Policz na zamrożonych kohortach v1–v11, per grupa `eligible`: czy istnieje
kandydat czysty (format, rt@20, `entity_preservation==1.0`, `pool_margin>0`,
bez copy-risk) oraz kandydaci defektowi per oś (A: `entity_preservation<1` lub
brak rt@100; B: górne pasmo overlapu przy spełnionym rt; C: wstępnie po starych
etykietach focus, z adnotacją o ich wadach). Wynik: raport liczności par
osiągalnych per oś i kohorta. Jawnie zadeklarowany jako **wejście projektowe**
polityki v2 (czyta pola jakości; nie jest quality-blind i nie udaje, że jest);
nie buduje par. Kryterium akceptacji: tabela podaży + rozkłady komponentów,
`final_tests_used=[]`.

### V2-01 — sędzia odpowiadalności (ADR + kontrakt + kalibracja)

- Model: lokalny, przypięty `qwen3.6-27b` Q4 (preferencja specyfikacji:
  przypięte wagi zamiast API). Uwaga sprzętowa: Q4 27B jest na styk na 16 GB —
  dopuszczalny fallback to llama.cpp z częściowym offloadem; batch cap 8.
  Sędzia ≠ generator (Bielik) i ≠ teacher ablacji (model API), więc bez
  self-preference.
- Zamrożony prompt: pytanie „czy na to zapytanie można odpowiedzieć wyłącznie
  na podstawie tego pasażu?” z odpowiedzią `yes/no/uncertain`; `uncertain`
  wyklucza kandydata z roli `chosen`, ale **nie** liczy się jako defekt.
- Kalibracja (przed użyciem w parach, prospektywny ADR): zgodność z polami
  `answerable_a/b` z ukończonego audytu dual-LLM (2×500 etykiet per strona już
  zebranych) oraz sanity na klasach konstrukcyjnych korpusu walidacyjnego
  (klasa `ungrounded` → `no`). Raport zgodności z CI.
- Koszt: certyfikacja tylko kandydatów faktycznie wchodzących do par
  (`chosen` + rejected osi A), nie całych kohort — rząd kilku–kilkunastu
  tysięcy wywołań lokalnych, nie 200 tys.

### V2-02 — `focus_v2`: naprawiony splitter i labeler (CPU)

Nowy, wersjonowany komponent (spaCy sentencizer albo reguły z listą polskich
skrótów: „np.”, „r.”, „łac.”, „dr”, „p.n.e.”, numeracja), **obok** starego —
stare artefakty i zamrożone pomiary Tasków 04–06 pozostają nietknięte, zgodnie
z zastrzeżeniem z raportu korpusu walidacyjnego. Wyniki `focus_v2` trafiają do
nowych artefaktów relabelingu z własnym provenance. Raport porównawczy: odsetek
zmienionych etykiet, odsetek abstencji przed/po. Kryterium: abstencja focusa
istotnie niższa niż zmierzone 26% bez ruszania czegokolwiek zamrożonego.

### V2-03 — ADR i moduł `pair_policy_v2` (CPU)

- Prospektywny ADR zamraża: kwoty osi, progi (pasmo overlapu osi B, pewność
  focusa osi C, cap konstruowanych), tie-breaki (w tym decyzję o wariancie
  DivPO), listę autoryzowanych kohort, seed próbki audytowej — **przed**
  zbudowaniem jakiejkolwiek pary v2 i po odczycie pełnego wyniku audytu v1.
- Moduł na rusztowaniu v1 (to rusztowanie jest sprawdzone): pinowanie SHA-256
  wejść, atomowa publikacja, odmowa nadpisania, jawne pola
  `margin_used_for_ordering=false`, `axis` per para, pełne komponenty obu
  stron, `final_tests_used=[]`.
- ADR rejestruje **predykcje** dla audytu v2 (rozdział 6).

### V2-04 — generator konstruowanych rejected (CPU + ewentualnie lokalny model)

Podmiana encji (słownikowa, z encji pasażu), generalizacja (usunięcie
encji/liczb), wstrzyknięcie kopii (n-gram pasażu), przesunięcie focusa
(wymaga generacji — lokalny model, nie sędzia). Każdy rekord z flagą
konstrukcji i typem defektu. Limit udziału wg ADR V2-03.

### V2-05 — rozwojowy audyt dual-LLM par v2 (API, ~2 dni budżetu)

Ta sama zamrożona maszyneria co v1 (kontrakt Groq bez zmian, nowy katalog
eksportu, 500 par, orientacja kontrbalansowana, zobowiązanie przed oceną).
Porównanie v2 vs v1 na prerejestrowanych predykcjach — to jest **bramka**
przejścia par v2 dalej. Fail-closed: predykcje niedowiezione → v2 nie zastępuje
niczego i wraca do projektowania.

### V2-06 — ablacja DPO (GPU, wymaga osobnej autoryzacji właściciela Task 07)

Trzy ramiona na 1.5B: DPO na parach marginesowych (v1.1), DPO na parach
defektowych (v2), continued-SFT control (samych chosen — obowiązkowa kontrola
ze specyfikacji). Ewaluacja: probe embeddera z guardrailem M-03, **5 seedów**,
statystyka na sparowanych różnicach per-seed. To jest jedyny arbiter — audyt
LLM waliduje pary, ale o wartości danych decyduje probe na naturalnych
zapytaniach. Bez tej ablacji nie wolno twierdzić, że v2 jest lepsze do DPO.

### V2-07 — notatka przekrojowa do Task 08 (dokumentacja)

Skład wielokryterialnej nagrody GRPO w świetle pomiarów: `entity_preservation`
jako kara halucynacji (nie nagroda specyficzności), format + guard wtrącenia,
komponent set-level różnorodności (distinct-n/self-BLEU na zbiorze per pasaż),
odpowiadalność z sędziego V2-01. Bez zmiany statusu Task 08 (`BLOCKED`,
wymaga `enable_grpo.md`).

## 6. Prerejestrowane predykcje (szkic; wartości zamrozi ADR V2-03)

Po pełnym audycie v1 (500 par) ADR V2-03 zamrozi predykcje w rodzaju:

1. odsetek `chosen` uznanych za nieodpowiadalne przez sędziów spada z ~18%
   do ≤ progu (kandydat: 5%);
2. `consensus_supports_automatic` rośnie względem finalnej wartości v1 o
   zadeklarowany margines;
3. `consensus_contradicts_automatic` spada co najmniej o połowę;
4. odsetek remisów **nie musi** spaść (pary defektowe mogą nadal być subtelne)
   — remis nie jest porażką osi B/C, o ile kontrast defektu jest raportowany.

Predykcje mają jawne mianowniki i CI; ich niedowiezienie zamyka ścieżkę v2 bez
poluzowania czegokolwiek.

## 7. Czego ta specyfikacja nie robi

- **Nie unieważnia** polityki v1/v1.1, par, eksportu ani audytu — pozostają
  zamrożonym pomiarem i punktem odniesienia dla predykcji v2.
- **Nie zmienia** `format.py`, bramki różnorodności, splitów, progu
  `source_en_score >= 23.50`, zamrożonego kontraktu Groq ani rubryki sędziów
  (zmiana rubryki wymagałaby osobnej, prerejestrowanej wersji audytu).
- **Nie autoryzuje** treningu DPO (V2-06 wymaga osobnej komendy właściciela)
  ani żadnego runu GPU poza wskazanymi w zadaniach.
- **Nie otwiera** testów finalnych; `final_tests_used=[]` w każdym artefakcie.
- Nie dotyka artefaktów innych sesji.

## 8. Kolejność wykonania

```
0. dokończyć audyt v1 (dzień 2: 128 + 43 requesty) i przeliczyć analizę na 500 parach
1. V2-00 (inwentarz podaży)          [CPU, minuty]
2. V2-01 (sędzia odpowiadalności)    [ADR + lokalny GPU, godziny]   ─┐ równolegle
   V2-02 (focus_v2)                  [CPU, godziny]                 ─┘
3. V2-03 (ADR + moduł pair_policy_v2, predykcje zamrożone)          [CPU]
4. V2-04 (konstruowane rejected, jeśli kwoty niedomknięte)          [CPU/lokalny]
5. budowa par v2 + V2-05 (audyt dual-LLM, bramka predykcji)         [API, ~2 dni]
6. decyzja właściciela → V2-06 (ablacja DPO, probe + M-03, 5 seedów) [GPU]
```

Zależności twarde: oś C czeka na V2-02; ADR V2-03 czeka na pełny wynik audytu
v1 i na kalibrację V2-01; V2-06 czeka na pozytywną bramkę V2-05 i osobną
autoryzację.
