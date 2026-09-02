# Screen probe Task 07: wyniki częściowe (8/13 ramion) i słownik ramion (2026-09-02)

## Status

Preregistrowany screen probe embedderów
([preregistracja](../preregistrations/task07_probe_screen_v1.md)): dla każdego
ramienia generatora ten sam zamrożony probe (`polish-reranker-base-ranknet`,
bi-encoder, 500 kroków) trenowany na identycznych 1 038 slotach
(pasaż → zapytanie ramienia), oceniany retrievalem na **6 598 naturalnych
zapytaniach dev** po pełnym korpusie 2,4 mln dokumentów. Δ = parowana różnica
`corpus_recall_at_10` względem `start`, bootstrap CI 95% po zapytaniach.
Policzone 8/13 ramion (dysk zapełnił się przy dziewiątym; kolejka wznowiona,
reszta w trakcie). Jeden seed — to screening, nie confirm. `final_tests_used=[]`.

## 1. Wyniki

| ramię | co to jest (jedno zdanie) | recall@10 | Δ vs start (CI 95%) | werdykt |
|---|---|---|---|---|
| [divch](#divch) | DPO na parach wad z **różnicowanym chosen** w grupie — wariant antykolapsowy | **0,1253** | **+0,0432** [+0,036; +0,050] | **istotnie lepszy** |
| [defect_csft](#defect_csft) | zwykły continued SFT wyłącznie na potwierdzonych dobrych `chosen` z kohorty wad | 0,1149 | +0,0328 [+0,026; +0,040] | istotnie lepszy |
| [nearmiss_dpo](#nearmiss_dpo) | DPO na parach, gdzie `rejected` to niemal-dobry kandydat (drugi z rankingu) | 0,1025 | +0,0204 [+0,014; +0,027] | istotnie lepszy |
| [rpo](#rpo) | DPO na parach wad + regularyzator NLL na `chosen` (λ=1,0) | 0,0969 | +0,0148 [+0,009; +0,021] | istotnie lepszy |
| [beta02](#beta02) | DPO na parach wad z mocniejszą kotwicą KL (β=0,2 zamiast 0,1) | 0,0959 | +0,0138 [+0,007; +0,021] | istotnie lepszy |
| [bottom_dpo](#bottom_dpo) | DPO na parach v3, gdzie `rejected` to najgorszy kandydat (kontrast tematyczny) | 0,0852 | +0,0032 [−0,003; +0,010] | nieistotny |
| [start](#start) | lipcowy SFT Bielika 4.5B na 50 tys. par MS MARCO — punkt odniesienia | 0,0821 | — | baseline |
| [defect_dpo](#defect_dpo) | goły DPO na parach wad (β=0,1) — bez zabiegów antykolapsowych | 0,0784 | −0,0036 [−0,010; +0,003] | nieistotny |

Rozstrzygnięcie na szczycie: **divch > defect_csft** parowane Δ=+0,0104,
CI [+0,003; +0,018] — istotne.

**Kluczowa obserwacja:** ranking podąża za duplikatami w generacji (kolaps),
nie za marginesem preferencji na dev. `defect_dpo` (22,2% duplikatów, margines
0,921) daje zero; `divch` (10,0% duplikatów, te same pary wad) daje +0,043.
Surogaty treningowe nie przeniosły się na wynik; różnorodność generacji tak.

## 2. Słownik ramion

Wspólny fundament wszystkich ramion: adapter startowy
`D01-4.5B-STYLE-50K-S42` (zob. [start](#start)), QLoRA na budżecie 8 GB VRAM,
seed 42, budżety kroków/tokenów dopasowane między ramionami wewnątrz kohorty.
Trzy **kohorty par** (bottom, near_miss, defect) × trzy **algorytmy**
(`dpo`, `csft`, `wsft`) + trzy warianty antykolapsowe na kohorcie defect.

### start

Punkt odniesienia: Bielik-4.5B-v3.0-Instruct + adapter SFT
`D01-4.5B-STYLE-50K-S42` z połowy lipca — uczony na 50 tys. par
pasaż→zapytanie z przetłumaczonego MS MARCO (z kontrolą stylu/formy).
Wszystkie pozostałe ramiona to dalszy trening TEGO adaptera; pytanie
kampanii brzmi „czy cokolwiek poprawia ten punkt startowy na prawdziwym
kryterium". Audyt pełnej puli (2026-09-01) wykazał, że ~12,6% jego danych
treningowych ma twarde wady (nieodpowiadalne/niesensowne).

### bottom_dpo

Kohorta **v3 bottom**: 2 730 par z zamrożonej polityki v3, gdzie `chosen` to
zwycięzca turnieju kandydatów dla pasażu, a `rejected` to kandydat z **dna
rankingu**. Diagnoza kontrastu (2026-08-28): w 75,3% par `rejected` jest po
prostu nie na temat pasażu (pokrycie 0,200) — para uczy więc głównie
rozróżnienia tematycznego, rzadziej jakościowego (obie strony ugruntowane
tylko w 7,1% par). DPO sigmoid, β=0,1, logproby referencji precomputowane.
Wynik probe (+0,003, nieistotny) potwierdza diagnozę: kontrast tematyczny
nie uczy niczego, czego start już nie umiał.

### nearmiss_dpo

Kohorta **near_miss**: 2 428 par tej samej polityki, ale `rejected` to
**niemal-dobry** kandydat (drugi w rankingu zamiast ostatniego) — kontrast
bliższy jakości: pokrycie `rejected` 0,333, par z obiema stronami
ugruntowanymi 1,5× więcej niż w bottom. Ten sam algorytm DPO. Wynik (+0,020,
istotny) wobec zera dla bottom_dpo to czysty pomiar wartości **lepiej
skonstruowanego negatywu** przy wszystkim innym stałym.

### defect_dpo

Kohorta **defect** (pomysł właściciela): 1 794 pary, w których każda para
adresuje **nazwaną klasę wady** (`not_answerable`, `too_general`,
`wrong_form`, `copy_phrasing`) z metadanymi `defect_class`; negatywy głównie
organiczne (wykopane z realnych kandydatów, nie sztucznie mutowane),
każda para potwierdzona jednomyślnie 2/2 w obu kolejnościach; klasy
oblewające audyt anty-skrótowy (AUC>0,80) zablokowane. Goły DPO β=0,1.
Na dev wygląda świetnie (margines 0,921, accuracy 93,2%), ale generacja
**kolapsuje** (22,2% duplikatów przy 4,5% startu) i wynik probe jest zerowy
(−0,004) — wady par były dobrym pomysłem, goły DPO złym nośnikiem.

### beta02

Ramię antykolapsowe nr 1 (ADR `task07_anti_collapse_v1`): identyczne pary
defect, ale **β=0,2** — dwukrotnie mocniejsza kotwica KL do modelu
referencyjnego, hipoteza „mniejszy dryf = mniejszy kolaps". Efekt: duplikaty
19,7% (ledwo lepiej), NLL chosen 0,729, probe +0,014 (istotny, ale słaby).
Kotwica KL sama w sobie kolapsu nie leczy.

### rpo

Ramię antykolapsowe nr 2: DPO + **regularyzator NLL** na `chosen`
(`loss = DPO + 1,0·NLL(chosen)`, wariant RPO) na parach defect. Najlepszy
profil treningowy z całej rodziny DPO: NLL chosen 0,553 (lepszy niż start
0,624!), accuracy preferencji 94,4%, duplikaty 14,0%. Probe +0,015 —
porównywalny z beta02. Chroni jakość dopasowania, kolaps ogranicza
połowicznie.

### divch

Ramię antykolapsowe nr 3 — **zwycięzca screena**: te same 1 794 pary defect,
ale `chosen` **różnicowany wewnątrz grupy** (275 podmian): tam, gdzie grupa
wystawiała do trzech par z identycznym `chosen`, klasy wad dostają RÓŻNE
potwierdzone dobre zapytania (kandydaci po weryfikacji klasyfikacja `ok` +
answerability TAK), rotacja deterministyczna. Usuwa to drugą przyczynę
kolapsu z ADR: trening „jednej odpowiedzi na pasaż". Jedyne ramię poniżej
progu ADR (duplikaty 10,0% ≤ 12%), margines zachowany (0,845). Probe:
**+0,043** — najlepszy wynik, istotnie lepszy także od defect_csft.
Ograniczenie: `chosen_components` po podmianie nie opisują nowego chosen,
więc na tej kohorcie wolno trenować tylko DPO (weighted SFT zabroniony).

### defect_csft

Kontrola algorytmiczna kohorty defect: **continued SFT** — zwykły trening
językowy wyłącznie na stronach `chosen` (bez funkcji preferencji, bez
negatywów), budżet tokenów dopasowany do DPO. Wynik +0,033 (drugi!) to
ważny sygnał: sama ekspozycja na dodatkowe, potwierdzone dobre zapytania
daje większość zysku, którego goły DPO nie umiał dowieźć — i stawia
poprzeczkę każdej metodzie preferencyjnej.

### Ramiona w trakcie liczenia (5/13)

- **defect_wsft** — weighted SFT na kohorcie defect: continued SFT na chosen
  ważony `pool_margin` (jak mocno chosen wygrał turniej).
- **nearmiss_csft / nearmiss_wsft** — te same dwa algorytmy kontrolne na
  kohorcie near_miss.
- **bottom_csft / bottom_wsft** — i na kohorcie bottom.

## 3. Zastrzeżenia

Jeden seed na ramię, budżet 1 038 par (przecięcie slotów po kolapsie,
jednolite K=3 — kontrakt P-04), rzadkie etykiety MS MARCO (stąd analiza
parowana jako podstawa werdyktów). Ranking czołówki potwierdzi confirm
(wiele seedów), a przenośność probe na realistyczny trening — M-01
(średnia skala od bazy MLM, prospektywny ADR). Ten pomiar nie zamraża
finalistów i nie otwiera testów finalnych.

`final_tests_used=[]`
