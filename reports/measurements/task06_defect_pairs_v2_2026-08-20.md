# Pomiar: pary zakotwiczone w defektach (polityka v2) i ślepy eksport audytowy

Kontrakt budowy: `task06-defect-pairs-v2`, status `defect_pairs_built_not_audited`.
Kontrakt eksportu: `task06-defect-pair-audit-blind-export-v2`, status
`blind_export_frozen_not_reviewed`. Kod: `src/doc2query/preferences/pair_policy_v2.py`,
`src/doc2query/preferences/pair_audit_export_v2.py`,
`scripts/build_task06_defect_pairs.py`, `scripts/export_task06_defect_pair_audit.py`,
32 testy CPU. Polityka i wszystkie progi pochodzą z ADR
[`task06_defect_pair_policy_v2.md`](../decisions/task06_defect_pair_policy_v2.md),
zamrożonego **przed** zbudowaniem pierwszej pary (commit `acce46b`, poprzedzający ten
pomiar). Artefakty: `artifacts/task06/same_prompt_expansion_v{1,2,3}/defect_pairs_v2/`,
`artifacts/task06/preference_audit_v3_defect_pairs/`, agregat
[`task06/defect_pairs_v2/summary.json`](task06/defect_pairs_v2/summary.json).

Etap w całości CPU (budowa trzech kohort: 9,6 s). Werdykty odpowiadalności **odczytane**
z journala przypiętego po SHA-256 (`fe675f07…`, 23 676 rekordów), nie generowane tutaj.
Żadnego progu nie zmieniono, żadnej pary nie użyto do treningu,
`task07_training_authorized=false`, `final_tests_used=[]`.

## 1. Zbudowane pary

| kohorta | grupy | `eligible` | **pary** | oś A | oś B | udział z `eligible` |
|---|---|---|---|---|---|---|
| v1 | 500 | 362 | **204** | 189 | 15 | 56,4% |
| v2 | 500 | 466 | **274** | 244 | 30 | 58,8% |
| v3 | 3 000 | 2 791 | **1 800** | 1 653 | 147 | 64,5% |
| **razem** | **4 000** | **3 619** | **2 278** | **2 086** | **192** | **62,9%** |

Dwie kontrole zgodności z wcześniejszym pomiarem podaży wychodzą dokładnie:

- podaż osi A zmierzona 2026-08-20 wynosiła **2 253** pary parowalnych grup; zbudowano
  2 086 par osi A **plus** 192 osi B, czyli 2 278 = 2 253 + **25** grup, które są
  parowalne wyłącznie na osi B. Reszta osi B (167 par) to grupy parowalne na obu osiach,
  które hasz skierował do B;
- rozkład werdyktów na reprezentantach jest identyczny z pomiarem podaży: `yes` 10 804,
  `no` 12 804, `uncertain` 68 (0,3%). **Zero kandydatów bez werdyktu**;
- bramka różnorodności odrzuciła **jedną** parę (`near_duplicate_query_pair`, kohorta v2)
  — tyle samo, co przy pomiarze podaży.

Dominującą przyczyną braku pary jest **brak dopuszczalnego `chosen`** (1 300 grup): to
koszt wymogu „czysty **i** werdykt `yes`”, zmierzony wcześniej jako zachowanie 79,5%
czystych `chosen`. Brak defektowego `rejected` dotyczy tylko 83 grup — naturalna podaż
defektów potwierdza się w praktyce.

**2 278 par to ponad dwukrotność progu 1 000** wymaganego przed finalnym DPO. Dla
porównania: polityka v1.1 miała 2 012 par zbudowanych, ale po bramce audytu tylko **122**
akceptowalne. Uwaga z §8 ADR obowiązuje: **liczba par nie jest miarą jakości polityki**,
bo v2 zdjęła też veto shadow — o wartości danych rozstrzygnie audyt (V2-05), a ostatecznie
probe embeddera.

## 2. Oś B nie dowiozła swojej kwoty: 192 pary wobec nominalnych 250

To najważniejszy wynik negatywny tego etapu i zapisuję go wprost. ADR uzasadnił wybór
cięcia p75 nad p90 m.in. tym, że p90 „grozi zejściem poniżej kwoty 250 par osi B”.
**Przy p75 też zeszło** — podaż osi B to 192 pary.

Przyczyna nie jest w cięciu overlapu, tylko w **zamrożonej regule przypisania osi**.
Hasz daje kolejność prób, a grupa parowalna na obu osiach trafia na tę, którą hasz
postawił pierwszą; ponieważ parowalność osi A jest wysoka, a osi B rzadka, oś A absorbuje
większość grup mieszanych. Widać to w liczbie **954 par zbudowanych na osi zapasowej**
(41,9% wszystkich). Rozmiar straty da się oszacować: 167 grup parowalnych na osi A
**nie** wydało pary osi A (2 253 − 2 086), czyli hasz skierował je do B i tam były
parowalne; z symetrii hasza podobnie liczny zbiór grup parowalnych na obu osiach trafił
do A i tam pary osi B nie wydał. Grup parowalnych na osi B było więc około **359**
(≈ 334 mieszanych + 25 wyłącznie osi B), a do osi B trafiła nieco ponad połowa. To była
**przewidywalna konsekwencja reguły,
której nie skwantyfikowałem przed zamrożeniem** — i nie mogłem, bo policzenie jej
wymagałoby zbudowania par. Reguły nie zmieniam, cięcia nie ruszam, kwoty nie obniżam.

Skutek dla próbki audytowej jest dokładnie taki, jaki przewiduje §6.2 ADR: oś B wchodzi
całą podażą, niewykorzystana kwota przechodzi do osi A.

| oś | kwota nominalna | kwota efektywna | podaż | wzięte |
|---|---|---|---|---|
| A | 250 | **308** | 2 086 | 308 |
| B | 250 | **192** | 192 | 192 |

`sampled_pair_count = 500`, `shortfall_pair_count = 0`, `development_gate_met = true`.
Predykcja P4 (kontrast osi) będzie więc mierzona na 192 parach osi B — mianownik mniejszy,
niż zakładała symetryczna kwota, ale wystarczający dla progu +20 pp.

## 3. Margines primary faktycznie przestał porządkować

Nie jest to deklaracja z configu, ale zmierzony fakt na zbudowanych parach:

| miara | wartość |
|---|---|
| pary z **ujemnym** `primary_margin_delta` (margines `chosen` **niższy** niż `rejected`) | **617 / 2 278 = 27,1%** |
| `primary_margin_delta` p25 | −0,011 (v1) / −0,016 (v2) / −0,137 (v3) |
| `primary_margin_delta` p50 | +1,12 / +1,43 / +1,37 |
| minimum | −11,12 |

Polityka v1 wymagała po stronie `chosen` marginesu **wyższego o co najmniej 1,0**, więc
**żadnej** z tych 617 par nie potrafiłaby zbudować, a mediana +1,3 pokazuje, że nawet w
większości par zgodnych kierunkowo zapas nad progiem v1 był niewielki. Zmiana sygnału
budującego jest zatem realna, a nie kosmetyczna.

Etykieta `shadow_agrees` występuje w 1 665 par (73,1%), czyli w **613 parach (26,9%)
shadow jest odwrotny do primary**. Wszystkie te pary v1 unieważniłaby wetem (i to licząc
tylko połowę wetową po marginesie, bez inwersji rangi). To jest zmierzona cena zdjęcia
weta — zapisana, żeby nikt nie przypisał wzrostu podaży samemu sygnałowi defektu.

## 4. Rozkład defektów i pułapka jednej etykiety

| etykieta `rejected` (raportowa, nieselekcyjna) | pary | udział |
|---|---|---|
| `possible_ambiguous_query` | 2 270 | **99,6%** |
| `shadow_agrees` | 1 665 | 73,1% |
| `lower_primary_margin` | 1 661 | 72,9% |
| **`judge_unanswerable`** | **1 570** | 68,9% |
| **`weak_corpus_round_trip`** | **1 024** | 45,0% |
| `judge_rank_disagreement` | 401 | 17,6% |
| **`high_lexical_overlap`** | **192** | 8,4% |
| `copy_risk` | 93 | 4,1% |

`possible_ambiguous_query` jest praktycznie **stałą** (99,6%), więc jako wymiar slice'owania
w analizie audytu jest bezwartościowa — zapisuję to teraz, przed audytem, żeby nie została
później zinterpretowana jako sygnał. Trzy etykiety osiowe (`judge_unanswerable`,
`weak_corpus_round_trip`, `high_lexical_overlap`) mają sensowne mianowniki.

Cięcia osi B trzymają się w danych: `chosen` osi B ma `content_jaccard` p75 = 0,045–0,051
(pułap 0,0556), a `rejected` osi B minimum 0,0857 (dokładnie cięcie) i medianę 0,103–0,111.

## 5. Ślepy eksport audytowy

Katalog **nowy**: `artifacts/task06/preference_audit_v3_defect_pairs/`; eksporty
`preference_audit_v1/` i `preference_audit_v2/` **nietknięte**.

- populacja 2 278 par (A 2 086 / B 192), próbka **500** par, ziarno **20260820**,
  12 strat (`cohort_id × axis × requested_form`), alokacja proporcjonalna metodą
  największych reszt, porządek po `pair_id`;
- **pasma marginesu nie występują** w stratach (`margin_used_for_stratification=false`);
- orientacja kontrbalansowana **250/250**, zobowiązanie
  `sha256(sól ‖ pair_id ‖ orientacja)` podjęte przed jakąkolwiek oceną, sól opublikowana
  w manifeście, **500/500 zobowiązań zweryfikowanych** maszynowo;
- ślepe rekordy mają **dokładnie pięć** dozwolonych pól (`audit_id`, `passage`, `query_a`,
  `query_b`, `orientation_commitment`) — ten sam zestaw, co w zamrożonym eksporcie v1;
  kontrola wykazała **zero** wycieku `chosen`/`rejected`/osi/werdyktów/marginesów do
  ślepych rekordów, zero duplikatów `audit_id` i zero par o identycznych zapytaniach;
- klucz odślepiający nie zawiera żadnego tekstu zapytań ani pasażu;
- skład próbki: oś A 308 / oś B 192; werdykty `rejected` w próbce `no` 231 / `yes` 269
  (zero `uncertain`); formy `full_question` 310 / `keyword_query` 190; intencje
  `fact_lookup` 169, `entity_lookup` 145, `procedure` 141, `definition` 45;
- `audit_ids_fingerprint = 2b1ad9b69714d88032e829f3be8d2693d401dd629193c3d9a55d1f568e63a38a`.

## 6. Czego ten etap nie robi

- **Audytu v2 nie uruchomiono.** Predykcje P1–P4 pozostają zamrożone i nieodczytane;
  żadna liczba z tego raportu nie może być użyta do ich korekty.
- Przypomnienie długu z §12 ADR: czytnik audytu Groq stratyfikuje po
  `primary_margin_gap_band`, którego eksport v2 celowo nie produkuje. Uruchomienie V2-05
  wymaga **czytnikowej** adaptacji zapisanej osobnym amendmentem — bez zmiany promptu,
  rubryki, modeli, limitów ani reguł decyzyjnych.
- Nie zbudowano par z kohort v4–v11 (czekają na pozytywny audyt), nie ruszono polityki
  v1/v1.1 ani jej artefaktów, nie zmieniono `format.py`, bramki różnorodności, splitów,
  progu `source_en_score ≥ 23,50` ani rubryki sędziów.
- Nie uruchomiono GPU, nie trenowano niczego, nie otwarto testów finalnych.
