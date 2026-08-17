# Pomiar: kalibracja proxy odpowiadalności v1 — **kryterium niedowiezione**

Kontrakt: `task06-answerability-proxy-v1`. ADR zamrażający protokół:
[`task06_answerability_proxy_v1.md`](../decisions/task06_answerability_proxy_v1.md)
(commit `8bec836`, **zamrożony przed** policzeniem czegokolwiek, co wiąże cechę z
etykietą). Artefakt: `artifacts/task06/answerability_proxy_v1/answerability_proxy_v1.json`.
Kod: `src/doc2query/preferences/answerability_proxy.py`,
`scripts/run_task06_answerability_proxy.py`, 10 testów CPU.
`human_evidence_claimed=false`, `task07_training_authorized=false`,
`final_tests_used=[]`.

## Wynik jednym zdaniem

**Proxy nie przeszło.** Na holdoucie czystość predykcji „odpowiadalne” wynosi
**0,8707** wobec zamrożonego progu **0,88** (podaż przeszła: `recall_yes` 0,9078
wobec progu 0,50). Zgodnie z §7 ADR proxy **nie jest** używane jako filtr strony
`chosen`, a oś A polityki par v2 powstanie **bez kontroli odpowiadalności**.

To jest realizacja predykcji zapisanej w ADR przed odczytem: „spodziewam się, że
P1 nie przejdzie albo przejdzie z CI obejmującym 0,88”. CI faktycznie obejmuje
próg ([0,816; 0,918]), a punktowa wartość leży pod nim — więc wynik nie jest ani
zaskoczeniem, ani porażką przyrządu.

## Migawka etykiet (pinowana)

Audyt v1 jest niedokończony (`incomplete_quota_deferred`; próba wznowienia
2026-08-17 18:20 UTC wykonała **0 requestów**, bo dzienne budżety obu modeli są
wyczerpane do 00:00 UTC). Kalibracja użyła migawki dnia 1, pinowanej po SHA-256:

| plik | SHA-256 |
|---|---|
| `sample.jsonl` | `179bbc62020f6318…` |
| `machine_key.jsonl` | `6222273fdc15ece0…` |
| `groq_dual_llm/pair_verdicts.jsonl` | `f2d2e702241f86b7…` |

| własność zbioru etykiet | wartość |
|---|---|
| strony z dwiema ocenami | 488 |
| strony z etykietą konsensusu | **392** (`yes` 306, `no` 86) |
| strony odrzucone: sędziowie rozjechani | 96 (19,7%) |
| strony bez pełnej pary ocen | 512 |
| zgodność sędziów co do odpowiadalności (sufit szumu) | **0,8033** |
| baza klasy większościowej `yes` | **0,7806** |

Wszystkie te liczby były **znane przed** zamrożeniem ADR i są tam jawnie
wyliczone — bez nich progu 0,88 nie dałoby się uczciwie uzasadnić (kryterium
„accuracy” byłoby wobec bazy 0,78 puste). Żadna cecha nie została zestawiona z
etykietą przed zamrożeniem.

## Podział i wybrana reguła

Podział deterministyczny po `sha256(audit_id)`, obie strony pary w tej samej
połowie (pasaż nie przecieka): **fit 212 stron** (165 `yes`), **holdout 180
stron** (141 `yes`). Holdout odczytany **dokładnie raz**.

Przestrzeń: 13 zamrożonych cech × 2 kierunki × decyle połowy fit, atom
pojedynczy albo koniunkcja dwóch atomów — **14 920 reguł**, z czego **253**
spełniły kryterium na fit. Zwycięzca wg zamrożonego celu (max `recall_yes` przy
`precision_yes ≥ 0,88` i `recall_yes ≥ 0,50`):

```
longest_copied_ngram <= 3  AND  pool_positive_score >= 7.777
```

| zbiór | `precision_yes` | `recall_yes` | `accuracy` | `balanced_accuracy` |
|---|---|---|---|---|
| fit (n=212) | 0,9042 | 0,9152 | 0,8585 | 0,7874 |
| **holdout (n=180)** | **0,8707** | **0,9078** | 0,8222 | 0,7103 |

95% CI bootstrap na holdoucie: `precision_yes` [0,8163; 0,9184],
`recall_yes` [0,8582; 0,9504]. Macierz pomyłek holdoutu: TP 128, FP 19, TN 20,
FN 13.

Spadek czystości fit → holdout (0,904 → 0,871) o ~3,4 pp przy 14 920
przeszukanych regułach jest dokładnie tym, po co holdout istniał: część przewagi
na fit była wyborem progu pod szum.

## Co to znaczy merytorycznie

1. **Kierunek sygnału jest realny, siła — niewystarczająca.** Baza `yes` na
   holdoucie to 0,7833; reguła podnosi czystość do 0,8707, czyli zbija udział
   nieodpowiadalnych wśród zatrzymanych z 21,7% do 12,9% (redukcja masy defektu
   o ~40% względnie). To **mniej** niż wymagane ≥45% i nie zbliża się do
   ambicji „≤5% nieodpowiadalnych `chosen`” ze szkicu predykcji V2-03.
2. **Reguła jest interpretowalna i zgodna z osią B, nie z osią A.**
   `longest_copied_ngram ≤ 3` to warunek antykopiowania, a
   `pool_positive_score ≥ 7,78` to absolutny score sędziego primary dla pasażu
   (nie margines!). Innymi słowy: to, co w tych polach da się złowić, to
   „pasaż w ogóle pasuje do zapytania”, a nie „pasaż zawiera odpowiedź”. Jest to
   spójne z wynikiem audytu v1, że `corpus_round_trip` nie różnicuje
   odpowiadalności — cała rodzina cech leksykalno-rankingowych mierzy
   odzyskiwalność, nie odpowiadalność.
3. **Absolutny score bije margines.** Zwycięzca używa `pool_positive_score`,
   a nie `pool_margin`; wśród 253 dopuszczalnych reguł najwyższą czystość na fit
   (1,000 przy `recall` 0,382) dała para `longest_copied_ngram ≤ 3` i
   `pool_positive_score ≥ 11,855`. To niezależne potwierdzenie tezy polityki v2:
   cross-encoder działa jako **filtr absolutnego score** (InPars, Promptagator),
   nie jako ranking dwóch zapytań między sobą.
4. **Zastrzeżenie na korzyść proxy, którego nie wolno użyć jako furtki.**
   W przekroju per rola czystość na stronach `chosen` wynosi 0,9024 (n=91), a na
   `rejected` 0,8308 (n=89). Kusi zawężenie kryterium do roli `chosen`, bo tam
   proxy „przechodzi”. **Nie robimy tego**: rola jest funkcją porządku
   marginesowego, którego v2 świadomie porzuca, przekrój nie był w ADR
   kryterium, a wybór podzbioru po zobaczeniu wyniku to dokładnie ta operacja,
   której cała ta procedura ma zapobiegać.
5. **Granica pomiaru.** Holdout zawiera wyłącznie strony **konsensusowe**, więc
   jest łatwiejszy niż zadanie sędziego, na którym sufit szumu 0,8033 był
   mierzony. Na 96 stronach spornych proxy nie jest oceniane w ogóle. Nie wolno
   więc czytać „0,871 > 0,803” jako „proxy lepsze od sędziego”.

## Konsekwencje (wprost z §7 ADR, bez negocjacji)

1. Oś A polityki par v2 powstaje **bez filtra odpowiadalności po stronie
   `chosen`** — wyłącznie na round-tripie i pozostałych warunkach czystości.
2. ADR V2-03 **nie może** przewidywać spadku udziału nieodpowiadalnych `chosen`
   do 5%. Dopuszczalna predykcja to brak pogorszenia względem zmierzonej
   wartości v1 (z **pełnego** audytu), a poprawa odpowiadalności wraca dopiero z
   przypiętym sędzią lokalnym.
3. Luka odpowiadalności pozostaje **otwartym, nazwanym długiem** osi A. Harness
   V2-01 jest gotowy i fail-closed; brakuje wyłącznie wag 27B na maszynie z
   16 GB VRAM. Klauzula zastąpienia z §1 ADR pozostaje wiążąca.
4. Reguła i artefakt **zostają zapisane** jako punkt odniesienia: gdy sędzia
   lokalny będzie kalibrowany, jego zgodność z tymi samymi etykietami będzie
   porównywalna z 0,8707/0,9078 tej reguły. Proxy nie jest usuwane z kodu, ale
   nie ma prawa niczego filtrować.

## Nietknięte

`format.py`, bramka różnorodności, polityka par v1/v1.1 i jej artefakty,
kontrakt audytu Groq, rubryka sędziów, próg `source_en_score ≥ 23,50`, splity,
`artifacts/task06/teacher_claude_v1/`. Żadnej pary nie zbudowano, żadnego progu
nie dostrojono po odczycie, `final_tests_used=[]`.
