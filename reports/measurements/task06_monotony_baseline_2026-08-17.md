# Pomiar: baseline monotonii zapytań (oś D) na 11 zamrożonych kohortach

Kontrakt: `task06-query-monotony-baseline-v1`, status
`design_input_measured_no_thresholds`. Kod:
`src/doc2query/evaluation/query_monotony.py`,
`scripts/run_task06_monotony_baseline.py`, 9 testów CPU. Artefakt:
`reports/measurements/task06/monotony_baseline_v1/summary.json` (wejścia pinowane
po SHA-256). `task07_training_authorized=false`, `final_tests_used=[]`.

Populacja: **224 000 wygenerowanych zapytań** z kohort `same_prompt_expansion_v1`
… `v11` (4000 + 4000 + 9 × 24 000), czyli wszystkie kandydatury, nie tylko
reprezentanci bramki. Czas liczenia: 19 s CPU.

To jest **zadeklarowane wejście projektowe**, jak V2-00: czyta wygenerowane teksty
jawnie, ale nie zamraża żadnego progu, nie buduje pary, niczego nie promuje i nie
jest bramką. Tokenizacja jest jawnie prymitywna (regex na słowach, lowercase,
**bez lematyzacji**), bo żaden polski model spaCy nie jest tu przypięty — raport
mówi to wprost, zamiast udawać różnorodność na poziomie lemm.

## Wynik główny: monotonia słów początkowych jest **dyktowana kontrolką**, nie kolapsem modelu

Pooled na `v3` (24 000 zapytań) dwa pierwsze słowa zbierają **połowę całej
populacji**: `definicja` 25,18% i `jak` 25,03%. Rozkład ten jest identyczny we
wszystkich 11 kohortach (top-1 udział 0,251–0,253). Przyczyna jest w przekroju po
kontrolce `intent`:

| `intent` | n (v3) | distinct pierwszych słów | top-1 udział | pierwsze słowo | znorm. entropia | średnia dł. (słowa) |
|---|---|---|---|---|---|---|
| `procedure` | 6000 | **1** | **1,0000** | `jak` | **0,0000** | 6,26 |
| `definition` | 6000 | 10 | **0,9963** | `definicja` | 0,0129 | 3,47 |
| `fact_lookup` | 6000 | 34 | 0,1965 | `czy` | 0,6682 | 5,99 |
| `entity_lookup` | 6000 | 816 | 0,1815 | `nazwa` | 0,6280 | 4,65 |

Powtarzalność między kohortami jest niemal doskonała: `procedure` ma **distinct=1
i entropię 0,0000 w 8 z 11 kohort** (w pozostałych trzech distinct=2, top-1
0,9990–0,9998), a `definition` top-1 0,9948–1,0000. Dla kontrastu
`entity_lookup` ma 93–885 różnych słów początkowych, a `fact_lookup` 17–36.

**Interpretacja, która zmienia adresata problemu P3.** „Pytania zawsze zaczynają
się tak samo” nie jest w tych danych stochastycznym kolapsem generatora, który
dałoby się wyprostować celem treningowym — to **deterministyczna konsekwencja
szablonu kontrolki**. Połowa populacji zaczyna się od `jak` albo `definicja`,
ponieważ dwie z czterech kontrolek intencji praktycznie wymuszają otwarcie. Cel
parowy (DPO) ani set-level nagroda (GRPO) nie mają tu czego naprawiać: model
robi dokładnie to, o co go proszono. To wzmacnia rozdział 3 specyfikacji v2 —
monotonia mieszka w **kontrolkach**, nie w parach — ale przesuwa punkt ciężkości:
nie chodzi o „dodać różnorodność do generacji”, a o to, że **dwie kontrolki
intencji są napisane jak szablony**.

Konsekwencja operacyjna dla M-05 (obowiązkowa ewaluacja trybu produkcyjnego bez
selektora): raport rozkładu słów początkowych **musi** być cięty per `intent`.
Policzony globalnie „odkryje” szablon kontrolki i zostanie zinterpretowany jako
monotonia modelu — co byłoby błędem atrybucji.

## Kontrolka długości nie została w tych kohortach w ogóle użyta

We **wszystkich 11 kohortach** jedyną zaobserwowaną wartością `control.length`
jest `medium`. Kontrolka `short`/`long` nigdy nie była żądana, więc:

1. **nie da się** z tych danych ocenić posłuszeństwa kontrolce długości ani
   przypisać monotonii długości modelowi — to kontrolka nieprzetestowana, nie
   kontrolka nieskuteczna;
2. zaobserwowana ciasnota rozkładu (pooled: średnia 5,09–5,15 słowa, mediana 5,
   p05–p95 = **2–9**, identycznie w v3…v11) jest w takim samym stopniu własnością
   żądania, jak modelu.

Kontrolka `form` natomiast **wyraźnie separuje** rozkład, co jest dowodem, że
kontrolki w ogóle działają: `full_question` 6,12 słowa (p05–p95 3–10, 34 różne
słowa początkowe), `keyword_query` 4,06 słowa (p05–p95 2–7, 816 różnych słów
początkowych).

## Baseline set-level dla nagrody GRPO (notatka V2-07)

Distinct-n liczone **na zbiorze zapytań per pasaż** (grupa same-prompt), czyli
dokładnie w formie kandydującego komponentu set-level nagrody z Task 08:

| kohorta | grup | distinct-1 (średnia) | distinct-2 (średnia) |
|---|---|---|---|
| v1 | 500 | **0,326** | **0,455** |
| v2 | 500 | 0,479 | 0,669 |
| v3–v11 | 3000 każda | 0,470–0,477 | 0,659–0,668 |

Rozkład w v3 (n=3000 grup): distinct-1 p05 0,227 / p50 0,476 / p95 0,731;
distinct-2 p05 0,333 / p50 0,688 / p95 0,933.

Dwa wnioski. Po pierwsze, **v1 wyraźnie odstaje w dół** (0,326 vs ~0,474) — jest
to niezależne potwierdzenie zmierzonego wcześniej kolapsu różnorodności v1
(`duplicate_rate` 0,399), tym razem innym przyrządem i na innej jednostce
(tokeny, nie duplikaty zapytań). Po drugie, szerszy decoding kohort v2+ podniósł
różnorodność wewnątrzgrupową o ~45% względnie i **ustabilizował** ją: dziewięć
niezależnych kohort po 3000 grup mieści się w przedziale 0,470–0,477. To daje
nagrodzie GRPO **stabilny punkt odniesienia**, a nie jednorazowy pomiar.

## Czego ten pomiar nie robi

Nie zamraża progów, nie buduje par, nie jest bramką, nie zmienia
`format.py`, bramki różnorodności, polityki par v1/v1.1, kontrolek generacji ani
statusu Tasków 07/08. Nie otwiera testów finalnych. Nie reaktywuje
`entity_preservation` jako detektora halucynacji — to wymagało backendu spaCy
(`pl_core_news`), a modele spaCy są hostowane na GitHubie, który jest z tej
maszyny nieosiągalny (connect timeout); pozostaje to niewykonaną częścią
opcjonalnego kroku 6.
