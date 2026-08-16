# Pomiar: generacja kohorty teachera API v1 (2026-08-16)

ADR (prerejestrowany przed generacją):
[`task06_claude_teacher_ablation_v1.md`](../decisions/task06_claude_teacher_ablation_v1.md).
Kontrakt: `configs/preferences/task06_claude_teacher_ablation_v1.yaml`.
Instrukcja autorska (pełna treść przekazana podsesjom):
`artifacts/task06/teacher_claude_v1/authoring_instruction.md`.

Ten raport dotyczy **wyłącznie etapu generacji**. Scoring zamrożonym kontraktem
Task 06 nie został wykonany i nie jest tym ADR autoryzowany, więc nie ma tu
żadnego twierdzenia o tym, czy teacher jest lepszy od lokalnego generatora.

## Co powstało

| pozycja | wartość |
|---|---|
| pasaże | 600 (prefiks kohorty `same_prompt_expansion_v3` w porządku `sha256(cluster_id)`) |
| kontrolki na pasaż | 4 (c0–c3, zamrożony zestaw D01) |
| kandydatury na kontrolkę | 4 |
| rekordy | **9600 / 9600** |
| shardy | 24 / 24 kompletne |
| artefakt | `artifacts/task06/teacher_claude_v1/candidates.jsonl` (`sha256=40f7a6d6f85bb14b…`) |
| walidacja | `artifacts/task06/teacher_claude_v1/cohort.validation.json` |

Walidacja przeszła bez błędów: każdy klaster ma dokładnie 16 rekordów, w obrębie
pary (`cluster_id`, `control_id`) cztery zapytania są różne po normalizacji,
`full_question` zawsze kończy się `?`, `keyword_query` ma 2–6 tokenów `\w+` i nie
kończy się `?`. Globalnie 9593/9600 zapytań jest unikalnych (7 kolizji między
różnymi pasażami, wszystkie to krótkie frazy encyjne). Mediana długości
zapytania: 39 znaków.

Dwa rekordy poprawiono ręcznie po walidacji (`shard_005:167`,
`shard_009:336`): przekraczały limit tokenów `\w+`, bo numer telefonu
`1-800-433-3405` liczy się jako cztery tokeny, a `Chick-fil-A` jako trzy.
Przepisano je na krótsze, nadal gruntowane w pasażu. Zmiana jest odnotowana tu,
a nie ukryta w danych.

## Wynik uboczny o wartości metodologicznej: zamrożone kontrolki D01 często nie mają pokrycia w pasażu

Autor oznaczał `intent_fit=strained`, gdy pasaż nie dawał materiału na zadaną
intencję i trzeba było napisać najbliższe sensowne zapytanie gruntowane w
tekście (zamiast zmyślać treść).

| kontrolka | intencja @ bucket | `strained` |
|---|---|---|
| c0 | fact_lookup @ beginning | 332 / 2400 = **13.8%** |
| c1 | definition @ middle | 662 / 2400 = **27.6%** |
| c2 | procedure @ end | 1454 / 2400 = **60.6%** |
| c3 | entity_lookup @ middle | 684 / 2400 = **28.5%** |
| razem | — | 3132 / 9600 = **32.6%** |

Kontrolka `procedure @ end` jest niewykonalna dla większości pasaży
`msmarco_pl`: to w przeważającej części notki faktograficzne, biogramy, cenniki
i hasła słownikowe, w których żadnej procedury nie ma. Round-robin przypisuje ją
jednak co czwartemu pasażowi. To jest przesłanka dla przyszłego projektu
kontrolek (nie decyzja): warunkowe przypisywanie intencji do pasaży, które mają
na nią pokrycie, byłoby prawdopodobnie skuteczniejsze niż stała rotacja. Zmiana
kontrolek wymaga własnego prospektywnego ADR i nie dotyka niczego zamrożonego.

`focus_fit=degenerate` wystąpiło w 1004/9600 rekordów (10.5%) — pasaże
jedno- i dwuzdaniowe, w których wskazany bucket po prostu nie istnieje.

## Jakość materiału źródłowego

`passage_quality_note` ustawiono dla **266 z 600 pasaży (44.3%)**. Powtarzalne
kategorie usterek, zgłaszane niezależnie przez wszystkie 24 podsesje:

- dosłownie zduplikowane zdania lub całe akapity w obrębie pasażu;
- zdania rozcięte na skrótach (`np.`, `r.`, `łac.`, `dr.`, `Inc.`, `St.`,
  inicjały nazwisk), co daje pseudo-zdania typu `.`, `1.`, `26.`;
- mojibake i uszkodzone kodowanie (`â`, `Â£`, transkrypcje fonetyczne, znaki CJK);
- resztki interfejsu strony w treści (paski nawigacji, „Tagi:”, „Kontynuuj
  czytanie”, liczniki głosów, podpisy zdjęć);
- błędy tłumaczenia nazw własnych (`Georgia` → „Gruzja”, `Lebanon` → „Liban”,
  `Destin` → „przeznaczenia”, `Frostproof` → „mrozoodporność”).

To jest niezależne potwierdzenie ryzyka translationese, które właściciel
świadomie zaakceptował w Task 03 (P06-T anulowany). Raportuję je jako obserwację
jakościową, nie jako podstawę do zmiany frozen train ani progu
`source_en_score >= 23.50`.

Ta sama obserwacja o segmentacji zdań pojawiła się w pomiarze korpusu
walidacyjnego nagrody
([`task06_reward_validation_corpus_v1.md`](task06_reward_validation_corpus_v1.md)),
gdzie odpowiada za 46/180 nierozstrzygniętych focusów.

## Granice

- Teacher nie ma przypiętych wag (transport API), więc kohorta jest
  nieodtwarzalna bit-exact; pozostaje **osobnym ramieniem ablacyjnym**.
- Kandydatury nie pochodzą z samplingu, tylko z intencji autora, więc kohorta
  **nie** wchodzi do bramki różnorodności same-prompt i jej metryki
  różnorodności nie są porównywalne z metrykami studenta.
- Kohorta nie weszła do frozen train, do żadnej kohorty preferencyjnej ani do
  par; niczego na niej nie trenowano.
- Scoring GPU (primary/shadow/corpus), budowa par i audyt dual-LLM pozostają
  nieautoryzowane. `final_tests_used=[]`.

## Następny krok (wymaga decyzji właściciela)

GPU zwolniło się 2026-08-16 po zakończeniu kolejki bezobsługowej. Kohortę
teachera można przescorować **tym samym** zamrożonym kontraktem co kandydatów
studenta i policzyć prerejestrowane kryteria z ADR (odsetek pasaży, w których
najlepszy teacher bije najlepszego studenta na tej samej kontrolce, według
primary, shadow i corpus round-trip, plus raport niezgodności). To jest jedyny
sposób, by ablacja cokolwiek znaczyła — i nadal nie autoryzuje budowy par.
