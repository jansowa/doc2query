# Task 04 — Kompletny harness ewaluacyjny

> [Centralny rejestr zadań i statusów](README.md). Każda zmiana statusu lub zakresu tego zadania musi aktualizować rejestr w tym samym commicie.

## Status

`IN PROGRESS`

Centralny harness, zamrożone manifesty/ID, metryki i slice’y, raporty
HTML/Markdown, ślepy eksport A/B, bootstrap oraz zamrożona recepta probe
embeddera są zaimplementowane i przetestowane. P-03 ma gotowy kod i testy
kontraktu negatywów. Dev-only kalibracja progu i train-corpus BM25 są już
zmierzone, zweryfikowane i przypięte. Dodano kompletny one-command runner
P-03, który zamraża train ID, wznawia generację i trening, tworzy wspólną
kohortę HN0/HN0+filter/HN1, zrównuje jawny budżet tokenów, ocenia wyłącznie
zamrożony dev i wykonuje paired-query bootstrap przez dedykowany comparator.
W05 pozostaje pomiarowo zablokowany, ponieważ projektowy cache nadal nie
zawiera przypiętych bazowych wag Bielik 1.5B; dry-run zatrzymuje się przed
generacją i nie omija gated access.
Na lokalnej karcie 8 GB rzeczywiście oceniono W03, W05 i W06 w trybie
deterministic oraz diverse na tym samym zamrożonym panelu 100 rekordów z co
najmniej 10 hard negative’ami; wykonano też nieporównywalny 2-step smoke
probe’a.

Pozostają pełne, porównywalne runy probe natural/copy/W03/W05/W06, niezależny
BGE shadow judge, embeddingowe miary diversity, oceny ludzi dla co najmniej
300 przypadków, pełny test rank10/embedder i S00 prompting baseline. Bez tych
pomiarów bramka Fazy B i główny ranking generatorów nie są zamknięte.
Dotychczasowy zakres W03/W05 opisuje
`docs/experiments/task04_8gb_evaluation_2026-07-18.md`, a porównanie z W06
`docs/experiments/task03_w06_vs_1_5b_2026-07-19.md`.

Shortlista nowych baz probe'a (`mmlw-roberta-base`,
`polish-distilroberta`, Ettin 32M/17M), ryzyka oraz redukowana procedura wyboru
recepty v2 są zapisane w
`docs/decisions/probe_embedder_candidates.md`. P-03 zachował model i budżet
recepty v1, ale jawnie podbił wersję kontraktu do `probe-v1.1-p03`; nie wpisano
niewykonanych wyników.

19 lipca ten sam kontrakt intrinsic zastosowano do finalnego checkpointu W06
4.5B/50k. Powstały komplet 500 generacji, raporty i sparowane bootstrapy W06
vs W03/W05. W06 ma potwierdzoną przewagę greedy MRR/nDCG nad W05, ale diverse
retrieval pozostaje nierozróżnialny, a miary różnorodności są gorsze. Nie
zastępuje to nadal niewykonanych pełnych probe embedderów.

Audyt z 18 lipca wykazał, że dotychczasowy kontrakt nie rozdziela rankingu
w małej puli od retrievalu korpusowego, nie ma natywnego polskiego holdoutu
i nie definiuje polityki fałszywych negatywów. Dlatego zadanie ponownie ma
status `IN PROGRESS`, a wyniki W03/W05/W06 są diagnostyczne. Nie wolno na ich
podstawie wybierać finalisty ani uruchamiać pierwszych porównawczych probe.

19 lipca zaimplementowano P-01 Harness v1.1. Kontrakty
`candidate_pool_ranking` i `corpus_retrieval` mają rozłączne prefiksy
`pool_`/`corpus_`, każdy rekord i blok metryk podaje rozmiar puli, a wspólna
walidacja odrzuca Recall@K dla puli mniejszej niż K. Probe korzysta teraz z
jawnego pełnego pliku dokumentów zamiast puli złożonej z rekordów testowych.
Dodano dyskowy BM25 na cachowanej analizie tekstu i brute-force zamrożonego
bi-encodera z revision, licencją i fingerprintami, deterministyczny backfill
puli diagnostycznej oraz korpusowy round-trip@1/5/20/100 ze specyficznością,
marginesem i korelacją z marginesem rerankera. Testy tanie przeszły; nie
zbudowano jeszcze pełnoskalowych indeksów korpusu i nie uruchomiono probe.

19 lipca zaimplementowano kontrakt i kod P-02. Audyt źródeł pierwotnych
wybrał test PolQA jako natywny kandydat oraz odrzucił całe PIRB/MAUPQA jako
jednorodny „native” holdout. Zamrożono rzeczywisty
`test_translated_msmarco_pl` (16 272 rekordy) z profilami `quick=100`,
`medium=500`, `full=16 272`, hashami list ID i fingerprintami. Importer PolQA,
profile kosztu, weryfikacja immutable manifestu, osobne raportowanie
native/translated oraz jawny model-free sygnał `translationese-surface-v1`
są gotowe i przetestowane bez modeli/GPU.

Po usunięciu przejściowego problemu sieciowego domknięto artefakty P-02:
zamrożono 956 pytań `test_native_pl`, pełny korpus PolQA z 7 097 288
dokumentami oraz trzy profile z rzeczywistymi hashami ID i fingerprintami.
Audyt exact-match nie znalazł wspólnych query ani dokumentów z translated
MS MARCO-PL; near-duplicate pozostaje jawnie `NOT MEASURED`. Manifest przeszedł
pełne `--verify` i nie ma blockerów. Nie zbudowano indeksu ani nie uruchomiono
probe. Stan ten był punktem wejścia do implementacji P-03; P-04 nadal
pozostaje nierozpoczęte.

19 lipca zaimplementowano bezpieczną część P-03. Recepta
`probe-v1.1-p03`/`probe-negatives-v1` definiuje deterministyczne HN0,
HN0+filter i HN1 BM25, polityki `drop | demote | keep+log` z domyślnym
`drop`, scoring naturalnych i syntetycznych query przez zamrożony primary,
raport flag per query source/generator oraz komplet provenance w manifestach.
Porównania odrzucają różne wersje recepty, strategie, polityki, progi,
identyfikatory/fingerprinty kalibracji i fingerprinty BM25. Testy jednostkowe
i smoke korzystają wyłącznie z mockowanego rerankera, bez modeli i GPU.

19 lipca domknięto oba pierwotne artefakty P-03. Pełny frozen dev (16 272
query) scored primary dał query-macro próg Youdena `8.617486953735352`;
artefakt ma fingerprint `9ee4280f…3b3f4` i nie używa żadnego testu. Zamrożony
`train-corpus-v1` zawiera 2 211 463 dokumenty; BM25 spaCy ma integrity check
`ok` i fingerprint `e5df2432…2119`. Oba są przypięte w recepcie.

Preflight ujawnił kolejny brak: W05 ma tylko generacje panelu testowego, z
zerowym pokryciem train ID, zaś bazowych wag Bielik 1.5B nie ma w lokalnym
cache. Dodany runner odtwarza brakujące train-query legalnie z checkpointu
po dostępności przypiętego snapshotu; nie używa testu ani naturalnych query
jako substytutu. HN0/HN0+filter/HN1 nadal nie uruchomiono i nie ma
rozstrzygnięcia. Blocker:
`reports/blockers/task04_p03_w05_sensitivity.md`. P-04 nie został rozpoczęty.

## Harness v1.1 — blokery po audycie

Poniższy pakiet jest następnym zadaniem projektu. Kolejność wykonania:
`P-01 → P-02 → P-03 → P-04`. Uzasadnienie historyczne znajduje się w
[`docs/plan_poprawek_po_audytach.md`](../docs/plan_poprawek_po_audytach.md);
operacyjny zakres i status są utrzymywane tutaj oraz w `tasks/README.md`.

### P-01 — rozdzielone protokoły retrieval — `IMPLEMENTED`

- `candidate_pool_ranking`: pozytyw(y) i odziedziczone lub deterministycznie
  uzupełnione negatywy; metryki z prefiksem `pool_`, diagnostyka generatora;
- `corpus_retrieval`: pełny zamrożony `documents.parquet`; metryki z prefiksem
  `corpus_`, główna ocena probe i round-trip generatora;
- indeks korpusowy BM25 oraz zamrożony pomocniczy bi-encoder
  (FAISS albo brute-force), z revision, licencją i fingerprintem;
- `effective_candidate_count`, margines do najlepszego niepozytywnego
  dokumentu i `possibly_ambiguous_query`;
- każda metryka raportuje rozmiar puli; pipeline odrzuca `recall@K`, gdy
  pula ma mniej niż K dokumentów;
- raportuje `corpus_round_trip@1/5/20/100` i jego korelację z marginesem
  rerankera.

Implementacja i testy jednostkowe/smoke są gotowe. Zbudowano diagnostyczny
train-corpus BM25 wymagany przez P-03. Nadal nie zbudowano pełnego indeksu
porównawczego nad całym korpusem ani pomocniczego bi-encodera; ich throughput
oraz round-trip W03/W05/W06 nie zostały zmierzone.

### P-02 — natywny polski holdout — `IMPLEMENTED`

- audyt PIRB, PolQA i ewentualnie MAUPQA w
  `docs/datasets/native_pl_holdout.md`: licencja, pochodzenie języka,
  kontaminacja i overlap z `msmarco_pl`;
- zamrożone `test_native_pl` oraz `test_translated_msmarco_pl`, opcjonalnie
  `test_transfer_ood`, wraz z fingerprintami i hashami ID;
- `evaluate embedder` i raport pokazują native i translated osobno;
- native nie jest używany do strojenia; brak wyniku native oznacza raport
  niekompletny;
- dodać tani, jawnie opisany sygnał „translationese”.

Gotowe: audyt PIRB/PolQA/MAUPQA, przypięte revisions i licencje, bezpieczny
importer test-only, trzy deterministyczne profile kosztu, frozen native i
translated split z fingerprintami/hashami ID, pełny korpus PolQA, weryfikacja
manifestu, osobne sloty native/translated w `evaluate embedder` i raporcie,
status `incomplete` bez zmierzonego native oraz jawny sygnał translationese.
Szczegóły:
[`docs/datasets/native_pl_holdout.md`](../docs/datasets/native_pl_holdout.md).

Zmierzono exact overlap z MS MARCO-PL (zero identycznych query i dokumentów);
near-duplicate pozostaje jawnie niezmierzony i nie jest zastępowany założeniem.
Nie uruchomiono probe, benchmarku PIRB, pełnego indeksu ani żadnego wyniku
eksperymentalnego. Kolejną bramką Harness v1.1 jest P-03.

### P-03 — probe recipe v1 i false negatives — `IMPLEMENTED / MEASUREMENT IN PROGRESS`

- dla naturalnych i syntetycznych query primary reranker flaguje odziedziczony
  negatyw jako `possible_false_negative` według progu kalibracyjnego z Task 02;
- polityka `drop | demote | keep+log`, domyślnie `drop`, identyczna dla
  wszystkich wariantów; raportuje odsetek flag per generator;
- wersja recepty jest częścią manifestu, a porównanie odmawia pracy dla
  różnych wersji;
- jednorazowy sensitivity check W05: HN0, HN0+filter i HN1 BM25. Istotna
  różnica wymaga ADR przed dalszymi porównaniami;
- pełne HN0/HN0+filter/HN1/HN2/HN3 pozostaje bramką przed Task 09.

Kod, konfiguracja, manifesty, walidacja porównań i testy są gotowe. Domyślna
recepta działa fail-closed: wymaga przypiętego artefaktu Task 02 utworzonego
wyłącznie na dev i weryfikuje jego ID, fingerprint, fingerprint danych,
SHA-256 score’ów, rewizję primary, przestrzeń score’u, operator, próg oraz
metodę jego wyboru. HN1 dodatkowo wymaga przypiętego fingerprintu indeksu BM25.

Dev-only kalibracja oraz train-corpus BM25 są gotowe i przypięte. Przypięty
snapshot `speakleash/Bielik-1.5B-v3` revision
`4b25049621bf3952a1fc9314c89773102eda0333` został legalnie skopiowany do
projektowego cache i właściwy sensitivity check W05 rozpoczął generację.
Wynik nie jest jeszcze dostępny. Gotowy runner
`scripts/run_p03_w05_sensitivity.sh` wykonuje fail-closed preflight, zamraża
deterministyczne train ID i fingerprint, generuje dokładnie jedno greedy query
z checkpointu `runs/W05-1.5B-50K-8GB/checkpoint-3125` z resume bez duplikatów,
materializuje wspólną kohortę legalnych negatywów, trenuje trzy ramiona z
identycznym modelem/LR/batchem/seedem/max_length/max_steps i stałym padded
token budgetem, wznawia trening oraz ocenia tylko `dev_intrinsic_rank10`.
Dedykowany comparator zezwala na drift strategii HN i jej artefaktu BM25, ale
odrzuca wszystkie pozostałe różnice kontraktu; raportuje paired-query
bootstrap 95% CI, flag/drop rates, throughput i peak VRAM. Wynik istotny albo
nierozstrzygalny automatycznie zapisuje ADR bez wyboru recepty. Mock smoke i
testy kontraktowe są gotowe. Runner raportuje w konsoli i wspólnym logu
postęp, throughput, czas i ETA generacji, przygotowania ramion, treningów
probe i ewaluacji dev. Nie wolno uznać P-03 za pomiarowo zamknięte ani przejść
do P-04 lub porównań generatorów przed zakończeniem właściwego runu.

Pierwsza próba zapisała komplet 10k generacji, lecz zatrzymała się przy HN1,
bo `.venv-gpu` nie zawierało przypiętego przez indeks
`pl_core_news_lg==3.8.0`. Zależność jest już częścią bootstrapu i przechodzi
smoke. Przygotowanie zapisuje teraz osobny atomowy cache po każdym ramieniu.
Utracone HN0+filter trzeba przeliczyć raz; zbiorczy, jawnie przypięty scoring
GPU zachował wszystkie flagi na panelu zgodności i przyspieszył smoke około
15× względem pierwotnej ścieżki. Szczegóły:
[`task04_p03_runtime_recovery_2026-07-20.md`](../docs/experiments/task04_p03_runtime_recovery_2026-07-20.md).

### P-04 — kontrakt statystyczny i budżetowy

Przed pierwszym porównaniem utworzyć ADR z:

- główną metryką (propozycja: `corpus_ndcg@10` probe na `test_native_pl`);
- metrykami i marginesami non-inferiority dla grounding, answerability
  i formatu;
- minimalnym praktycznym efektem, seedami successive halving oraz osobnym
  raportowaniem wariancji między treningami i bootstrapu po query;
- budżetem liczonym równocześnie w tokenach, parach, unikalnych pasażach
  i K query/pasaż;
- regułami dev oraz jednorazowego otwarcia finalnych testów.

Pipeline porównań musi cytować wersję ADR i odmawiać porównania niezgodnych
definicji budżetu. Dopiero spełnienie P-01…P-04 zezwala na porównawcze probe,
eksperymenty D00–D12 oraz Task 06.

## Cel

Zbudować centralny system ewaluacji generatora oraz end-to-end wpływu syntetycznych query na embedder.

## Zależności

Taski 01–03. Część infrastrukturalna może być rozwijana równolegle z Taskiem 03.

## Zbiory ewaluacyjne

Utwórz i zamroź:

1. `dev_intrinsic` — do strojenia promptów i progów;
2. `test_intrinsic` — naturalne query, niewidziane dokumenty;
3. `test_adversarial` — ręczne przypadki z Tasku 02;
4. `test_human_panel` — próbka do ocen A/B;
5. `test_embedder` — pełny retrieval test, niewykorzystywany do strojenia generatora.

Zapisz wersje, fingerprint i hash listy ID.

## Generacja ewaluacyjna

Każdy model ma być oceniany w co najmniej dwóch trybach:

- deterministic: greedy lub niska temperatura;
- diverse: ustalona temperatura/top-p i K próbek.

Parametry generacji są częścią identyfikatora runu.

## Intrinsic metrics

### Retrieval/grounding

- rank pozytywnego pasażu wśród co najmniej 10 hard negative’ów;
- Recall@1, Recall@5;
- MRR;
- nDCG@10;
- średni i percentylowy reranker margin;
- sentence-level source hit.

### Lexical copying

- content lemma Jaccard;
- query lemma precision/recall względem passage;
- longest common n-gram;
- normalized LCS;
- copy density;
- entity/number preservation;
- rozkład, nie tylko średnia.

### Diversity

Dla K query na dokument:

- distinct-1/2;
- Self-BLEU;
- mean/max pairwise lemma Jaccard;
- mean/max pairwise embedding cosine;
- duplicate rate;
- semantic cluster count;
- style entropy;
- focus entropy.

### Format i język

- empty/multiple query rate;
- prefiks/metakomentarz;
- długość;
- language ID;
- znaki niedozwolone;
- JSON validity w trybie multi-query.

### Focus

- predicted sentence index;
- bucket distribution;
- accuracy względem kontrolki;
- first-sentence concentration;
- Gini/entropy.

## Slice’y

Wszystkie kluczowe metryki rozbij według:

- overlap kwantyla naturalnego query;
- długości passage;
- liczby zdań;
- target sentence position;
- domeny;
- query style;
- obecności encji/liczb;
- liczby pozytywów;
- trudności rerankera;
- doc near-duplicate cluster size.

## Probe embedder

Zaimplementuj `train_probe_embedder.py` z zamrożoną receptą. Celem nie jest stworzenie najlepszego embeddera, tylko porównanie generatorów.

Wymagania:

- ten sam model bazowy i tokenizer dla wszystkich wariantów;
- ten sam budżet kroków/tokenów;
- identyczny sampling pozytywów i hard negative’ów;
- te same seedy;
- zapis pełnego configu;
- trening na naturalnych query jako gold-data control;
- trening na prostych kopiach/heurystycznych query jako negative control;
- trening na syntetycznych query każdego generatora.

Loss może być MultipleNegativesRankingLoss, CachedMNRL, contrastive margin lub recepta zgodna z docelowym embedderem, ale raz wybrana musi być zamrożona na czas porównań.

## Retrieval evaluation embeddera

Raportuj:

- Recall@1/5/10/100;
- MRR@10;
- nDCG@10;
- MAP;
- hard-negative win rate;
- latency i rozmiar indeksu, jeśli istotne.

Dla każdego porównania wykonaj bootstrap po query:

- różnica metryki;
- 95% CI;
- odsetek bootstrapów, w których wariant wygrywa.

## Raport HTML/Markdown

`build_report.py` tworzy:

- executive summary;
- tabelę eksperymentów;
- wykresy rozkładów;
- Pareto grounding–copying–diversity–embedder score;
- slice’y;
- statystykę istotności;
- co najmniej 100 przykładów side-by-side;
- sekcję „reward hacking / failure modes”.

Raport ma wyraźnie oznaczać metryki niewykonane.

## Human evaluation export

Eksport CSV/JSONL bez nazwy modelu:

- passage;
- query A/B;
- kolejność losowa;
- pytania oceniające;
- hidden experiment IDs.

Importuj oceny i licz Cohen/Fleiss kappa albo Krippendorff alpha zależnie od liczby oceniających.

## Testy

- ręcznie znane rankingi dają poprawne MRR/nDCG;
- duplikaty dają oczekiwaną karę diversity;
- bootstrap jest deterministyczny dla seed;
- slice’y sumują się do całości;
- brak danych nie jest zamieniany na zero;
- porównanie runów sprawdza zgodność test fingerprintu.
- protokoły `pool_*` i `corpus_*` mają rozłączne nazwy;
- `recall@K` jest odrzucane dla puli mniejszej niż K;
- porównanie runów sprawdza wersję recepty probe i kontraktu budżetowego;
- manifest native/translated odrzuca zmianę źródła, rekordów lub profilowej
  listy ID;
- importer native przyjmuje wyłącznie test i nie wymaga sieci/modelu/GPU;
- raport embeddera bez zmierzonego `test_native_pl` ma status `incomplete`;
- sygnał translationese ujawnia składowe i nie deklaruje dowodu tłumaczenia;

## Kryteria akceptacji

- jeden command ocenia checkpoint i generuje komplet artefaktów;
- raport generatora i probe embeddera można odtworzyć z manifestu;
- pipeline odrzuca porównanie runów na różnych wersjach testu;
- raport bez wyniku `test_native_pl` jest oznaczany jako niekompletny;
- główny ranking wariantów może używać wyniku probe embeddera, nie tylko rewardu.
