# Audyt koncepcyjny programu Bielik doc2query

> Dokument roboczy do planowania kolejnych zmian w zadaniach i eksperymentach.

**Data przeglądu:** 18 lipca 2026
**Zakres:** cele projektu, rejestr i treść zadań, `doc2query_research.md`,
metodologia eksperymentalna oraz zgodność z aktualnymi pracami o generowaniu
syntetycznych zapytań i treningu retrieverów.
**Poza zakresem:** szczegółowy code review modułów `.py`.

## 1. Wniosek wykonawczy

Projekt ma mocny i przemyślany szkielet metodologiczny. Szczególnie dobrze
rozwiązuje problemy, które często są pomijane w projektach doc2query:

- odróżnia preferencje między dokumentami od preferencji między query
  wygenerowanymi dla tego samego promptu;
- traktuje wynik końcowego embeddera jako główne kryterium, a metryki
  generatora jako pomocnicze;
- utrzymuje zamrożony primary i shadow judge zamiast współtrenowania
  generatora i rerankera;
- wymaga continued SFT jako kontroli dla DPO;
- blokuje GRPO do czasu wykazania jakości rewardów i konkretnej potrzeby;
- rozdziela `IMPLEMENTED` od `DONE` i nie przypisuje niewykonanych wyników.

Największym ryzykiem nie jest obecnie wybór Bielika ani metoda optymalizacji
generatora. Jest nim możliwość wybrania generatora na podstawie zbyt wąskiej
ewaluacji: rankingu w puli jednego pozytywu i około dziesięciu negatywów,
pochodzących z tego samego tłumaczonego źródła i wydobytych pierwotnie dla
naturalnego, a nie syntetycznego query.

Przed kosztownymi runami 4.5B i przed Taskiem 06 należy wzmocnić Task 04 o:

1. corpus-wide retrieval dla probe embeddera;
2. niezależny native-Polish holdout;
3. negatywy wydobywane ponownie dla syntetycznych query;
4. eksperymenty mieszające query naturalne i syntetyczne;
5. brakujący baseline seq2seq klasy mT5/plT5.

## 2. Stan programu według rejestru

Źródłem prawdy pozostaje [`tasks/README.md`](../tasks/README.md).

| Obszar | Stan | Ocena |
|---|---|---|
| Task 00 — infrastruktura | `DONE` | Fundament wystarczający do dalszej pracy. |
| Task 01 — dane i splity | `IMPLEMENTED` | Dobra kontrola leakage; pozostają percentyle tokenizerów i decyzja o rekordach z mniej niż 10 negatywami. |
| Task 02 — rerankery i rewardy | `IMPLEMENTED` | Integracja działa, ale brakuje pełnego benchmarku primary/shadow na dev/test oraz pełnego ręcznego potwierdzenia. |
| Task 03 — SFT/QLoRA | `IMPLEMENTED` | Wykonalność 1.5B potwierdzona; loss nie otwiera bramki jakościowej. Brakuje S00, oceny downstream i porównań 4.5B. |
| Task 04 — ewaluacja | `IN PROGRESS` | Prawidłowe bieżące wąskie gardło. Kontrakt wymaga uzupełnień opisanych poniżej. |
| Taski 05–07 | `TODO` | Kolejność style/focus → preferences → DPO jest poprawna. |
| Task 08 | `OPTIONAL / BLOCKED` | Powinien pozostać zablokowany. |
| Taski 09–10 | `BLOCKED` | Poprawnie zależą od dowodów z wcześniejszych faz. |
| Task 11 | `OPTIONAL` | Pełny audyt może być późny, ale minimalna separacja sędziów powinna wejść wcześniej. |

## 3. Mocne strony obecnego planu

### 3.1. Poprawna hierarchia dowodów

Plan nie wybiera generatora wyłącznie na podstawie lossu, lexical overlapu ani
logitu rerankera. Główną bramką ma być probe embedder trenowany przy zamrożonej
recepcie i oceniany na naturalnych query. To właściwa hierarchia:

```text
format i sanity checks
        ↓
intrinsic grounding / copying / diversity
        ↓
niezależny judge i panel człowieka
        ↓
retrieval końcowego embeddera
```

### 3.2. Ostrożne użycie rerankera

Task 02 słusznie traktuje reranker jako zamrożone proxy, nie jako dowód
entailmentu. Raportowanie disagreement zamiast ukrywania go w średniej jest
dobrą praktyką.

### 3.3. Poprawna konstrukcja DPO

Task 06 buduje `chosen/rejected` dla tego samego passage i promptu, zachowuje
źródła kandydatów, komponenty score oraz near-miss rejected. Task 07 wymaga
continued SFT o porównywalnym budżecie. Bez tej kontroli poprawy po DPO nie
dałoby się przypisać objective preferencyjnemu.

### 3.4. Kontrolowana różnorodność

Jawne style, focus i coverage-aware selection są lepsze niż próba uzyskania
różnorodności samą temperaturą. Wymóg `style_applicable` chroni przed
wymuszaniem niepasujących intencji.

### 3.5. Bramka przed RL

GRPO jest opcjonalne, zaczyna się od 1.5B, ma dry run rewardów, stop conditions,
shadow judge i obowiązkowe porównanie z best-of-N. Ten zakres nie wymaga obecnie
rozszerzania.

## 4. Luki krytyczne — P0

Poniższe punkty należy rozwiązać przed uznaniem Tasku 04 za gotowy i przed
kosztowną kampanią 4.5B.

### P0.1. Rozdzielić candidate-pool ranking od corpus-wide retrieval

#### Problem

Task 04 opisuje:

- ranking pozytywu wśród co najmniej 10 hard negative’ów;
- Recall@1/5, MRR i nDCG@10 dla tej puli;
- Recall@1/5/10/100 dla probe embeddera.

Jeżeli `test_embedder_rank10` jest całą pulą ewaluacyjną, Recall@100 jest
trywialne, a wynik nie mierzy zdolności wyszukania dokumentu w dużym korpusie.

#### Zalecana zmiana

Zdefiniować dwa jawne protokoły:

```text
candidate_pool_ranking:
  zastosowanie: reranker, grounding, diagnostyka generatora
  pula: 1+ pozytywów oraz 10+ kontrolowanych negatywów
  metryki: margin, rank, MRR, nDCG@10, Recall@1/5

corpus_wide_retrieval:
  zastosowanie: główna ocena probe/docelowego embeddera
  pula: pełny zamrożony korpus dokumentów
  metryki: Recall@k, MRR@10, nDCG@10, MAP, latency
```

Candidate-pool ranking pozostaje wartościową diagnostyką, ale nie może być
głównym dowodem jakości embeddera.

#### Warunki akceptacji

- raport jednoznacznie podaje rozmiar przeszukiwanego korpusu;
- metryki z małej puli i pełnego korpusu mają różne nazwy;
- Recall@100 nie jest raportowane dla puli mniejszej niż 100 kandydatów;
- pełny test obsługuje wiele pozytywów na query;
- wynik corpus-wide jest podstawą porównania generatorów.

### P0.2. Dodać niezależny native-Polish holdout

#### Problem

Lokalny split jest wolny od bezpośredniego leakage między train/dev/test, ale
wszystkie części nadal pochodzą z jednego źródła o charakterystyce MS MARCO.
Dokumentacja `msmarco_pl` wymaga oddzielenia ewaluacji tłumaczonej od
native-Polish lub ręcznie zweryfikowanego holdoutu, lecz Task 04 nie czyni tego
twardym kryterium.

Tłumaczone query mogą zawierać translationese i rozkład intencji odmienny od
naturalnych zapytań polskich. Ponadto lokalny split nie chroni przed obecnością
treści MS MARCO w pretrainingu modeli.

#### Zalecana zmiana

Dodać zamrożone zbiory:

- `test_native_pl_in_domain`;
- `test_native_pl_ood`;
- `test_translated_msmarco_pl`;
- opcjonalnie prywatny/post-cutoff holdout, jeśli dostępny.

Jako publiczne źródło kandydatów rozważyć wybrane zadania retrieval z PIRB,
po audycie licencji, overlapu i charakteru danego podzbioru. BEIR-PL może być
testem transferu na dane tłumaczone, ale nie zastępuje native holdoutu.

#### Obowiązkowe slice’y

- `synthetic_positive`;
- `source_en_difficulty`;
- query native kontra tłumaczone;
- artefakty tłumaczenia i kodowania;
- domena i źródło;
- naturalne pytania kontra pozostałe intencje wyszukiwawcze.

### P0.3. Ponownie wydobywać negatywy dla syntetycznych query

#### Problem

Hard negative’y w zbiorze wejściowym zostały wydobyte dla naturalnego query.
Wygenerowane query może wskazywać inny aspekt passage. W konsekwencji:

- dawny hard negative może przestać być trudny;
- dawny negative może stać się pozytywem dla nowego query;
- najtrudniejsze dokumenty dla syntetycznego query mogą nie występować w
  oryginalnej dziesiątce;
- porównania generatorów mogą mierzyć zgodność z historyczną pulą, a nie jakość
  danych treningowych.

#### Zalecana zmiana

Dla każdego zaakceptowanego syntetycznego query wykonać:

1. BM25 mining z pełnego train corpus;
2. mining zamrożonym bi-encoderem;
3. deduplikację dokumentów i wykluczenie znanych pozytywów;
4. positive-aware filtering przez zamrożony cross-encoder;
5. oznaczenie niepewnych kandydatów jako `possible_false_negative`;
6. zapis provenance negatywu i modelu, który go wydobył.

#### Ablacja dla probe embeddera

| Wariant | Negatywy |
|---|---|
| HN0 | oryginalne negatywy naturalnego query |
| HN1 | BM25 re-mining dla syntetycznego query |
| HN2 | frozen bi-encoder re-mining |
| HN3 | union BM25 + bi-encoder, cross-encoder filtering |
| HN4 | curriculum: łatwiejsze → trudniejsze negatywy |

Nie należy zakładać, że wariant z „najtrudniejszymi” negatywami będzie
najlepszy; silniejszy miner zwiększa również ryzyko false negatives.

### P0.4. Testować mieszanki naturalnych i syntetycznych query

#### Problem

Obecny Task 04 porównuje natural-only, heurystyczne query i query syntetyczne.
Natural-only jest nazwane `upper/control baseline`, choć nie musi być górnym
ograniczeniem. Syntetyczne query mogą uzupełnić rzeczywiste dane o nowe intencje,
ale mogą też zaszkodzić przez styl generatora.

#### Zalecana macierz

Przy stałej liczbie par lub tokenów treningowych:

| ID | Natural | Synthetic | Cel |
|---|---:|---:|---|
| MIX0 | 100% | 0% | gold-data control |
| MIX1 | 75% | 25% | mała augmentacja |
| MIX2 | 50% | 50% | zbalansowana mieszanka |
| MIX3 | 25% | 75% | dominacja syntetyków |
| MIX4 | 0% | 100% | synthetic-only |

Osobno można zmierzyć `natural + synthetic` przy większym całkowitym budżecie,
ale nie wolno mieszać tego wyniku z porównaniem budget-matched.

#### Dodatkowe wymagania

- zachować ten sam rozkład unikalnych dokumentów;
- raportować liczbę par oraz liczbę unikalnych passage;
- nie pozwolić, aby K query dla jednego passage niejawnie zwiększało jego wagę;
- wykonać krzywą jakości względem K query per passage.

### P0.5. Dodać baseline seq2seq

#### Problem

`doc2query_research.md` słusznie wymaga baseline’u docT5query, ale Task 03
porównuje wyłącznie modele z rodziny Bielik. Bez małego encoder-decoder baseline
nie będzie wiadomo, czy zysk wynika z architektury decoder-only i kontrolek, czy
głównie ze skali modelu.

#### Zalecane baseline’y

1. gotowy angielski docT5query jako historyczny weak baseline;
2. mT5/plT5 lub porównywalny model seq2seq dostrojony na tych samych polskich
   parach;
3. Bielik bez treningu, prompting;
4. Bielik SFT.

Porównanie powinno wyrównać:

- dane i split;
- maksymalny budżet generacji;
- liczbę próbek per passage;
- filtering/selection;
- końcowy budżet danych dla probe embeddera;
- koszt generacji.

## 5. Luki ważne — P1

### P1.1. Wdrożyć prawdziwe concept coverage

Task 05 ma style, focus i selektor zestawu, ale nie ma pełnego mechanizmu
rekomendowanego w researchu:

```text
koncepcje dokumentu
→ koncepcje już pokryte
→ koncepcje niepokryte
→ następne query warunkowane na brakującym aspekcie
```

Focus sentence nie jest równoważny pokryciu koncepcji. Kilka zdań może dotyczyć
tego samego aspektu, a jedna koncepcja może być rozproszona po kilku zdaniach.

#### Proponowane eksperymenty Tasku 05

- `D08`: ekstrakcja koncepcji, fraz i encji;
- `D09`: stateful concept-aware generation;
- `D10`: concept-aware consistency filtering;
- `D11`: marginal gain dla K = 1, 2, 4, 8, 16;
- `D12`: MMR versus submodular coverage selector.

Ekstraktor koncepcji również wymaga ręcznego audytu. Słabe lub nadmiernie
ogólne koncepcje mogą tylko przenieść błąd do generatora.

### P1.2. Rozdzielić style językowe od intencji retrieval

`full_question` i `keyword_query` opisują formę, natomiast `definition`,
`how_to` i `comparison` częściowo opisują intencję. Warto jawnie rozdzielić:

```yaml
form:
  - full_question
  - keyword_query

intent:
  - fact_lookup
  - definition
  - entity_lookup
  - procedure
  - comparison
  - document_lookup
  - recommendation

retrieval_task:
  - web_passage
  - legal_document
  - scientific_abstract
  - product_or_service
  - faq_or_support
```

Nie należy wymuszać jednego globalnego rozkładu. Rozkład docelowy powinien być
kalibrowany na realnych query lub świadomie zdefiniowany dla zastosowania.

### P1.3. Rozszerzyć focus z jednego zdania do evidence set

Query może wymagać informacji z kilku zdań albo relacji między nimi. Schemat
powinien opcjonalnie wspierać:

```json
{
  "evidence_sentence_ids": [2, 3],
  "evidence_spans": [],
  "evidence_type": "single_sentence|multi_sentence|section|global",
  "evidence_confidence": 0.84
}
```

Single-sentence focus pozostaje baseline’em, lecz nie powinien być twardą prawdą
dla wszystkich przykładów.

### P1.4. Mierzyć corpus ambiguity i specificity query

Wysoki score query–source passage nie wystarcza. Query może być tak ogólne, że
pasuje do setek dokumentów. Należy mierzyć:

- liczbę dokumentów powyżej progu relevance;
- retrieval entropy lub effective candidate count;
- margines do najlepszego dokumentu spoza znanych pozytywów;
- udział query, dla których inne dokumenty są prawdopodobnie równie poprawne;
- lexical i semantic specificity.

Niepewne query nie powinny automatycznie otrzymywać etykiety „source passage
jest jedynym pozytywem”.

### P1.5. Wcześniej rozdzielić builder judge i evaluator judge

Primary reranker może obecnie:

- przypisywać focus;
- kalibrować reward;
- filtrować query;
- budować preferencje;
- oceniać intrinsic grounding.

To tworzy kołowość. Minimalny wariant Tasku 11 należy przenieść do Tasków 04
i 06:

- primary: budowa danych;
- shadow: metryka potwierdzająca/veto;
- corpus retrieval: niezależny sygnał;
- human panel: kalibracja i analiza błędów.

Pełny audyt reward hackingu może nadal pozostać późnym Taskiem 11.

### P1.6. Ustalić kontrakt statystyczny przed kampanią

Successive halving jest rozsądny kosztowo, ale może wzmacniać selection bias.
Przed Taskiem 09 należy zamrozić:

- jedną główną metrykę;
- ważne metryki bezpieczeństwa/non-inferiority;
- minimalny praktycznie istotny efekt;
- tolerowany spadek grounding;
- liczbę seedów dla etapów selekcyjnych;
- sposób agregacji wariancji między seedami i query;
- zasady użycia dev i jednorazowego otwarcia final test.

Bootstrap po query nie zastępuje wariancji między niezależnymi treningami.

## 6. Eksperymenty opcjonalne — P2

Te pomysły są interesujące, ale nie powinny blokować SFT, Tasku 04 ani
podstawowej ścieżki DPO.

### P2.1. Teacher distillation

Do puli kandydatów w Tasku 06 można dodać query od większego, zamrożonego
teachera i porównać:

- SFT tylko na naturalnych query;
- SFT na naturalnych + filtrowanych query teachera;
- student best-of-N;
- DPO/CPO na student versus teacher versus reference query.

Wymagane są jawne koszty, licencja, prywatność danych i provenance.

### P2.2. Listwise preference optimization

Task 06 zachowuje listę 4–8 kandydatów, a DPO redukuje ją do pary. Po
uruchomieniu baseline’u DPO można porównać jedną metodę listwise, np. LiPO lub
PRO. Nie należy wykonywać szerokiego gridu objective’ów.

Rekomendowana kolejność:

```text
best-of-N
→ filtered/score-weighted continued SFT
→ DPO
→ jedna metoda listwise, tylko jeśli ranking kandydatów jest wiarygodny
```

### P2.3. Iteracyjne noisy self-training

Można przetestować cykl:

```text
generator → synthetic query → retriever → nowe negatywy/pseudo-labels
→ ponowny trening retrievera
```

Jest to jednak późny eksperyment, ponieważ łatwo tworzy niestacjonarną pętlę i
wzmacnia błędy generatora lub retrievera.

### P2.4. Kontrfaktyczne hard negatives

Można generować prawie trafne negatywy przez kontrolowaną zmianę encji, liczby,
relacji lub warunku w passage. Najpierw trzeba jednak mieć stabilny baseline
corpus-mined HN. Generowane negatywy mogą wnosić artefakty źródła, po których
embedder rozpoznaje „syntetyczność” zamiast relevance.

### P2.5. Dekodowanie

Po wprowadzeniu kontroli intencji i koncepcji warto wykonać małą ablacją:

- stałe top-p;
- min-p, jeśli backend wspiera;
- miks temperatur;
- niezależne sample versus stateful coverage.

Nie należy oczekiwać, że dobór top-k lub temperatury da większy efekt niż
kontrola zadania, jakości negatywów i mieszanki danych.

## 7. Proponowane zmiany w poszczególnych taskach

### Task 01

- Zachować obecną politykę splitów.
- Uzupełnić raport o wymagane percentyle tokenizerów.
- Dla rekordów z mniej niż 10 negatywami preferować:
  - corpus-wide evaluation, gdzie problem znika; lub
  - deterministyczne uzupełnienie negatywów z tego samego splitu.
- Nie usuwać dużej części dev/test wyłącznie po to, aby zachować arbitralne
  `10+`, bez raportu wpływu na rozkład trudności.

### Task 02

- Dokończyć benchmark primary/shadow na dev/test.
- Dodać ręczne etykiety answerability i false-negative dla próbki trudnych
  przypadków.
- Zdefiniować builder judge oraz confirmatory judge.
- Nie zmieniać zakazu treningu własnego rerankera.

### Task 03

- Dokończyć S00.
- Dodać mały baseline mT5/plT5.
- Nie wybierać checkpointu na podstawie samego eval loss.
- Wstrzymać pełne porównania 4.5B do czasu pierwszego wiarygodnego probe.

### Task 04

- Rozdzielić candidate-pool i corpus-wide retrieval.
- Dodać native-Polish/OOD holdout.
- Dodać slice’y `synthetic_positive` i `source_en_difficulty`.
- Zmienić nazwę natural-only z `upper` na `gold-data control`.
- Dodać eksperymenty MIX0–MIX4.
- Zaplanować re-mining negatywów.
- Dla finalistów potwierdzić wynik na drugim backbone embeddera lub na
  docelowym embedderze.

### Task 05

- Rozdzielić `form`, `intent` i `retrieval_task`.
- Dodać concept extraction i stateful coverage.
- Dodać `evidence_sentence_ids`.
- Zmierzyć marginal gain względem K.
- Porównać selection top-N, MMR i coverage-aware.

### Task 06

- Re-minować negatywy dla kandydatów syntetycznych.
- Dodać corpus ambiguity/specificity do score.
- Zachować pełne listwise rankingi i score ciągłe.
- Nie używać jednego judge’a jako jedynego źródła preferencji i ewaluacji.
- Dodać teacher candidates wyłącznie jako jawną ablacją.

### Task 07

- Zachować obowiązkowy continued SFT.
- Dodać score-weighted/filtered SFT jako tani baseline.
- Jedną metodę listwise testować dopiero po stabilnym DPO.
- Nie rozszerzać teraz pełnej siatki lossów.

### Task 08

- Pozostawić `OPTIONAL / BLOCKED`.
- Nie uruchamiać bez corpus-wide probe i dodatniej korelacji reward–human.
- Best-of-N powinno pozostać obowiązkową kontrolą.

### Task 09

- Zamrozić primary metric, non-inferiority margins i minimalny efekt.
- Wymagać native-Polish wyniku dla każdego finalisty.
- Raportować wariancję między seedami, nie tylko bootstrap po query.
- Budżetować jednocześnie:
  - liczbę tokenów;
  - liczbę par;
  - liczbę unikalnych passage;
  - liczbę query per passage.

### Task 10

- Finalny release powinien podawać osobno wyniki translated i native Polish.
- Model/data card powinny opisywać sposób mining HN oraz mieszankę
  natural/synthetic.
- Koszt generacji powinien obejmować również filtering, reranking i re-mining.

### Task 11

- Pełny audyt może pozostać opcjonalny.
- Nazwa pliku `optional_alternating_cotraining` jest myląca, ponieważ treść
  dotyczy audytu sędziego i jawnie zabrania cotrainingu. Warto rozważyć zmianę
  nazwy przy najbliższej aktualizacji linków.

## 8. Korekty do `doc2query_research.md`

Research jest wartościowy, ale miesza trzy różne zastosowania:

1. document expansion dla BM25/sparse retrieval;
2. generowanie par treningowych dla embeddera;
3. generowanie keywordów reklamowych.

Projekt dotyczy przede wszystkim punktu 2. Zalecane jest oznaczenie pozostałych
części jako „kontekst” lub „poza bieżącym zakresem”.

Należy również poprawić następujące niespójności:

- research wymaga docT5query, lecz Task 03 go nie zawiera;
- research rekomenduje stateful concept coverage, lecz Task 05 opisuje głównie
  style i focus;
- research proponuje trening rerankera na syntetycznych parach w fazie
  downstream; jest to sprzeczne z nadrzędnym zakazem projektu i powinno być
  oznaczone jako osobny, niedozwolony obecnie zakres;
- dual-index, dopisywanie query do dokumentów i koszt indeksu są istotne dla
  document expansion, ale nie powinny obciążać krytycznej ścieżki generatora
  danych dla embeddera;
- „docT5query” dla polskiego należy rozumieć jako historyczny baseline plus
  uczciwy polski/multilingual seq2seq baseline, nie jako założenie, że angielski
  checkpoint będzie konkurencyjny.

## 9. Minimalna kolejność dalszych decyzji

```text
1. Popraw kontrakt Tasku 04.
2. Dokończ benchmark primary/shadow z Tasku 02.
3. Uruchom S00 i baseline seq2seq.
4. Oceń W03/W05:
   - candidate pool,
   - corpus-wide,
   - translated test,
   - native-Polish test.
5. Uruchom pierwszy probe:
   - natural-only,
   - W05 synthetic-only,
   - jedna mieszanka natural/synthetic.
6. Dopiero na tej podstawie zdecyduj o 4.5B.
7. Rozszerz Task 05 o concept coverage i krzywą K.
8. W Tasku 06 re-minuj negatywy dla syntetycznych query.
9. Porównaj best-of-N i weighted continued SFT.
10. Dopiero później uruchom DPO.
11. GRPO pozostaw zablokowane do spełnienia zapisanej bramki.
```

## 10. Proponowana minimalna macierz nowych eksperymentów

| ID | Eksperyment | Priorytet | Warunek |
|---|---|---|---|
| A00 | candidate pool versus corpus-wide evaluation | P0 | przed ukończeniem Tasku 04 |
| A01 | translated versus native-Polish test | P0 | przed wyborem generatora |
| A02 | static versus re-mined hard negatives | P0 | przed poważnym probe |
| A03 | natural/synthetic mixture ratios | P0 | przed Taskiem 09 |
| A04 | mT5/plT5 versus Bielik SFT | P0 | przed kosztownym 4.5B |
| A05 | concept-aware versus focus/style-only | P1 | Task 05 |
| A06 | K = 1/2/4/8/16 marginal gain | P1 | Task 05 |
| A07 | primary-built versus shadow-confirmed selection | P1 | Task 06 |
| A08 | fixed HN versus BM25/bi-encoder/filtered HN | P1 | probe embedder |
| A09 | best-of-N versus weighted SFT versus DPO | P1 | Task 07 |
| A10 | pairwise DPO versus jedna metoda listwise | P2 | tylko po stabilnym DPO |
| A11 | teacher distillation | P2 | po ocenie kosztu/licencji |
| A12 | kontrfaktyczne syntetyczne negatywy | P2 | po stabilnym mining baseline |

## 11. Checklista bramki przed 4.5B

- [ ] S00 prompting bez treningu został oceniony.
- [ ] Istnieje seq2seq baseline.
- [ ] Primary i shadow mają pełny benchmark na projekcie.
- [ ] Candidate-pool i corpus-wide metrics są rozdzielone.
- [ ] Recall@100 nie jest liczony w puli 11 dokumentów.
- [ ] Native-Polish holdout jest zamrożony.
- [ ] W03 i W05 mają intrinsic oraz corpus-wide evaluation.
- [ ] Uruchomiono co najmniej jeden porównywalny probe embedder.
- [ ] Natural-only i synthetic-only mają identyczny budżet.
- [ ] Uruchomiono co najmniej jedną mieszankę natural/synthetic.
- [ ] Znany jest wpływ statycznych i re-mined negatywów.
- [ ] Decyzja o 4.5B opiera się na wyniku retrieval, nie eval loss.

## 12. Checklista bramki przed DPO

- [ ] Stabilny checkpoint SFT wygrywa z promptingiem.
- [ ] Task 05 ustalił użyteczny wariant style/focus/concept coverage.
- [ ] Wygenerowano kilka wiarygodnych kandydatów per prompt.
- [ ] Preferencje nie pochodzą wyłącznie z jednego sędziego.
- [ ] Corpus ambiguity i possible false negatives są raportowane.
- [ ] Ręczna zgodność preferencji jest wystarczająca.
- [ ] Best-of-N zostało ocenione.
- [ ] Filtered/weighted continued SFT zostało ocenione.
- [ ] Probe embedder działa na naturalnym, zamrożonym teście.
- [ ] Nie ma podstaw, by oczekiwać, że DPO naprawia wyłącznie artefakt scorera.

## 13. Źródła

### Fundamenty i generowanie query

- Nogueira et al., *Document Expansion by Query Prediction*:
  <https://arxiv.org/abs/1904.08375>
- Dai et al., *Promptagator: Few-shot Dense Retrieval From 8 Examples*:
  <https://arxiv.org/abs/2209.11755>
- Gospodinov et al., *Doc2Query--: When Less is More*:
  <https://arxiv.org/abs/2301.03266>
- Basnet et al., *DeeperImpact*:
  <https://arxiv.org/abs/2405.17093>
- Lee et al., *Disentangling Questions from Query Generation for
  Task-Adaptive Retrieval*:
  <https://arxiv.org/abs/2409.16570>
- Kang et al., *Concept Coverage-based Query Set Generation*:
  <https://arxiv.org/abs/2502.11181>
- Kuo et al., *Doc2Query++*:
  <https://arxiv.org/abs/2510.09557>

### Dane syntetyczne i trening embeddera

- Wang et al., *GPL: Generative Pseudo Labeling*:
  <https://arxiv.org/abs/2112.07577>
- Wang et al., *Improving Text Embeddings with Large Language Models*:
  <https://arxiv.org/abs/2401.00368>
- Moreira et al., *NV-Retriever: Effective Hard-Negative Mining*:
  <https://arxiv.org/abs/2407.15831>
- Cai et al., *Hard Negatives or False Negatives*:
  <https://arxiv.org/abs/2209.05072>
- Jiang et al., *Noisy Self-Training with Synthetic Queries*:
  <https://arxiv.org/abs/2311.15563>

### Preferencje

- Krastev et al., *InPars+*:
  <https://arxiv.org/abs/2508.13930>
- Liu et al., *LiPO: Listwise Preference Optimization*:
  <https://arxiv.org/abs/2402.01878>
- Song et al., *Preference Ranking Optimization*:
  <https://arxiv.org/abs/2306.17492>

### Ewaluacja polska i wielodomenowa

- Dadas et al., *PIRB: Polish Information Retrieval Benchmark*:
  <https://arxiv.org/abs/2402.13350>
- Poświata et al., *PL-MTEB*:
  <https://arxiv.org/abs/2405.10138>
- Wojtasik et al., *BEIR-PL*:
  <https://arxiv.org/abs/2305.19840>
- Bonifacio et al., *mMARCO*:
  <https://arxiv.org/abs/2108.13897>
- Thakur et al., *BEIR*:
  <https://arxiv.org/abs/2104.08663>

## 14. Status rekomendacji

Ten dokument jest audytem i backlogiem decyzyjnym. Nie zmienia statusu żadnego
tasku i nie oznacza automatycznej zgody na rozszerzenie zakresu. Przy wdrażaniu
rekomendacji należy:

1. przenieść zaakceptowane punkty do odpowiednich plików `tasks/*.md`;
2. zaktualizować `tasks/README.md` w tym samym commicie;
3. jawnie odrzucić lub odłożyć pozostałe punkty;
4. nie oznaczać zadań jako `DONE` bez wymaganych runów i artefaktów.
