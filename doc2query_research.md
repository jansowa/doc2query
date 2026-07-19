# Doc2query i syntetyczne generowanie zapytań — wnioski z sesji

**Stan przeglądu:** 18 lipca 2026
**Zakres:** generowanie zapytań/keywordów na podstawie dokumentu, trenowanie generatorów, dekodowanie, filtrowanie oraz plan eksperymentów.

> Najważniejszy wniosek: współczesny pipeline nie powinien polegać wyłącznie na SFT `dokument → zapytanie` i zwiększaniu temperatury. Najlepszy praktyczny układ to: **dobry generator + kontrola pokrycia intencji/koncepcji + wielokrotne próbkowanie + selekcja jakościowa + ewaluacja downstream**.

> **Granice bieżącego projektu:** fragmenty o ekspansji indeksu, dual-index
> fusion i keywordach reklamowych są kontekstem badawczym, nie elementem
> backlogu. Projekt trenuje generator i probe/docelowy embedder. Trening
> rerankera na syntetycznych parach byłby osobnym projektem produktowym;
> primary i shadow judge tutaj pozostają zamrożone.

---

## 1. Dwa podobne, ale różne zastosowania

### A. Document expansion / rozszerzanie indeksu [kontekst — poza zakresem]

Generator tworzy zapytania, które są dopisywane do dokumentu przed indeksowaniem albo przechowywane w osobnym indeksie. Celem jest ograniczenie **vocabulary mismatch** — użytkownik może nazwać potrzebę inaczej niż autor dokumentu.

Najważniejsze ryzyka:

- powiększenie indeksu i czasu wyszukiwania;
- halucynowane lub nadmiernie ogólne terminy;
- duplikaty semantyczne;
- pogorszenie precyzji przy zbyt agresywnej ekspansji.

### B. Synthetic query generation / tworzenie danych treningowych

Generator tworzy pary `(syntetyczne zapytanie, dokument)`, na których trenowany jest retriever lub reranker.

W bieżącym projekcie odbiorcą danych jest embedder. Wzmianki o treningu
rerankera opisują literaturę i nie zezwalają na aktualizację wag sędziów.

Najważniejsze ryzyka:

- model downstream uczy się stylu generatora zamiast realnego rozkładu zapytań;
- fałszywie pozytywne pary query–document;
- zbyt łatwe, powtarzalne przykłady;
- brak dobrych negatywów lub kontrastów.

**Konsekwencja:** dobra metoda generacji może działać w obu scenariuszach, ale kryterium selekcji powinno być inne. Przy indeksowaniu liczy się dodatkowo koszt indeksu i latency; przy danych treningowych — jakość sygnału uczącego i transfer do nowych domen.

---

## 2. Najważniejsze prace

### Fundamenty doc2query

1. **Document Expansion by Query Prediction** — Nogueira, Yang, Lin, Cho (2019)
   [Paper — arXiv:1904.08375](https://arxiv.org/abs/1904.08375)
   [Repozytorium docTTTTTquery / docT5query](https://github.com/castorini/docTTTTTquery)

   Klasyczna metoda: model seq2seq jest trenowany na parach query–document, a następnie generuje potencjalne zapytania dla każdego dokumentu. Zapytania są dopisywane do dokumentu przed indeksowaniem.

   **Wniosek praktyczny:** to nadal dobry baseline. Każdy nowy pipeline powinien być porównany z prostym T5/docT5query, a nie tylko z „brakiem ekspansji”.

2. **From doc2query to docTTTTTquery** — opis ewolucji do T5
   [Repozytorium i materiały](https://github.com/castorini/docTTTTTquery)

   **Wniosek praktyczny:** wielokrotne próbkowanie jest integralną częścią podejścia; pojedyncze „najlepsze” zapytanie nie daje wystarczającego pokrycia słownictwa.

### Większe generatory i learned sparse retrieval

3. **DeeperImpact: Optimizing Sparse Learned Index Structures** — Basnet, Gou, Mallia, Suel (2024)
   [Paper — arXiv:2405.17093](https://arxiv.org/abs/2405.17093)
   [Kod](https://github.com/basnetsoyuj/improving-learned-index)

   Najważniejsze elementy:

   - Llama 2 7B zamiast T5 jako generator zapytań;
   - LoRA zamiast pełnego fine-tuningu;
   - wagi ładowane w 8 bitach;
   - trening na ok. 532 tys. parach MS MARCO;
   - 80 próbek na dokument;
   - łączone top-k i top-p podczas inferencji;
   - dalsze ulepszenia modelu sparse: hard negatives, distillation i inicjalizacja CoCondenser.

   **Wniosek praktyczny:** większy decoder-only LLM może poprawić pokrycie i recall, ale sama jakość wygenerowanych tekstów nie jest jedynym źródłem wyniku — istotny jest cały model downstream i jego objective.

   **Ważna obserwacja:** w ich ustawieniu filtrowanie Doc2Query-- nie poprawiło DeepImpact. Filtracja nie jest więc uniwersalnie korzystna; zależy od sposobu wykorzystania rozszerzeń.

### Filtrowanie jakości

4. **Doc2Query--: When Less is More** — Gospodinov, MacAvaney, Macdonald (2023)
   [Paper — arXiv:2301.03266](https://arxiv.org/abs/2301.03266)
   [Kod PyTerrier Doc2Query](https://github.com/terrierteam/pyterrier_doc2query)
   [Dokumentacja PyTerrier](https://pyterrier.readthedocs.io/en/stable/ext/pyterrier-doc2query/index.html)

   Metoda ocenia wygenerowane pary query–document modelem relevance i usuwa słabe zapytania przed indeksowaniem.

   Raportowane korzyści w badanym pipeline obejmowały jednocześnie poprawę retrieval, mniejszy indeks i krótszy czas wykonania zapytania.

   **Wniosek praktyczny:** filtrowanie jest szczególnie obiecujące dla BM25 i prostego dopisywania tekstu. Trzeba jednak wykonać ablation, ponieważ learned sparse model może sam nauczyć się ignorować część szumu.

### Generowanie danych syntetycznych dla retrieverów i rerankerów

5. **InPars: Data Augmentation for Information Retrieval using Large Language Models** — Bonifacio et al. (2022)
   [Paper — arXiv:2202.05144](https://arxiv.org/abs/2202.05144)
   [Kod](https://github.com/zetaalphavector/inpars)

   Few-shot prompting dużego modelu służy do generowania domenowych par query–document.

   **Wniosek praktyczny:** syntetyczne dane domenowe mogą być lepsze od prostego transferu modelu wytrenowanego wyłącznie na MS MARCO.

6. **Promptagator: Few-shot Dense Retrieval From 8 Examples** — Dai et al. (2022/ICLR 2023)
   [Paper — arXiv:2209.11755](https://arxiv.org/abs/2209.11755)

   Najważniejsze elementy:

   - prompt opisujący konkretne zadanie retrieval;
   - kilka przykładów query–document z docelowej domeny;
   - generowanie danych dla retrievera;
   - consistency filtering: syntetyczne zapytanie zostaje zachowane, gdy retrieval prowadzi z powrotem do dokumentu źródłowego;
   - osobny trening retrievera i rerankera.

   **Wniosek praktyczny:** struktura promptu i wiedza o zadaniu są często ważniejsze niż podnoszenie temperatury.

7. **InPars-v2: Large Language Models as Efficient Dataset Generators for Information Retrieval** — Jeronymo et al. (2023)
   [Paper — arXiv:2301.01820](https://arxiv.org/abs/2301.01820)

   Wykorzystuje otwarty generator oraz mocny reranker, np. monoT5, do selekcji syntetycznych par.

   **Wniosek praktyczny:** wygenerowanie dużej puli i zachowanie najlepiej ocenianej części jest mocnym, prostym baseline’em.

8. **InPars Toolkit: A Unified and Reproducible Synthetic Data Generation Pipeline for Neural Information Retrieval** — Abonizio et al. (2023)
   [Paper — arXiv:2307.04601](https://arxiv.org/abs/2307.04601)
   [Kod](https://github.com/zetaalphavector/InPars)

   Pipeline obejmuje generowanie, filtrowanie, trening rerankera i ewaluację na BEIR.

   **Wniosek praktyczny:** najlepszy punkt startowy do reprodukowalnego eksperymentowania zamiast budowania całej infrastruktury od zera.

### Preference optimization generatora

9. **InPars+: Supercharging Synthetic Data Generation for Information Retrieval Systems** — Krastev et al. (2025)
   [Paper — arXiv:2508.13930](https://arxiv.org/abs/2508.13930)
   [Kod i materiały](https://github.com/danilotpnta/IR2-project)

   Najważniejsze elementy:

   - student: Llama 3.1 8B Instruct;
   - teacher: Llama 3.1 Nemotron 70B Instruct;
   - fine-tuning na 100 tys. przykładów MS MARCO;
   - dla dokumentu powstaje kilka kandydatów: m.in. odpowiedź studenta, teachera i query referencyjne;
   - kandydaci są oceniani funkcją relevance łączącą sygnał embeddingowy i BM25;
   - najwyżej oceniany kandydat jest `preferred`, najniżej oceniany `dispreferred`;
   - generator jest uczony przez Contrastive Preference Optimization (CPO);
   - osobny kierunek pracy dotyczy optymalizacji promptów przez DSPy.

   **Wniosek praktyczny:** nowoczesny generator można trenować nie tylko na „złotym” query przez cross-entropy, lecz także bezpośrednio uczyć preferencji pomiędzy lepszymi i gorszymi kandydatami.

   **Ostrożność:** jest to nowsza praca o mniejszej liczbie niezależnych replikacji niż klasyczne docT5query, InPars lub Promptagator. Warto traktować ją jako obiecujący kierunek i reprodukować na własnej domenie.

### Kontrola pokrycia zamiast czystej losowości

10. **Improving Scientific Document Retrieval with Concept Coverage-based Query Set Generation (CCQGen)** — Kang et al., WSDM 2025
    [Paper — arXiv:2502.11181](https://arxiv.org/abs/2502.11181)
    [DOI](https://doi.org/10.1145/3701551.3703544)

    Metoda:

    - identyfikuje kluczowe koncepcje dokumentu na poziomie tematów i fraz;
    - mierzy, które koncepcje pokryły już wcześniejsze syntetyczne zapytania;
    - następne zapytanie jest warunkowane na koncepcjach jeszcze niepokrytych;
    - stosuje consistency filtering wspierany informacją o koncepcjach.

    **Wniosek praktyczny:** kolejne próbki powinny być komplementarne, a nie tylko niezależnie losowe. To bezpośrednio odpowiada problemowi, w którym wysoka temperatura daje dużo parafraz tego samego intentu.

11. **Doc2Query++: Topic-Coverage based Document Expansion and its Application to Dense Retrieval via Dual-Index Fusion** — Kuo et al. (2025/2026)
    [Paper — arXiv:2510.09557](https://arxiv.org/abs/2510.09557)

    Łączy kontrolę pokrycia tematów i keywordów z generowaniem LLM. W dense retrieval oddziela reprezentację oryginalnego tekstu od reprezentacji zapytań i łączy wyniki z dwóch indeksów, zamiast bezpośrednio sklejać wszystko w jeden tekst.

    **Wniosek praktyczny:** dla dense retrieval osobny indeks sygnału ekspansji może być bezpieczniejszy niż dopisywanie potencjalnie zaszumionych zapytań do dokumentu przed embeddingiem.

    **Ostrożność:** to bardzo świeża praca; należy sprawdzić kod, status publikacji, szczegóły zbiorów i reprodukowalność przed traktowaniem jej jako ustalonego standardu.

    **Zakres projektu:** dual-index fusion pozostaje kontekstem i nie wchodzi
    do bieżącego planu wykonawczego.

---

## 3. Zalecana metoda treningowa

### Etap 1: przygotowanie danych

Zbudować pary:

```text
input:  dokument / fragment dokumentu
output: prawdopodobne zapytanie użytkownika
```

Dobre źródła danych:

- rzeczywiste query–document z logów lub ocen relevance;
- MS MARCO jako duży bootstrap;
- przykłady z docelowej domeny;
- zapytania zaakceptowane przez ludzi;
- syntetyczne zapytania od mocniejszego teachera, po filtracji.

Dla długich dokumentów warto trenować na logicznych fragmentach. Zapytanie powinno być możliwe do obsłużenia przez wejściowy fragment; inaczej model uczy się par z niejasnym związkiem.

### Etap 2: baseline SFT

Model klasy 7–8B instruction-tuned można dostroić przez LoRA/QLoRA. Baseline:

- objective: token-level cross-entropy;
- wejście: instrukcja + dokument;
- wyjście: jedno krótkie zapytanie;
- mieszanka domenowa: dane ogólne + dane docelowe;
- walidacja po dokumentach, aby uniknąć przecieku podobnych fragmentów.

Należy zachować co najmniej dwa baseline’y:

1. gotowy docT5query;
2. współczesny LLM bez fine-tuningu, tylko z promptingiem.

### Etap 3: budowa danych preferencyjnych

Dla każdego dokumentu wygenerować kilku kandydatów:

- query referencyjne, jeżeli istnieje;
- query od studenta;
- query od teachera;
- query z różnych promptów/intencji;
- opcjonalnie celowo słabszy lub mniej relewantny kandydat.

Ocenić je przez kombinację:

- cross-encoder relevance;
- round-trip retrieval;
- lexical/BM25 relevance;
- pokrycie kluczowych koncepcji;
- kara za kopiowanie dokumentu;
- kara za semantyczną redundancję;
- reguły domenowe.

Następnie utworzyć trójki:

```text
(document, preferred_query, rejected_query)
```

I zastosować CPO/DPO lub inną metodę preference optimization.

### Etap 4: kontrolowane generowanie zestawu

Zamiast prosić model 20 razy o „wygeneruj query”, utrzymywać stan:

```text
- koncepcje dokumentu
- pokryte koncepcje
- niepokryte koncepcje
- wygenerowane intencje
- wykryte duplikaty
```

Każda następna generacja powinna otrzymać warunek, np.:

```text
Wygeneruj realistyczne zapytanie użytkownika dotyczące dokumentu.
Skoncentruj się na niepokrytych aspektach: [X, Y].
Nie powtarzaj intencji ani sformułowania wcześniejszych zapytań: [lista].
```

To zwykle daje bardziej użyteczną różnorodność niż samo zwiększenie temperatury.

---

## 4. Top-k czy temperatura?

### Mechanika

**Temperatura** zmienia kształt całego rozkładu prawdopodobieństwa:

- niższa: bardziej deterministyczne i typowe odpowiedzi;
- wyższa: większa eksploracja, ale także większe ryzyko słabego ogona.

**Top-k** pozostawia dokładnie `k` najbardziej prawdopodobnych tokenów na każdym kroku:

- jest proste;
- nie dostosowuje się do pewności modelu;
- to samo `k` może być za duże w łatwym kroku i za małe w niepewnym.

**Top-p / nucleus sampling** pozostawia najmniejszy zbiór tokenów o zadanej sumie prawdopodobieństwa. Jest adaptacyjne względem kształtu rozkładu.

- [The Curious Case of Neural Text Degeneration / nucleus sampling — arXiv:1904.09751](https://arxiv.org/abs/1904.09751)

**Min-p** odcina tokeny względem prawdopodobieństwa najlepszego tokenu, dzięki czemu próg zależy od pewności modelu.

- [Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs — arXiv:2407.01082](https://arxiv.org/abs/2407.01082)

### Rekomendacja

Nie warto domyślnie generować jednej partii z `top_k=10`, drugiej z `top_k=30`, a trzeciej z `top_k=100`.

Lepszy punkt startowy:

1. wybrać jeden mechanizm ograniczenia ogona — zwykle top-p albo min-p;
2. wykonać wiele niezależnych próbek;
3. zastosować 2–3 kontrolowane temperatury, jeżeli potrzebna jest mieszanka head i long-tail;
4. większą część różnorodności uzyskać przez kontrolę intencji/koncepcji w promptach;
5. filtrować i deduplikować wyniki.

### Proponowany budżet dla doc2query

Dla 16 kandydatów na dokument:

| Liczba | Temperatura | Cel |
|---:|---:|---|
| 8 | 0.65–0.75 | typowe, bezpieczne zapytania |
| 6 | 0.80–0.90 | parafrazy i long-tail |
| 2 | 1.00–1.05 | kontrolowana eksploracja |

Punkt startowy dla truncation:

```yaml
do_sample: true
top_p: 0.90-0.95
# alternatywnie min_p, jeżeli backend go wspiera
```

Nie są to wartości uniwersalnie optymalne. Powinny wejść do małego gridu eksperymentalnego, a nie być traktowane jako prawda dla każdego modelu.

### Co testować osobno

Nie zmieniać jednocześnie temperatury, top-p, promptu i liczby próbek. W przeciwnym razie nie będzie wiadomo, co faktycznie poprawiło wynik.

Minimalna ablation:

| Wariant | Sampling | Kontrola pokrycia | Filtrowanie |
|---|---|---|---|
| A | `T=0.75`, stałe top-p, 16 próbek | nie | nie |
| B | miks 3 temperatur, stałe top-p | nie | nie |
| C | stała temperatura, różne top-k | nie | nie |
| D | `T=0.75`, stałe top-p | tak | nie |
| E | miks temperatur | tak | relevance + deduplikacja |
| F | jak E | tak | round-trip retrieval |

Hipoteza do sprawdzenia: **D/E/F powinny dawać lepsze pokrycie przy mniejszej redundancji niż samo zmienianie top-k lub temperatury.**

---

## 5. Wnioski dla generowania keywordów reklamowych [kontekst — poza zakresem]

Intuicyjne używanie różnych temperatur miało sens: tworzyło mieszaninę bezpiecznych i bardziej eksploracyjnych fraz. Problem polega na tym, że temperatura nie kontroluje rodzaju różnorodności.

Wysoka temperatura może dać:

- inne słowa, ale tę samą intencję;
- nietrafione produkty lub cechy;
- frazy zbyt szerokie;
- język mało naturalny dla wyszukiwarki;
- trudne do zauważenia przesunięcie znaczenia.

Lepszy pipeline:

### 1. Najpierw zdefiniować kubełki intencji

Przykłady:

- generyczne kategorie;
- konkretne produkty/usługi;
- problem lub potrzeba użytkownika;
- zakup / cena / oferta;
- porównanie i alternatywy;
- zastosowanie lub scenariusz;
- lokalizacja;
- long-tail;
- pytania informacyjne prowadzące do zakupu;
- wykluczenia i potencjalne negative keywords.

### 2. Generować osobno dla każdego kubełka

Dla każdej intencji wykonać kilka niezależnych próbek przy umiarkowanej temperaturze.

Przykładowo:

```yaml
temperature: 0.70-0.85
top_p: 0.90-0.95
samples_per_intent: 3-5
```

### 3. Deduplikować na dwóch poziomach

- tekstowym: lowercase, normalizacja fleksji, znaków i kolejności;
- semantycznym: embeddingi + clustering lub wybór MMR.

### 4. Oceniać relevance względem reklamy i landing page’a

Ocena powinna uwzględniać:

- czy landing page faktycznie odpowiada na frazę;
- zgodność produktu, lokalizacji i grupy docelowej;
- commercial intent;
- zbyt szerokie znaczenie;
- ryzyko pomylenia z innym produktem;
- pokrycie nowych intencji względem już wybranych keywordów.

### 5. Walidować danymi zewnętrznymi

LLM powinien tworzyć kandydatów, nie prognozować popytu. Kandydatów warto później sprawdzić w danych kampanii, narzędziu do planowania słów kluczowych lub innym źródle wolumenu i kosztu.

---

## 6. Filtrowanie i selekcja

### Relevance filtering

Dla każdej pary query–document policzyć score cross-encodera. Odrzucić kandydatów poniżej progu albo wybrać top-N per dokument.

Zaleta top-N per dokument: każdy dokument zachowuje pewną liczbę zapytań.
Zaleta globalnego progu: jakość jest bardziej jednolita, ale część dokumentów może stracić wszystkie próbki.

Warto porównać oba warianty.

### Round-trip consistency

1. Wygenerować query z dokumentu `d`.
2. Uruchomić retrieval dla query.
3. Zachować parę, jeżeli `d` wraca w top-K.

Testować różne K, np. 1, 5, 10 i 100. Zbyt restrykcyjne K może preferować zapytania kopiujące dokument; zbyt szerokie K może przepuszczać słabe pary.

### Semantic deduplication

Prosty wariant:

1. obliczyć embedding każdego query;
2. pogrupować zapytania o cosine similarity powyżej progu;
3. z każdego klastra zachować kandydat o najwyższym relevance score.

Bardziej elastyczny wariant: **MMR** — kolejno wybierać zapytania o wysokiej jakości, ale niskim podobieństwie do już wybranych.

### Coverage score

Zbudować listę koncepcji/intencji dokumentu i mierzyć:

- ile z nich pokrywa cały zestaw query;
- jak równomierne jest pokrycie;
- ile nowych koncepcji wnosi kolejna próbka;
- jaki procent próbek jest redundantny.

---

## 7. Co mierzyć

### Jakość pojedynczego query

- relevance query–document;
- answerability przez dokument;
- naturalność i podobieństwo do realnych zapytań;
- długość;
- lexical overlap z dokumentem;
- halucynowane encje, produkty, liczby i właściwości.

### Jakość zestawu query

- liczba unikalnych intencji;
- semantic diversity;
- concept/topic coverage;
- odsetek duplikatów;
- marginal gain: ile wnosi próbka numer 2, 4, 8, 16 itd.;
- rozkład head vs long-tail.

### Retrieval downstream

- nDCG@10;
- MRR@10;
- Recall@100 i Recall@1000;
- MAP, jeśli pasuje do zbioru;
- wyniki zero-shot na BEIR lub domenowym benchmarku;
- różnica względem BM25 bez ekspansji;
- różnica względem docT5query.

### Koszt i wydajność

- tokeny i czas generowania na dokument;
- liczba wygenerowanych i zachowanych query;
- rozmiar indeksu;
- postings per document;
- latency wyszukiwania;
- koszt rerankingu/filteringu;
- koszt okresowego przebudowania indeksu.

### Dla keywordów reklamowych

- udział kandydatów zaakceptowanych przez człowieka;
- pokrycie intencji;
- udział fraz nietrafionych lub zbyt szerokich;
- liczba nowych, nieduplikujących się grup tematycznych;
- późniejsze metryki kampanii — analizowane osobno od jakości językowej generatora.

---

## 8. Proponowany plan eksperymentu

### Faza 1: tani benchmark dekodowania

Na 500–2000 dokumentach porównać:

1. greedy/beam jako baseline jakościowy;
2. 16 próbek przy stałym `T=0.75`, `top_p=0.95`;
3. miks temperatur 0.65/0.85/1.0;
4. stała temperatura i różne top-k;
5. stałe sampling + promptowane kubełki intencji;
6. iteracyjne generowanie z listą niepokrytych koncepcji.

Policzyć relevance, diversity, coverage i redundancy. Dopiero najlepsze warianty przepuścić przez pełny retrieval benchmark.

### Faza 2: filtracja

Dla najlepszego wariantu generacji porównać:

- bez filtracji;
- cross-encoder top-N;
- globalny próg relevance;
- round-trip top-K;
- relevance + semantic dedup;
- relevance + MMR;
- relevance + coverage.

### Faza 3: trening generatora

Porównać:

- prompting bez treningu;
- SFT;
- SFT + dane teachera;
- SFT + CPO/DPO;
- CPO/DPO ze score’em relevance;
- CPO/DPO ze score’em relevance + coverage + kara za kopiowanie.

### Faza 4: downstream

Osobne eksperymenty dla:

- BM25 z dopisanymi query;
- learned sparse retrieval;
- dense retrieval z dopisanym tekstem;
- dense retrieval z dual-index/fusion;
- retrievera trenowanego na syntetycznych parach;
- rerankera trenowanego na syntetycznych parach.

Nie należy wnioskować, że konfiguracja najlepsza dla BM25 będzie najlepsza dla dense retrieval lub dla tworzenia danych treningowych.

---

## 9. Minimalny pipeline rekomendowany na start

```text
1. Podziel dokumenty na samodzielne semantycznie fragmenty.
2. Zbuduj baseline docT5query oraz prompting współczesnego LLM.
3. Dla każdego fragmentu wyodrębnij tematy, encje i intencje.
4. Wygeneruj 12–16 query:
   - większość przy T≈0.7–0.85,
   - mała część przy T≈1.0,
   - stałe top-p lub min-p,
   - osobne prompty dla niepokrytych aspektów.
5. Usuń duplikaty tekstowe.
6. Oceń relevance cross-encoderem.
7. Zastosuj semantic dedup/MMR.
8. Sprawdź round-trip retrieval.
9. Zachowaj 4–8 najlepszych i komplementarnych query.
10. Oceń downstream oraz koszt indeksu.
11. Dopiero potem trenuj generator przez SFT lub preference optimization.
```

---

## 10. Pytania, które warto rozstrzygnąć eksperymentalnie

1. Czy większy generator poprawia wynik po wyrównaniu liczby parametrów, kosztu i liczby próbek?
2. Czy lepszy jest globalny próg jakości, czy stałe top-N per dokument?
3. Ile query na dokument daje jeszcze dodatni marginal gain?
4. Czy kontrola koncepcji zastępuje wysoką temperaturę?
5. Czy query syntetyczne powinny przypominać realne logi, czy maksymalizować recall?
6. Jak karać kopiowanie terminów z dokumentu, nie tracąc keywordów specjalistycznych?
7. Czy cross-encoder relevance nie preferuje zapytań z dużym lexical overlap?
8. Jak zbudować negatywy: losowe, BM25 hard negatives, teacher negatives czy generowane „prawie trafne” query?
9. Czy preferencyjny score powinien łączyć relevance, coverage i novelty?
10. Jak stabilny jest wynik między różnymi seedami samplingu?
11. Czy oddzielny indeks query daje lepszy kompromis niż dopisywanie tekstu do dokumentu?
12. Jak wynik przenosi się z MS MARCO/BEIR na konkretną domenę i język polski?
13. Czy generator wielojęzyczny powinien generować po polsku, po angielsku i w wariantach mieszanych?
14. Czy morfologiczna różnorodność języka polskiego wymaga deduplikacji na lematyzowanych formach?
15. Czy dla keywordów biznesowych warto osobno optymalizować relevance i potencjalną wartość komercyjną?

---

## 11. Kolejność czytania

### Szybkie wejście

1. [Document Expansion by Query Prediction](https://arxiv.org/abs/1904.08375)
2. [DeeperImpact](https://arxiv.org/abs/2405.17093)
3. [Doc2Query--](https://arxiv.org/abs/2301.03266)

### Synthetic data i filtracja

4. [Promptagator](https://arxiv.org/abs/2209.11755)
5. [InPars-v2](https://arxiv.org/abs/2301.01820)
6. [InPars Toolkit](https://arxiv.org/abs/2307.04601)
7. [InPars+](https://arxiv.org/abs/2508.13930)

### Pokrycie i różnorodność

8. [CCQGen](https://arxiv.org/abs/2502.11181)
9. [Doc2Query++](https://arxiv.org/abs/2510.09557)
10. [Nucleus sampling](https://arxiv.org/abs/1904.09751)
11. [Min-p sampling](https://arxiv.org/abs/2407.01082)

---

## 12. Podsumowanie decyzji

- **Nie zmieniać top-k intuicyjnie między próbkami jako głównego źródła różnorodności.**
- **Wykonywać wiele niezależnych samplingów przy stałym top-p/min-p.**
- **Używać 2–3 temperatur tylko jako kontrolowanej mieszanki exploitation/exploration.**
- **Różnorodność semantyczną uzyskiwać przede wszystkim przez intencje, koncepcje i pamięć wcześniejszych generacji.**
- **Generować więcej, niż potrzeba, a następnie filtrować relevance i deduplikować semantycznie.**
- **Dla danych treningowych stosować round-trip consistency i rozważyć preference optimization.**
- **Dla BM25 mierzyć jakość razem z rozmiarem indeksu i latency.**
- **Dla dense retrieval sprawdzić osobny indeks/fusion zamiast prostego doklejania tekstu.**
- **Każdą metodę porównywać z docT5query, promptingiem bez fine-tuningu i wariantem bez ekspansji.**
- **Nie istnieje jeden uniwersalny „topowy doc2query”: najlepszy pipeline zależy od downstream — BM25, learned sparse, dense retrieval lub synthetic training data.**
