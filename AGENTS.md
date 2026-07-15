# Program badawczo-inżynieryjny: Bielik doc2query dla treningu polskiego embeddera

## 1. Rola tego pliku

Ten plik jest nadrzędną instrukcją dla systemu agentowego Codex. Codex ma przeczytać go przed rozpoczęciem pracy, a następnie wykonywać zadania z katalogu `tasks/` w kolejności wynikającej z zależności.

Nie traktuj tego projektu jako pojedynczego treningu. To program eksperymentalny, którego celem jest ustalenie, **jaka procedura generowania zapytań rzeczywiście poprawia końcowy embedder**, a nie tylko daje dobrze wyglądające pytania.

Codex ma:

1. utworzyć kompletny, testowalny projekt Python;
2. przygotować skrypty do walidacji danych, treningu, generacji, scoringu, DPO/GRPO i ewaluacji;
3. wdrożyć rejestrowanie eksperymentów i odtwarzalność;
4. uruchomić tanie testy i smoke testy;
5. przygotować komendy i konfiguracje dla drogich treningów;
6. nigdy nie deklarować wyniku eksperymentu, którego faktycznie nie uruchomiono;
7. podejmować decyzje dopiero na podstawie ustalonych bramek eksperymentalnych.

Oficjalna dokumentacja Codex zaleca umieszczanie trwałych reguł repozytorium w `AGENTS.md`; ten plik pełni dokładnie tę rolę.

---

## 2. Kontekst i cel

Dostępne jest około 500 tys. przykładów. Każdy przykład zawiera:

- jedno naturalne zapytanie użytkownika;
- co najmniej jeden pozytywny pasaż;
- co najmniej dziesięć trudnych negatywów.

Docelowo trzeba wytrenować model `doc2query`, który dla pasażu generuje jedno lub kilka polskich zapytań nadających się do budowy danych treningowych dla embeddera.

Podstawowe modele generatora:

- `speakleash/Bielik-1.5B-v3` lub wariant instruct — wyłącznie do taniego prototypowania procedur;
- `speakleash/Bielik-4.5B-v3.0-Instruct` — domyślny model rozwojowy na 16 GB VRAM;
- `speakleash/Bielik-Minitron-7B-v3.0-Instruct` oraz, dla korpusu wyłącznie polskiego, wariant z polskim tokenizatorem `speakleash/Bielik-PL-Minitron-7B-v3.0-Instruct` — kandydaci do treningu finalnego.

Bielik 4.5B ma około 4.6 mld parametrów, a Bielik Minitron 7B około 7.35 mld. Repozytoria modeli mogą wymagać zaakceptowania warunków dostępu na Hugging Face. Nie omijaj tego mechanizmu i nie kopiuj wag do repozytorium.

### Główny cel jakościowy

Generator ma produkować zapytania, które jednocześnie:

1. są ugruntowane w pasażu i można na nie odpowiedzieć na jego podstawie;
2. prowadzą do wysokiego wyniku pozytywnego pasażu względem hard negative’ów;
3. nie są mechaniczną kopią pasażu;
4. obejmują różne fragmenty dokumentu;
5. mają zróżnicowany styl wyszukiwawczy;
6. poprawiają końcową jakość embeddera na naturalnych, niewidzianych zapytaniach.

**Metryki powierzchniowe generatora są pomocnicze. Ostatecznym kryterium jest wynik embeddera wytrenowanego na danych syntetycznych.**

---

## 3. Najważniejsze decyzje metodologiczne

### 3.1. Zacznij od QLoRA SFT, nie od RL

Na pojedynczej karcie 16 GB:

- używaj 4-bitowego NF4, podwójnej kwantyzacji, LoRA, gradient checkpointingu i batch size 1;
- akumuluj gradienty do żądanego efektywnego batcha;
- ogranicz długość wejścia po analizie percentyli długości, zwykle zaczynając od 768–1024 tokenów;
- generowane zapytanie powinno mieć zwykle maksymalnie 64–96 nowych tokenów;
- najpierw doprowadź SFT do stabilnego i mierzalnego baseline’u.

Nie trenuj własnego rerankera w tym projekcie. Użyj gotowego, zamrożonego modelu jako proxy jakości i osobnego shadow judge do kontroli. Jednoczesne dostrajanie generatora i sędziego tworzyłoby niestacjonarną nagrodę, utrudniało interpretację wyników i sprzyjało reward hackingowi.

### 3.2. Pozytywny i negatywny dokument nie są bezpośrednią parą DPO

DPO wymaga dla **tego samego promptu** preferowanej i odrzuconej odpowiedzi:

```json
{
  "prompt": "<pasaż i instrukcja>",
  "chosen": "dobre zapytanie",
  "rejected": "gorsze zapytanie"
}
```

Obecne dane zawierają preferencję między dokumentami dla zapytania, nie między zapytaniami dla dokumentu. Hard negative’y wykorzystaj do:

- benchmarku i kalibracji gotowego, zamrożonego rerankera;
- obliczania marginesu `score(query, positive) - score(query, hard_negative)`;
- ewaluacji źródłowego pasażu;
- tworzenia części trudnych, ale niebanalnych odrzuconych zapytań.

Aby zastosować DPO:

1. wytrenuj SFT;
2. wygeneruj 4–8 kandydatów dla tego samego pasażu, z różnymi kontrolami i temperaturami;
3. oceń kandydatów wielokryterialnie;
4. wybierz `chosen` i `rejected`, zachowując minimalny margines jakości i unikając odrzuceń całkowicie losowych;
5. ręcznie sprawdź próbkę par preferencji;
6. dopiero wtedy uruchom DPO.

### 3.3. Nie minimalizuj bezwzględnie pokrycia lematów

Niektóre wspólne lematy są konieczne: nazwy własne, terminy domenowe, liczby i jednostki. Nagroda za „jak najmniej wspólnych lematów” będzie zachęcała do halucynacji i nieprecyzyjnych parafraz.

Zamiast tego zastosuj **docelowy przedział pokrycia** skalibrowany na naturalnych zapytaniach walidacyjnych:

- kara za bardzo wysokie pokrycie treściowe i długie skopiowane ciągi;
- kara za skrajnie niskie pokrycie, gdy zanika związek z dokumentem;
- brak kary lub dodatnia nagroda w środkowym przedziale;
- osobne traktowanie encji, liczb i terminów specjalistycznych;
- stopwordy i najczęstsze lematy ignoruj.

W pierwszej wersji wdrożenia użyj taniej normalizacji tekstowej. Pełną lematyzację spaCy/Stanza uruchamiaj na CPU i cachuj. Pasaże lematyzuj offline; online lematyzuj tylko krótkie wygenerowane zapytania.

### 3.4. Reranker jest zamrożonym proxy ugruntowania, nie dowodem logicznego entailmentu

Użyj gotowego polskiego cross-encodera jako zamrożonego sędziego. Domyślnym kandydatem jest `sdadas/polish-reranker-roberta-v3`; porównaj go z co najmniej jednym niezależnym modelem kontrolnym. Nie aktualizuj wag żadnego z nich na danych projektu. Głównym sygnałem ma być margines rankingowy źródłowego pasażu względem negatywów.

Nie utożsamiaj pojedynczego wysokiego logitu rerankera z gwarancją, że odpowiedź znajduje się w pasażu. W ewaluacji dodaj:

- ranking źródłowego pasażu wśród hard negative’ów;
- scoring na poziomie zdań;
- opcjonalne generowanie krótkiej odpowiedzi lub identyfikatora zdania dowodowego;
- ręczną kontrolę, czy na query można odpowiedzieć z pasażu;
- testy adwersarialne, w których zapytanie dotyczy podobnego tematu, ale nieobecnego faktu.

### 3.5. Kontrola różnorodności ma być jawna

Nie polegaj wyłącznie na temperaturze. Wprowadź kontrolki:

- `style`: `full_question`, `keyword_query`, `entity_lookup`, `definition`, `how_to`, `comparison`, `fact_lookup`;
- `focus`: numer zdania, bucket `beginning/middle/end` lub jawnie oznaczone zdanie docelowe;
- opcjonalnie `length`: `short/medium`;
- opcjonalnie `needs_entity`: `true/false`.

Domyślna ścieżka to **jedno kontrolowane zapytanie na jedno wywołanie**. Kilka zapytań dla dokumentu generuj przez macierz kontrolek i deduplikację. Trening jednego wyjścia zawierającego listę wielu pytań jest osobnym eksperymentem, nie baseline’em.

---

## 4. Architektura rozwiązania

Utwórz projekt o strukturze zbliżonej do:

```text
.
├── AGENTS.md
├── README.md
├── pyproject.toml
├── uv.lock
├── Makefile
├── configs/
│   ├── data/
│   ├── model/
│   ├── reranker/
│   ├── train/
│   ├── generation/
│   ├── rewards/
│   └── experiments/
├── src/doc2query/
│   ├── cli.py
│   ├── schemas.py
│   ├── data/
│   │   ├── validate.py
│   │   ├── normalize.py
│   │   ├── deduplicate.py
│   │   ├── split.py
│   │   ├── invert.py
│   │   ├── style_labels.py
│   │   └── focus_labels.py
│   ├── models/
│   │   ├── load_generator.py
│   │   ├── lora.py
│   │   └── templates.py
│   ├── reranker/
│   │   ├── load.py
│   │   ├── infer.py
│   │   ├── benchmark.py
│   │   └── calibrate.py
│   ├── rewards/
│   │   ├── lexical.py
│   │   ├── grounding.py
│   │   ├── diversity.py
│   │   ├── style.py
│   │   └── composite.py
│   ├── training/
│   │   ├── sft.py
│   │   ├── weighted_sft.py
│   │   ├── dpo.py
│   │   └── grpo.py
│   ├── generation/
│   │   ├── candidates.py
│   │   ├── controlled.py
│   │   └── deduplicate.py
│   ├── preferences/
│   │   ├── score.py
│   │   └── build.py
│   ├── evaluation/
│   │   ├── generator.py
│   │   ├── retrieval.py
│   │   ├── embedder_probe.py
│   │   ├── slices.py
│   │   ├── bootstrap.py
│   │   └── report.py
│   └── utils/
│       ├── hardware.py
│       ├── reproducibility.py
│       ├── tracking.py
│       └── io.py
├── scripts/
│   ├── train_sft.py
│   ├── benchmark_rerankers.py
│   ├── calibrate_reranker.py
│   ├── generate_candidates.py
│   ├── score_candidates.py
│   ├── build_preferences.py
│   ├── train_dpo.py
│   ├── train_grpo.py
│   ├── evaluate_generator.py
│   ├── train_probe_embedder.py
│   └── build_report.py
├── tests/
├── tasks/
└── reports/
```

Nazwy mogą się nieznacznie różnić, ale publiczne CLI, konfiguracje i odpowiedzialności modułów muszą pozostać rozdzielone.

---

## 5. Kontrakt danych

Obsłuż wejściowy JSONL/Parquet przez jawny mapping kolumn w konfiguracji. Kanoniczny rekord:

```json
{
  "example_id": "q-123",
  "query": "jak działa pompa ciepła",
  "positives": [
    {"doc_id": "d-10", "text": "...", "metadata": {}}
  ],
  "hard_negatives": [
    {"doc_id": "d-11", "text": "...", "metadata": {}}
  ],
  "metadata": {
    "source": "...",
    "domain": "...",
    "language": "pl"
  }
}
```

Po odwróceniu do doc2query:

```json
{
  "pair_id": "q-123::d-10",
  "doc_id": "d-10",
  "passage": "...",
  "query": "jak działa pompa ciepła",
  "query_style": "keyword_query",
  "focus_sentence_id": 2,
  "focus_bucket": "middle",
  "content_lemma_overlap": 0.23,
  "negative_doc_ids": ["d-11", "..."],
  "split": "train"
}
```

Wymagania:

- zachowuj wszystkie pozytywy, ale kontroluj nadreprezentację zapytań z wieloma pozytywami;
- nie pozwól, aby ten sam dokument lub jego near-duplicate trafił do wielu splitów;
- testowy pozytyw nie może występować w treningu jako hard negative;
- komponenty grafu query–positive-document oraz klastry near-duplicate mają być podstawową jednostką splitu;
- raportuj konflikty i przykłady odrzucone;
- zapisuj fingerprint danych, wersję kodu i konfigurację.

---

## 6. Szablon SFT

Domyślnie użyj prompt-completion i licz loss wyłącznie na completion.

Przykładowy prompt:

```text
Wygeneruj jedno polskie zapytanie wyszukiwawcze, na które można odpowiedzieć wyłącznie na podstawie podanego pasażu.
Nie kopiuj długich fragmentów pasażu. Zachowaj konieczne nazwy własne, liczby i terminy.
Styl: {style}
Docelowy fragment: {focus_instruction}
Długość: {length_control}

Pasaż:
{passage}

Zapytanie:
```

Completion ma zawierać samo zapytanie, bez komentarza, numeracji i prefiksu.

Porównaj eksperymentalnie:

1. base model vs instruct model;
2. brak kontrolek vs kontrolki stylu;
3. brak kontrolek focus vs focus bucket vs oznaczone zdanie;
4. losowy sampling danych vs sampling równoważący styl, focus i pokrycie lematów;
5. zwykłe SFT vs ważone SFT;
6. 1 zapytanie na completion vs lista 3 zapytań w JSON.

---

## 7. Gotowy reranker i proxy jakości

W tym projekcie nie trenuj własnego rerankera. Minimum:

- integracja gotowego polskiego rerankera jako zamrożonego primary judge;
- co najmniej jeden niezależny shadow judge lub jawne uzasadnienie jego braku;
- walidacja rankingowa na odseparowanym zbiorze z pozytywem i 10+ hard negative’ami;
- kalibracja wyjść bez aktualizacji wag, np. robust z-score, percentyle, Platt scaling lub isotonic regression;
- wydajna inferencja CPU albo osobny offline scoring, jeżeli online reward jest zbyt wolny.

Kandydaci do porównania:

- `sdadas/polish-reranker-roberta-v3` jako domyślny polski primary judge;
- `BAAI/bge-reranker-v2-m3` lub inny silny model wielojęzyczny jako shadow judge;
- opcjonalnie drugi gotowy polski reranker jako dodatkowy punkt odniesienia.

Dla każdego modelu przypnij revision, sprawdź licencję, odnotuj `trust_remote_code`, długość kontekstu, truncation, koszt i throughput. Surowe logity modeli nie są bezpośrednio porównywalne.

Reranker musi zwracać:

- score pary query–passage;
- margines względem najtrudniejszego negatywu;
- ranking źródłowego pasażu;
- score najlepszego zdania w pasażu;
- flagi nietypowych przypadków, np. wszystkie wyniki bliskie sobie;
- disagreement względem shadow judge.

Jeśli gotowy model słabo koreluje z ręcznym holdoutem, najpierw sprawdź format wejścia, truncation, kalibrację, ensemble, dodatkowy answerability checker i ograniczenie rerankera do offline best-of-N. Dostrajanie rerankera nie jest częścią bieżącego zakresu.

---

## 8. Nagrody i scoring kandydatów

Każdy komponent nagrody ma być osobnym, testowalnym modułem. Nie mieszaj skali komponentów bez normalizacji na zbiorze kalibracyjnym.

### 8.1. Grounding/retrieval reward

Preferowany składnik:

```text
R_ground = calibrated(score(q, positive))
           + lambda_margin * calibrated(score(q, positive) - max_j score(q, negative_j))
```

Dodatkowo licz MRR/Recall pozytywnego pasażu w puli negatywów.

### 8.2. Lexical-copy reward

Licz co najmniej:

- Jaccard lematów treściowych;
- precision/recall pokrycia lematów;
- maksymalny skopiowany n-gram;
- udział tokenów należących do długiego wspólnego podciągu;
- zachowanie encji, liczb i jednostek.

Nagroda ma mieć kształt pasmowy, np. maksimum w przedziale wyznaczonym z danych naturalnych. Progi nie mogą być zakodowane na sztywno bez raportu kalibracyjnego.

### 8.3. Diversity reward

Dla grupy K zapytań do jednego dokumentu licz:

- pairwise lemma Jaccard;
- pairwise embedding cosine;
- Self-BLEU lub analogiczną miarę n-gramową;
- distinct-1 i distinct-2;
- liczbę unikalnych klastrów semantycznych;
- różnorodność stylów i focus bucketów.

### 8.4. Focus coverage reward

Przypisz każdemu zapytaniu zdanie lub bucket o najwyższym score. Dla grupy K zapytań licz:

- entropię rozkładu pozycji;
- udział zapytań przypisanych do pierwszego zdania;
- liczbę unikalnych bucketów;
- zgodność z żądaną kontrolką focus.

### 8.5. Style reward

W pierwszej wersji użyj reguł i lekkiego klasyfikatora stylu. Pełne pytanie, fraza wyszukiwawcza i pozostałe style muszą być rozróżnialne. Karz niezgodność z kontrolką, ale nie wymuszaj jednego globalnego rozkładu; rozkład docelowy ma być konfigurowalny.

### 8.6. Pozostałe ograniczenia

Karz:

- pusty output;
- kilka odpowiedzi zamiast jednego query, jeśli tryb jest single-query;
- metakomentarze;
- oczywistą halucynację formatu;
- zbyt długie query;
- duplikaty;
- zapytania zawierające odpowiedź zamiast intencji wyszukiwawczej.

---

## 9. Ewaluacja

### 9.1. Intrinsic generator evaluation

Raportuj:

- source passage Recall@1, Recall@5, MRR, nDCG@10;
- reranker score i margin;
- rozkład pokrycia lematów, kopiowanych n-gramów i długości;
- format validity;
- answerability proxy;
- style accuracy i rozkład stylów;
- focus accuracy, pozycję zdania i entropię pokrycia;
- distinct-n, Self-BLEU, pairwise cosine i odsetek duplikatów;
- throughput, peak VRAM, czas i rozmiar adaptera.

Każdy raport ma zawierać slice’y:

- niski/średni/wysoki overlap naturalnego query z pasażem;
- krótki/długi pasaż;
- pozycja odpowiedniego zdania;
- encje/liczby/brak encji;
- domena i źródło;
- pełne pytanie vs fraza;
- liczba pozytywów;
- trudność mierzona baseline’em rerankera.

### 9.2. Extrinsic embedder evaluation — metryka główna

Dla każdego poważnego wariantu generatora:

1. wygeneruj tę samą liczbę syntetycznych par;
2. wytrenuj ten sam mały „probe embedder” przy identycznym budżecie, seedach i hard negative’ach;
3. oceń go na naturalnym, zamrożonym teście;
4. raportuj Recall@k, MRR@10, nDCG@10, MAP i hard-negative win rate;
5. wykonaj bootstrap po zapytaniach i podaj 95% CI;
6. dopiero potem uznaj wariant generatora za lepszy.

Nie wybieraj modelu tylko dlatego, że ma mniejszy overlap lub wyższy score rerankera.

### 9.3. Ewaluacja człowieka

Przygotuj ślepy formularz dla co najmniej 300 przypadków w finalnej fazie. Oceniane pola:

- czy można odpowiedzieć na query z pasażu;
- czy query brzmi naturalnie;
- czy query jest użyteczne wyszukiwawczo;
- czy nie jest nadmiernie skopiowane;
- czy nie zdradza odpowiedzi;
- preferencja A/B;
- fragment pasażu, którego dotyczy.

Mierz zgodność oceniających, a nie tylko średnią ocenę.

---

## 10. Plan faz i bramki decyzyjne

### Faza A — infrastruktura i dane

Zadania `00–02`.

Brama:

- walidator i testy przechodzą;
- raport danych nie wykazuje leakage;
- reranker ma sensowny wynik rankingowy na naturalnym dev/test;
- wszystkie pipeline’y działają na syntetycznym mini-zbiorze.

### Faza B — baseline SFT

Zadania `03–04`.

Uruchom najpierw 1.5B na 10–50 tys. par, potem 4.5B na 50–100 tys. Nie zaczynaj pełnych 500 tys., dopóki generator i ewaluacja nie są stabilne.

Brama:

- SFT przewyższa prompting bez treningu w source retrieval;
- nie pogarsza drastycznie rozkładu overlapu względem naturalnych query;
- ma stabilny format;
- probe embedder na syntetycznych query jest co najmniej konkurencyjny względem baseline’u.

### Faza C — kontrolowana różnorodność

Zadanie `05`.

Brama:

- spada odsetek grup, w których wszystkie query dotyczą początku pasażu;
- rośnie entropia focus i różnorodność stylów;
- grounding nie spada więcej niż tolerancja ustalona z CI;
- probe embedder nie traci jakości.

### Faza D — preference optimization

Zadania `06–07`.

Brama:

- preferencje mają wysoką zgodność automatycznego score z ludźmi;
- DPO daje poprawę co najmniej jednej ważnej wady bez istotnego pogorszenia retrieval;
- efekt utrzymuje się dla co najmniej dwóch seedów na redukowanym eksperymencie.

### Faza E — RL/GRPO

Zadanie `08` tylko wtedy, gdy SFT i DPO nie osiągnęły celu.

Brama:

- reward nie wykazuje łatwego hackowania;
- korelacja rewardu z oceną człowieka i wynikiem embeddera jest dodatnia;
- trening jest stabilny na 1.5B;
- dopiero potem dopuszczony jest 4.5B.

Nie planuj GRPO 7B na 16 GB jako domyślnej ścieżki. Dla 7B przewiduj większe zasoby lub offline preference optimization.

### Faza F — wybór i finalny trening

Zadania `09–10`.

Wybór modelu wykonuj na macierzy Pareto: jakość końcowego embeddera, ugruntowanie/answerability, różnorodność, kopiowanie i koszt generacji.

---

## 11. Macierz eksperymentów minimalnych

Każdy eksperyment ma unikalny identyfikator, config, seed, commit i raport.

| ID | Generator | Dane | Cel |
|---|---|---|---|
| E00 | Bielik bez treningu | 5k dev | baseline promptingu |
| E01 | 1.5B QLoRA SFT | 10k | smoke i dobór parametrów |
| E02 | 1.5B QLoRA SFT | 50k | stabilny tani baseline |
| E03 | 4.5B QLoRA SFT | 50k | wpływ skali |
| E04 | 4.5B base vs instruct | 50k | wybór checkpointu startowego |
| E05 | 4.5B balanced/weighted SFT | 50k | kontrola overlapu i stylu |
| E06 | 4.5B + style controls | 50k | różnorodność stylu |
| E07 | 4.5B + focus controls | 50k | pokrycie dokumentu |
| E08 | single-query vs multi-query JSON | 50k | strategia generacji K query |
| E09 | filtered continued SFT | pref subset | kontrola dla DPO |
| E10 | DPO | 20–100k par | preference optimization |
| E11 | 1.5B GRPO | 5–20k promptów | wykonalność RL |
| E12 | 4.5B GRPO | tylko po E11 | test RL na modelu docelowym |
| E13 | 4.5B pełne dane | 500k | finalista lokalny |
| E14 | 7B/7B-PL pełne dane | 500k | finalista na większych zasobach |

Dla eksperymentów selekcyjnych stosuj 3 seedy, chyba że koszt jest nieproporcjonalny. Dla bardzo drogich finalnych treningów minimum 2 seedy lub pełna replikacja skróconego runu plus jeden pełny run.

---

## 12. Budżet 16 GB VRAM

Domyślne ustawienia startowe, które agent ma traktować jako punkt wyjścia, nie dogmat:

```yaml
quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: nf4
  bnb_4bit_use_double_quant: true
  bnb_4bit_compute_dtype: bf16_or_fp16_after_capability_check

lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules: auto_detect_linear_attention_and_mlp

training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
  gradient_checkpointing: true
  max_length: 1024
  use_cache: false
  optimizer: paged_adamw_8bit
  warmup_ratio: 0.03
  lr_scheduler_type: cosine
  learning_rate: 0.0001
```

Wymagane ablacją:

- `r ∈ {8,16,32}`;
- LR dla SFT `5e-5, 1e-4, 2e-4`;
- max length `512, 768, 1024` zależnie od pokrycia danych;
- target modules: attention-only vs all-linear;
- packing on/off;
- LoRA dropout `0, 0.05`.

Nie używaj CPU offloadingu całego generatora jako domyślnej ścieżki treningowej. Dopuszczalne na CPU/RAM:

- lematyzacja;
- offline scoring rerankera;
- precompute referencyjnych log-probów dla DPO;
- przygotowanie danych i indeksów;
- ewaluacja małymi batchami.

Przed każdym runem wykonaj krótki memory probe. Loguj `torch.cuda.max_memory_allocated()` i `max_memory_reserved()`. W przypadku OOM redukuj kolejno: długość, batch generacyjny, liczbę generacji, rank LoRA; dopiero później rozważ bardziej agresywny offload.

---

## 13. DPO na 16 GB

Domyślna procedura:

- start od adaptera SFT;
- QLoRA;
- `precompute_ref_log_probs=true`, jeżeli biblioteka i format na to pozwalają;
- batch 1, gradient accumulation;
- `max_length` zgodny z SFT;
- krótkie completion;
- testuj `beta ∈ {0.05, 0.1, 0.2}`;
- preferuj klasyczne `sigmoid`, a inne lossy traktuj jako późniejszą ablacją;
- zawsze porównuj z continued SFT na samych `chosen`.

Nie buduj preferencji na podstawie jednego arbitralnego score. Wymagaj co najmniej:

- dodatniego marginesu rerankera;
- poprawnego formatu;
- akceptowalnego overlapu;
- braku duplikatu;
- różnicy całkowitego score większej od progu kalibracyjnego;
- odrzucenia par o niepewnej kolejności.

---

## 14. GRPO/RL

GRPO ma być eksperymentem późnej fazy. Używaj kilku nagród obsługiwanych jako osobne funkcje i jawnych wag.

Startowy skład:

```text
R = 1.00 * R_ground
  + 0.35 * R_margin
  + 0.20 * R_overlap_band
  + 0.15 * R_focus
  + 0.10 * R_style
  + 0.10 * R_format
  + 0.20 * R_group_diversity
  - 0.20 * R_duplicate
```

Wagi są tylko punktem startowym. Każdy komponent znormalizuj na zbiorze kalibracyjnym, monitoruj osobno i wykonaj ablacją „leave-one-reward-out”.

Dla 16 GB:

- zacznij od 1.5B;
- `num_generations=4`, a nie 8;
- `max_completion_length=64`;
- nie uruchamiaj colocated vLLM, jeżeli memory probe nie zostawia bezpiecznego marginesu;
- preferuj zwykłą generację Transformers lub ciągłe batchowanie wspierane przez używaną wersję;
- przy `beta=0` nie jest potrzebny model referencyjny, ale monitoruj dryf względem SFT;
- zachowuj regularne checkpointy i próbki generacji;
- zatrzymuj run po wykryciu reward hackingu.

Przykłady reward hackingu:

- bardzo ogólne pytania uzyskujące wysoki score na wielu dokumentach;
- kopiowanie jednego terminu i usuwanie całej reszty;
- zapytania sztucznie dopasowane do rerankera;
- identyczne szablony z podmienioną encją;
- ukrywanie odpowiedzi w query;
- preferowanie pierwszego zdania mimo formalnej kontrolki.

---

## 15. Audyt odporności zamrożonego sędziego

Nie dostrajaj rerankera na outputach generatora. Zamiast tego wykonaj późny eksperyment odporności:

1. zamroź primary i shadow reranker z przypiętymi revision;
2. wygeneruj adwersarialne query oraz przypadki o wysokim primary score;
3. znajdź disagreement między sędziami, retrieval probe i oceną człowieka;
4. porównaj single judge, ensemble, shadow veto, offline best-of-N i wariant bez online rewardu rerankera;
5. zmieniaj reward, filtry, prompt lub strategię generacji, ale nie wagi sędziego;
6. oceń na niezmiennym naturalnym teście i ręcznej próbce.

Jeżeli dwa silne gotowe rerankery powtarzalnie zawodzą na domenowym holdoucie, przygotuj jedynie ADR dla osobnego przyszłego projektu adaptacji. Nie uruchamiaj tego treningu bez osobnej decyzji użytkownika.

---

## 16. Rejestrowanie eksperymentów

Każdy run zapisuje:

- ID eksperymentu;
- git commit i stan dirty;
- pełny config po rozwiązaniu interpolacji;
- wersje bibliotek, CUDA, sterownika i GPU;
- fingerprint datasetu i splitu;
- seed;
- model bazowy i revision;
- liczbę parametrów LoRA;
- metryki treningowe i walidacyjne;
- peak VRAM i throughput;
- próbki generacji w stałym panelu dokumentów;
- ścieżki do adaptera, predykcji i raportu.

Wybierz jeden backend trackingu, np. MLflow lub Weights & Biases, ale zawsze zapisuj też lokalne JSON/JSONL/Parquet, aby wyniki nie zależały od usługi zewnętrznej.

---

## 17. Testy i definicja ukończenia

Każde zadanie ma własne kryteria akceptacji. Globalnie wymagane są:

- `ruff check`;
- formatowanie;
- type checking przynajmniej dla publicznych API i schematów;
- `pytest`;
- testy deterministyczności danych;
- testy braku leakage;
- testy rewardów na ręcznie zdefiniowanych przykładach po polsku;
- smoke training 5–20 kroków na tiny modelu;
- smoke generation i scoring;
- CLI z `--help`;
- README z komendami od zera;
- brak dużych danych i wag w Git.

Testy CI nie mogą pobierać Bielika ani wymagać GPU. Użyj małego publicznego modelu testowego lub mocków. Osobne testy GPU oznacz markerem.

---

## 18. Kolejność zadań

1. `tasks/00_repository_bootstrap.md`
2. `tasks/01_data_contract_audit_and_splits.md`
3. `tasks/02_reranker_and_reward_proxies.md`
4. `tasks/03_sft_qlora_baselines.md`
5. `tasks/04_evaluation_harness.md`
6. `tasks/05_controlled_diversity_and_multiquery.md`
7. `tasks/06_candidate_scoring_and_preference_data.md`
8. `tasks/07_dpo_training.md`
9. `tasks/08_grpo_multiobjective_rl.md`
10. `tasks/09_experiment_campaign.md`
11. `tasks/10_final_scaleup_inference_release.md`
12. `tasks/11_reranker_robustness_and_fallback.md`

Codex może delegować niezależne moduły subagentom, ale integracja, kontrakty danych i ostateczna walidacja należą do głównego agenta. Zadania zależne od wyników eksperymentów nie mogą być „zaliczone” na podstawie założonych rezultatów.

---

## 19. Zasady bezpieczeństwa badawczego

- Nie zmieniaj zamrożonego testu po obejrzeniu wyników.
- Nie używaj testu do strojenia progów rewardu.
- Nie mieszaj syntetycznych i naturalnych query w raporcie bez jawnego oznaczenia.
- Nie usuwaj trudnych przykładów tylko dlatego, że obniżają metrykę.
- Nie raportuj średniej bez rozkładów i slice’ów.
- Nie przyjmuj, że większy model jest lepszy.
- Nie przyjmuj, że mniejszy overlap jest zawsze lepszy.
- Nie przyjmuj, że reranker jest obiektywnym sędzią.
- Nie publikuj danych, które mogą zawierać informacje prywatne lub objęte ograniczeniami licencyjnymi.
- Przed publikacją adaptera sprawdź licencję modelu bazowego i danych.

---

## 20. Źródła techniczne

- OpenAI Codex, `AGENTS.md`: https://developers.openai.com/codex/agent-configuration/agents-md
- Bielik 4.5B base: https://huggingface.co/speakleash/Bielik-4.5B-v3
- Bielik 4.5B Instruct: https://huggingface.co/speakleash/Bielik-4.5B-v3.0-Instruct
- Bielik Minitron 7B Instruct: https://huggingface.co/speakleash/Bielik-Minitron-7B-v3.0-Instruct
- Bielik PL Minitron 7B Instruct: https://huggingface.co/speakleash/Bielik-PL-Minitron-7B-v3.0-Instruct
- QLoRA: https://arxiv.org/abs/2305.14314
- PEFT/QLoRA: https://huggingface.co/docs/peft/developer_guides/quantization
- BitsAndBytes NF4: https://huggingface.co/docs/transformers/quantization/bitsandbytes
- TRL SFTTrainer: https://huggingface.co/docs/trl/sft_trainer
- TRL DPOTrainer: https://huggingface.co/docs/trl/dpo_trainer
- TRL GRPOTrainer: https://huggingface.co/docs/trl/grpo_trainer
- TRL dataset formats: https://huggingface.co/docs/trl/dataset_formats
- sdadas Polish reranker RoBERTa v3: https://huggingface.co/sdadas/polish-reranker-roberta-v3
- sdadas Polish rerankers collection: https://huggingface.co/collections/sdadas/polish-rerankers
- Polish reranker generalization study: https://arxiv.org/abs/2402.14318
- Sentence Transformers CrossEncoder: https://sbert.net/docs/cross_encoder/usage/usage.html
- spaCy Polish models: https://spacy.io/models/pl
- Doc2Query: https://arxiv.org/abs/1904.08375
- GPL: https://arxiv.org/abs/2112.07577