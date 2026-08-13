# Task 10 — Finalny trening, generacja korpusu i release

> [Centralny rejestr zadań i statusów](README.md). Każda zmiana statusu lub zakresu tego zadania musi aktualizować rejestr w tym samym commicie.

## Status

`BLOCKED`

Zadanie oczekuje na wyniki Task 09 i zatwierdzony finalny ADR.

Aktualizacja 2026-08-13 (rozszerzenie specyfikacji, decyzja właściciela):
dodano wymóg prerejestrowanej reguły decyzyjnej testów finalnych świadomej
mocy statystycznej (M-04, sekcja poniżej) oraz wymóg przenośności pipeline'u
treningowego dla większych modeli na sprzęcie zewnętrznym. Definicje
M-01–M-05: AGENTS.md §9.2.

## Cel

Przenieść wybraną strategię na pełne dane i ewentualnie większe zasoby, następnie przygotować niezawodny pipeline generacji oraz dokumentację modelu.

## Zależności

Task 09 i zatwierdzony ADR.

## Preflight

Przed treningiem:

- zamroź config;
- przypnij model revision;
- sprawdź licencje;
- potwierdź fingerprint pełnych danych;
- potwierdź brak test leakage;
- wykonaj 100-step rehearsal na docelowym sprzęcie treningu (dotyczy także
  treningów na sprzęcie zewnętrznym);
- sprawdź save/resume;
- oszacuj miejsce na checkpointy i cache;
- przygotuj monitoring.

## Finalny trening

Warianty dopuszczone przez ADR:

- 4.5B QLoRA SFT lub SFT+DPO;
- 7B/7B-PL QLoRA na większym GPU;
- RL tylko jeśli przeszedł wszystkie bramki.

Zapisuj checkpointy w sposób umożliwiający wybór najlepszego punktu bez używania testu.

## Merge i eksport

Przygotuj:

- adapter LoRA;
- opcjonalnie merged model, jeśli licencja i zasoby pozwalają;
- tokenizer i chat/prompt template;
- `generation_config.json`;
- skrypt walidujący zgodność adaptera z base revision;
- opcjonalny eksport kwantyzowany do inferencji, oddzielony od artefaktu treningowego.

## Pipeline generacji korpusu

CLI:

```bash
doc2query generate-corpus \
  --input documents.parquet \
  --output generated_queries/ \
  --config configs/generation/final.yaml \
  --num-queries 4 \
  --resume
```

Wymagania:

- shardowanie;
- resumable i idempotent;
- deterministyczne seedy per doc/control;
- deduplikacja per passage i globalna;
- zapis score każdego query;
- retry z limitem;
- rejection log;
- obsługa OOM przez adaptacyjny batch;
- throughput i ETA w logach, bez utraty wyników po przerwaniu;
- możliwość best-of-N i coverage-aware selection;
- opcja generowania jednego query dla minimalnego kosztu.

## Format outputu

```json
{
  "doc_id": "d-10",
  "passage_hash": "...",
  "generator_run_id": "...",
  "queries": [
    {
      "text": "...",
      "style": "keyword_query",
      "focus": "middle",
      "seed": 123,
      "scores": {},
      "accepted": true
    }
  ]
}
```

## Reguła decyzyjna testów finalnych (M-04)

`test_native_pl` ma 956 query; dla efektów rzędu +0.01 nDCG@10 półszerokość
sparowanego 95% CI wynosi około ±0.016 (ekstrapolacja z analizy czułości dla
n=591), więc sam test natywny nie ma mocy, aby potwierdzić efekt przechodzący
bramki dev. Przed zamrożeniem finalistów — a najpóźniej przed jednorazowym
otwarciem testów — prerejestruj w ADR regułę decyzyjną, która:

- czerpie moc statystyczną z licznego `test_translated_msmarco_pl`
  (16 272 query) lub `test_embedder`;
- używa `test_native_pl` jako kontroli kierunku i spójności (np. wymaganie
  nieujemnego efektu punktowego / braku istotnej szkody), a nie jako
  samodzielnego progu istotności;
- określa z góry postępowanie przy wynikach rozbieżnych między testami;
- zachowuje zasadę jednorazowego otwarcia testów finalnych.

Poprawa wyłącznie na teście tłumaczonym bez potwierdzenia kierunku na teście
natywnym nadal nie wystarcza do release bez jawnego ADR (zgodnie z Task 09).

## Przenośność pipeline'u treningowego

Trening finalny większych modeli (7B+) odbywa się na sprzęcie zewnętrznym
przez ten sam pipeline: wszystkie parametry zależne od sprzętu (batch, max
length, kwantyzacja, offload, liczba GPU) są jawnymi polami configu, preflight
i memory probe działają na docelowej maszynie, a rehearsal 100 kroków jest
obowiązkowy przed pełnym runem. Artefakty i manifesty muszą być przenośne
między maszynami: ścieżki względne oraz fingerprinty danych zamiast ścieżek
absolutnych.

## Finalna ewaluacja

- pełny intrinsic test raz, po zamrożeniu;
- pełny probe lub docelowy embedder trening;
- osobne wyniki `test_native_pl` i `test_translated_msmarco_pl`;
- human panel;
- porównanie z natural-only, heuristic i najlepszym baseline’em;
- slice’y;
- koszt wytworzenia miliona query;
- analiza błędów.

## Model card / data card

Udokumentuj:

- cel i niezalecane użycia;
- model bazowy i revision;
- sposób treningu;
- dane i ograniczenia;
- metryki;
- oddzielne metryki natywnego i tłumaczonego holdoutu;
- różnorodność stylów;
- ryzyko halucynacji i query, na które nie można odpowiedzieć z pasażu;
- licencje;
- wymagania sprzętowe;
- dokładny prompt i przykład inferencji;
- wersję rerankera/lemmatyzera użytego do selekcji.

## Kryteria akceptacji

- pełny run można wznowić;
- output ma provenance i score;
- finalny embedder jest oceniony na naturalnym teście;
- artefakty nie zawierają sekretów ani danych prywatnych;
- release ma model card, config, checksums i instrukcję reprodukcji;
- znane porażki są opisane, nie ukryte.
