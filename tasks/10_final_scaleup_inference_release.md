# Task 10 — Finalny trening, generacja korpusu i release

> [Centralny rejestr zadań i statusów](README.md). Każda zmiana statusu lub zakresu tego zadania musi aktualizować rejestr w tym samym commicie.

## Status

`BLOCKED`

Zadanie oczekuje na wyniki Task 09 i zatwierdzony finalny ADR.

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
- wykonaj 100-step rehearsal na identycznym sprzęcie;
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

## Finalna ewaluacja

- pełny intrinsic test raz, po zamrożeniu;
- pełny probe lub docelowy embedder trening;
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
