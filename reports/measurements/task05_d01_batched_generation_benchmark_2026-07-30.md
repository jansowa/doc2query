# D01 batched generation — microbenchmark GPU 2026-07-30

## Decyzja

Przypięto `generation_batch_size=16` dla matched baseline W05 1.5B i W06
4.5B. Jest to zmiana wyłącznie wykonawcza; pełna jakość nie została zmierzona.

| Model | Passage | Batch | Passage/s | Peak allocated VRAM | Query | Próby |
|---|---:|---:|---:|---:|---:|---:|
| W05 1.5B | 16 | 4 | 0.882 | 1.16 GB | 63 | 69 |
| W05 1.5B | 16 | 8 | 1.133 | 1.29 GB | 63 | 70 |
| W05 1.5B | 16 | 16 | 1.775 | 1.54 GB | 63 | 69 |
| W06 4.5B | 8 | 4 | 0.398 | 2.95 GB | 32 | 36 |
| W06 4.5B | 8 | 8 | 0.598 | 3.12 GB | 32 | 35 |
| W06 4.5B | 16 | 16 | 0.856 | 3.47 GB | 64 | 71 |

Sprzęt: NVIDIA GeForce RTX 3060 Ti, 8192 MiB VRAM;
generacja BF16, pinned modele/adapters i frozen `dev_intrinsic_rank10`.
`elapsed_seconds` obejmuje właściwą generację po załadowaniu modelu.

Wcześniejsza ścieżka batch-1 obserwowana podczas pełnego runu miała około
0.25 passage/s i niskie GPU-util. Batch 16 daje orientacyjnie około 7.1x dla
1.5B i 3.4x dla 4.5B względem tej obserwacji, ale krótkie próbki, różne
długości i retry nie pozwalają traktować tych wartości jako gwarantowanego ETA.
Przy zmierzonych szybkościach sama generacja 6598 passage to około 62 min dla
1.5B i 128 min dla 4.5B, przed narzutami i możliwymi odchyleniami pełnego runu.

## Odtwarzalność i ograniczenia

- Każdy prompt ma osobny generator Torch i logiczny seed; retry innych promptów
  nie konsumują jego strumienia RNG.
- Batch, sampler i harmonogram retry są częścią generation identity. Wznowienie
  z innym batchem jest fail-closed.
- Journal zatwierdza atomowo stałą porcję pasaży; po przerwaniu powtarzana jest
  co najwyżej ostatnia niezatwierdzona porcja.
- Wyniki batch 1 i batch N nie muszą być bitowo identyczne: padding i kształt
  batcha mogą nieznacznie zmienić logity. Microbenchmark to potwierdził.
- Stary journal 220 passage pod `uncontrolled.full.jsonl.journal.jsonl` nie
  został wykorzystany ani nadpisany. V2 używa `uncontrolled.batched_v2.jsonl`.
- Nie otwarto finalnych testów i nie zinterpretowano jakości wygenerowanych
  zapytań.
