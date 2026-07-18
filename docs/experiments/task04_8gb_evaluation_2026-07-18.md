# Task 04: rzeczywista ewaluacja W03/W05 na 8 GB, 2026-07-18

## Zamrożone dane

Nie zmieniono ani nie zregenerowano splitów v1. Przed oceną modeli utworzono
lokalne listy ID i manifest `task04-v1`; ich śledzony skrót znajduje się w
`configs/evaluation/task04_v1_fingerprints.json`.

| Zbiór | Liczba | Wykluczenia | Fingerprint rekordów |
|---|---:|---:|---|
| `dev_intrinsic` | 16 272 | 0 | `cc4489a23003…fddf9` |
| `dev_intrinsic_rank10` | 6 598 | 9 674 | `235d9b81e04d…ffab6` |
| `test_intrinsic` / `test_embedder` | 16 272 | 0 | `3aef27e01c68…b10f` |
| `test_intrinsic_rank10` / `test_embedder_rank10` | 6 632 | 9 640 | `a9697cecb7e1…61fb` |
| `test_generator_panel_rank10` | 100 | 6 532 poza panelem | `ce3a6fbd43c9…85dd` |
| `test_human_panel` | 300 | 6 332 poza panelem | `10234c3cce0c…9880` |
| `test_adversarial` | 150 | 0 | `48a0c7ff1782…aa44` |

9 674 rekordy dev i 9 640 rekordów test wykluczono wyłącznie z metryk
wymagających co najmniej 10 hard negative’ów, ponieważ po cleanupie
cross-split Tasku 01 mają ich mniej. Rekordy pozostały w pełnych splitach v1.

## Wykonane inference

Użyto istniejących adapterów W03 i W05. Nie wykonano ponownego SFT, backward ani
aktualizacji wag generatora. Dla każdego checkpointu i każdego z 100 rekordów
zamrożonego panelu wygenerowano:

- 1 query deterministic/greedy;
- 4 query diverse, temperatura 0,8 i top-p 0,95;
- maksymalnie 64 nowe tokeny.

| Run | Generacje | Czas | Generacje/s | Peak allocated | Peak reserved |
|---|---:|---:|---:|---:|---:|
| W03 | 500 | 234,57 s | 2,132 | 1,129 GiB | 1,318 GiB |
| W05 | 500 | 226,37 s | 2,209 | 1,129 GiB | 1,322 GiB |

Primary judge:
`sdadas/polish-reranker-roberta-v3@e6471da541f4e7be33845b6d57248a8d8bde27e8`,
zamrożony, inference na GPU. Każde query oceniono względem pozytywnego pasażu i
co najmniej 10 istniejących hard negative’ów. Scoring trwał odpowiednio 81,16 s
i 83,33 s.

Lokalne, ignorowane przez Git artefakty znajdują się w
`reports/evaluation/W03-1.5B-10K-LR2E4-S42/`,
`reports/evaluation/W05-1.5B-50K-8GB/` oraz
`reports/evaluation/W03-vs-W05.bootstrap.json`. Finalny smoke probe’a znajduje
się w `runs/task04-probe-smoke-natural-v3/`.

## Wyniki intrinsic

### Deterministic

| Run | R@1 | R@5 | MRR | nDCG@10 | Mean margin | Jaccard | Copy density | Format | Sentence source hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| W03 | 0,9400 | 0,9900 | 0,9650 | 0,9738 | 4,6414 | 0,0669 | 0,3767 | 1,0000 | 0,9500 |
| W05 | 0,9200 | 0,9700 | 0,9433 | 0,9568 | 4,5596 | 0,0627 | 0,3755 | 1,0000 | 0,9200 |

### Diverse

Wartości retrieval są makro po 100 naturalnych rekordach; najpierw uśredniono
cztery kandydaty danego rekordu.

| Run | R@1 | R@5 | MRR | nDCG@10 | Mean margin | Jaccard | Copy density | Format | Sentence source hit | Duplicate rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| W03 | 0,9350 | 0,9825 | 0,9562 | 0,9667 | 4,4302 | 0,0597 | 0,3338 | 1,0000 | 0,9225 | 0,1200 |
| W05 | 0,9225 | 0,9825 | 0,9509 | 0,9629 | 4,4442 | 0,0577 | 0,3371 | 1,0000 | 0,8975 | 0,1500 |

Ogółem koncentracja na pierwszym zdaniu wyniosła 0,418 dla W03 i 0,410 dla
W05. Entropia bucketów focus wyniosła odpowiednio 1,524 i 1,544. W panelu 99
dokumentów ma klaster near-duplicate rozmiaru 1, a jeden rozmiaru 2; metryki
slice zapisano w `summary.json`.

## Bootstrap W03 vs W05

Wykonano 5 000 sparowanych bootstrapów po 100 rekordach. Różnica to W05 minus
W03. Dla diverse cztery kandydaty uśredniono wewnątrz rekordu przed
resamplingiem.

| Tryb | Metryka | Różnica | 95% CI |
|---|---|---:|---:|
| deterministic | R@1 | -0,0200 | [-0,0700; 0,0300] |
| deterministic | MRR | -0,0218 | [-0,0614; 0,0133] |
| deterministic | nDCG@10 | -0,0170 | [-0,0473; 0,0096] |
| deterministic | margin | -0,0818 | [-0,5539; 0,3709] |
| diverse | R@1 | -0,0125 | [-0,0425; 0,0175] |
| diverse | MRR | -0,0053 | [-0,0244; 0,0133] |
| diverse | nDCG@10 | -0,0038 | [-0,0182; 0,0102] |
| diverse | margin | 0,0141 | [-0,2522; 0,2879] |

Wszystkie wymienione przedziały obejmują zero. Te pomiary nie dowodzą przewagi
W03 ani W05 i nie zastępują głównej metryki probe embeddera.

## Probe smoke i niewykonane pomiary

Pipeline probe embeddera przeszedł osobny, nieporównywalny smoke: 2 kroki na 16
naturalnych parach, peak allocated 3,06 GiB, następnie retrieval na 100
zamrożonych rekordach względem ich wspólnego korpusu 1 123 dokumentów. Wyniki
tego smoke nie są rankingiem generatorów.

Nie wykonano:

- pełnych probe’ów natural/copy/W03/W05 według budżetu 1 000 kroków;
- przygotowania równolicznych syntetycznych zbiorów treningowych probe’a;
- niezależnego shadow judge BGE (jego pinned weights nie były lokalne);
- embedding cosine i semantic clustering query;
- ocen ludzi; wyeksportowano jedynie ślepy formularz A/B dla 100 rzeczywiście
  wygenerowanych przypadków, podczas gdy finalna faza wymaga co najmniej 300;
- pomiaru na pełnych 6 632 rekordach rank10 ani pełnym `test_embedder`;
- porównania S00 prompting bez treningu.

Z tego powodu Task 04 ma status `IMPLEMENTED`, nie `DONE`, a bramka Fazy B
dotycząca wpływu na końcowy embedder pozostaje otwarta.
