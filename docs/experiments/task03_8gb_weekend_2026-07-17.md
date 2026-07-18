# Task 03: kampania Bielik 1.5B na 8 GB, 2026-07-17/18

## Zakres i środowisko

- GPU: NVIDIA GeForce RTX 3060 Ti, 8 GB;
- model: `speakleash/Bielik-1.5B-v3`;
- revision: `4b25049621bf3952a1fc9314c89773102eda0333`;
- QLoRA: NF4, double quant, BF16, LoRA r=8/alpha=16, all-linear;
- max length: 512, batch 1, gradient accumulation 16;
- prompt: baseline B1, bez chat template i bez kontrolek style/focus;
- commit kampanii: `ecc6883`;
- wszystkie runy zakończyły się kodem 0, bez OOM i tracebacków.

## Wyniki

| Run | Przykłady | LR | Seed | Kroki | Eval loss | Ostatni train loss | Czas | Peak reserved VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| W01 | 10 000 | 1e-4 | 42 | 625 | 1.2640 | 1.1982 | 1h 38m | 1.771 GiB |
| W02 | 10 000 | 5e-5 | 42 | 625 | 1.2914 | 1.2218 | 1h 38m | 1.771 GiB |
| W03 | 10 000 | 2e-4 | 42 | 625 | **1.2505** | **1.1885** | 1h 37m | 1.771 GiB |
| W04 | 10 000 | 1e-4 | 43 | 625 | 1.2595 | 1.2214 | 1h 38m | 1.777 GiB |
| W05 | 50 000 | 1e-4 | 42 | 3125 | **1.1457** | **1.1723** | 8h 51m | 1.857 GiB |

Memory probe 512 zakończył się przy 1.508 GiB peak reserved VRAM. Nie wykonano
jeszcze probe 768/1024.

## Wnioski dopuszczone przez dane

1. QLoRA Bielika 1.5B jest stabilna i ma duży margines pamięci na tej karcie.
2. Na budżecie 10k LR `2e-4` ma najlepszy eval loss; LR `5e-5` jest najsłabszy.
3. Dwa runy LR `1e-4` dla seedów 42/43 mają zbliżony eval loss, ale nie jest to
   pełna replikacja wieloseedowa.
4. Zwiększenie danych do 50k poprawia eval loss względem wszystkich runów 10k.
5. Do dalszej ewaluacji należy zachować W03 jako najlepszy tani baseline 10k
   oraz W05 jako najlepszy dotąd model 50k.

Nie wolno na tej podstawie uznać W05 za finalnie najlepszy generator. Nie
wykonano jeszcze retrieval/grounding, lexical-copy, diversity ani probe
embeddera z Task 04.

## Panel testowy i ograniczenia

Po treningu wygenerowano deterministyczny panel 100 unikalnych dokumentów ze
splitu test. Żaden z tych dokumentów nie występuje jako pozytyw treningowy,
100/100 outputów ma poprawny jednoliniowy format, a 15/100 jest identycznych z
naturalną referencją. Kontrola ręczna pokazuje zarówno sensowne parafrazy, jak
i błędy wyboru intencji, dlatego panel nie zastępuje Task 04.

Pierwotne panele treningowe używały domyślnego dodatkowego tokenu specjalnego
podczas inferencji. Commit `3755b0d` wyrównał kodowanie promptu do SFT; za
miarodajny do oglądania należy uznać ponownie wygenerowany panel testowy.

Logi zawierały ostrzeżenie, że deterministyczne algorytmy CuBLAS wymagają
`CUBLAS_WORKSPACE_CONFIG`. Kolejne runy ustawiają `:4096:8`; zakończone runy
należy traktować jako seedowane, ale nie gwarantowane bitowo deterministyczne.

## Artefakty, które należy zachować

Artefakty są lokalne i celowo ignorowane przez Git. Minimum do zachowania:

- `runs/W03-1.5B-10K-LR2E4-S42/`;
- `runs/W05-1.5B-50K-8GB/`;
- `reports/W05-1.5B-50K-8GB.test-100.greedy.jsonl`;
- `reports/W05-1.5B-50K-8GB.test-100.greedy.md`;
- małe `sft_summary.json`, `run_manifest.json`, `resume_identity.json` i panele
  pozostałych runów, nawet jeżeli ich adaptery/checkpointy zostaną usunięte.

W03 fingerprint: `95736afcbd7de4e143865bc3355f869789538cee0cd94f46f0d380f97054ff56`.
W05 fingerprint: `017a26ebcf6c5811d5c84498d44881d943c919680e9eed482a649409dfc06b73`.

## Diagnostyczny scoring panelu W05

Panel 100 query–pasaż oceniono modelem
`sdadas/polish-reranker-base-ranknet` w revision
`a7c66d41a8097ca02e75616d0951c941d94ff6a1`, z limitem 512 tokenów i
surowym logitem (`Identity`). Wyniki nie są skalibrowanym
prawdopodobieństwem i nie wolno ich porównywać bezpośrednio z logitami
`sdadas/polish-reranker-roberta-v3`.

- minimum: `-0.8532`;
- mediana: `3.7889`;
- średnia: `3.8961`;
- maksimum: `7.9319`.

Pełne, rosnąco posortowane wyniki lokalne:

- `reports/W05-1.5B-50K-8GB.test-100.greedy.polish-reranker-base-ranknet.scored.jsonl`;
- `reports/W05-1.5B-50K-8GB.test-100.greedy.polish-reranker-base-ranknet.scored.md`;
- `reports/W05-1.5B-50K-8GB.test-100.greedy.polish-reranker-base-ranknet.report.json`.

To jest szybka diagnostyka trafności źródłowego pasażu, nie pełna ewaluacja
retrieval: nie obejmuje hard negative'ów, marginesów ani shadow judge.

Ten sam panel oceniono następnie primary judge
`sdadas/polish-reranker-roberta-v3` w revision
`e6471da541f4e7be33845b6d57248a8d8bde27e8`, z limitem 8192 tokenów:

- minimum: `7.2237`;
- mediana: `12.1936`;
- średnia: `12.0387`;
- maksimum: `16.3261`.

Rozrzut obu modeli jest podobny (odchylenie standardowe `1.9730` dla base i
`2.0690` dla v3), ale poziom bezwzględny logitów jest przesunięty i nie jest
wspólną skalą. Korelacja Pearsona score'ów wynosi `0.6154`, a Spearmana
rankingów `0.6298`. Dolne dziesiątki mają tylko trzy wspólne pary. V3 poprawił
pozycje oczywiście trafnych par: marchewka/Safeway z 2. na 72., SMS Export z
3. na 51., a CCIM z 10. na 33. pozycję rosnącą.

Pełne lokalne artefakty v3 i porównania:

- `reports/W05-1.5B-50K-8GB.test-100.greedy.polish-reranker-roberta-v3.scored.jsonl`;
- `reports/W05-1.5B-50K-8GB.test-100.greedy.polish-reranker-roberta-v3.scored.md`;
- `reports/W05-1.5B-50K-8GB.test-100.greedy.polish-reranker-roberta-v3.report.json`;
- `reports/W05-1.5B-50K-8GB.test-100.greedy.reranker-comparison.json`.
