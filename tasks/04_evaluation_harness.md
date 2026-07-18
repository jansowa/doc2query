# Task 04 — Kompletny harness ewaluacyjny

> [Centralny rejestr zadań i statusów](README.md). Każda zmiana statusu lub zakresu tego zadania musi aktualizować rejestr w tym samym commicie.

## Status

`IMPLEMENTED`

Centralny harness, zamrożone manifesty/ID, metryki i slice’y, raporty
HTML/Markdown, ślepy eksport A/B, bootstrap oraz zamrożona recepta probe
embeddera są zaimplementowane i przetestowane. Na lokalnej karcie 8 GB
rzeczywiście oceniono W03 i W05 w trybie deterministic oraz diverse na tym samym
zamrożonym panelu 100 rekordów z co najmniej 10 hard negative’ami; wykonano też
nieporównywalny 2-step smoke probe’a.

Pozostają pełne, porównywalne runy probe natural/copy/W03/W05, niezależny BGE
shadow judge, embeddingowe miary diversity, oceny ludzi dla co najmniej 300
przypadków, pełny test rank10/embedder i S00 prompting baseline. Bez tych
pomiarów bramka Fazy B i główny ranking generatorów nie są zamknięte. Rzeczywisty
zakres, fingerprinty, wyniki i 95% CI opisuje
`docs/experiments/task04_8gb_evaluation_2026-07-18.md`.

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
- trening na naturalnych query jako upper/control baseline;
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

## Kryteria akceptacji

- jeden command ocenia checkpoint i generuje komplet artefaktów;
- raport generatora i probe embeddera można odtworzyć z manifestu;
- pipeline odrzuca porównanie runów na różnych wersjach testu;
- główny ranking wariantów może używać wyniku probe embeddera, nie tylko rewardu.
