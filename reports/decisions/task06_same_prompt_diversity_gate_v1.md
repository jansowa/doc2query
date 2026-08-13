# Task 06 — bramka różnorodności same-prompt (ADR v1, 2026-08-13)

## Kontekst

Run `same_prompt_expansion_v1` zakończył się technicznym sukcesem: 500 promptów
D01 × 8 odpowiedzi na dokładnie ten sam prompt, 4000/4000 wygenerowanych i
4000/4000 ocenionych kandydatów (`artifacts/task06/same_prompt_expansion_v1`).
Grupowe metryki różnorodności pokazały jednak kolaps przy identycznym promptcie:

| metryka (500 grup) | mean | p25 | p50 | p75 |
|---|---|---|---|---|
| `duplicate_rate` | 0.39875 | 0.25 | 0.375 | 0.625 |
| `self_bleu` | 0.60298 | 0.42985 | 0.59513 | 0.77234 |
| `max_pairwise_lemma_jaccard` | 0.97520 | 1.0 | 1.0 | 1.0 |
| `distinct_1` | 0.32595 | 0.23037 | 0.31366 | 0.41095 |

Dla porównania pilot 512×8 (różne prompty per slot) miał `duplicate_rate`
średnio 0.0049. Różnica jest konstrukcyjna: przy wspólnym promptcie jedynym
źródłem wariancji jest sampling, a D01 jest silnie skalibrowany na kontrolki.

Pary DPO zbudowane z niemal identycznych kandydatów kodują szum sędziego, nie
różnicę jakości. `AGENTS.md` §9.2 i sekcja „Bramka różnorodności same-prompt”
w `tasks/06_candidate_scoring_and_preference_data.md` (rozszerzenie
specyfikacji z 2026-08-13, decyzja właściciela) wymagają zatem osobnego ADR
zamrażającego próg **przed odczytem jakichkolwiek par**.

## Decyzja

Zamraża się bramkę `task06-same-prompt-diversity-gate-v1`, zaimplementowaną w
`src/doc2query/preferences/diversity_gate.py` i uruchamianą przez
`scripts/apply_task06_same_prompt_diversity_gate.py`. Polityka progów jest
przypięta w `configs/preferences/task06_same_prompt_diversity_gate_v1.yaml`.

Bramka jest **quality-blind**: czyta wyłącznie `generations.jsonl` (teksty,
kontrolki, decoding, provenance). Nie otwiera `scoring/`, nie czyta score'ów
primary/shadow/corpus, nie rankuje kandydatów i nie emituje `chosen`/`rejected`.

### Definicje metryk (per grupa same-prompt)

1. `distinct_normalized_count` — liczba unikalnych kandydatów po normalizacji
   `task06_whitespace_casefold` (`normalize_task06_query`: strip, kolaps
   whitespace, casefold). `duplicate_rate = 1 - distinct/K`.
2. `effective_candidate_count` — liczba klastrów near-duplicate: union-find po
   parach, które są równe po normalizacji **lub** mają Jaccard treściowych
   lemmatów (`simple_pl:v1:nfkc:stopwords-v1`) `>= 0.90`. Reprezentantem
   klastra jest kandydat o najmniejszym `candidate_index` (wybór indeksowy, nie
   jakościowy).
3. `effective_self_bleu` — self-BLEU (n=1..4, wygładzone, brevity penalty) na
   reprezentantach klastrów, tą samą funkcją co metryki runu.
4. `min_pairwise_representative_query_jaccard` — minimalny Jaccard tokenów
   `\w+` casefold między reprezentantami, dokładnie tą definicją, którą stosuje
   frozen `SelectionPolicy.max_normalized_query_jaccard` w
   `preferences/build.py`.

### Progi (zamrożone)

| kryterium | próg | uzasadnienie |
|---|---|---|
| `require_exact_same_prompt` | `true` | kontrakt DPO: identyczny prompt w grupie (identyczny `prompt` i `prompt_sha256`) |
| `expected_candidates_per_group` | `8` | kontrakt zamrożonego runu expansion |
| `min_effective_candidates` | `3` | wymóg wprost ze specyfikacji: `chosen`, `rejected` i co najmniej jeden zapas na odrzucenia dalszych filtrów jakości |
| `max_duplicate_rate` | `0.50` | grupa, w której ponad połowa slotów to dokładne powtórzenia, jest kolapsem decodingu, nie rozkładem odpowiedzi |
| `max_effective_self_bleu` | `0.75` | po deduplikacji reprezentanci muszą różnić się n-gramowo; 0.75 to granica, powyżej której różnica jest głównie permutacją tych samych n-gramów |
| `max_min_pairwise_query_jaccard` | `0.85` | istnieje co najmniej jedna para, którą przepuściłaby już zamrożona `SelectionPolicy` (`max_normalized_query_jaccard = 0.85`); próg nie jest nową liczbą, lecz spójnością z istniejącym kontraktem parowania |

Grupa jest `eligible` wtedy i tylko wtedy, gdy spełnia wszystkie kryteria.
Kody odrzucenia: `prompt_mismatch`, `unexpected_group_size`,
`insufficient_effective_candidates`, `duplicate_rate_above_threshold`,
`self_bleu_above_threshold`, `no_pairable_candidate_pair`.

### Co było widoczne przed zamrożeniem progów

Widoczne były wyłącznie **grupowe** rozkłady różnorodności z
`d01_controlled/scoring/summary.json` (tabela wyżej) — to one uzasadniły
istnienie bramki i są cytowane w specyfikacji. Nie użyto: score'ów
primary/shadow/corpus, rankingów kandydatów, marginesów, żadnej pary
`chosen/rejected`, ani liczby grup przechodzących bramkę przy jakimkolwiek
kandydacie progu. Progi wynikają z wymagań kontraktu (min. 3 efektywne
kandydatury, spójność z `SelectionPolicy`) i z okrągłych granic (0.50, 0.75),
nie z docelowego poziomu przejść.

## Konsekwencje

- Odsetek grup odrzuconych jest raportowany jawnie, razem z histogramem
  przyczyn; nie wolno go ukrywać ani obchodzić.
- Progów nie wolno zmieniać po zobaczeniu wyniku bramki. Zmiana wymaga nowego,
  prospektywnego ADR z własnym uzasadnieniem.
- Bramka nie autoryzuje budowy par. Materializacja tentative preferences i
  audyt dual-LLM Groq pozostają zamknięte do osobnej decyzji właściciela,
  podjętej po odczytaniu raportu bramki.
- Dozwolone osie naprawy w ramach *tego samego* promptu (rozszerzenie
  specyfikacji 2026-08-13): szerszy rozkład decodingu (temperatury, min-p/top-p,
  seedy) oraz większe K z deduplikacją. Każda z nich wymaga nowej generacji na
  GPU i własnej autoryzacji; kontrakt zakończonego runu expansion pozostaje
  niezmieniony i nie wolno go nadpisywać.
- Prospektywna ablacja teachera (lokalny, przypięty `Qwen3.6-27B` Q4) jako
  dodatkowe źródło `chosen` wymaga własnego ADR i osobnego provenance; par z
  kandydatem teachera nie może audytować sędzia tożsamy z teacherem.

`final_tests_used=[]`. Bramka nie dotyka splitów dev/test, nie ładuje modeli i
nie uruchamia GPU.
