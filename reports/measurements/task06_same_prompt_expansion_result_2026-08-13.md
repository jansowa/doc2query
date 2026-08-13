# Task 06 — wynik expansion 500×8 i zastosowanie bramki różnorodności (2026-08-13)

Ten raport domyka dwa etapy: (1) faktyczny wynik zamrożonego runu
`same_prompt_expansion_v1`, który wcześniej był tylko „uruchomiony odłączony”,
oraz (2) pierwsze zastosowanie bramki różnorodności same-prompt zamrożonej ADR
[`task06_same_prompt_diversity_gate_v1.md`](../decisions/task06_same_prompt_diversity_gate_v1.md).

Par `chosen/rejected` nie zbudowano, audytu Groq nie uruchomiono, Task 07
pozostaje zamknięty. `final_tests_used=[]`.

## 1. Run expansion (GPU, zakończony)

Kontrakt: `task06-same-prompt-preference-expansion-v1`, config
`configs/preferences/task06_same_prompt_expansion_v1.yaml`, artefakty w
`artifacts/task06/same_prompt_expansion_v1`.

| pozycja | wartość |
|---|---|
| prompty (passage × jedna kontrolka D01) | 500 |
| odpowiedzi na ten sam prompt | 8 (temperatury 0.3–1.0, top_p 0.90/0.95, 8 seedów) |
| wygenerowane / oczekiwane | 4000 / 4000 (`same_prompt_generation_complete`) |
| ocenione primary/shadow/corpus | 4000 / 4000 (`status: measured`) |
| generacja | 1509.9 s, bf16, peak VRAM alloc. 3.43 GB / reserved 4.44 GB |
| scoring | 1307.6 s, batch 8, primary+shadow na `cuda`, BM25 8 workerów |
| `output_sha256` | `4ce7e1902077db9f660179816bb2f79d605b442a055f34e0c2d7915533470fe1` |
| resume | `resumed_generation_count=0` (run przeszedł bez przerwania) |

Kontrolki i format (4000 kandydatów): `form_accuracy` 0.96575 przy abstencji
0.02275, `intent_accuracy_excluding_unknown` 0.79253 (62 `unknown`),
`valid_rate` 1.0, `empty_rate` 0.0, `metacomment_rate` 0.0,
`multiple_query_rate` 0.0, mediana długości 4.5 słowa.

Sędziowie (surowe skale raportowane osobno, nigdy nie łączone): primary
`sdadas/polish-reranker-roberta-v3` margin mean 3.4053 (p05 −1.9280, p50
3.4147), shadow `BAAI/bge-reranker-v2-m3` margin mean 4.9825;
`rank_disagreement_rate` 0.12. Corpus round-trip: @1 0.20825, @5 0.40475,
@20 0.56675, @100 0.70175. Niezmierzone pozostają
`pairwise_embedding_cosine`, `semantic_cluster_count`, `human_answerability`
i `probe_embedder`.

## 2. Kolaps różnorodności przy identycznym promptcie

Grupowe metryki (500 grup) potwierdziły problem, który uzasadnił bramkę:

| metryka | mean | p25 | p50 | p75 |
|---|---|---|---|---|
| `duplicate_rate` | 0.39875 | 0.25 | 0.375 | 0.625 |
| `self_bleu` (bez deduplikacji) | 0.60298 | 0.42985 | 0.59513 | 0.77234 |
| `max_pairwise_lemma_jaccard` | 0.97520 | 1.0 | 1.0 | 1.0 |
| `distinct_1` | 0.32595 | 0.23037 | 0.31366 | 0.41095 |

Pilot 512×8 (różne prompty w slotach) miał `duplicate_rate` 0.0049 — różnica
jest konstrukcyjna, nie regresją runu: przy wspólnym promptcie jedynym źródłem
wariancji jest sampling.

## 3. Bramka różnorodności same-prompt (CPU, quality-blind)

Uruchomienie: `scripts/apply_task06_same_prompt_diversity_gate.py` z polityką
`configs/preferences/task06_same_prompt_diversity_gate_v1.yaml`
(`policy_sha256=ddbd9c8334e397611da4a639508689089b96ddeacf001cd7966dc5ec96d9f2c7`).
Artefakt: `artifacts/task06/same_prompt_expansion_v1/diversity_gate`
(`manifest.json`, `report.json`, `group_verdicts.jsonl` — 500 rekordów).
Czas: 0.7 s CPU.

Bramka przeczytała wyłącznie `generations.jsonl` (teksty, kontrolki, decoding,
provenance). Manifest zapisuje `judge_scores_read=false`,
`candidates_ranked=false`, `pairs_built=false`,
`model_loading_performed=false`, `final_tests_used=[]`, status
`diversity_gate_applied_not_paired`.

### Wynik

| pozycja | wartość |
|---|---|
| grupy wejściowe / kandydaci | 500 / 4000 |
| grupy `eligible` | **362 (72.4%)** |
| grupy odrzucone | **138 (27.6%)** |

Histogram przyczyn odrzucenia (grupa może mieć kilka):

| kod | grup |
|---|---|
| `duplicate_rate_above_threshold` (> 0.50) | 133 |
| `insufficient_effective_candidates` (< 3) | 72 |
| `self_bleu_above_threshold` (> 0.75 po deduplikacji) | 38 |
| `no_pairable_candidate_pair` (min Jaccard reprezentantów > 0.85) | 24 |

Żadna grupa nie odpadła na `prompt_mismatch` ani `unexpected_group_size`:
kontrakt „dokładnie ten sam prompt, osiem odpowiedzi” został utrzymany w
500/500 grupach.

### Rozkłady po deduplikacji near-duplicate (próg lemma Jaccard 0.90)

| metryka | wszystkie 500 | grupy `eligible` (362) |
|---|---|---|
| `effective_candidate_count` mean / p50 | 4.66 / 5 | 5.54 / 5 |
| `duplicate_rate` mean / p95 | 0.3987 / 0.75 | 0.2856 / 0.50 |
| `effective_self_bleu` mean / p95 | 0.4242 / 0.6873 | 0.3940 / 0.6004 |
| `min_pairwise_representative_query_jaccard` mean / p50 | 0.1725 / 0.125 | 0.1161 / 0.0909 |

Deduplikacja jest istotna interpretacyjnie: surowe `self_bleu` 0.603 spada do
0.424 po zredukowaniu każdego klastra near-duplicate do jednego reprezentanta.
Kolaps polega więc głównie na powtarzaniu tych samych odpowiedzi, a nie na tym,
że wszystkie odpowiedzi są parafrazami jednej treści — w 362 grupach zostaje co
najmniej trzy efektywnie różne kandydatury.

## 4. Konsekwencje i następny krok

- Zamrożona procedura Task 06 dopuszcza najwyżej **jedną** tentative preference
  na prompt, więc ta kohorta daje maksymalnie **362 pary** przed filtrami
  jakości (margin, format, answerability, konflikt sędziów). To **mniej niż
  wymagane 500 par** dla rozwojowej bramki ślepego audytu dual-LLM
  (i mniej niż 1000 par wymaganych przed finalnym DPO).
- Zakończonego runu expansion nie wolno nadpisywać ani powtarzać; jego kontrakt
  i artefakty pozostają zamrożone.
- Uzupełnienie deficytu wymaga **nowej generacji na GPU** i osobnej decyzji
  właściciela. Dwie dozwolone osie naprawy w ramach tego samego promptu:
  1. **większe K z deduplikacją** na tych samych 500 promptach (np. dodatkowe
     8 slotów o szerszym rozkładzie decodingu — min-p/top-p, wyższe
     temperatury, nowe seedy); zachowuje kohortę, ale nie zwiększa liczby
     promptów, więc podnosi jedynie odsetek grup przechodzących bramkę;
  2. **nowe grupy same-prompt** z legalnej puli 307309 pasaży (quality-blind
     wybór, nowy prospektywny ADR i nowy fingerprint kohorty); przy zmierzonym
     odsetku 72.4% do przekroczenia 500 par potrzeba około 190 nowych promptów
     (~1520 generacji, szacunkowo ~10 min generacji + ~8 min scoringu przy
     przepustowości tego runu).
- Progów bramki nie wolno zmieniać po zobaczeniu tego wyniku; zmiana wymaga
  nowego, prospektywnego ADR.
- Budowa par, audyt dual-LLM Groq oraz Task 07 pozostają zamknięte.
  `task07_training_authorized=false`, `final_tests_used=[]`.

## 5. Walidacja repozytorium

Ruff (`check` + `format`), `mypy src` (113 plików, bez błędów), pełny
`make typecheck` z 19 wcześniejszymi błędami w sześciu niezmienianych plikach
testowych oraz nowy zestaw 20 testów CPU bramki (`pytest -q`) — bez GPU i bez
sieci.
