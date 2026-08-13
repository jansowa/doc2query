# Task 06 — zamrożenie kohorty same-prompt v2 (2026-08-13)

Etap 1 ADR
[`task06_same_prompt_expansion_v2_2026-08-13.md`](../decisions/task06_same_prompt_expansion_v2_2026-08-13.md):
quality-blind ID freeze i materializacja nowej kohorty. Bez GPU, bez ładowania
modeli, bez budowy par. `final_tests_used=[]`.

## Wykonanie

`scripts/freeze_task06_same_prompt_expansion_v2.py` z configiem
`configs/preferences/task06_same_prompt_expansion_v2.yaml`
(`sha256=ed69b94cdee4dac6afa15dfb3550c490cf8b42e52109e2673090292b93c8e263`),
29.7 s CPU. Artefakt: `artifacts/task06/same_prompt_expansion_v2`
(`cohort.ids.json`, `cohort.records.jsonl`, `cohort.manifest.json`).

| pozycja | wartość |
|---|---|
| legalna pula par po wykluczeniach | 291463 |
| wykluczone klastry wspólnej selekcji 50k SFT | 49352 |
| wykluczone klastry wcześniejszych kohort Task 06 (smoke 32 + pilot 512) | 544 |
| nakładanie się wybranej kohorty z wykluczeniami | 0 |
| wybrane pasaże / unikalne klastry | 500 / 500 |
| `ids_fingerprint` | `194dd528c577965e3c4433a5e3cb72ce9adcd442c76831806c72f275c691fcf8` |
| `records_sha256` | `8e17c862ea9bbeca27621471f0b1c7706fdb45d1998c45f725bb6596cdff682f` |
| `cluster_ids_sha256` | `637563698ff6b7c0e4090584ff9619525de48a3a7a988741c685bfd037437f56` |
| pinned design | `e332793388a376b461c1469e0f8bbc012433e54d8781fb4adc992ee3100f6f23` (read-only, niezmieniony) |

Manifest zapisuje `quality_fields_used_for_selection=[]`,
`generation_started=false`, `scoring_started=false`,
`diversity_gate_applied=false`, `pairs_built=false`,
`model_loading_performed=false`.

Kolejność była zachowana: najpierw ID freeze (`cohort.ids.json`,
`status=ids_frozen_before_text_materialization`), potem weryfikacja
rozłączności, potem materializacja tekstu. Wybór użył wyłącznie
`pair_id`/`example_id`/`doc_id`/`cluster_id` i deterministycznego skrótu
`sha256(seed:cluster_id:pair_id)` przy seedzie `20260814`. Kohorta pilota 512
jest nadzbiorem 500 pasaży kohorty v1, więc v2 jest rozłączna także z v1.

## Gotowość etapu GPU

Ścieżka generacji v2 została zweryfikowana na prawdziwych artefaktach do
momentu ładowania modelu (zaślepione `load_tokenizer`/`load_generator`):
kontrakt, autoryzacja, zgodność `records_sha256` i `config_sha256` z zamrożonym
manifestem oraz 500 unikalnych klastrów przeszły, po czym zapisano
`generations.jsonl.identity.json` z
`identity_sha256=804e342554dc7efb4ba37a71cdd75a593a87a15ebb45fd5774efe570a28145a8`
i kontraktem `task06-same-prompt-preference-expansion-v2`. Generacji nie
uruchomiono.

Jedna komenda dla etapu 2 (po zwolnieniu GPU):

```bash
bash scripts/run_task06_same_prompt_expansion_v2.sh
```

Skrypt kolejno generuje 4000 kandydatów, ocenia je primary/shadow/corpus i
stosuje **niezmienioną** politykę bramki różnorodności
(`sha256=ddbd9c8334e397611da4a639508689089b96ddeacf001cd7966dc5ec96d9f2c7`).
Szacowany koszt na podstawie v1: ~25 min generacji + ~22 min scoringu, peak
VRAM ok. 3.4 GB; wyższe temperatury v2 mogą ten czas nieco wydłużyć.

Uwaga operacyjna: `config_sha256` jest przypięty w manifeście kohorty, więc
każda późniejsza edycja configu v2 wymaga świadomego ponownego zamrożenia
kohorty (usunięcie katalogu i ponowne uruchomienie skryptu, ~30 s).

## Walidacja

Ruff, `mypy src` (114 plików), pełny pytest `512 passed` (9 nowych testów
kohorty i dyspozycji kontraktu v2, bez GPU i bez sieci).
