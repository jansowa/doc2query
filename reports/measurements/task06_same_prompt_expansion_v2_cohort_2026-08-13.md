# Task 06 — zamrożenie kohorty same-prompt v2 (2026-08-13)

Etap 1 ADR
[`task06_same_prompt_expansion_v2_2026-08-13.md`](../decisions/task06_same_prompt_expansion_v2_2026-08-13.md):
quality-blind ID freeze i materializacja nowej kohorty. Bez GPU, bez ładowania
modeli, bez budowy par. `final_tests_used=[]`.

## Wykonanie

`scripts/freeze_task06_same_prompt_expansion_v2.py` z configiem
`configs/preferences/task06_same_prompt_expansion_v2.yaml`
(`sha256=9ee019945a82c934058b0347c0efb2b947ecfe88745d895bc2db4843d9cbb95d`),
~30 s CPU. Kohortę zamrożono dwukrotnie: po dopisaniu do configu
`generator.experiment_id` i `generation_batch_size` (wcześniej dekoracyjnego)
ponowne zamrożenie dało **identyczne** `ids_fingerprint` i `records_sha256`,
co potwierdza, że wybór zależy wyłącznie od bloku `cohort` i designu.
Artefakt: `artifacts/task06/same_prompt_expansion_v2` (`cohort.ids.json`,
`cohort.records.jsonl`, `cohort.manifest.json`).

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
`identity_sha256=be33ec5cf48b0de17b0a779ee8c2d80b7281324659d6809daf6dc10aa94037e0`
i kontraktem `task06-same-prompt-preference-expansion-v2`. Generacji nie
uruchomiono.

### Wznawialność (zweryfikowana testem, nie tylko lekturą kodu)

Oba kosztowne etapy prowadzą fsyncowane journale i wznawiają się z trwałego
prefiksu, ucinając wyłącznie niedokończoną ostatnią linię:

- generacja — `generations.jsonl.journal.jsonl`; wznowienie ma granulację
  jednego batcha (domyślnie 8 generacji, ~3 s pracy), a `evaluation_id` musi być
  dokładnym oczekiwanym prefiksem, inaczej run odmawia startu. Seedy zależą od
  indeksu absolutnego, nie od batcha, więc wynik nie zależy od miejsca przerwy;
- scoring — `scoring.journal.jsonl` + `scoring.resume.json`; ta sama granulacja
  8 rekordów (~2.6 s). Tożsamość resume obejmuje rekordy, sędziów, korpus i
  `experiment_id`, ale **nie** rozmiar batcha, więc batch scoringu można
  obniżyć w trakcie i wznowić bez utraty pracy;
- bramka różnorodności (0.7 s) sprząta staging przy przerwaniu i nie nadpisuje
  gotowego artefaktu; runner pomija ją, jeśli `diversity_gate/manifest.json`
  już istnieje.

Test `test_v2_generation_resumes_after_an_interruption_without_losing_work`
symuluje `KeyboardInterrupt` po 10 z 32 wierszy i sprawdza, że po ponownym
uruchomieniu dogenerowane zostaje dokładnie 22 wiersze, prefiks 10 wierszy jest
identyczny znak w znak, a trzecie uruchomienie nie wykonuje już żadnej pracy.

### Rozmiar batcha

`generation_batch_size` z configu i `scoring.max_batch_size` z designu były
wcześniej dekoracyjne (kod używał literału 8) — teraz są faktycznie
respektowane, z walidacją zakresu 1–8 dla generacji przed ładowaniem modelu.
Efektywny batch pozostaje 8, więc identity zakończonego runu v1 nie zmienia się.
Obniżenie batcha generacji zmienia `identity_sha256`, więc dla nieuruchomionej
kohorty v2 wymaga ponownego zamrożenia kohorty (~30 s), a dla runu już
rozpoczętego — amendmentu ADR, zgodnie z praktyką Task 05.

Pomiar v1 przy batchu 8: peak VRAM 3.43 GB allocated / 4.44 GB reserved na
karcie 8 GB, czyli ok. 55% budżetu. Wyczerpanie VRAM objawia się wyjątkiem
CUDA OOM, nie wyłączeniem komputera; nagłe zaniki zasilania przy dużym batchu
wskazują na skoki poboru mocy lub temperaturę, na co właściwą reakcją jest limit
mocy karty (`nvidia-smi -pl`), a nie zmiana kontraktu eksperymentu.

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
kohorty (usunięcie katalogu i ponowne uruchomienie skryptu, ~30 s). Skrypt
runnera można przerwać w dowolnym momencie i uruchomić ponownie tą samą
komendą — szczegóły w sekcji o wznawialności wyżej.

## Walidacja

Ruff, `mypy src` (114 plików), pełny pytest `514 passed` (11 nowych testów
kohorty, dyspozycji kontraktu v2, wznawiania po przerwaniu i walidacji batcha —
bez GPU i bez sieci).
