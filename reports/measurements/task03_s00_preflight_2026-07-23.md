# Audyt i preflight S00 — 2026-07-23

Status: `TOOLING_READY_RUN_UNEXECUTED`

Commit wejściowy: `135181f9a00168e39323adebbd753a07aa1de1de`

Finalne dane: nieotwarte; `final_tests_used=[]`

## Audyt stanu wejściowego

Istniał config `configs/experiments/s00_prompting.yaml`, lecz nie istniał
runner ani artefakt S00. Config wskazywał odwrócony dev i B0. Ogólny
`scripts/generate_panel.py` wybierał tylko 100 rekordów, obsługiwał zero-shot
bez demonstracji, nie wiązał wejścia z frozen manifestem i nie miał dziennika
wznowienia. Nie spełniało to definicji S00 (5k, zero/few-shot, greedy/sampling,
Harness v1.1).

## Wykonany preflight CPU

- zweryfikowano niezmieniony manifest `task04-v1`; `dev_intrinsic_rank10` ma
  6 598 rekordów i fingerprint `235d9b81…ffab6`;
- prospektywna kohorta `dev_s00_5000` ma 5 000 rekordów, fingerprint
  `93313d5a9a0d46d5bdbe19b66f3ba749452fb36d757b0654b744eaef87ddb284`
  i hash listy ID
  `0eabda2851bb70e55d4bc9a7b2c77866e9ff8b0f68e04a1b66c315bd70c40290`;
- wybrano 6 demonstracji: 3 `full_question` i 3 `keyword_query`;
- przecięcie demonstracji z kohortą wynosi 0 po `example_id` i 0 po
  pozytywnym `doc_id`;
- kontrakt ma fingerprint
  `77438ecad9a7d2722b9afdab6c54653f387af142f53a0c0c8ca0f4a89a34ecbc`;
- manifest i listy robocze zapisano pod ignorowanym przez Git
  `runs/S00-prompting-v1/cohort/`; są odtwarzalne z kontraktu;
- primary/shadow config są dostępne; pełny indeks
  `data/processed/v1/evaluation/corpus-bm25-v1` nie istnieje i zostanie
  zbudowany jako pierwszy etap właściwego runnera.

## Zgodność metodologiczna

S00 korzysta wyłącznie z zamrożonego dev i nie akceptuje nazw zawierających
`test_native`, `test_translated`, `test_embedder` ani `final_test`. Pochodny
manifest zachowuje fingerprint rodzica i nie modyfikuje Task 01/Task 04.
Candidate-pool oraz corpus retrieval korzystają z istniejącego Harness v1.1.

Kontrakt P-04 pozostaje niezmieniony. S00 jest intrinsic baseline'em na dev,
więc nie może sam spełnić głównej metryki P-04 ani autoryzować `dev_confirm`.
Włączenie S00 do probe wymagałoby późniejszego, prospektywnego wyrównania
tokenów, par, unikalnych pasaży oraz K query/passage.

## Tooling i testy

Dodano kontrakt S00, fail-closed materializator kohorty, wersjonowany prompt,
stratyfikowane demonstracje, deterministyczne seedy per rekord/ramię,
wznawialny SQLite journal, atomowe JSONL, postęp z ETA i runner etapowy.

Krótkie testy CPU: 20 testów przeszło (`test_s00_prompting.py`,
`test_config.py`, `test_statistical_contract.py`). Sprawdzają między innymi
blokadę finalnego markera, budżet promptu i wznowienie po kontrolowanym
przerwaniu. Właściwe 50 tys. completionów i scoring nie zostały uruchomione;
nie ma wyniku S00.
