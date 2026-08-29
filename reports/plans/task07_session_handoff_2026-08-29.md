# Prompt przekazania sesji — Task 07 / pipeline wad (2026-08-29)

Do wklejenia jako pierwsza wiadomość nowej sesji. Stan zweryfikowany w chwili
spisania; nowa sesja ma sprawdzić artefakty, nie wierzyć na słowo.

---

Przeczytaj najpierw `AGENTS.md` i `tasks/README.md` (wiersz Task 07), potem:

- `reports/measurements/task07_arms_dev_result_2026-08-29.md` — sześć ramion
  DPO/continued/weighted × bottom/near_miss wytrenowanych; metryki dev są
  pomocnicze, wynik właściwy (probe embedder) NIE jest policzony.
- `reports/decisions/task06_defect_pair_pipeline_v1.md` — zamrożony prospektywny
  ADR pipeline'u par z wadami nazwanymi z konstrukcji (klasy, progi, populacje
  negatywów, klasa pary lexical_contrast, bramki w tym audyt anty-skrótowy
  AUC 0,80). Projekt i eksploracje:
  `reports/plans/task07_defect_pair_pipeline_design_2026-08-29.md`.
- `reports/measurements/task07_pair_contrast_diagnostic_2026-08-28.md` i
  `reports/measurements/task07/lemma_overlap_v1/` — dlaczego obecne pary są
  łatwe (75,3% rejected nie na temat; chosen mediana pokrycia lematycznego
  0,667) i po co nowa kohorta.

Stan operacyjny w chwili przekazania:

1. **Serwer inferencji (druga maszyna)**: właściciel uruchamia
   `scripts/run_defect_pipeline.sh <BASE_URL> <MODEL> <API_KEY> [CONCURRENCY]`
   po `git pull` i rozpakowaniu `defect_pipeline_input.tar.gz` w katalogu repo
   (2 730 grup, ~50-65k wywołań, wznawialne po journalu). Wynik do przeniesienia
   z powrotem: `artifacts/task06/defect_pipeline_v1/verdicts/`.
2. **GPU lokalnie**: kolejka `scripts/queue_task07_generator_eval.sh`
   (log `logs/task07_generator_eval.log`) liczy intrinsics generatora na
   subsecie ROZWOJOWYM `dev_intrinsic_rank10` dla 7 punktów (start + 6 ramion),
   wyniki w `runs/task07_generator_eval_v1/<punkt>/result.json`; restart
   bezpieczny (gotowe punkty pomijane).
3. **Niewypchnięte commity** na master (kilkanaście) — push wymaga wyłączonego
   VPN; spróbuj `git -c http.version=HTTP/1.1 push origin master` i jawnie
   raportuj, co niewypchnięte.

Zadania po powrocie verdictów z serwera (w tej kolejności):

1. Napisz **lokalne składanie par** wg ADR §2–§6: filtry deterministyczne
   (LCS ≥5 → copy_phrasing; Jaccard vs chosen ≤0,6; długości; forma), wymogi
   answerability per klasa, jednomyślność potwierdzeń 2/2, klasa
   `lexical_contrast` na lematach (stanza pl, grupa zależności `nlp`; progi
   0,6/0,4), metadane `defect_class`/`negative_population`/`pair_class`,
   limit ≤1 pary na (grupę, klasę). Potem bramki ADR §7: raport pass-rate,
   audyt anty-skrótowy (AUC >0,80 blokuje klasę), ślepy spot-check ≥30 par dla
   właściciela (`scripts/task07_owner_spot_check.py` jako wzorzec ślepości).
2. **Trening na nowej kohorcie wymaga OSOBNEJ autoryzacji właściciela** —
   autoryzacja z 2026-08-28 pokrywa tylko kohorty v3 (bottom + near_miss).
   Łańcuch: handoff (`scripts/build_task07_handoff_v3.py` ze wskazaniem par) →
   token lengths → plan → precompute → `scripts/run_task07_arms.sh`-podobny
   runner; wszystko istnieje i jest sparametryzowane.
3. **Ewaluacja probe** sześciu (docelowo dziewięciu) adapterów — właściwy wynik
   Task 07; przejrzyj kontrakty `training/comparison_preflight.py`
   (protokół, outcome evidence) zanim cokolwiek uruchomisz.

Twarde zasady bez zmian: zbiory testowe zamknięte (`final_tests_used=[]`),
kohorty v4-v11 zamknięte, `source_en_score >= 23.50` nietykalne, statusy tylko
z artefaktami, wyniki niezmierzone nie istnieją, commity po polsku bez wzmianek
o asystencie i bez Co-Authored-By, walidacja przed commitem: ruff + mypy +
pytest (856 zielonych w chwili przekazania), nowe raporty whitelistowane w
`.gitignore`. GPU przez `.venv-gpu/bin/python` (główne `.venv` ma torch
CPU-only). Groq wymaga nagłówka User-Agent i `reasoning_effort: "none"`
(oba już w kodzie); serwer właściciela używa `chat_template_kwargs`.
Przed pushem daj znać właścicielowi (VPN blokuje GitHub i Groq).

---
