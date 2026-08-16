# Task 06 — autoryzacja scoringu GPU kohort tokenowych (amendment, 2026-08-16)

## Kontekst

Dwa ADR-y z okna tokenowego celowo odroczyły etap GPU, bo maszyna była zajęta
kolejką bezobsługową:

- [`task06_reward_validation_corpus_v1.md`](task06_reward_validation_corpus_v1.md)
  — predykcja **P8** (`ungrounded` i `too_general` wobec primary/shadow/corpus)
  zapisana prospektywnie, `gpu_measurement_authorized: false`;
- [`task06_claude_teacher_ablation_v1.md`](task06_claude_teacher_ablation_v1.md)
  — kryteria porównania teacher vs student zapisane prospektywnie,
  `scoring_authorized: false`.

Kolejka bezobsługowa zakończyła się 2026-08-16 (25/25 zadań), a właściciel
komendą z 2026-08-16 udostępnił GPU z jednym warunkiem: **skrypty muszą być
sensownie wznawialne, bez utraty większej ilości pracy**.

## Decyzja

Autoryzuję **wyłącznie scoring** obu kohort tokenowych zamrożonym kontraktem
Task 06 (`task06_candidate_execution_design_v1.yaml`,
`sha256=e332793388a376b461c1469e0f8bbc012433e54d8781fb4adc992ee3100f6f23`):
primary jako builder, shadow jako niezależna kontrola, corpus round-trip na
zamrożonym indeksie BM25, batch 8.

Żadna z zapisanych wcześniej predykcji ani żaden próg nie ulega zmianie. To
amendment o **dostępie do sprzętu**, nie o metodzie: P8 i kryteria ablacji były
zamrożone, zanim GPU się zwolniło, i są mierzone dokładnie w zapisanym brzmieniu.

### Wznawialność (warunek właściciela)

Etap scoringu korzysta z **istniejącej, sprawdzonej** ścieżki
`evaluate_intrinsic_records`, która utrzymuje `scoring.journal.jsonl` oraz
`scoring.resume.json` i wznawia się od dokładnego prefiksu fsyncowanego
dziennika; niezgodność tożsamości wejścia jest błędem, a nie cichym nadpisaniem.
Do tego:

- materializacja wejść scoringu jest deterministyczna i idempotentna: ponowne
  uruchomienie daje bajtowo ten sam plik albo kończy się błędem, jeśli
  istniejący plik ma inną treść;
- runner bierze `flock` na katalogu wyjściowym, więc dwa procesy nie mogą
  ścigać się o ten sam dziennik;
- maksymalna strata przy zabiciu procesu to jeden batch (8 rekordów);
- kohorty są rozłączne, więc scoring teachera i korpusu nagrody można wznawiać
  niezależnie.

### Zakres

| etap | status |
|---|---|
| scoring korpusu walidacyjnego nagrody (1440 rekordów) | **autoryzowany** |
| scoring kohorty teachera (9600 rekordów) | **autoryzowany** |
| pomiar P8 i prerejestrowanych kryteriów ablacji | **autoryzowany** |
| budowa par `chosen/rejected` | nieautoryzowana (wymaga własnego ADR) |
| audyt dual-LLM (Groq) | nieautoryzowany |
| trening czegokolwiek na tych danych | nieautoryzowany |
| testy finalne | zamknięte, `final_tests_used=[]` |

Scoring korpusu walidacyjnego nagrody jest **diagnostyką komponentów**, a nie
kalibracją: jego wynik nie może zmienić progów bramki różnorodności ani progu
`source_en_score >= 23.50`.

Kandydatów teachera ocenia primary/shadow — modele lokalne, rozłączne z autorem
kandydatów, więc zakaz self-preference bias jest zachowany. Audyt dual-LLM par
(gdy powstanie) nadal nie może używać modelu tożsamego z teacherem.

## Artefakty

- wejścia: `artifacts/task06/<kohorta>/scoring_inputs/generations.jsonl`
  (materializowane w zamrożonym schemacie rekordów generacji);
- wyniki: `artifacts/task06/<kohorta>/scoring/` (`per_generation.jsonl`,
  `summary.json`, dziennik i plik wznowienia).
