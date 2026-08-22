# Task 04 / M-03 — tryb obserwacyjny detektora zapadnięć (amendment, 2026-08-22)

Status: **zamrożony prospektywnie, przed uruchomieniem pierwszego runu serii nocnej.**
Amendment do [`task04_m03_in_run_collapse_detection_v1.md`](task04_m03_in_run_collapse_detection_v1.md).
Config: `configs/evaluation/task04_m03_in_run_collapse_detection_shadow_v1.yaml`.
Autoryzacja właściciela: 2026-08-22 (okno nocne 8–10 h, GPU wolne).

## 1. Po co

Pomiar
[`task04_m03_in_run_collapse_detection_2026-08-21.md`](../measurements/task04_m03_in_run_collapse_detection_2026-08-21.md)
zdał wszystkie pięć prerejestrowanych kryteriów, ale ma jedną jawnie słabą
liczbę: zero fałszywych alarmów na **0/3 runach** (górna granica 95% CI ≈ 0,63).
Swoistość detektora nie jest więc wykazana, a bez niej detektora nie wolno
uznać za bezpieczny w porównaniach ramion.

Trybu przerywającego **nie da się** do tego użyć: przerwany przebieg nigdy nie
dostaje werdyktu guardraila M-03, więc nie istnieje prawda odniesienia, wobec
której liczy się fałszywy alarm. Potrzebny jest wariant, w którym run zawsze
kończy się normalnie.

## 2. Decyzja

Kontrakt `task04-m03-in-run-collapse-detection-v1` zyskuje jedno pole:

```yaml
mode: abort_and_reseed | shadow_observe_only   # domyślnie abort_and_reseed
```

W trybie `shadow_observe_only` kontrole pośrednie są liczone i zapisywane
dokładnie tak samo, ale **run nigdy nie jest przerywany ani reseedowany**;
wykrycie ląduje w dzienniku jako obserwacja.

Czego amendment **nie** zmienia: progów, reguł, wymogu dwóch kolejnych trafień,
definicji zbioru pośredniego, polityki reseedu w trybie abort, zapisu
`loss_based_guardrail_permitted: false` ani żadnego artefaktu. Domyślną
wartością pozostaje `abort_and_reseed`, a bez flagi `--collapse-detection-config`
probe nadal zachowuje się jak przed 2026-08-21.

## 3. Seria nocna (zamrożona przed uruchomieniem)

Jedno ramię (baseline W06), wejścia, hiperparametry i budżet **identyczne** z
serią wariancji S47–S51 i serią walidacyjną; różni je wyłącznie seed i tryb.

- seedy **52–71** (20 nowych, nieużywanych; bez kolizji z 42–51 i z reseedami
  1000+);
- tryb `shadow_observe_only`, więc każdy run kończy się i dostaje werdykt M-03;
- kolejka `configs/probe_inrun_collapse_shadow_queue_2026-08-22.tsv`,
  wznawialna, checkpoint co 64 kroki, encode batch 8;
- katalog `runs/task04_probe_shadow_collapse_v1/`; po **ukończonym** runie
  kasowany jest wyłącznie `corpus_embedding_cache` (odtwarzalny z modelu i
  korpusu), żeby 20 runów zmieściło się na dysku;
- żadnego crona, żadnego wyłączania maszyny, żadnego dotykania `artifacts/task06/`
  ani katalogów `runs/task04_probe_variance_v1/` i
  `runs/task04_probe_inrun_collapse_v1/`.

Kolejka może nie dobiec do końca (okno 8–10 h, run ~25 min). To jest w porządku:
analiza liczy się na tylu runach, ile faktycznie się ukończyło, a liczba runów
jest raportowana.

## 4. Co będzie raportowane (zamrożone przed odczytem)

1. **Macierz pomyłek detektora wobec zamrożonego guardraila M-03** liczonego
   post hoc na tej serii: prawda odniesienia = werdykt `converged`, predykcja =
   czy reguła trafiłaby dwa razy z rzędu. Osobno dla reguły retrievalowej,
   osobno dla kierunku straty, osobno dla ich alternatywy (czyli faktycznej
   reguły przerwania).
2. **Odsetek zapadnięć** na 20 niezależnych seedach z 95% CI (Clopper-Pearson),
   jako aktualizacja szacunku 5/27 = 18,5% — tym razem przy niezmienionym
   budżecie i jednym ramieniu.
3. **Nieselekcjonowany rozkład `corpus_ndcg_at_10`** ramienia W06: średnia, sd i
   wynikające półszerokości 95% CI przy n = 3, 5, 10, 20, jako podstawa
   planowania mocy przyszłych porównań wobec niezmienionego progu `+0,01`.
   Rozkład jest nieselekcjonowany właśnie dlatego, że tryb obserwacyjny nie
   reseeduje.
4. Najwcześniejszy krok, na którym reguła trafiłaby, dla każdego zapadniętego
   runu — czyli realna oszczędność czasu, gdyby tryb był przerywający.

## 5. Czego ta seria nie robi

- **Nie kalibruje żadnego progu.** Jeżeli macierz pomyłek wyjdzie zła, jest to
  wynik negatywny i tak zostanie zaraportowany; zmiana progów wymaga nowego,
  prospektywnego ADR.
- **Nie jest porównaniem ramion** i nie promuje ani nie degraduje niczego:
  wszystkie runy to jedno ramię W06.
- **Nie unieważnia ani nie przelicza** żadnego zamkniętego pomiaru; artefakty
  M-03, confirmu TriviaQA, sweepu budżetu, serii wariancji i serii walidacyjnej
  pozostają nietknięte.
- Runy obserwacyjne **nie są** porównywalne z runami w trybie abort jako
  składniki jednego porównania (te drugie mają warunkowany rozkład seedów);
  jako oszacowanie własności ramienia nadają się natomiast lepiej.
- **Nie autoryzuje treningu Task 07** ani kampanii Task 09.

`final_tests_used=[]`. `task07_training_authorized=false`.
