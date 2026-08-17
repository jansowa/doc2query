# Pomiar: kalibracja guardraila M-03 i diagnostyka pięciu seedów (2026-08-16)

ADR: [`task04_m03_probe_convergence_guardrail_v1.md`](../decisions/task04_m03_probe_convergence_guardrail_v1.md).
Kontrakt: `configs/evaluation/task04_m03_probe_convergence_guardrail_v1.yaml`.
Artefakt: [`task04/m03_probe_convergence_v1/summary.json`](task04/m03_probe_convergence_v1/summary.json).
Etap w całości CPU, bez treningu i bez ładowania modeli. `final_tests_used=[]`.

Ten artefakt jest **diagnostyką**. Nie unieważnia zamkniętego TriviaQA
`dev_confirm`, nie zleca jego powtórzenia i niczego nie promuje
(`promotion_authorized=false` w każdym porównaniu).

## Co zmierzono

22 zakończone runy probe, po `corpus_recall_at_100` (sygnał zbieżności) i
`corpus_ndcg_at_10` (metryka decyzyjna), w trzech porównaniach:

| porównanie | runy | mediana sygnału | podłoga | poziom losowy | min sygnał |
|---|---|---|---|---|---|
| TriviaQA confirm S42–46 | 10 | 0,15750 | 0,078751 | 0,0007154 | 0,000452 |
| sweep budżetu 1024 | 6 | 0,07734 | 0,038671 | 0,0007154 | 0,001434 |
| sweep budżetu 2048 | 6 | 0,14528 | 0,072640 | 0,0007154 | 0,005753 |

W każdym porównaniu wiązała podłoga medianowa (połowa mediany), nigdy podłoga
losowa — ta pozostaje zabezpieczeniem na wypadek zapadnięcia się całego
porównania.

## Wynik kalibracji: guardrail oznaczył 4 z 22 runów

Cztery runy `non_converged` to dokładnie **cztery najniższe** runy po
`corpus_recall_at_100` w całym zbiorze:

| run | `corpus_recall_at_100` | `corpus_ndcg_at_10` | krotność poziomu losowego |
|---|---|---|---|
| D01B-TRIVIA-CONFIRM-W06-4.5B-S43 | 0,000452 | 0,0000190 | **0,63×** (poniżej losowego) |
| PROBE-BUDGET-1024-W06-S42 | 0,001434 | 0,000104 | 2,0× |
| PROBE-BUDGET-2048-HYBRID-S43 | 0,005753 | 0,001092 | 8,0× |
| PROBE-BUDGET-1024-HYBRID-S44 | 0,032927 | 0,009882 | 46× |

Pierwszy run **niezaznaczony** ma sygnał 0,074877, czyli **2,3× więcej** niż
najwyższy run zaznaczony. Separacja jest więc czysta, a nie progowa na styk. Dwa
zaznaczone runy (2× i 0,63× poziomu losowego) nie odzyskują praktycznie nic —
`ndcg@10` rzędu 1e-5–1e-4 potwierdza to niezależnie.

Kalibracja jest retrospektywna: progi dobrano tak, aby oddzielić runy widocznie
zapadnięte od jedynie słabych, z zapasem, po czym je zamrożono. Zastosowanie do
tych samych 22 runów nie jest nową decyzją.

## Diagnostyka pięciu seedów confirmu TriviaQA

Sparowane różnice per-seed `Hybrid − W06` dla `corpus_ndcg_at_10`, z surowych
podsumowań runów:

| seed | Hybrid | W06 | różnica |
|---|---|---|---|
| 42 | 0,10525077 | 0,07821245 | **+0,02703832** |
| 43 | 0,10239832 | 0,00001897 | **+0,10237935** (W06 niezbieżny) |
| 44 | 0,07376647 | 0,05958430 | **+0,01418217** |
| 45 | 0,06846219 | 0,04976049 | **+0,01870170** |
| 46 | 0,06662049 | 0,06953726 | **−0,00291677** |

| agregat | n | średnia | sd | 95% CI bootstrapu po seedach | test znakowy `p` | najmniejsze osiągalne `p` |
|---|---|---|---|---|---|---|
| bez filtra | 5 | **+0,03187695** | 0,04090017 | [+0,006494, +0,068004] | 0,15625 | 0,03125 |
| tylko zbieżne (bez S43) | 4 | **+0,01425135** | 0,01262354 | [+0,002488, +0,023824] | 0,31250 | 0,06250 |

**Werdykt reguły M-03: `insufficient_converged_seeds`.** Po odrzuceniu seeda 43
jako pary zostają cztery zbieżne pary, czyli poniżej wymaganego minimum pięciu.
Przy czterech parach dokładny jednostronny test znakowy ma najmniejsze osiągalne
`p = 0,0625 > 0,05`, więc reguła jest wtedy **konstrukcyjnie nierozstrzygalna** —
to ta sama arytmetyka, która uzasadnia minimum pięciu seedów w ADR.

Co z tego wynika, powiedziane wprost:

- nagłówek zamkniętego confirmu (`+0,0479`, 3 seedy, agregacja per-query przed
  bootstrapem zapytań) był podniesiony przez seed 43, w którym ramię kontrolne
  W06 **nie zbiegło**. Pięcioseedowa różnica bez filtra to `+0,0319`, a po
  odrzuceniu niezbieżnej pary `+0,0143` — nadal dodatnia, ale z dolną granicą CI
  `+0,0025`, czyli **poniżej** niezmiennego progu `+0,01`;
- confirm pozostaje zamknięty i ważny w swoim własnym, zamrożonym kontrakcie.
  Ten pomiar go nie zastępuje: liczy inną statystykę (bootstrap po seedach, nie
  po zapytaniach) na innej liczbie seedów;
- żadne ramię nie zostaje wypromowane ani zdegradowane. Hybrid pozostaje
  zachowany do `finalist-freeze review` dokładnie na tej podstawie, na jakiej był.

## Diagnostyka sweepu budżetu

| porównanie | odrzucone seedy | n zbieżnych | średnia zbieżnych | średnia bez filtra | status |
|---|---|---|---|---|---|
| budżet 1024 | 42, 44 | 1 | +0,03760434 | +0,02115850 | `insufficient_converged_seeds` |
| budżet 2048 | 43 | 2 | +0,01547214 | −0,00227787 | `insufficient_converged_seeds` |

Oba sweepy miały z założenia tylko trzy seedy, więc nie mogły spełnić reguły
M-03 i nie były do tego przeznaczone. Warto jednak zauważyć kierunek: przy
budżecie 2048 filtr **zmienia znak** średniej (z −0,0023 na +0,0155), co jest
najlepszą ilustracją, dlaczego wynik bez filtra jest w kontrakcie obowiązkowy do
raportowania.

## Granice tego pomiaru

- Guardrail działa na artefaktach końcowych, bo żaden z 22 runów nie ma
  pośredniej ewaluacji retrievalowej ani utrwalonej krzywej straty
  (`embedder_probe.py` zapisuje wyłącznie `losses[0]` i `losses[-1]`, a
  `checkpoint.pt` z pełną listą usuwa po sukcesie).
- Podłoga medianowa zależy od składu porównania; przy sześciu runach jest szumna.
  Reguła wymaga minimum pięciu par seedów, czyli minimum dziesięciu runów.
- Filtr zbieżności oparty na tym samym zbiorze ewaluacyjnym może obciążać
  porównanie. Sparowane odrzucanie, podłoga niezależna od ramienia i obowiązkowy
  wynik bez filtra ograniczają to, ale nie usuwają.
- Nie zmieniono żadnego zamrożonego progu, definicji budżetu ani configu
  Tasków 04–05. `final_tests_used=[]`.
