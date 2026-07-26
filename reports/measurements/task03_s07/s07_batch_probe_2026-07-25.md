# S07 plT5 — wybór microbatcha na RTX 3060 Ti 8 GB

Data: 2026-07-25

Status: `MEASURED / BS16 SELECTED`

Zakres danych: wyłącznie frozen train/dev; `final_tests_used=[]`,
`dev_confirm_opened=false`, `p06_opened=false`.

## Reguła wyboru

Próby zachowują effective batch 16, seed 42, source/target 512/64,
gradient checkpointing i pełny fine-tuning przypiętego
`allegro/plt5-base@56379680948ce8b42d3d48df86569cfc210d3060`.
Zmieniają wyłącznie microbatch i liczbę kroków akumulacji. Wariant większy
jest odrzucany przy OOM, niefinitywnym lossie albo gdy podwojenie microbatcha
daje mniej niż 10% przyspieszenia; przy małym zysku preferowany jest wariant
mniejszy.

## Wynik krótkiego sweepu

Każdy wariant wykonał 6 optimizer steps, czyli 96 przykładów.

| Microbatch | Grad accum | Effective batch | examples/s | Zysk vs poprzedni | Peak reserved GiB | Eval loss | Stan |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 16 | 16 | 3,098 | — | 2,717 | 9,204 | OK |
| 2 | 8 | 16 | 5,685 | +83,5% | 2,793 | 8,789 | OK |
| 4 | 4 | 16 | 9,903 | +74,2% | 2,803 | 9,253 | OK |
| 8 | 2 | 16 | 14,855 | +50,0% | 2,816 | 7,834 | OK |
| 16 | 1 | 16 | 18,394 | +23,8% | 2,809 | 7,765 | OK |

Surowy train loss raportowany przez bieżący stos Transformers skaluje się z
liczbą kroków akumulacji, dlatego nie jest porównywany liczbowo pomiędzy
ramionami. Wszystkie przebiegi miały skończone train/eval loss i gradienty.

## Potwierdzenie BS8 vs BS16

Oba warianty powtórzono przez 30 optimizer steps, po 480 przykładów, aby
zmniejszyć wpływ rozgrzewki i końcowej ewaluacji.

| Microbatch | Grad accum | examples/s | Peak allocated/reserved GiB | Eval loss | Stan |
|---:|---:|---:|---:|---:|---|
| 8 | 2 | 16,468 | 2,746 / 2,818 | 7,193 | OK |
| 16 | 1 | 28,684 | 2,784 / 3,256 | 6,747 | OK |

BS16 jest o 74,2% szybszy od BS8 w próbie potwierdzającej i wykorzystuje
39,7% nominalnych 8 GiB według peak reserved Torch. Nie wystąpił OOM, NaN ani
Inf. Microbatcha nie zwiększa się dalej, ponieważ 16 jest prospektywnie
zamrożonym effective batchem wspólnym z W05.

## Decyzja i wznowienie

Wybrano `per_device_train_batch_size=16` oraz
`gradient_accumulation_steps=1`. Effective batch, liczba par, liczba optimizer
steps, scheduler i pozostały budżet S07 nie zmieniają się. Gradient
checkpointing pozostaje włączony dla zapasu stabilności.

Po wyborze ponowiono także oficjalny dwukrokowy memory gate na pełnym
kontrakcie 50k/1k. Odtworzył fingerprint W05
`017a26ebcf6c5811d5c84498d44881d943c919680e9eed482a649409dfc06b73`,
zakończył się kodem 0 i zmierzył peak allocated/reserved
`2,699/2,805 GiB`. Wcześniejszy gate BS1 zachowano w sąsiednim katalogu z
sufiksem `-bs1`.

Przerwany techniczny run BS1 nie może być wznowiony po zmianie trajectory
identity. Jego kompletny `checkpoint-50` zachowano bez modyfikacji w
`runs/task03_s07/interrupted/S07-PLT5-BASE-50K-S42-bs1-step50`, a log w
`logs/task03_s07/train-bs1-step50.log`. Właściwy run BS16 musi rozpocząć się
od kroku 0 w standardowym katalogu S07.

Pełny S07, Harness v1.1 i downstream probe nadal są niewykonane. Ta decyzja
nie otwiera P-06, `dev_confirm` ani finalnych testów.
