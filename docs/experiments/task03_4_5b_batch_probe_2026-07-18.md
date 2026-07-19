# Task 03: memory/throughput probe Bielika 4.5B na 8 GB

## Stałe warunki

- model: `speakleash/Bielik-4.5B-v3.0-Instruct`;
- QLoRA NF4, double quant, LoRA r=8/alpha=16, all-linear;
- długość 512, efektywny batch 16, 3 optimizer steps;
- 128 przykładów train i 32 eval;
- ta sama karta RTX 3060 Ti 8 GB, seed i kod;
- peak mierzony przez PyTorch; throughput pochodzi z `Trainer`.

## Wyniki

| Microbatch | Gradient accumulation | Efektywny batch | Przykłady/s | Peak reserved | Wynik |
|---:|---:|---:|---:|---:|---|
| 2 | 8 | 16 | 1,177 | 4,06 GB | OK |
| 4 | 4 | 16 | 1,493 | 4,85 GB | OK |
| 8 | 2 | 16 | 1,546 | 6,43 GB | OK |
| 16 | 1 | 16 | — | co najmniej 7,52 GB przed nieudaną alokacją | OOM przy dodatkowej alokacji 566 MiB |

BS8 jest najszybszym bezpiecznym wariantem przy zachowaniu identycznego
efektywnego batcha. Zysk względem BS4 wynosi około 3,5%, więc krzywa throughput
jest już bliska plateau. BS16 nie daje użytecznej konfiguracji na tej karcie.
Do pełnego W06 wybrano BS8/L512 z gradient accumulation 2.

Smoke służy do wyboru konfiguracji sprzętowej, nie do porównania jakości:
trzykrokowe loss/eval loss nie są miarodajnym wynikiem modelu.
