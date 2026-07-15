# ADR 0001: stos projektu i odtwarzalność

- Status: zaakceptowano
- Data: 2026-07-15
- Zakres: task 00

## Kontekst

Projekt ma działać na CPU w CI, a później obsłużyć QLoRA na pojedynczej karcie
16 GB. Musi zachować stabilne kontrakty CLI i konfiguracji bez pobierania modeli
w testach. Środowisko badawcze powinno być odtwarzalne, a ciężkie opcje
inference nie mogą obciążać instalacji bazowej.

## Decyzja

Używamy Pythona 3.11 i `uv` z commitowanym lockfile. Python 3.11 zapewnia szeroką
zgodność stosu PyTorch/Hugging Face. Pakiet ma układ `src`, CLI buduje Typer, a
hierarchiczne YAML-e obsługują Hydra/OmegaConf. Granica wejściowa konfiguracji
jest dodatkowo walidowana ścisłymi modelami Pydantic, dzięki czemu błędy są
wykrywane przed runem.

PyTorch w lockfile wskazuje oficjalny indeks CPU. Jest to deterministyczna i
lżejsza baza dla CI; środowisko GPU nadpisuje wyłącznie wariant koła PyTorch
zgodnie z lokalnym CUDA. Do śledzenia wybrano Weights & Biases ze względu na
prosty tryb offline, ale niezależny lokalny `run_manifest.json` jest obowiązkowy
i nie importuje SDK trackera.

Ruff odpowiada za lint i format, mypy pracuje w trybie strict, a pytest za testy.
Duże dodatki (`flash-attn`, vLLM, ONNX Runtime, Optimum, OpenVINO) są extras i nie
wchodzą do obowiązkowej instalacji CPU.

## Alternatywy

- Poetry/pip-tools: poprawne, lecz `uv` daje jeden szybki resolver, zarządzanie
  Pythonem i grupami zależności.
- Click/argparse: mniej wygodne dla zagnieżdżonego, typowanego CLI.
- czysta Hydra bez Pydantic: kompozycja jest dobra, ale komunikaty o naruszeniu
  kontraktu runu byłyby mniej jednoznaczne.
- MLflow: mocny lokalny backend, ale znacznie cięższa obowiązkowa instalacja.
- Pyright: dobry checker; wybrano mypy ze względu na dojrzałe stuby ekosystemu.

## Ryzyka i konsekwencje

Koła CUDA, sterownik NVIDIA, `bitsandbytes` i PyTorch muszą tworzyć zgodny zestaw;
sam lockfile CPU nie gwarantuje działania GPU. `bitsandbytes` może zainstalować
się na CPU, lecz jego ścieżki kwantyzacji wymagają osobnego GPU smoke testu.
BF16 zależy od capability urządzenia, dlatego nie jest włączany na podstawie
samej deklaracji configu. Każdy kosztowny run poprzedza `doctor` i memory probe.

Opcjonalne `flash-attn` i vLLM mają silne ograniczenia wersji CUDA/PyTorch i mogą
wymagać kompilacji; pozostają poza lockiem akceptacyjnym. Aktualizacje
Transformers/TRL/PEFT należy wykonywać razem i potwierdzać smoke testami, bo ich
API trenerów oraz formaty datasetów zmieniają się szybciej niż core projektu.
