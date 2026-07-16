# Task 00 — Bootstrap repozytorium i odtwarzalność

> [Centralny rejestr zadań i statusów](README.md). Każda zmiana statusu lub zakresu tego zadania musi aktualizować rejestr w tym samym commicie.

## Status

`DONE`

## Cel

Utworzyć szkielet projektu, środowisko, konfiguracje, CLI, testy i zasady rejestrowania eksperymentów. Ten task nie pobiera pełnych danych ani Bielika i nie uruchamia kosztownego treningu.

## Zależności

Brak.

## Wymagane rezultaty

1. `pyproject.toml` z Pythonem 3.11 lub wersją zgodną ze stosem GPU.
2. Lockfile wygenerowany przez `uv`.
3. Pakiet `src/doc2query` i entrypoint `doc2query`.
4. Hydra/OmegaConf albo równoważny system hierarchicznych konfiguracji.
5. `Makefile` lub `justfile` z komendami:
   - `setup`;
   - `lint`;
   - `typecheck`;
   - `test`;
   - `smoke`;
   - `data-audit`;
   - `train-sft`;
   - `eval-generator`.
6. Moduł `utils/reproducibility.py` ustawiający seedy Pythona, NumPy i PyTorch.
7. Moduł `utils/hardware.py` wykrywający GPU, BF16, CUDA, VRAM i zapisujący raport JSON.
8. Moduł `utils/tracking.py`, który zawsze zapisuje lokalny `run_manifest.json`, nawet gdy tracking online jest wyłączony.
9. Schemat konfiguracji i walidacja błędów przed rozpoczęciem runu.
10. `README.md` z instalacją CPU i GPU.
11. `.gitignore` wykluczający dane, wagi, cache HF, checkpointy i sekrety.
12. `.env.example` bez prawdziwych tokenów.
13. CI CPU uruchamiające lint, typecheck i testy.

## Zależności Python

Wybierz kompatybilne stabilne wersje i zapisz je w lockfile. Minimalne grupy:

- core: `pydantic`, `pyyaml`/`omegaconf`, `hydra-core`, `typer`, `rich`;
- data: `datasets`, `pyarrow`, `pandas`, `numpy`, `scikit-learn`, `xxhash`, `datasketch`;
- training: `torch`, `transformers`, `accelerate`, `peft`, `trl`, `bitsandbytes`;
- retrieval: `sentence-transformers`, `ranx` lub `pytrec-eval`;
- NLP: `spacy`, opcjonalnie `stanza`;
- evaluation: `sacrebleu`, `scipy`, `statsmodels` lub własny bootstrap;
- tracking: `mlflow` albo `wandb`, z trybem offline;
- quality: `pytest`, `pytest-cov`, `ruff`, `mypy` lub `pyright`.

`flash-attn`, `vllm`, `onnxruntime`, `optimum` i `openvino` umieść w extras, nie w obowiązkowej instalacji CPU.

## Publiczne CLI

Utwórz co najmniej:

```bash
doc2query doctor
doc2query config validate --config ...
doc2query data validate --config ...
doc2query train sft --config ...
doc2query train reranker --config ...
doc2query generate --config ...
doc2query preferences build --config ...
doc2query train dpo --config ...
doc2query train grpo --config ...
doc2query evaluate generator --config ...
doc2query evaluate embedder --config ...
```

Komendy nie muszą być jeszcze w pełni zaimplementowane, ale muszą mieć stabilne sygnatury, walidację configu i jasny komunikat o brakującym module.

## Testy

- test ustawiania seedów;
- test manifestu runu;
- test raportu sprzętowego bez GPU;
- test walidacji configu;
- test CLI `--help`;
- test importu wszystkich modułów;
- test, że CI nie pobiera dużego modelu.

## Kryteria akceptacji

```bash
uv sync --all-groups
uv run ruff check .
uv run pytest -q
uv run doc2query doctor
```

Wszystkie komendy przechodzą na CPU. `doctor` raportuje brak GPU jako stan wspierany, a nie wyjątek.

## Dokumentacja decyzji

Utwórz `docs/adr/0001-project-stack.md` z uzasadnieniem wyboru narzędzi i wersji. Zapisz alternatywy i ryzyka kompatybilności CUDA/bitsandbytes.
