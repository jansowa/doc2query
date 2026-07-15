# Bielik doc2query

Odtwarzalny szkielet programu badawczo-inżynieryjnego do trenowania polskiego
generatora doc2query. Aktualny zakres obejmuje infrastrukturę z Task 00 oraz
zamrożone rerankery, reward proxies i adapter `speakleash/msmarco_pl` z Task 02.
Pełny pipeline splitów z Task 01 oraz pipeline'y treningu pozostają jeszcze
niezaimplementowane.

## Wymagania

- Git;
- `uv >= 0.11`;
- CPU z systemem Linux/macOS lub GPU NVIDIA z poprawnie zainstalowanym
  sterownikiem. Python 3.11 jest instalowany i zarządzany przez `uv`.

## Instalacja CPU

```bash
uv python install 3.11
uv sync --all-groups
uv run doc2query doctor
```

Pakiet PyTorch jest pobierany z oficjalnego indeksu CPU, dzięki czemu zwykła
instalacja i CI nie wymagają CUDA ani nie pobierają modeli Hugging Face.

## Instalacja GPU

Bazowy lockfile pozostaje przenośny i CPU-only. Dla stacji GPU zainstaluj koło
PyTorch pasujące do sterownika/CUDA zgodnie z selektorem na pytorch.org, a potem
zsynchronizuj projekt bez nadpisywania środowiska albo utwórz osobne środowisko
GPU. Przykład dla obsługiwanej wersji CUDA (numer indeksu trzeba dobrać do
lokalnego sterownika):

```bash
uv sync --all-groups
uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cu128
uv run doc2query doctor --output reports/hardware.json
```

`bitsandbytes`, QLoRA i BF16 muszą zostać potwierdzone przez `doctor` oraz krótki
memory probe przed treningiem. Dostęp do checkpointów Bielika może wymagać
zaakceptowania warunków na Hugging Face; repozytorium nie omija tego mechanizmu.

## Jakość i smoke test

```bash
make lint
make typecheck
make test
make smoke
```

Równoważne kryteria akceptacji tasku 00:

```bash
uv sync --all-groups
uv run ruff check .
uv run pytest -q
uv run doc2query doctor
```

## Konfiguracja i CLI

Konfiguracje są plikami YAML walidowanymi przez Pydantic przed uruchomieniem.
Minimalny przykład znajduje się w `configs/base.yaml`:

```bash
uv run doc2query config validate --config configs/base.yaml
uv run doc2query data validate --config configs/base.yaml
uv run doc2query train sft --config configs/base.yaml
uv run doc2query train reranker --config configs/base.yaml
uv run doc2query generate --config configs/base.yaml
uv run doc2query preferences build --config configs/base.yaml
uv run doc2query train dpo --config configs/base.yaml
uv run doc2query train grpo --config configs/base.yaml
uv run doc2query evaluate generator --config configs/base.yaml
uv run doc2query evaluate embedder --config configs/base.yaml
```

Poza `doctor` i `config validate` komendy zachowują już stabilne sygnatury, ale
kończą się jasnym komunikatem, dopóki odpowiedni późniejszy task nie dostarczy
implementacji. Wyjątkiem jest wymagana przez kontrakt tasku 00 komenda
`train reranker`: pozostaje kompatybilnościowym stubem, który waliduje config,
ale zawsze odmawia treningu. Zgodnie z `AGENTS.md` task 02 integruje wyłącznie
zamrożone modele primary/shadow oraz implementuje ich benchmark i kalibrację.

Każdy przyszły run powinien utworzyć lokalny `run_manifest.json`, niezależnie od
trackingu online.

Przypięty kontrakt i sposób bezpiecznego przygotowania przykładowego zbioru
opisuje [dokumentacja `msmarco_pl`](docs/datasets/msmarco_pl.md). Źródłowe
score'y tego zbioru pochodzą z angielskich tekstów i nie są polskim rewardem.

Sekrety przechowuj wyłącznie w lokalnym `.env` (wzór: `.env.example`). Nie
commituj danych, modeli, adapterów ani checkpointów.

Decyzje stosu opisuje [ADR 0001](docs/adr/0001-project-stack.md).
