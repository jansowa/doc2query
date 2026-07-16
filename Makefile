.PHONY: setup lint format typecheck test smoke data-audit train-sft memory-probe eval-generator

CONFIG ?= configs/base.yaml
RESUME ?= --resume-if-available

setup:
	uv sync --all-groups

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy

test:
	uv run pytest -q

smoke:
	uv run doc2query doctor
	uv run doc2query config validate --config $(CONFIG)

data-audit:
	uv run doc2query data validate --config $(CONFIG)

train-sft:
	uv run doc2query train sft --config $(CONFIG) $(RESUME)

memory-probe:
	uv run python scripts/run_memory_probe.py --config $(CONFIG) --lengths 512 768 1024

eval-generator:
	uv run doc2query evaluate generator --config $(CONFIG)
