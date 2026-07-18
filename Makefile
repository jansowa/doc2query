.PHONY: setup lint format typecheck test smoke data-audit train-sft memory-probe eval-generator freeze-eval

CONFIG ?= configs/base.yaml
RESUME ?= --resume-if-available
EVAL_MANIFEST ?= data/processed/v1/evaluation/task04-v1/manifest.json
EVAL_OUTPUT ?= reports/evaluation/manual
PRIMARY_JUDGE ?= configs/reranker/primary_polish_roberta_v3.yaml

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
	uv run doc2query evaluate generator --config $(CONFIG) \
		--frozen-manifest $(EVAL_MANIFEST) --output-dir $(EVAL_OUTPUT) \
		--primary-judge $(PRIMARY_JUDGE)

freeze-eval:
	uv run python scripts/freeze_evaluation_sets.py
