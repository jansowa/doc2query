.PHONY: setup lint format typecheck test smoke data-audit train-sft eval-generator

CONFIG ?= configs/base.yaml

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
	uv run doc2query train sft --config $(CONFIG)

eval-generator:
	uv run doc2query evaluate generator --config $(CONFIG)
