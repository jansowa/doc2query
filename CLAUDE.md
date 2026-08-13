# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A reproducible research program (in Polish) for training a Polish doc2query generator (Bielik models) to produce synthetic queries that improve a downstream embedder. It is **not** a single training project — it is a gated experiment campaign with strict research-integrity rules.

Two documents govern all work and must be read before making changes:

- **`AGENTS.md`** — the authoritative instruction set: methodology, data contract, reward design, VRAM budget, experiment matrix, phase gates, and research-safety rules.
- **`tasks/README.md`** — the central task registry and the *only* source of truth for task status and ordering (`00 → 01 → … → 11`). Any change to a task's state must, in the same commit, update the table there, the `Status` section of the task file, and describe remaining runs/gates.

## Commands

Everything runs through `uv` (Python 3.11, CPU-only lockfile; PyTorch comes from the CPU index by default).

```bash
uv sync --all-groups          # install (make setup)
make lint                     # uv run ruff check .
make format                   # uv run ruff format .
make typecheck                # uv run mypy   (strict mode, covers src + tests)
make test                     # uv run pytest -q
make smoke                    # doc2query doctor + config validate
```

Run a single test: `uv run pytest tests/test_d01_probe.py -q` or `uv run pytest -k <pattern> -q`.

CI (`.github/workflows/ci.yml`) runs ruff, mypy, pytest, and `doc2query doctor` on CPU with `HF_HUB_OFFLINE=1` — tests must never download models or require a GPU. Use tiny fixtures/mocks; GPU-only tests get the `gpu` pytest marker.

### CLI

The package exposes a Typer app (`src/doc2query/cli.py`), always driven by a Pydantic-validated YAML config:

```bash
uv run doc2query doctor
uv run doc2query config validate --config configs/base.yaml
uv run doc2query data validate --config configs/base.yaml
uv run doc2query train sft|dpo|grpo --config ...
uv run doc2query generate --config ... [--adapter ...]
uv run doc2query preferences build --config ...
uv run doc2query evaluate generator|embedder --config ...
```

Some subcommands are deliberate stubs until their task delivers the implementation; `train reranker` is a compatibility stub that always refuses (rerankers are frozen judges, never trained here).

## Architecture

- **`src/doc2query/`** — the library. Pipeline stages are separated by module: `data/` (validate → normalize → dedup → split → invert to doc2query pairs, style/focus labels), `models/` (generator loading, LoRA, prompt templates), `reranker/` (frozen primary/shadow judges: load, infer, benchmark, calibrate), `rewards/` (independent, testable components: lexical, grounding, diversity, style, composite), `training/` (SFT, weighted SFT, DPO, GRPO — QLoRA on a 16 GB VRAM budget), `generation/` (controlled candidate generation + dedup), `preferences/` (candidate scoring → DPO pair construction), `evaluation/` (generator intrinsics, retrieval, probe embedder — the *primary* metric, bootstrap CIs, slices), `schemas.py` + `config.py` (Pydantic data/config contracts).
- **`configs/`** — YAML configs by concern (data, model, reranker, train, generation, rewards, evaluation, preferences, experiments). Every experiment has its own config, ID, and seed.
- **`scripts/`** — experiment runners, preflights, audits, and one-off gated procedures (e.g. `run_d01b_*`, `preflight_task06_*`). These encode prospective, preregistered experiment protocols — do not "fix" or rerun completed ones (the registry marks several as must-not-rerun).
- **`tasks/`** — one file per task with acceptance criteria; **`reports/`** (`decisions/`, `measurements/`, `preregistrations/`) — decision records and measured results; **`docs/`** — per-task documentation and ADRs.
- **`data/`, `artifacts/`, `runs/`** — local data, adapters, run outputs; never committed.

## Research-integrity rules (enforced, not aspirational)

- **Never report an experiment result that was not actually run.** Statuses (`TODO`/`IN PROGRESS`/`IMPLEMENTED`/`DONE`/`BLOCKED`/`OPTIONAL`) must be backed by artifacts; `IMPLEMENTED` ≠ `DONE` (code ready vs. gates and measurements passed).
- Frozen things stay frozen: test sets (never touched after viewing results; `final_tests_used=[]` until finalists are frozen), splits, reranker weights, and the `source_en_score >= 23.50` threshold on `msmarco_pl` positives (no mass rescoring with local rerankers, no local threshold, no drop/weighted variants without a new prospective ADR).
- The final criterion is the probe-embedder result on natural frozen queries, not surface metrics (reranker score, overlap) of the generator.
- Gates open work, not implementation completeness: Tasks 08–10 are `BLOCKED`/`OPTIONAL` until their recorded decision documents exist. Check `tasks/README.md` before starting anything experimental — many runs are explicitly closed or one-shot.
- No large data or model weights in git; secrets only in local `.env`.

Repository documentation, reports, and commit-facing prose are written in Polish.
