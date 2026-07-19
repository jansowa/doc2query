# Task 03 — resumable Bielik 1.5B base technical queue

## Scope

This queue prepares technical, single-factor QLoRA measurements for
`speakleash/Bielik-1.5B-v3` at pinned revision
`4b25049621bf3952a1fc9314c89773102eda0333`. It does not compare base against
instruct, select a final generator, open a final test, or start DPO/GRPO.

The existing completed runs already cover:

| Run | Technical factor |
|---|---|
| W01 | 10k, LR 1e-4, seed 42 |
| W02 | 10k, LR 5e-5, seed 42 |
| W03 | 10k, LR 2e-4, seed 42 |
| W04 | 10k, LR 1e-4, seed 43 |
| W05 | 50k baseline, LR 1e-4, seed 42 |

They are reused and never restarted.

## Queue order

1. Materialize the exact generator, probe and primary-judge snapshots in the
   project Hugging Face cache. An exact snapshot already present in the user
   cache is copied; otherwise the authenticated Hugging Face CLI downloads the
   pinned revision into the project partition.
2. Complete the dev-only W05 P-03 HN0/HN0+filter/HN1 sensitivity runner.
3. Measure the missing 768/1024 memory probes.
4. Run seven resumable 10k single-factor technical ablations against W03:
   length 768, length 1024, rank 16, rank 32, attention-only LoRA, effective
   batch 32 and LoRA dropout 0.

All new SFT runs use the same base revision, deterministic 10k selection,
seed 42, LR 2e-4, B1 prompt and completion-only loss. Rank runs preserve
`alpha/r=2`. The attention-only run keeps rank 8 so that target modules are
the only changed factor.

The queue intentionally does not pick two 50k finalists. Eval loss is logged
but is not an allowed final selection metric. P-04 and comparable probe
measurements must select finalists before expensive 50k/multi-seed expansion.

## Commands

Preflight the resolved configs without GPU work:

```bash
HF_HOME="$PWD/.cache/huggingface" \
UV_CACHE_DIR="$PWD/.uv-cache" \
bash scripts/run_base_1_5b_campaign.sh --dry-run
```

Start or resume the complete queue:

```bash
HF_HOME="$PWD/.cache/huggingface" \
UV_CACHE_DIR="$PWD/.uv-cache" \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_base_1_5b_campaign.sh
```

The first real invocation creates or repairs `.venv-gpu` on the project
partition. It pins the same CUDA stack recorded by the completed W03/W05/W06
runs (`torch 2.6.0+cu124`, Transformers 5.13.1, PEFT 0.19.1, TRL 0.29.1 and
bitsandbytes 0.49.2). The regular `.venv` intentionally remains CPU-only for
tests and must not be used for campaign training. Package downloads and the
environment stay under the project through `UV_CACHE_DIR` and `.venv-gpu`.

An already prepared CUDA interpreter can be selected explicitly with
`DOC2QUERY_PYTHON=/absolute/path/to/python`. The GPU preflight distinguishes a
CPU-only Torch build from a CUDA runtime unavailable to the selected process
and prints the diagnostic tail immediately on failure.

Interrupting the foreground command is safe. Re-running the same command
resumes P-03 generation/probe training, completed memory probes and the latest
compatible SFT checkpoint. The queue lock prevents two concurrent owners.

Progress is written to `reports/base_1_5b_campaign/status.tsv`; combined logs
are written to `logs/base_1_5b_campaign.log`.
