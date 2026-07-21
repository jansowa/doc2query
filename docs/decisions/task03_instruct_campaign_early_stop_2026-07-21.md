# Task 03 decision: early-stop the 1.5B Instruct sweep

Date: 2026-07-21

Status: accepted for the technical queue; no generator winner selected.

## Decision

Interrupt I02 (`10k`, LR `5e-5`) and remove it from automatic execution.
Keep only I03 (`10k`, LR `2e-4`) as the remaining Instruct training arm.
Defer I04 (seed 43) until a candidate passes downstream development
screening, and defer I05 (`50k`) until the Instruct learning rate is selected
and the P-04 dev screen justifies expansion.

The queue must print deferred arms without constructing or executing their
training commands. S00 and S07 remain required and unexecuted. No final split
was opened for this decision.

## Evidence available at the decision point

All values below are completion-only eval loss on the same frozen dev input.
They are screening evidence, not the primary generator or embedder metric.

| Arm | Factor | Eval loss |
|---|---|---:|
| W02 | base, LR `5e-5` | 1.2914 |
| W01 | base, LR `1e-4`, seed 42 | 1.2640 |
| W04 | base, LR `1e-4`, seed 43 | 1.2595 |
| W03 | base, LR `2e-4` | 1.2505 |
| B05 | attention-only LoRA | 1.2929 |
| B06 | effective batch 32 | 1.2596 |
| B07 | dropout 0 | 1.2638 |
| I01 | Instruct, LR `1e-4` | 1.2225 |

Length 768 and 1024 produced 1.2488 and 1.2508, respectively, and therefore
did not materially improve W03 at length 512. Rank 16 and 32 reduced eval
loss to about 1.2416 and 1.2408, but this small technical gain does not
justify expanding the current Instruct matrix before downstream screening.

The known base seed difference at LR `1e-4` is about 0.0046, while I01 is
about 0.0415 below the matched W01. This makes I03 a useful final LR bracket,
but it does not establish that Instruct improves retrieval. On the shared
100-example greedy panel, normalized exact match was 7% for W01 and 9% for
I01; both had 100% format validity. The panel is too small and too intrinsic
to authorize a 50k expansion.

## Cost and stopping rule

I01 took about 2.18 GPU-hours. I02, I03 and I04 have similar expected cost;
I05 would add roughly 9–11 GPU-hours. Running only I03 saves approximately
13–15 GPU-hours relative to finishing I02/I03/I04/I05.

After I03, compare I01 and I03 using development-only intrinsic/retrieval
artifacts. Do not choose a final generator from eval loss. A 50k Instruct run
or seed expansion requires a favorable P-04-compatible development screen.
The first P-05 comparison remains the budget-matched natural-only, W05
synthetic-only and natural+synthetic 50/50 cohort. Comparative probes, P-05,
P-06 and final tests remain unexecuted at this decision point.

## Research-safety statement

This is a resource-allocation decision, not a result claim. It uses only
training/dev artifacts already produced by Task 03. `final_tests_used=[]`.
