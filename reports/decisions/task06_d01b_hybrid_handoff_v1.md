# ADR: confirmed D01b Hybrid handoff to Task 06

Date: 2026-08-12

Status: accepted by the owner after the external-development confirm

## Evidence and decision

The owner accepts the completed D01b 4.5B TriviaQA development-confirm and
authorizes preparation of the next prospective Task 06 stage. The confirm
compared matched-budget probe embedders trained on W06-only data versus data
selected by the D01b Hybrid procedure. Hybrid passed the frozen primary gate
and all guardrails and remains positive in the labelled post-hoc 42+44 seed
diagnostic. W06 seed 43 non-convergence remains a mandatory stability caveat.

Freeze the following roles:

- **Task 06 data-generation procedure:** form a candidate pool from the pinned
  W06 uncontrolled adapter and the pinned D01 controlled adapter, then apply
  the pinned safe-anchor selector. W06 supplies the four-query safety anchor;
  D01 supplies controlled form/intent diversity. The selector chooses four of
  eight candidates without using the reserved shadow judge.
- **Task 07 optimization start:** use the pinned D01 controlled 4.5B SFT
  adapter. This preserves the explicit controls that Task 07 is intended to
  improve. W06 remains an anchor and candidate source, not a second set of
  weights merged into the DPO model.

This is a handoff decision, not a claim that the confirm compared alternative
DPO starting checkpoints. Task 07 must retain continued-SFT and score-weighted
continued-SFT controls and must evaluate whether DPO improves over them.

## Frozen identities

Both adapters share `speakleash/Bielik-4.5B-v3.0-Instruct` at revision
`4b1220a9d745bdd874c44347075ef25484ef322b`, with
`trust_remote_code=false`:

- W06 anchor: `runs/W06-4.5B-INSTRUCT-50K-8GB-BS8-L512/adapter`, artifact
  fingerprint `9253810026385c8749bfbb4de9b3520e1b0a73fd16020c98e728d8ff405d73e2`;
- D01 controlled and Task 07 start:
  `runs/D01-4.5B-STYLE-50K-S42/adapter`, artifact fingerprint
  `71937228ea977d9d6a89613fe6f802fc3711dba9499a8e23c6c1e4e21e77a867`.

The selector remains the implementation and contract already pinned by the
scale pilot: 8 candidates, 4 selected, all subsets enumerated, four W06 slots
as the safety anchor, no loss in grounding/format feasibility, no increase in
copy-risk count, fixed objective weights and deterministic identity tie-break.

## Prospective boundary

This ADR authorizes only a model-free handoff preflight and preparation of a
separate Task 06 candidate-generation design. It does not yet choose a Task 06
train/dev passage cohort, candidate count, decoding matrix, calibration data,
score normalization, weights, thresholds, human-audit panel or execution
budget. Therefore it does not authorize model loading, candidate generation,
scoring, preference selection, DPO training, Task 09 promotion or final-test
access.

The next Task 06 ADR must be accepted before any expensive execution and must
prospectively pin the non-test cohort and leakage exclusions, K=4–8 request
matrix, at least two generation seeds, primary/shadow revisions, component
calibration, human evidence, output namespaces, crash recovery and one
explicit operator command. TriviaQA remains internal-evaluation-only and must
not become Task 06 training data.

Mandatory state after this handoff:

- `task06_handoff_approved=true`;
- `task06_generation_authorized=false`;
- `task06_scoring_authorized=false`;
- `task07_training_authorized=false`;
- `task09_promotion_authorized=false`;
- `four_point_five_b_full_authorized=false`;
- `final_tests_used=[]`.
