# P-05 common-cohort materializer

This is CPU-only preparation tooling. It does not run planner commands, train a
probe, generate queries, or read final tests. Real project data was not
materialized while implementing it.

`scripts/materialize_p05_cohort.py` requires explicit paths for canonical
natural pairs, W05 generations, their two fingerprint manifests, the P-04
budget, all three outputs and the output manifest. JSONL and Parquet are
supported. Omitting `--queries-per-passage` enforces K=1; supplying it requires
that exact uniform K in both the data and P-04 budget.

Each fingerprint manifest must contain `artifact_path`, the artifact `sha256`,
`splits` restricted to `train`/`dev`, and `final_tests_used: []`. The W05
manifest additionally requires `generator_id: W05-1.5B-50K-8GB` and
`source_data_sha256` equal to the canonical natural-pairs SHA-256.

Example shape (paths are intentionally placeholders, not commands to run now):

```text
python scripts/materialize_p05_cohort.py \
  --natural-pairs <train-or-dev-natural.jsonl> \
  --w05-generations <w05-train-or-dev.jsonl> \
  --natural-fingerprint <natural.fingerprint.json> \
  --w05-fingerprint <w05.fingerprint.json> \
  --budget <p04-budget.json> \
  --gold-output <gold.jsonl> \
  --synthetic-output <w05.jsonl> \
  --mixed-output <mixed.jsonl> \
  --manifest-output <materialization.json> \
  --seed 42
```

The planner then receives the three exact outputs and
`--p05-materialization-manifest`. It validates each declared path and SHA-256;
missing or drifted artifacts produce blockers and no `execution_commands`.
S00 zero/few-shot and S07 remain `required_unexecuted`.

The tooling being ready does not mean that the campaign, real materialization,
P-05/P-06, comparative probes, or final tests have been executed.
