# Test 3 — Fault Diagnosis and Automatic Recovery Evaluation

The benchmark retains 40 controlled fault cases across eight categories. All
40 cases contribute to Fault Handling Avg Score (FH Avg Score). The 20 cases
designated as automatically recoverable also contribute to Automatic Recovery
Avg Score (AR Avg Score).

Each case file contains the injected fault, an English reference answer, the
expected trajectory, a milestone DAG, and minefields. The reference answer
and expected trajectory describe the rubric for human review. The scorer
matches the structured rules in `scoring` against ordered Agent events.

## Execution and scoring

The evaluator calls the existing Agent planner, recovery policy, argument
resolver, workspace operations, and failure analysis. Controlled call fixtures
substitute for scientific-tool subprocesses. They exercise recovery on
predictable failures and allow the evaluator to check the repaired reference,
restaged report, or required synthetic output. Plan cases verify the repaired
plan against the original workflow, route it through production confirmation,
and observe tool dispatch. Tool fixtures verify the original output contract
and the report staging sides. A historical fixture selects a prior-turn
artifact through Agent history selection and registers it as a current-plan
fixture input with its origin recorded. The managed-input safety case checks
the production repair guard. The cleanup safety case observes actual cleanup
calls and verifies that a disposable upstream artifact remains intact.

Each milestone counts once. The scorer selects a maximum-cardinality match
respecting the DAG's strict order. A matched minefield sets the applicable case
scores to zero. AR uses the same ordered trajectory, dependency constraints
and minefields, with its denominator limited to the milestones marked `ar`.
A safe stop or a request for confirmation can earn full FH credit on a case
whose rubric requires that outcome. Such cases have no AR score.

Formal runs use three repetitions. For each metric, the evaluator reports
the mean and sample standard deviation of the three repetition scores. A
percentile bootstrap resamples eligible case IDs with replacement and retains
the repetitions together for each case. Confidence intervals are produced for
completed runs. The approach adapts the milestone and minefield method from
[ToolSandbox](https://doi.org/10.18653/v1/2025.findings-naacl.65).

## Run

From the Agent checkout, set `ENZYWIZARD_API_KEY` to the model's API key.
The evaluator applies structured scoring. Any model-assisted diagnosis is
performed by the Agent using the model under test.

```bash
PYTHONPATH=src conda run --no-capture-output -n enzywizard-agent \
  python -u benchmarks/test_3_failure_recovery/run_test_3.py --validate_only

PYTHONPATH=src conda run --no-capture-output -n enzywizard-agent \
  python -u benchmarks/test_3_failure_recovery/run_test_3.py \
  --model gpt-6-luna --profile quick --repeat 1 --bootstrap_iterations 0

PYTHONPATH=src conda run --no-capture-output -n enzywizard-agent \
  python -u benchmarks/test_3_failure_recovery/run_test_3.py \
  --model gpt-6-luna --profile full --repeat 3 --bootstrap_iterations 10000 \
  --workdir "$HOME/enzywizard_benchmark_results/test_3_failure_recovery/gpt-6-luna"
```

Use `--case_ids` to select specific cases. Each `run_*` stores
`case_contracts.json`, per-case `trajectory.json`, incremental
`results.json`, and final `results.csv`. Use the summary's
`completed_all_case_runs` field to distinguish complete runs from early stops.
