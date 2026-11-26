# Test 1 — Workflow Planning and Execution

The benchmark contains 60 workflow cases. Its `cases/`, `resources/`, and
`case_catalog.yaml` are self-contained. The execution harness calls the
Agent's routing, history, planning, confirmation, and tool execution nodes.

Each case lists its English question, any follow-up question, a descriptive
reference answer for each accepted workflow, and an example of its expected
trajectory. Those examples show one valid ordering. The scoring contract in
the same case declares the milestone DAG, minefields, and SR checks. The
reference answer and example trajectory are for review. The scorer checks
recorded events against the declared scoring contract.

## Review the cases

- [Single-tool requests](cases/01_single_tool_requests.yaml)
- [Multi-tool requests](cases/02_multi_tool_requests.yaml)
- [Parameters and constraints](cases/03_parameters_constraints.yaml)
- [Historical artifact reuse](cases/04_historical_reuse.yaml)
- [Plan revision](cases/05_plan_revision.yaml)

Within each case, `after` lists the milestone dependencies. The example
trajectory describes what a successful run can look like. The run's
`trajectory.json` records what the Agent actually did.

## Scoring

- **Avg Score** uses one declared milestone DAG for each accepted reference workflow.
  A milestone earns 1 for an exact match of its required structured fields.
  The scorer maximizes the number of matched milestones over distinct trajectory
  events subject to the DAG order. Unmatched milestones earn 0. A forbidden
  tool in a plan, extra tool execution, wrong historical input, or execution of
  a revoked plan sets the score for that alternative to 0. The highest score
  across the accepted alternatives is the case score.
- **SR** checks the final execution and task-specific output requirements,
  including expected artifact type, side, producer, file validity, and any
  required scientific settings or reuse. A case earns 1 if an accepted
  alternative meets every success criterion. A missing or failed execution is
  scored 0 in an execution run. SR is independent of partial planning credit.

Plan-only runs report the diagnostic **Plan Score**. Use `--execute` for
Avg Score and SR. The evaluator uses structured programmatic checks throughout.
The milestone approach follows
[ToolSandbox](https://aclanthology.org/2025.findings-naacl.65/) and the
task-specific binary success evaluation follows
[ScienceAgentBench](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f12b4df26344f3be803c06b555252efe-Abstract-Conference.html).

## Run

```bash
PYTHONPATH=src python benchmarks/test_1_workflow_planning/run_test_1.py \
  --validate_only

PYTHONPATH=src python benchmarks/test_1_workflow_planning/run_test_1.py \
  --model gpt-6-luna --reasoning_effort medium --ontology_mode retrieve \
  --profile full --execute --repeat 3 --bootstrap_iterations 1000
```

Use the existing `ENZYWIZARD_API_KEY` and scientific-tool environment. The
default output root is `~/enzywizard-benchmark-results/test_1_workflow_planning`.
The runner saves the active scoring contracts, per-case trajectories, scientific
artifacts, planning diagnostics, and metric summaries. Initial history fixtures
are recorded as inputs. Execution scoring uses new tool calls and their output
artifacts. The graph ends at tool execution. The final state records whether the
workflow succeeded and which artifacts it produced. The separate user-facing
explanation generated in a normal Agent session is outside this benchmark.

Each repetition is macro-averaged over the cases. The summary reports the mean
and sample standard deviation of the repetition scores. Bootstrap samples case
IDs with replacement, preserves each sampled case's repetitions, and computes
the mean across repetition scores for each sample.
