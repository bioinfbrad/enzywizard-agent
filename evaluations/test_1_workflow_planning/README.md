# Test 1: Scientific Goal Translation and Dynamic Workflow Construction

This evaluation contains 68 English-language cases for testing translation of a
scientific request into a valid EnzyWizard workflow. It evaluates planning and,
when enabled by the runner, real end-to-end execution.

## Case categories

| Category | Cases | Distribution |
| --- | ---: | --- |
| Single-tool requests | 12 | 4 each for 1MXT, 1IU4, and 1TCA |
| Multi-tool requests | 12 | 4 each for 1MXT, 1IU4, and 1TCA |
| Parameters and constraints | 12 | 4 each for 1MXT, 1IU4, and 1TCA |
| Historical artifact reuse | 12 | 4 each for 1MXT, 1IU4, and 1TCA |
| User intervention and replanning | 12 | 4 each for 1MXT, 1IU4, and 1TCA |
| Boundary and non-tool requests | 8 | Not enzyme-balanced |

The first five categories contain different tasks rather than three copies of
the same request with the enzyme name replaced. Most requests were migrated
from the existing Test 1 and request-routing suites. Two additional multi-turn
revision cases were added to complete the balanced replanning category.

## Files

- `case_catalog.yaml` defines reusable input, plan, metric, and provenance
  profiles.
- `cases/index.yaml` fixes category order and expected counts.
- `cases/01_*.yaml` through `cases/06_*.yaml` contain the individual cases.
- `resources/resource_manifest.csv` records the three representative enzymes,
  mutations, substrates, and local resource paths.
- `resources/enzymes/` contains fixed WT/MUT structures, sequences, and MSAs.
- `resources/history_fixtures/` contains successful historical artifacts used
  to construct reproducible prior turns.

All paths stored in the case definitions are relative to this directory. The
source structures are successful cleaned runtime-evaluation resources, but are
treated as supplied input structures; an expected workflow still includes
`enzywizard-clean` or `enzywizard-mut-clean` where required.

Input profiles contain only fields accepted by `AgentInputs`; sequence files
are not passed as direct Agent inputs. Each plan profile defines an exact
tool-call multiset through `expected_calls`, including WT/MUT/shared side, and
partial-order dependencies through `order_constraints`. RAC expectations are
resolved only from the fixed `required_artifact_groups`, whose entries include
artifact type, side, and logical count. They are never inferred from the plan
produced by the Agent under evaluation. RAC includes every required non-log
output declared by each expected tool. Runtime logs are evaluated through
process provenance rather than counted as scientific result artifacts.

## Metric applicability

Workflow cases use TSA, PCVR, WOVR, RAC, E2E-SR, and PPCR. Boundary and
non-tool cases use TSA only: their expected tool-call multiset is empty, and
the remaining metrics are not applicable and are excluded from denominators.

For revision cases, TSA, PCVR, and WOVR are evaluated for both the initial and
final plans. RAC and E2E-SR are evaluated only for the final executed plan.
PPCR covers the initial-plan record, revision request, plan delta, and final
approval record.

PPCR requirements are selected through `provenance_profile`. Recovery-action
provenance is deliberately excluded from Test 1 and belongs to Test 3.

## Runner

Validate all 68 case definitions and local resource references without an API
call:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
python evaluations/test_1_workflow_planning/run_test_1.py --validate_only
```

Run a reproducible planning-only sample. This records final approval for PPCR
but deliberately does not execute tools, so RAC and E2E-SR are not applicable:

```bash
export ENZYWIZARD_API_KEY="YOUR_API_KEY"
PYTHONPATH=src \
python evaluations/test_1_workflow_planning/run_test_1.py \
  --profile quick \
  --quick_case_count 8 \
  --seed 20261110 \
  --repeat 3 \
  --show_plans
```

Run the formal full evaluation with real tool execution:

```bash
export ENZYWIZARD_API_KEY="YOUR_API_KEY"
PYTHONPATH=src \
python evaluations/test_1_workflow_planning/run_test_1.py \
  --profile full \
  --repeat 3 \
  --execute \
  --show_plans \
  --bootstrap_iterations 5000
```

The default model configuration is `gpt-6-luna` with reasoning effort
`medium`; every provider, model, endpoint, and reasoning setting can be
overridden through command-line options. Results are written incrementally as
JSON and as a flat per-run CSV. The JSON summary reports every repetition,
mean, sample standard deviation, and a case-cluster bootstrap 95% confidence
interval for each applicable metric.

The default uses 1,000 bootstrap replicates, which is sufficient for routine
evaluation of this 68-case suite. Use `--bootstrap_iterations 0` during rapid
debugging, or 5,000 replicates for the final manuscript tables. Increasing the
replicate count reduces Monte Carlo noise in the interval estimate but does not
increase the effective sample size.
