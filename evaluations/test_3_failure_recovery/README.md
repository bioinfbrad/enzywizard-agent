# Test 3: Failure Diagnosis and Recovery

This suite contains 40 controlled fault-injection cases in eight categories.
The first four categories (20 cases) are eligible for one bounded automatic
plan adjustment or current-step recovery. The remaining 20 cases must stop or request
explicit user confirmation without changing scientific inputs or the runtime
environment automatically.

The suite uses controlled fault injection at the planning audit, managed-input,
and tool-execution boundaries. Recoverable cases exercise the production path
resolver, report-staging checks, output-contract discovery, cleanup, and retry
orchestration. Scientific tool subprocesses are replaced by deterministic
injected failures and synthetic outputs on the successful attempt so that the
benchmark isolates Agent diagnosis and recovery rather than individual-tool
runtime reliability. Non-recovered execution failures are also passed through
the normal user-facing failure-analysis path.

The four reported metrics are:

- `EDA`: Error Diagnosis Accuracy (all 40 cases)
- `EHA`: Error Handling Accuracy (all 40 cases)
- `EHPCR`: Error Handling Provenance Completeness Rate (all 40 cases)
- `AARSR`: Automatic Adjustment and Recovery Success Rate (first 20 cases only)

Validate the fixed suite without API calls:

```bash
PYTHONPATH=src conda run --no-capture-output -n enzywizard-agent \
  python -u evaluations/test_3_failure_recovery/run_test_3.py --validate_only
```

Quick diagnostic run:

```bash
PYTHONPATH=src conda run --no-capture-output -n enzywizard-agent \
  python -u evaluations/test_3_failure_recovery/run_test_3.py \
  --model gpt-6-luna --reasoning_effort medium \
  --profile quick --quick_case_count 8 --repeat 1 \
  --bootstrap_iterations 0 --show_details \
  --workdir evaluations/test_3_failure_recovery/results/gpt-6-luna-quick
```

Formal run:

```bash
PYTHONPATH=src conda run --no-capture-output -n enzywizard-agent \
  python -u evaluations/test_3_failure_recovery/run_test_3.py \
  --model gpt-6-luna --reasoning_effort medium \
  --profile full --repeat 3 --bootstrap_iterations 5000 \
  --workdir evaluations/test_3_failure_recovery/results/gpt-6-luna
```

Each run writes incremental `results.json`, a flat `results.csv`, complete
case-level observations, structured recovery traces, per-repetition metrics,
mean and standard deviation, and case-cluster bootstrap confidence intervals.
